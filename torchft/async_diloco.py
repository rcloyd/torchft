# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
AsyncDiLoCo — Real Asynchronous DiLoCo
=======================================
A central parameter server holds the authoritative global model weights and
applies the outer optimizer. Workers independently push pseudo-gradients
after every window of inner steps and pull updated global parameters back.

Reference: https://arxiv.org/pdf/2401.09135
"""

import dataclasses
import itertools
import json
import logging
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import parse_qs, urlparse

import torch
import torch.profiler
from torch import nn, optim

from torchft.http import _IPv6HTTPServer
from torchft.parameter_server import ParameterServer
from torchft.process_group import ProcessGroup, ProcessGroupGloo

logger: logging.Logger = logging.getLogger(__name__)

_worker_counter = itertools.count()


class DelayedNesterovOptimizer(optim.Optimizer):
    """
    Outer optimizer implementing Delayed Nesterov Momentum (DN) for async DiLoCo.

    Standard Nesterov applied to every worker push amplifies gradient staleness:
    each stale pseudo-gradient accumulates momentum in a stale direction. DN
    decouples the gradient application from the momentum correction:

      - Between milestones: each push applies ``ε · g / N`` (pure gradient,
        no momentum) so workers always receive updated params.
      - Every N-th push (milestone): additionally applies ``ε · β · m_new``
        (Nesterov correction) where ``m_new = β·m + avg_grad``.

    Total parameter change over N pushes equals ``-ε · (avg_grad + β · m_new)``,
    identical to standard Nesterov applied once to the *average* of N gradients.
    Momentum is never amplified by individual stale updates.

    Setting ``nesterov_period=1`` recovers standard Nesterov-SGD.

    **Async multi-worker note**: in a multi-worker server, ``grad_buffer``
    accumulates pushes from all workers interleaved. With heterogeneous workers
    a fast worker may dominate the buffer before a slow one contributes, making
    ``avg_grad`` at a milestone a cross-worker mixture rather than a single
    worker's window average. Set ``nesterov_period`` ≥ number of workers so
    each milestone includes at least one push per worker on average.

    Reference: Algorithm 3, "Asynchronous Local SGD" (2024),
        https://arxiv.org/abs/2401.09135
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
        nesterov_period: int = 10,
    ) -> None:
        """
        Args:
            params: Model parameters (same signature as any ``optim.Optimizer``).
            lr: Outer learning rate ε.
            momentum: Nesterov momentum coefficient β.
            nesterov_period: Number of worker pushes N between momentum
                corrections. Higher values reduce staleness amplification
                at the cost of less frequent momentum updates.
        """
        if nesterov_period < 1:
            raise ValueError("nesterov_period must be >= 1")
        defaults = dict(lr=lr, momentum=momentum, nesterov_period=nesterov_period)
        super().__init__(params, defaults)
        self._push_count: int = 0

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """
        Process one worker pseudo-gradient push.

        Expects ``p.grad`` to be set to the worker's pseudo-gradient before
        calling (same contract as ``AsyncDiLoCoServer.forward``).
        """
        self._push_count += 1

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            N: int = group["nesterov_period"]
            is_milestone: bool = (self._push_count % N == 0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]

                if "grad_buffer" not in state:
                    state["grad_buffer"] = torch.zeros_like(p)
                    # float32 avoids precision loss when accumulating momentum across many pushes
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)

                state["grad_buffer"].add_(g)

                if is_milestone:
                    avg_grad = state["grad_buffer"] / N
                    m: torch.Tensor = state["m"]
                    m.mul_(beta).add_(avg_grad)
                    p.add_(-(g / N + beta * m), alpha=lr)
                    state["grad_buffer"].zero_()
                else:
                    # Pure gradient descent: apply 1/N fraction, no momentum
                    p.add_(-g / N, alpha=lr)


@dataclasses.dataclass
class _GraceBatch:
    """One grace-period aggregation window.

    Created by the first worker to arrive. Subsequent workers that arrive
    before ``deadline`` append their pseudo-gradients to ``grads_list``.
    The first thread whose ``deadline`` has passed becomes the *processor*:
    it applies each worker's update sequentially (matching paper Algorithm 2:
    θ ← sync(θ, w.update) for each w in arrival order), fills ``snapshot``
    and ``pool_max_speed``, then calls ``notify_all`` so every other thread
    in the batch can continue.
    """
    grads_list: List[Dict[str, torch.Tensor]]  # per-worker pseudo_grads, arrival order
    speeds: List[float]                         # per-worker speeds (for DyLU pool)
    deadline: float                             # time.monotonic() deadline
    count: int = 0                              # workers accumulated so far
    claimed: bool = False                       # processor has been elected
    done: bool = False                          # results are ready
    snapshot: Optional[Dict[str, torch.Tensor]] = None
    pool_max_speed: float = 0.0                 # max speed in pool after step


class AsyncDiLoCoServer(ParameterServer):
    """
    Central parameter server for AsyncDiLoCo.

    Stores the authoritative global model weights and outer optimizer state.
    Each worker connects via a new HTTP session, performs one push-pull cycle:
      1. Worker sends pseudo-gradients (outer_params - local_params)
      2. Server applies outer optimizer step
      3. Server sends updated global params back to the worker

    Thread-safe: concurrent worker sessions serialize around the optimizer step.
    The model passed here acts as the global (outer) model and should be on CPU.
    The outer_optimizer must reference this model's parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: optim.Optimizer,
        port: int = 0,
        store_port: int = 0,
        dylu_H: int = 0,
        dylu_timeout: float = 300.0,
        heartbeat_timeout: float = 15.0,
        should_quantize: bool = False,
        grace_period: float = 0.0,
    ) -> None:
        """
        Args:
            model: The global (outer) model on CPU. Its parameters are the
                authoritative weights shared across all workers.
            outer_optimizer: Outer optimizer bound to ``model.parameters()``.
            port: HTTP port for the session endpoint (0 = OS-assigned).
            store_port: TCPStore port (0 = OS-assigned).
            dylu_H: Maximum local steps H for Dynamic Local Updates (DyLU).
                Per the paper (Eq. 6), each worker w is assigned
                ``floor(v(w) / max_{w'∈W} v(w') * H)`` steps so slower
                workers finish each window in roughly the same wall-clock
                time as the fastest worker.
                Set to 0 (default) to disable DyLU; workers keep their own
                ``sync_every`` unchanged.
            dylu_timeout: Seconds after which a worker that has not synced
                is removed from the active set W. Defaults to 300 s.
            heartbeat_timeout: Seconds without a heartbeat before a worker
                is considered departed and removed from the active set.
                Workers send heartbeats every ``heartbeat_interval`` seconds
                (configured on the worker side; default 2 s). Defaults to 15 s.
            should_quantize: If True, send/receive parameter tensors as float16
                over the wire and cast back to float32 on each side. Halves Gloo
                transfer bandwidth at the cost of ~1e-3 precision loss per
                sync. Must match the worker's ``should_quantize`` setting.
            grace_period: Seconds the server waits after the first worker
                delivers pseudo-gradients before applying the outer step.
                Workers arriving within the window have their gradients
                averaged into a single outer step (§3.3 of the paper).
                0.0 (default) disables grace-period aggregation.
        """
        self._lock = threading.Lock()
        self._model = model
        self._outer_optimizer = outer_optimizer
        self._param_names: List[str] = [n for n, _ in model.named_parameters()]
        self._quantize: bool = should_quantize
        self._grace_period: float = grace_period
        self._grace_batch: Optional[_GraceBatch] = None
        self._grace_cond: threading.Condition = threading.Condition()
        self._dylu_H: int = dylu_H
        self._dylu_timeout: float = dylu_timeout
        # DyLU speed pool: list of (v(w), timestamp) from all recent sessions.
        # Per-worker identity not needed — max across the pool is sufficient.
        self._worker_speeds: List[Tuple[float, float]] = []
        # Heartbeat registry: worker_id → last_seen monotonic timestamp
        self._heartbeats: Dict[str, float] = {}
        self._heartbeat_timeout: float = heartbeat_timeout

        # Standalone HTTP server just for heartbeats — separate port, no changes
        # to ParameterServer needed.  Handler uses a closure over `self`.
        server_ref = self

        class _HeartbeatHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                wids = qs.get("worker_id", [])
                if parsed.path != "/heartbeat" or not wids:
                    self.send_response(400)
                    self.end_headers()
                    return
                worker_id = wids[0]
                with server_ref._lock:
                    is_new = worker_id not in server_ref._heartbeats
                    server_ref._heartbeats[worker_id] = time.monotonic()
                    n = len(server_ref._heartbeats)
                if is_new:
                    logger.info(
                        f"Worker joined: {worker_id[:8]}... ({n} active)"
                    )
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress default access-log noise

        self._hb_server = _IPv6HTTPServer(("", 0), _HeartbeatHandler)
        self._hb_server.daemon_threads = True
        threading.Thread(
            target=self._hb_server.serve_forever, daemon=True
        ).start()

        # Daemon monitor that logs join/leave events and evicts stale entries
        threading.Thread(
            target=self._run_heartbeat_monitor, daemon=True
        ).start()
        super().__init__(port=port, store_port=store_port)

    def heartbeat_address(self) -> str:
        """
        Return the URL workers should send heartbeats to.

        Pass this to :class:`AsyncDiLoCo` as ``heartbeat_address``::

            server = AsyncDiLoCoServer(model, outer_opt)
            with AsyncDiLoCo(server.address(), model, inner_opt, sync_every=100,
                             heartbeat_address=server.heartbeat_address()):
                ...
        """
        port = self._hb_server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/heartbeat"

    def _run_heartbeat_monitor(self) -> None:
        """
        Background daemon: evict workers whose heartbeats have expired and
        log the departure. Runs at half the heartbeat_timeout cadence so
        departures are detected within ~1.5× heartbeat_timeout of the last
        heartbeat (same trade-off as the Lighthouse eviction loop).
        """
        while True:
            time.sleep(self._heartbeat_timeout / 2)
            departed: List[str] = []
            with self._lock:
                cutoff = time.monotonic() - self._heartbeat_timeout
                for wid, ts in list(self._heartbeats.items()):
                    if ts < cutoff:
                        del self._heartbeats[wid]
                        departed.append(wid)
                n = len(self._heartbeats)
            for wid in departed:
                logger.info(
                    f"Worker departed (no heartbeat): {wid[:8]}... ({n} active)"
                )

    def active_workers(self) -> Dict[str, float]:
        """
        Return ``{worker_id: last_seen_timestamp}`` for all workers whose
        most recent heartbeat arrived within ``heartbeat_timeout`` seconds.

        The timestamp is from ``time.monotonic()``; subtract from the current
        ``time.monotonic()`` to get the staleness in seconds.
        """
        with self._lock:
            cutoff = time.monotonic() - self._heartbeat_timeout
            return {
                wid: ts
                for wid, ts in self._heartbeats.items()
                if ts >= cutoff
            }

    def worker_count(self) -> int:
        """Number of workers currently sending heartbeats."""
        return len(self.active_workers())

    @classmethod
    def new_process_group(cls) -> ProcessGroup:
        return ProcessGroupGloo()

    def _grace_accumulate_and_wait(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
        worker_speed: float,
    ) -> Tuple[_GraceBatch, bool]:
        """Accumulate pseudo-grads into the current grace window and wait.

        Returns ``(batch, is_processor)``.  If ``is_processor`` is True the
        caller must:
          1. Fill ``batch.snapshot`` and ``batch.pool_max_speed``.
          2. Call :meth:`_grace_batch_publish` to unblock all waiting threads.
        Non-processor threads return only after ``batch.done`` is True.
        """
        i_am_processor = False
        with self._grace_cond:
            now = time.monotonic()
            if self._grace_batch is None or self._grace_batch.done:
                self._grace_batch = _GraceBatch(
                    grads_list=[{n: g.clone() for n, g in pseudo_grads.items()}],
                    speeds=[worker_speed],
                    deadline=now + self._grace_period,
                    count=1,
                )
            else:
                self._grace_batch.grads_list.append(
                    {n: g.clone() for n, g in pseudo_grads.items()}
                )
                self._grace_batch.speeds.append(worker_speed)
                self._grace_batch.count += 1

            batch = self._grace_batch

            while not batch.done:
                remaining = batch.deadline - time.monotonic()
                if remaining <= 0:
                    if not batch.claimed:
                        batch.claimed = True
                        i_am_processor = True
                    break
                self._grace_cond.wait(timeout=remaining)

            # Non-processor: another thread claimed it — wait for it to publish
            if not i_am_processor:
                while not batch.done:
                    self._grace_cond.wait(timeout=0.005)

        return batch, i_am_processor

    def _grace_batch_publish(self, batch: _GraceBatch) -> None:
        """Mark batch done and wake all waiting threads."""
        with self._grace_cond:
            batch.done = True
            self._grace_cond.notify_all()

    @torch.profiler.record_function("async_diloco.forward")
    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        """
        Handle one worker sync session.

        Protocol (worker is rank 1, server is rank 0):
          1. Worker → flag scalar: 0.0 = pull-only, 1.0 = full sync.
          2. Worker → speed scalar: worker's steps/sec (0.0 if unknown).
          3. If full sync: Worker → pseudo-grads (one tensor per param).
          4. Server → updated global params (one tensor per param).
          5. Server → new_steps scalar: DyLU-recommended local steps
             (0.0 means DyLU disabled — worker should keep its own sync_every).
        """
        flag = torch.zeros(1)
        pg.broadcast_one(flag, root=1).wait()
        is_full_sync = flag[0].item() > 0.5

        speed_buf = torch.zeros(1)
        pg.broadcast_one(speed_buf, root=1).wait()
        worker_speed = speed_buf[0].item()

        if is_full_sync:
            pseudo_grads: Dict[str, torch.Tensor] = {}
            wire_dtype = torch.float16 if self._quantize else None
            for name, p in self._model.named_parameters():
                if wire_dtype is not None:
                    buf = torch.zeros(p.data.numel(), dtype=wire_dtype)
                    pg.broadcast_one(buf, root=1).wait()
                    pseudo_grads[name] = buf.float().view_as(p.data)
                else:
                    buf = torch.zeros_like(p.data)
                    pg.broadcast_one(buf, root=1).wait()
                    pseudo_grads[name] = buf

            if self._grace_period > 0.0:
                batch, i_am_processor = self._grace_accumulate_and_wait(
                    pseudo_grads, worker_speed
                )

                if i_am_processor:
                    # Update DyLU pool once for all workers in the batch
                    with self._lock:
                        now = time.monotonic()
                        cutoff = now - self._dylu_timeout
                        for spd in batch.speeds:
                            if self._dylu_H > 0 and spd > 0:
                                self._worker_speeds.append((spd, now))
                        self._worker_speeds = [
                            (s, ts) for s, ts in self._worker_speeds if ts >= cutoff
                        ]
                        batch.pool_max_speed = max(
                            (s for s, _ in self._worker_speeds), default=0.0
                        )

                    # Apply each worker's update sequentially (paper Algorithm 2:
                    # θ ← sync(θ, w.update) for each w in arrival order)
                    for grads in batch.grads_list:
                        with self._lock:
                            with torch.no_grad():
                                for name, p in self._model.named_parameters():
                                    p.grad = grads[name]
                            self._outer_optimizer.step()
                            self._outer_optimizer.zero_grad()

                    with self._lock:
                        batch.snapshot = {
                            name: p.detach().clone()
                            for name, p in self._model.named_parameters()
                        }

                    self._grace_batch_publish(batch)

                snapshot = batch.snapshot
                if self._dylu_H > 0 and worker_speed > 0 and batch.pool_max_speed > 0:
                    new_steps = max(
                        1, int(worker_speed / batch.pool_max_speed * self._dylu_H)
                    )
                else:
                    new_steps = self._dylu_H
            else:
                with self._lock:
                    now = time.monotonic()
                    if self._dylu_H > 0 and worker_speed > 0:
                        self._worker_speeds.append((worker_speed, now))
                    # Expire stale entries
                    cutoff = now - self._dylu_timeout
                    self._worker_speeds = [
                        (spd, ts) for spd, ts in self._worker_speeds if ts >= cutoff
                    ]
                    if self._dylu_H > 0 and worker_speed > 0 and self._worker_speeds:
                        max_speed = max(spd for spd, _ in self._worker_speeds)
                        new_steps = max(1, int(worker_speed / max_speed * self._dylu_H))
                    else:
                        new_steps = self._dylu_H  # 0 → disabled

                    with torch.no_grad():
                        for name, p in self._model.named_parameters():
                            p.grad = pseudo_grads[name]
                    self._outer_optimizer.step()
                    self._outer_optimizer.zero_grad()
                    snapshot: Dict[str, torch.Tensor] = {
                        name: p.detach().clone()
                        for name, p in self._model.named_parameters()
                    }
        else:
            with self._lock:
                snapshot = {
                    name: p.detach().clone()
                    for name, p in self._model.named_parameters()
                }
            new_steps = self._dylu_H

        for name in self._param_names:
            buf = snapshot[name].half().flatten() if self._quantize else snapshot[name]
            pg.broadcast_one(buf, root=0).wait()

        pg.broadcast_one(torch.tensor([float(new_steps)]), root=0).wait()


class AsyncDiLoCo:
    """
    AsyncDiLoCo worker trainer.

    Wraps an inner training loop via a context manager. After every
    ``sync_every`` inner optimizer steps the worker:
      1. Computes pseudo-gradients: ``global_params - local_params``
      2. Pushes them to :class:`AsyncDiLoCoServer`
      3. Pulls the updated global parameters
      4. Resets the local model to the new global parameters

    Workers operate fully independently — no cross-worker communication.

    Example::

        server = AsyncDiLoCoServer(global_model, outer_optimizer)
        server_addr = server.address()

        with AsyncDiLoCo(server_addr, model, inner_optimizer, sync_every=100):
            for inputs, labels in dataloader:
                inner_optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                inner_optimizer.step()
    """

    def __init__(
        self,
        server_address: str,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        fragment_update_alpha: float = 0.0,
        heartbeat_address: Optional[str] = None,
        heartbeat_interval: float = 2.0,
        should_quantize: bool = False,
    ) -> None:
        """
        Args:
            server_address: HTTP address returned by
                :py:meth:`AsyncDiLoCoServer.address`.
            model: The local worker model (may live on GPU).
            inner_optimizer: Inner optimizer applied every step.
            sync_every: Number of inner steps between server syncs.
            backup_device: Device for storing the global parameter backup.
                Defaults to CPU.
            fragment_update_alpha: Blends local and global params after each
                sync: ``p = (1 - alpha) * global + alpha * local``.
                ``alpha=0`` (default): full reset to global params (standard DiLoCo).
                ``alpha=1``: keep fully local params (outer optimizer has no effect).
            heartbeat_address: URL returned by
                :py:meth:`AsyncDiLoCoServer.heartbeat_address`.
                When provided, a daemon thread pings this endpoint every
                ``heartbeat_interval`` seconds so the server can track
                which workers are active. Pass ``None`` (default) to
                disable heartbeats.
            heartbeat_interval: Seconds between heartbeat pings to the server.
                Must be well below the server's ``heartbeat_timeout``.
                Defaults to 2 s.
            should_quantize: If True, transfer tensors as float16 over the wire.
                Must match the server's ``should_quantize`` setting.
        """
        self._server_address = server_address
        self._model = model
        self._inner_optimizer = inner_optimizer
        self._sync_every = sync_every
        self._fragment_update_alpha = fragment_update_alpha
        self._quantize = should_quantize
        self._local_step = 0
        self._hooks: List[Any] = []
        self._window_start: float = 0.0
        backup = backup_device or torch.device("cpu")
        self._global_params: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                self._global_params[name] = p.detach().to(backup).clone()
        # Explicit ordered list so push/pull loops always match the server's order
        self._param_names: List[str] = list(self._global_params.keys())

        # Heartbeat: persistent daemon thread pinging the server while in context.
        # Disabled if heartbeat_address is None.
        self._heartbeat_interval = heartbeat_interval
        if heartbeat_address is not None:
            self._heartbeat_url: Optional[str] = (
                f"{heartbeat_address}?worker_id={next(_worker_counter)}"
            )
        else:
            self._heartbeat_url = None
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "AsyncDiLoCo":
        self._initial_pull()
        self._hooks.append(
            self._inner_optimizer.register_step_post_hook(self._step_post_hook)
        )
        if self._heartbeat_url is not None:
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._run_heartbeat, daemon=True
            )
            self._heartbeat_thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval * 2)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return False

    def _run_heartbeat(self) -> None:
        """
        Daemon thread: ping the server's /heartbeat endpoint every
        ``heartbeat_interval`` seconds while this worker is in context.

        Mirrors the ``_run_heartbeat`` loop in ``manager.rs`` (Lighthouse).
        Failures are logged at DEBUG and retried on the next tick — transient
        network hiccups should not kill the training loop.
        """
        assert self._heartbeat_stop is not None
        assert self._heartbeat_url is not None
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                with urllib.request.urlopen(
                    self._heartbeat_url, timeout=5.0
                ) as resp:
                    resp.read()
            except Exception as exc:
                logger.debug("Heartbeat failed (will retry): %s", exc)

    def _step_post_hook(
        self,
        _optim: optim.Optimizer,
        _args: Tuple[Any, ...],
        _kwargs: Dict[str, Any],
    ) -> None:
        self._local_step += 1
        if self._local_step >= self._sync_every:
            self._sync()
            self._local_step = 0
            self._window_start = time.monotonic()

    @torch.profiler.record_function("async_diloco.initial_pull")
    def _initial_pull(self) -> None:
        """Pull current global params from server without sending any pseudo-gradient.

        Called at __enter__ so the local model and _global_params are aligned
        with the server's authoritative weights before the first inner window starts.
        Also receives the server's DyLU H value as the initial sync_every hint.
        """
        pg = AsyncDiLoCoServer.new_session(self._server_address)
        try:
            pg.broadcast_one(torch.zeros(1), root=1).wait()  # flag=0 → pull-only
            pg.broadcast_one(torch.zeros(1), root=1).wait()  # speed=0 (unknown)
            new_global: Dict[str, torch.Tensor] = {}
            for name in self._param_names:
                ref = self._global_params[name]
                if self._quantize:
                    buf = torch.zeros(ref.numel(), dtype=torch.float16)
                    pg.broadcast_one(buf, root=0).wait()
                    new_global[name] = buf.float().view_as(ref)
                else:
                    buf = torch.zeros_like(ref)
                    pg.broadcast_one(buf, root=0).wait()
                    new_global[name] = buf
            steps_buf = torch.zeros(1)
            pg.broadcast_one(steps_buf, root=0).wait()
            new_steps = int(steps_buf[0].item())
        finally:
            pg.shutdown()

        with torch.no_grad():
            for name, p in self._model.named_parameters():
                self._global_params[name].copy_(new_global[name])
                p.data.copy_(new_global[name].to(p.device))
        # Params jumped discontinuously; stale optimizer momentum would bias
        # the first inner window. Clear it so the window starts from a clean state.
        self._inner_optimizer.state.clear()

        if new_steps > 0:
            self._sync_every = new_steps
        self._window_start = time.monotonic()

    @torch.profiler.record_function("async_diloco.sync")
    def _sync(self) -> None:
        """Push pseudo-gradients to server and pull new global params.

        Note: the outer step is committed on the server before the response is
        delivered. If the return broadcast fails, the server is one step ahead
        while _global_params remains stale, making the next sync's pseudo-gradient
        relative to the wrong baseline.
        """
        logger.info(f"AsyncDiLoCo syncing after {self._sync_every} inner steps")

        elapsed = time.monotonic() - self._window_start
        speed = self._local_step / elapsed if elapsed > 0 else 0.0

        # Snapshot local params for alpha blend (only needed when alpha > 0)
        need_local = self._fragment_update_alpha > 0.0
        pseudo_grads: Dict[str, torch.Tensor] = {}
        local_params: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                local_cpu = p.detach().cpu()
                if need_local:
                    local_params[name] = local_cpu
                pseudo_grads[name] = self._global_params[name] - local_cpu

        # Use self._param_names (fixed insertion-order list) to guarantee the
        # send and receive loops match the server's named_parameters() order.
        pg = AsyncDiLoCoServer.new_session(self._server_address)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()   # flag=1 → full sync
            pg.broadcast_one(torch.tensor([speed]), root=1).wait()
            for name in self._param_names:
                g = pseudo_grads[name]
                pg.broadcast_one(g.half().flatten() if self._quantize else g, root=1).wait()

            new_global: Dict[str, torch.Tensor] = {}
            for name in self._param_names:
                ref = self._global_params[name]
                if self._quantize:
                    buf = torch.zeros(ref.numel(), dtype=torch.float16)
                    pg.broadcast_one(buf, root=0).wait()
                    new_global[name] = buf.float().view_as(ref)
                else:
                    buf = torch.zeros_like(ref)
                    pg.broadcast_one(buf, root=0).wait()
                    new_global[name] = buf

            steps_buf = torch.zeros(1)
            pg.broadcast_one(steps_buf, root=0).wait()
            new_steps = int(steps_buf[0].item())
        finally:
            pg.shutdown()

        with torch.no_grad():
            for name, p in self._model.named_parameters():
                self._global_params[name].copy_(new_global[name])
                new_val = new_global[name].to(p.device)
                if need_local:
                    p.data.copy_(new_val)
                    p.data.lerp_(local_params[name].to(p.device), self._fragment_update_alpha)
                else:
                    p.data.copy_(new_val)

        # Params jumped discontinuously; stale optimizer momentum would bias
        # the next inner window. Clear it so the window starts from a clean state.
        self._inner_optimizer.state.clear()

        if new_steps > 0:
            logger.info(f"AsyncDiLoCo DyLU: sync_every updated {self._sync_every} → {new_steps}")
            self._sync_every = new_steps
