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
import json
import logging
import math
import os
import socket
import threading
import time
import urllib.request
import uuid
from datetime import timedelta
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
    at claim time the batch is detached from the server (late arrivals open
    a fresh batch), so the processor can safely iterate ``grads_list``. It
    applies each worker's update sequentially (matching paper Algorithm 2:
    θ ← sync(θ, w.update) for each w in arrival order), fills ``snapshot_flat``
    / ``revision`` / ``pool_speed``, then publishes so every other thread in
    the batch can continue. If processing fails, ``error`` is published
    instead so waiters fail fast rather than hanging.
    """
    grads_list: List[Dict[str, torch.Tensor]]  # per-worker pseudo_grads, arrival order
    speeds: List[float]                         # per-worker speeds (for DyLU pool)
    deadline: float                             # time.monotonic() deadline
    claimed: bool = False                       # processor has been elected
    done: bool = False                          # results (or error) are ready
    error: Optional[str] = None                 # set if the processor failed
    snapshot_flat: Optional[torch.Tensor] = None
    revision: int = 0                           # global-model revision of snapshot
    pool_speed: float = 0.0                     # DyLU pool speed after step


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

    Every committed outer step increments the server's global-model
    *revision*. Workers track the revision their pseudo-gradient is relative
    to; a push whose baseline revision is ahead of the server's (possible only
    after the server restored from an older checkpoint) is rejected and the
    worker resyncs instead of silently corrupting the outer trajectory.

    Deployment assumptions (inherited from the ParameterServer prototype):
      - **Reachability**: the session data plane (Gloo) forms point-to-point
        pairs and picks the connect direction by address comparison, so the
        server and workers must be *bidirectionally* dialable — workers
        behind NAT or asymmetric routing will hang during pair formation.
        Addresses are derived from ``socket.gethostname()``, so all hosts
        must resolve each other's hostnames.
      - **Security**: the HTTP endpoints and the Gloo/TCPStore tensor plane
        are unauthenticated plaintext. Run on a trusted, isolated network
        (VPC / WireGuard or similar); anyone who can reach the ports can
        read or poison the model.
    """

    # Explicit Gloo timeout for session process groups: a dead peer parks a
    # server thread for at most this long instead of the wrapper default.
    SESSION_TIMEOUT_SECONDS: float = 60.0

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: optim.Optimizer,
        port: int = 0,
        store_port: int = 0,
        monitor_port: int = 0,
        dylu_H: int = 0,
        dylu_timeout: float = 300.0,
        dylu_percentile: float = 0.9,
        heartbeat_timeout: float = 15.0,
        should_quantize: bool = False,
        grace_period: float = 0.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 10,
    ) -> None:
        """
        Args:
            model: The global (outer) model on CPU. Its parameters are the
                authoritative weights shared across all workers.
            outer_optimizer: Outer optimizer bound to ``model.parameters()``.
            port: HTTP port for the session endpoint (0 = OS-assigned).
            store_port: TCPStore port (0 = OS-assigned).
            monitor_port: HTTP port for the heartbeat/status endpoints
                (0 = OS-assigned). Set explicitly so the port can be
                pre-opened in firewalls / security groups and advertised
                statically.
            dylu_H: Maximum local steps H for Dynamic Local Updates (DyLU).
                Per the paper (Eq. 6), each worker w is assigned
                ``floor(v(w) / v_ref * H)`` steps (capped at H) so slower
                workers finish each window in roughly the same wall-clock
                time as the fastest workers, where ``v_ref`` is a high
                percentile of the recent speed pool (see ``dylu_percentile``).
                Set to 0 (default) to disable DyLU; workers keep their own
                ``sync_every`` unchanged.
            dylu_timeout: Seconds after which a worker that has not synced
                is removed from the active set W. Defaults to 300 s.
            dylu_percentile: Percentile of the speed pool used as the DyLU
                reference speed. Using a high percentile instead of the max
                keeps a single mis-measured outlier window from shrinking
                every worker's window until ``dylu_timeout`` expires it.
                Defaults to 0.9.
            heartbeat_timeout: Seconds without a heartbeat before a worker
                is considered departed and removed from the active set.
                Workers send heartbeats every ``heartbeat_interval`` seconds
                (configured on the worker side; default 2 s). Defaults to 15 s.
            should_quantize: If True, receive worker pseudo-gradients as
                float16 over the wire (halving upload bandwidth). The
                server→worker parameter download always stays float32:
                quantizing the authoritative params would compound error
                into every worker's baseline each sync. Must match the
                worker's ``should_quantize`` setting.
            grace_period: Seconds the server waits after the first worker
                delivers pseudo-gradients before applying the outer step.
                Workers arriving within the window have their gradients
                averaged into a single outer step (§3.3 of the paper).
                0.0 (default) disables grace-period aggregation.
            checkpoint_path: File path for periodic server-state checkpoints
                (global model, outer optimizer state, revision). If the file
                exists at construction time, state is restored from it.
                None (default) disables checkpointing.
            checkpoint_every: Outer steps between checkpoints when
                ``checkpoint_path`` is set. Defaults to 10.
        """
        self._lock = threading.Lock()
        self._model = model
        self._outer_optimizer = outer_optimizer
        self._param_names: List[str] = []
        self._param_shapes: List[torch.Size] = []
        self._param_numels: List[int] = []
        for name, p in model.named_parameters():
            self._param_names.append(name)
            self._param_shapes.append(p.shape)
            self._param_numels.append(p.numel())
        self._total_numel: int = sum(self._param_numels)

        self._quantize: bool = should_quantize
        self._grace_period: float = grace_period
        self._grace_batch: Optional[_GraceBatch] = None
        self._grace_cond: threading.Condition = threading.Condition()
        self._dylu_H: int = dylu_H
        self._dylu_timeout: float = dylu_timeout
        self._dylu_percentile: float = dylu_percentile
        # DyLU speed pool: list of (v(w), timestamp) from all recent sessions.
        # Per-worker identity not needed — a pool percentile is sufficient.
        self._worker_speeds: List[Tuple[float, float]] = []
        # Heartbeat registry: worker_id → last_seen monotonic timestamp
        self._heartbeats: Dict[str, float] = {}
        self._heartbeat_timeout: float = heartbeat_timeout

        # Global-model revision: incremented on every committed outer step.
        self._revision: int = 0
        self._applied_pushes: int = 0
        self._last_step_time: Optional[float] = None  # wall clock, for /status
        # One snapshot shared by all sessions at the same revision
        # (K concurrent syncs no longer cost K model-size clones).
        self._snapshot_cache: Optional[Tuple[int, torch.Tensor]] = None

        self._checkpoint_path: Optional[str] = checkpoint_path
        self._checkpoint_every: int = checkpoint_every
        self._last_checkpoint_revision: int = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self._shutdown_event = threading.Event()

        # Standalone HTTP server for heartbeats and the status endpoint —
        # separate port, no changes to the shared ParameterServer prototype
        # needed. Handler uses a closure over `self`.
        server_ref = self

        class _MonitoringHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/heartbeat":
                    wids = parse_qs(parsed.query).get("worker_id", [])
                    if not wids:
                        self.send_response(400)
                        self.end_headers()
                        return
                    worker_id = wids[0]
                    with server_ref._lock:
                        is_new = worker_id not in server_ref._heartbeats
                        server_ref._heartbeats[worker_id] = time.monotonic()
                        n = len(server_ref._heartbeats)
                    if is_new:
                        logger.info(f"Worker joined: {worker_id} ({n} active)")
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"ok")
                elif parsed.path == "/status":
                    payload = json.dumps(server_ref.status()).encode()
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress default access-log noise

        self._hb_server = _IPv6HTTPServer(("", monitor_port), _MonitoringHandler)
        self._hb_server.daemon_threads = True
        threading.Thread(
            target=self._hb_server.serve_forever, daemon=True
        ).start()

        super().__init__(port=port, store_port=store_port)

        # Daemon monitor that logs join/leave events and evicts stale entries
        threading.Thread(
            target=self._run_heartbeat_monitor, daemon=True
        ).start()

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

    def status_address(self) -> str:
        """URL of the JSON status endpoint (see :meth:`status`), served on
        the same port as the heartbeat endpoint."""
        port = self._hb_server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/status"

    def status(self) -> Dict[str, Any]:
        """
        Liveness/progress snapshot for external supervisors (also served as
        JSON at ``/status`` on the main HTTP port).
        """
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._heartbeat_timeout
            active = {
                wid: round(now - ts, 3)
                for wid, ts in self._heartbeats.items()
                if ts >= cutoff
            }
            return {
                "active_workers": active,  # worker_id → heartbeat staleness (s)
                "worker_count": len(active),
                "revision": self._revision,
                "applied_pushes": self._applied_pushes,
                "last_outer_step_time": self._last_step_time,
                "dylu_pool_size": len(self._worker_speeds),
            }

    def shutdown(self) -> None:
        """Stop both HTTP servers (session + heartbeat/status) and the
        heartbeat monitor, releasing their threads and sockets."""
        self._shutdown_event.set()
        self._hb_server.shutdown()
        self._hb_server.server_close()
        # The session server is owned by the ParameterServer prototype, which
        # has no teardown API; stop its serve_forever loop directly.
        self._server.shutdown()
        self._server.server_close()

    def _run_heartbeat_monitor(self) -> None:
        """
        Background daemon: evict workers whose heartbeats have expired and
        log the departure. Runs at half the heartbeat_timeout cadence so
        departures are detected within ~1.5× heartbeat_timeout of the last
        heartbeat (same trade-off as the Lighthouse eviction loop).
        """
        while not self._shutdown_event.wait(self._heartbeat_timeout / 2):
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
                    f"Worker departed (no heartbeat): {wid} ({n} active)"
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
        return ProcessGroupGloo(
            timeout=timedelta(seconds=cls.SESSION_TIMEOUT_SECONDS)
        )

    # ------------------------------------------------------------------ #
    # Checkpointing (R3)                                                  #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str) -> None:
        """
        Atomically persist the server's authoritative state: global model,
        outer optimizer state, and revision. Tensors are cloned under the
        lock; the (slow) disk write happens outside it.
        """
        with self._lock:
            state = {
                "model": {
                    k: v.detach().clone()
                    for k, v in self._model.state_dict().items()
                },
                "outer_optimizer": self._outer_optimizer.state_dict(),
                "revision": self._revision,
                "applied_pushes": self._applied_pushes,
            }
            # Optimizer state tensors are references — clone before leaving the lock
            state["outer_optimizer"] = _clone_tensors(state["outer_optimizer"])
        tmp = f"{path}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info(f"Checkpointed server state at revision {state['revision']} to {path}")

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self._model.load_state_dict(state["model"])
        self._outer_optimizer.load_state_dict(state["outer_optimizer"])
        self._revision = state["revision"]
        self._applied_pushes = state["applied_pushes"]
        self._last_checkpoint_revision = self._revision
        logger.info(f"Restored server state at revision {self._revision} from {path}")

    def _maybe_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            return
        with self._lock:
            due = (
                self._revision - self._last_checkpoint_revision
                >= self._checkpoint_every
            )
            if due:
                # Claim before saving so concurrent sessions don't double-save
                self._last_checkpoint_revision = self._revision
        if due:
            try:
                self.save_checkpoint(self._checkpoint_path)
            except Exception:
                logger.exception("periodic checkpoint failed; training continues")

    # ------------------------------------------------------------------ #
    # Flat-buffer helpers (R1: one coalesced transfer per direction)      #
    # ------------------------------------------------------------------ #

    def _unflatten(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split one flat buffer into per-parameter views (no copies)."""
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, numel in zip(
            self._param_names, self._param_shapes, self._param_numels
        ):
            out[name] = flat[offset : offset + numel].view(shape)
            offset += numel
        return out

    def _snapshot_flat(self) -> Tuple[torch.Tensor, int]:
        """
        Return ``(flat_params, revision)`` for the current global model.

        The snapshot is built at most once per revision and shared by all
        concurrent sessions — the cache is invalidated whenever an outer
        step commits.
        """
        with self._lock:
            cache = self._snapshot_cache
            if cache is not None:
                return cache[1], cache[0]
            snap = self._build_snapshot_locked()
            flat = torch.cat(
                [snap[name].detach().reshape(-1).float() for name in self._param_names]
            )
            self._snapshot_cache = (self._revision, flat)
            return flat, self._revision

    def _build_snapshot_locked(self) -> Dict[str, torch.Tensor]:
        """
        Parameters to send back to workers. Must be called with ``self._lock``
        held. Subclasses may override (e.g. HeLoCo's look-ahead shift).
        """
        return {name: p.data for name, p in self._model.named_parameters()}

    # ------------------------------------------------------------------ #
    # Outer-step application                                              #
    # ------------------------------------------------------------------ #

    def _commit_step_locked(self, grads: Dict[str, torch.Tensor]) -> None:
        """Apply one outer step. Must be called with ``self._lock`` held."""
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                p.grad = grads[name].to(p.dtype)
        self._outer_optimizer.step()
        self._outer_optimizer.zero_grad()
        self._revision += 1
        self._applied_pushes += 1
        self._last_step_time = time.time()
        self._snapshot_cache = None

    def _apply_one(self, pseudo_grads: Dict[str, torch.Tensor]) -> None:
        """
        Apply one worker's pseudo-gradient as one outer step. Subclasses may
        override to transform the gradient first (e.g. HeLoCo block
        correction) as long as they end with ``_commit_step_locked``.
        """
        with self._lock:
            self._commit_step_locked(pseudo_grads)

    def _record_speeds_locked(self, speeds: List[float]) -> None:
        """Add worker speeds to the DyLU pool. Must hold ``self._lock``."""
        if self._dylu_H <= 0:
            return
        now = time.monotonic()
        for spd in speeds:
            if spd > 0:
                self._worker_speeds.append((spd, now))

    def _pool_speed_locked(self) -> float:
        """
        DyLU reference speed: expire stale entries, then take the configured
        percentile of the pool (robust to a single outlier, unlike max).
        Must hold ``self._lock``.
        """
        cutoff = time.monotonic() - self._dylu_timeout
        self._worker_speeds = [
            (s, ts) for s, ts in self._worker_speeds if ts >= cutoff
        ]
        if not self._worker_speeds:
            return 0.0
        speeds = sorted(s for s, _ in self._worker_speeds)
        idx = math.ceil(self._dylu_percentile * (len(speeds) - 1))
        return speeds[idx]

    def _dylu_steps(self, worker_speed: float, pool_speed: float) -> int:
        """Recommended local steps for a worker (paper Eq. 6, capped at H)."""
        if self._dylu_H > 0 and worker_speed > 0 and pool_speed > 0:
            return min(
                self._dylu_H,
                max(1, int(worker_speed / pool_speed * self._dylu_H)),
            )
        return self._dylu_H  # 0 → disabled

    # ------------------------------------------------------------------ #
    # Grace-period aggregation                                            #
    # ------------------------------------------------------------------ #

    def _grace_accumulate_and_wait(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
        worker_speed: float,
    ) -> Tuple[_GraceBatch, bool]:
        """Accumulate pseudo-grads into the current grace window and wait.

        Returns ``(batch, is_processor)``.  If ``is_processor`` is True the
        caller must process the batch and call :meth:`_grace_batch_publish`
        (with an error on failure) to unblock all waiting threads. At claim
        time the batch is detached from ``self._grace_batch`` so workers
        arriving after the processor election open a fresh batch instead of
        racing the processor's iteration of ``grads_list``.

        Non-processor threads return only after the batch is published; they
        must check ``batch.error``.
        """
        i_am_processor = False
        with self._grace_cond:
            now = time.monotonic()
            if self._grace_batch is None:
                self._grace_batch = _GraceBatch(
                    grads_list=[pseudo_grads],
                    speeds=[worker_speed],
                    deadline=now + self._grace_period,
                )
            else:
                self._grace_batch.grads_list.append(pseudo_grads)
                self._grace_batch.speeds.append(worker_speed)

            batch = self._grace_batch

            while not (batch.done or batch.claimed):
                remaining = batch.deadline - time.monotonic()
                if remaining <= 0:
                    batch.claimed = True
                    # Detach: late arrivals open a fresh batch; grads_list is
                    # now safe for the processor to iterate without the lock.
                    self._grace_batch = None
                    i_am_processor = True
                    break
                self._grace_cond.wait(timeout=remaining)

            # Non-processor: another thread claimed it — wait for it to publish
            if not i_am_processor:
                while not batch.done:
                    self._grace_cond.wait()

        return batch, i_am_processor

    def _grace_batch_publish(
        self, batch: _GraceBatch, error: Optional[str] = None
    ) -> None:
        """Mark batch done (optionally with an error) and wake all waiters."""
        with self._grace_cond:
            batch.error = error
            batch.done = True
            self._grace_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Session protocol                                                    #
    # ------------------------------------------------------------------ #

    @torch.profiler.record_function("async_diloco.forward")
    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        """
        Handle one worker sync session.

        This is the canonical wire-protocol description; subclasses reuse it
        unchanged and customize behavior via :meth:`_apply_one` and
        :meth:`_build_snapshot_locked`.

        Protocol (worker is rank 1, server is rank 0). All parameter data
        moves as a single flat buffer per direction (params flattened and
        concatenated in ``named_parameters()`` order) so a sync costs a
        constant number of round trips instead of one per tensor:
          1. Worker → header ``float64[3]``:
             ``[flag (0=pull-only, 1=full sync), speed (steps/s, 0=unknown),
             baseline_revision]``.
          2. If full sync: Worker → flat pseudo-gradients
             (``float16`` if ``should_quantize`` else ``float32``).
          3. Server → header ``float64[3]``:
             ``[new_steps (DyLU recommendation, 0=disabled), revision,
             applied (1=push applied, 0=rejected → worker must resync)]``.
          4. Server → flat global params (always ``float32``).
        """
        header = torch.zeros(3, dtype=torch.float64)
        pg.broadcast_one(header, root=1).wait()
        is_full_sync = header[0].item() > 0.5
        worker_speed = header[1].item()
        baseline_revision = int(header[2].item())

        applied = False
        if is_full_sync:
            wire_dtype = torch.float16 if self._quantize else torch.float32
            flat_grads = torch.zeros(self._total_numel, dtype=wire_dtype)
            pg.broadcast_one(flat_grads, root=1).wait()
            pseudo_grads = self._unflatten(
                flat_grads.float() if self._quantize else flat_grads
            )

            with self._lock:
                stale = baseline_revision > self._revision
            if stale:
                # Only possible after this server restored from an older
                # checkpoint: the pseudo-gradient is relative to params we no
                # longer have continuity with. Reject; the worker resyncs.
                logger.warning(
                    f"Rejecting push with baseline revision {baseline_revision} "
                    f"ahead of server revision {self._revision} "
                    "(server restored from checkpoint?)"
                )
                new_steps = self._dylu_H
                snapshot_flat, revision = self._snapshot_flat()
            elif self._grace_period > 0.0:
                batch, i_am_processor = self._grace_accumulate_and_wait(
                    pseudo_grads, worker_speed
                )

                if i_am_processor:
                    try:
                        # Update DyLU pool once for all workers in the batch
                        with self._lock:
                            self._record_speeds_locked(batch.speeds)
                            batch.pool_speed = self._pool_speed_locked()

                        # Apply each worker's update sequentially (paper
                        # Algorithm 2: θ ← sync(θ, w.update) in arrival order)
                        for grads in batch.grads_list:
                            self._apply_one(grads)

                        batch.snapshot_flat, batch.revision = self._snapshot_flat()
                    except Exception as exc:
                        self._grace_batch_publish(
                            batch, error=f"{type(exc).__name__}: {exc}"
                        )
                        raise
                    self._grace_batch_publish(batch)
                    self._maybe_checkpoint()

                if batch.error is not None:
                    # Fail the session fast; the worker drops the push and
                    # resyncs rather than hanging until the Gloo timeout.
                    raise RuntimeError(
                        f"grace batch processing failed: {batch.error}"
                    )

                applied = True
                snapshot_flat, revision = batch.snapshot_flat, batch.revision
                new_steps = self._dylu_steps(worker_speed, batch.pool_speed)
            else:
                with self._lock:
                    self._record_speeds_locked([worker_speed])
                    pool_speed = self._pool_speed_locked()
                self._apply_one(pseudo_grads)
                applied = True
                snapshot_flat, revision = self._snapshot_flat()
                new_steps = self._dylu_steps(worker_speed, pool_speed)
                self._maybe_checkpoint()
        else:
            snapshot_flat, revision = self._snapshot_flat()
            new_steps = self._dylu_H

        resp = torch.tensor(
            [float(new_steps), float(revision), 1.0 if applied else 0.0],
            dtype=torch.float64,
        )
        pg.broadcast_one(resp, root=0).wait()
        pg.broadcast_one(snapshot_flat, root=0).wait()


def _clone_tensors(obj: Any) -> Any:
    """Recursively clone all tensors in a state-dict-like structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tensors(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tensors(v) for v in obj)
    return obj


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

    Fault tolerance: a failed sync never kills the training loop. The push is
    dropped, inner training continues on the current params, and the worker
    retries at subsequent window boundaries (with exponential backoff) using
    a pull-only resync once the server is reachable again.

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
        reset_inner_state: bool = False,
        resync_backoff_max: float = 60.0,
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
            should_quantize: If True, upload pseudo-gradients as float16 over
                the wire (the parameter download stays float32 — see
                ``AsyncDiLoCoServer.should_quantize``). Must match the
                server's setting.
            reset_inner_state: If True, clear the inner optimizer state after
                every sync. Standard DiLoCo persists inner AdamW state across
                windows (that persistence is load-bearing for convergence),
                so this defaults to False; enable only if you have evidence
                the reset helps for your workload.
            resync_backoff_max: Cap in seconds on the exponential backoff
                between resync attempts while the server is unreachable.
        """
        self._server_address = server_address
        self._model = model
        self._inner_optimizer = inner_optimizer
        self._sync_every = sync_every
        self._fragment_update_alpha = fragment_update_alpha
        self._quantize = should_quantize
        self._reset_inner_state = reset_inner_state
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
        self._total_numel: int = sum(
            t.numel() for t in self._global_params.values()
        )

        # Revision of the global model our params are based on (see
        # AsyncDiLoCoServer: lets the server detect pushes computed against a
        # baseline it no longer has continuity with).
        self._baseline_revision: int = 0

        # Failed-sync recovery state (see _step_post_hook)
        self._pending_resync: bool = False
        self._resync_at: float = 0.0
        self._resync_backoff: float = 1.0
        self._resync_backoff_max: float = resync_backoff_max
        # The window right after a resync started from stale params and an
        # unusual boundary — exclude it from DyLU speed measurement.
        self._skip_speed_report: bool = False

        # Heartbeat: persistent daemon thread pinging the server while in context.
        # Disabled if heartbeat_address is None.
        self._heartbeat_interval = heartbeat_interval
        # Unique per instance (uuid, not a module counter): every worker
        # process must register under a distinct id, hostname-prefixed for
        # readable logs.
        self._worker_id: str = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        if heartbeat_address is not None:
            self._heartbeat_url: Optional[str] = (
                f"{heartbeat_address}?worker_id={self._worker_id}"
            )
        else:
            self._heartbeat_url = None
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "AsyncDiLoCo":
        # Start heartbeats before the initial pull: on a large model the pull
        # is the worker's longest silent phase and it should be visible to
        # the server for all of it.
        if self._heartbeat_url is not None:
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._run_heartbeat, daemon=True
            )
            self._heartbeat_thread.start()
        try:
            self._initial_pull()
        except Exception:
            self._stop_heartbeat()
            raise
        self._hooks.append(
            self._inner_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self._stop_heartbeat()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return False

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval * 2)
            self._heartbeat_thread = None

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
        if self._local_step < self._sync_every:
            return

        if self._pending_resync:
            # Server was unreachable on a previous boundary: the dropped
            # push's window is gone, so just try to re-baseline (pull-only)
            # with backoff and keep training locally in the meantime.
            if time.monotonic() >= self._resync_at:
                self._try_resync()
        else:
            try:
                self._sync()
            except Exception as exc:
                # A transient server/network failure must not kill the
                # training loop (and must not leave a half-applied push: the
                # server may have committed our step before the return
                # transfer failed, so the push is dropped and we re-baseline
                # via a pull-only resync instead of retrying it).
                logger.warning(
                    "AsyncDiLoCo sync failed; dropping push and continuing "
                    "local training (will resync): %s",
                    exc,
                )
                self._pending_resync = True
                self._resync_backoff = 1.0
                self._resync_at = time.monotonic()

        self._local_step = 0
        self._window_start = time.monotonic()

    def _try_resync(self) -> None:
        """Attempt a pull-only re-baseline after a failed sync."""
        try:
            self._pull_global()
        except Exception as exc:
            self._resync_at = time.monotonic() + self._resync_backoff
            self._resync_backoff = min(
                self._resync_backoff * 2, self._resync_backoff_max
            )
            logger.warning(
                "AsyncDiLoCo resync failed (next attempt in %.0fs): %s",
                self._resync_at - time.monotonic(),
                exc,
            )
            return
        self._pending_resync = False
        self._resync_backoff = 1.0
        self._skip_speed_report = True
        logger.info(
            "AsyncDiLoCo resynced to server revision %d", self._baseline_revision
        )

    # ------------------------------------------------------------------ #
    # Session plumbing                                                    #
    # ------------------------------------------------------------------ #

    def _session_roundtrip(
        self, flag: float, speed: float, flat_grads: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, int, int, bool]:
        """
        One push/pull cycle against the server; see
        :meth:`AsyncDiLoCoServer.forward` for the protocol.

        Returns ``(flat_params, new_steps, revision, applied)``.
        """
        pg = AsyncDiLoCoServer.new_session(self._server_address)
        try:
            header = torch.tensor(
                [flag, speed, float(self._baseline_revision)],
                dtype=torch.float64,
            )
            pg.broadcast_one(header, root=1).wait()
            if flat_grads is not None:
                pg.broadcast_one(flat_grads, root=1).wait()

            resp = torch.zeros(3, dtype=torch.float64)
            pg.broadcast_one(resp, root=0).wait()
            flat_params = torch.zeros(self._total_numel)
            pg.broadcast_one(flat_params, root=0).wait()
        finally:
            pg.shutdown()

        new_steps = int(resp[0].item())
        revision = int(resp[1].item())
        applied = resp[2].item() > 0.5
        return flat_params, new_steps, revision, applied

    def _adopt_global(
        self,
        flat_params: torch.Tensor,
        revision: int,
        new_steps: int,
        blend_local: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Install newly pulled global params into the model and backup."""
        with torch.no_grad():
            offset = 0
            for name, p in self._model.named_parameters():
                n = p.numel()
                chunk = flat_params[offset : offset + n].view(p.shape)
                offset += n
                self._global_params[name].copy_(chunk)
                p.data.copy_(chunk.to(p.device))
                if blend_local is not None:
                    p.data.lerp_(
                        blend_local[name].to(p.device),
                        self._fragment_update_alpha,
                    )
        self._baseline_revision = revision

        if self._reset_inner_state:
            # Optional deviation from DiLoCo (which persists inner state
            # across windows); see the constructor docstring.
            self._inner_optimizer.state.clear()

        if new_steps > 0 and new_steps != self._sync_every:
            logger.info(
                f"AsyncDiLoCo DyLU: sync_every updated {self._sync_every} → {new_steps}"
            )
            self._sync_every = new_steps

    def _pull_global(self) -> None:
        """Pull current global params (flag=0) and adopt them wholesale."""
        flat_params, new_steps, revision, _ = self._session_roundtrip(
            flag=0.0, speed=0.0, flat_grads=None
        )
        self._adopt_global(flat_params, revision, new_steps)

    @torch.profiler.record_function("async_diloco.initial_pull")
    def _initial_pull(self) -> None:
        """Pull current global params from server without sending any pseudo-gradient.

        Called at __enter__ so the local model and _global_params are aligned
        with the server's authoritative weights before the first inner window starts.
        Also receives the server's DyLU H value as the initial sync_every hint.
        """
        self._pull_global()
        self._window_start = time.monotonic()

    @torch.profiler.record_function("async_diloco.sync")
    def _sync(self) -> None:
        """Push pseudo-gradients to server and pull new global params.

        Note: the outer step is committed on the server before the response is
        delivered. If the return transfer fails, the caller
        (:meth:`_step_post_hook`) drops the push and re-baselines via a
        pull-only resync — it never retries the push, so a committed-but-
        unacknowledged step can't be applied twice.
        """
        logger.info(f"AsyncDiLoCo syncing after {self._sync_every} inner steps")

        if self._skip_speed_report:
            speed = 0.0
            self._skip_speed_report = False
        else:
            elapsed = time.monotonic() - self._window_start
            speed = self._local_step / elapsed if elapsed > 0 else 0.0

        # Snapshot local params for alpha blend (only needed when alpha > 0)
        need_local = self._fragment_update_alpha > 0.0
        local_params: Dict[str, torch.Tensor] = {}
        grad_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            # self._param_names (fixed insertion-order list) guarantees the
            # flat layout matches the server's named_parameters() order.
            for name, p in self._model.named_parameters():
                local_cpu = p.detach().cpu()
                if need_local:
                    local_params[name] = local_cpu
                grad_chunks.append(
                    (self._global_params[name] - local_cpu).reshape(-1).float()
                )
        flat_grads = torch.cat(grad_chunks)
        if self._quantize:
            flat_grads = flat_grads.half()

        flat_params, new_steps, revision, applied = self._session_roundtrip(
            flag=1.0, speed=speed, flat_grads=flat_grads
        )

        if not applied:
            # Server rejected the push (stale baseline, e.g. after a server
            # checkpoint restore): treat the response as a pure resync.
            logger.warning(
                "AsyncDiLoCo push rejected by server (baseline revision %d); "
                "re-baselining to server revision %d",
                self._baseline_revision,
                revision,
            )
            self._adopt_global(flat_params, revision, new_steps)
            self._skip_speed_report = True
            return

        self._adopt_global(
            flat_params,
            revision,
            new_steps,
            blend_local=local_params if need_local else None,
        )
