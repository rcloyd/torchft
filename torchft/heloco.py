# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
HeLoCo — Heterogeneity-aware Low-Communication Training
=========================================================
Extends AsyncDiLoCo with two server-side modifications:

  1. Look-ahead worker initialization (Eq. 5):
     Workers receive θ̄ = θ − η·μ·m instead of θ, so they fine-tune
     from the predicted future outer-model position.

  2. Tensor-block directional correction (Algorithm 2):
     Each incoming pseudo-gradient block is compared against the current
     outer momentum. Aligned blocks pass through; anti-aligned blocks are
     shrunk; weakly-aligned blocks are rotated toward momentum while
     preserving the original block magnitude.

The worker class (HeLoCoWorker) is AsyncDiLoCo unchanged — both HeLoCo
modifications live entirely on the server.

Reference: HeLoCo paper https://arxiv.org/pdf/2606.00271.
"""

import logging
import queue
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.profiler
from torch import nn, optim

from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer, _GraceBatch
from torchft.process_group import ProcessGroup

logger: logging.Logger = logging.getLogger(__name__)


class HeLoCoOptimizer(optim.Optimizer):
    """
    Outer optimizer implementing HeLoCo's MLA update rule (Eqs. 18-19):

      m_{t+1} = μ·m_t + (1−μ)·G_t
      θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})

    Block correction is applied by HeLoCoServer *before* setting p.grad,
    so this optimizer receives the already-corrected gradient G_t and only
    applies the plain momentum-lookahead update.

    Momentum buffers are stored as float32 regardless of parameter dtype
    to avoid precision loss during accumulation.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """
        Apply one outer step using the corrected gradient already in p.grad.

        Eqs. 18-19:
          m_{t+1} = μ·m_t + (1−μ)·G_t
          θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})
        """
        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad.detach().float()
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                m: torch.Tensor = state["m"]
                m.mul_(mu).add_(G, alpha=1.0 - mu)       # Eq. 18
                p.add_(-(G + mu * m), alpha=lr)           # Eq. 19


@torch.profiler.record_function("heloco.block_correct")
def block_correct(
    pseudo_grads: Dict[str, torch.Tensor],
    momentum_buffers: Dict[str, Optional[torch.Tensor]],
    rho: float = 1.0,
    c_ok: float = 0.2,
    k_s: float = 0.5,
    k_d: float = 1.0,
    kappa: float = 3.0,
    beta_max: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Algorithm 2: Tensor-block directional correction (paper Eqs. 9-15).

    Three cases per block b:
      cos_b ≥ c_ok  → pass through (Eq. 9)
      cos_b < 0     → shrink: Δ̂_b = Δ_b − β_b·cos_b·‖Δ_b‖·v̂_b, β_b = clamp(k_s·(-cos_b)·conf_b, β_max) (Eqs. 10-11)
      otherwise     → rotate: Δ̂_b = ‖Δ_b‖·ũ_mix/‖ũ_mix‖, ũ_mix=(1−λ_b)û_b+λ_b·v̂_b (Eqs. 12-14)
      conf_b = ‖Δ_b‖/(‖Δ_b‖+κ‖m_b‖+ε) (Eq. 15)

    No .item() calls — all math stays in tensor land, enabling PyTorch to fuse
    ops and avoid per-scalar Python/C++ round-trips.
    """
    corrected: Dict[str, torch.Tensor] = {}

    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)
        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue

        delta_f = delta.float()
        m_f     = m.float()
        norm_d  = delta_f.norm()
        norm_m  = m_f.norm()
        safe_d  = norm_d.clamp(min=eps)
        safe_m  = norm_m.clamp(min=eps)

        cos_b  = torch.dot(delta_f.flatten(), m_f.flatten()) / (safe_d * safe_m)  # Eqs. 7-8
        conf_b = norm_d / (norm_d + kappa * norm_m + eps)                          # Eq. 15

        u_hat = delta_f / safe_d
        v_hat = m_f     / safe_m

        # Anti-aligned case (Eqs. 10-11)
        beta_b     = torch.clamp(k_s * (-cos_b) * conf_b, max=beta_max)
        block_anti = delta_f - beta_b * cos_b * norm_d * v_hat

        # Weakly-aligned case (Eqs. 12-14)
        lambda_b   = torch.clamp(k_d * (1.0 - cos_b) * conf_b, max=1.0)
        u_mix      = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
        norm_mix   = u_mix.norm()
        block_weak = torch.where(
            norm_mix > eps,
            norm_d * u_mix / norm_mix.clamp(min=eps),
            delta_f,
        )

        degen = (norm_d < eps) | (norm_m < eps)
        corrected_block = torch.where(
            degen | (cos_b >= c_ok), delta_f,
            torch.where(cos_b < 0.0, block_anti, block_weak),
        )
        corrected[name] = (rho * corrected_block).to(delta.dtype)

    return corrected


class HeLoCoServer(AsyncDiLoCoServer):
    """
    Parameter server for HeLoCo distributed training.

    Extends AsyncDiLoCoServer with:
      1. Look-ahead initialization: both pull-only and post-full-sync
         responses send θ̄ = θ − η·μ·m so workers train from the
         predicted future outer position (Algorithm 1, line 3).
      2. Block correction: before each outer step, the incoming
         pseudo-gradient is passed through block_correct() to align
         it with the outer momentum (Algorithm 2).

    Wire protocol is identical to AsyncDiLoCoServer — no changes on
    the worker side are required (HeLoCoWorker = AsyncDiLoCo).

    DyLU is inherited and continues to work when dylu_H > 0.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: HeLoCoOptimizer,
        port: int = 0,
        store_port: int = 0,
        dylu_H: int = 0,
        dylu_timeout: float = 300.0,
        heartbeat_timeout: float = 15.0,
        rho: float = 1.0,
        c_ok: float = 0.2,
        k_s: float = 0.5,
        k_d: float = 1.0,
        kappa: float = 3.0,
        beta_max: float = 0.5,
        eps: float = 1e-8,
        should_quantize: bool = False,
        grace_period: float = 0.0,
    ) -> None:
        """
        Args:
            model: Global (outer) model on CPU.
            outer_optimizer: HeLoCoOptimizer bound to model.parameters().
            port: HTTP port (0 = OS-assigned).
            store_port: TCPStore port (0 = OS-assigned).
            dylu_H: DyLU maximum local steps (0 = disabled).
            dylu_timeout: DyLU worker expiry in seconds (default 300).
            heartbeat_timeout: Seconds without a heartbeat before a worker
                is considered departed (default 15).
            rho: Arrival weight ρ applied after block correction.
                 Paper recommends 1/√K for K concurrent workers.
            c_ok: Alignment threshold (default 0.2).
            k_s: Anti-aligned shrinkage strength (default 0.5).
            k_d: Weakly-aligned rotation strength (default 1.0).
            kappa: Confidence factor momentum scale κ (default 3.0).
            beta_max: Shrinkage coefficient cap (default 0.5).
            eps: Numerical floor (default 1e-8).
            should_quantize: If True, transfer parameter tensors as float16
                over the wire. Must match the worker's ``should_quantize``
                setting.
            grace_period: Seconds to wait for additional workers after the
                first pseudo-gradient arrives before applying the outer step.
                0.0 (default) disables grace-period aggregation.
        """
        # Set HeLoCo attrs before super().__init__ launches the server thread
        self._rho = rho
        self._c_ok = c_ok
        self._k_s = k_s
        self._k_d = k_d
        self._kappa = kappa
        self._beta_max = beta_max
        self._eps = eps
        self._lookahead_cache: Optional[Dict[str, torch.Tensor]] = None
        # Pre-allocated grad buffers; reused across sessions to avoid page faults.
        self._grad_buf_q: queue.Queue = queue.Queue()
        self._grad_buf_q.put(
            {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}
        )
        super().__init__(
            model=model,
            outer_optimizer=outer_optimizer,
            port=port,
            store_port=store_port,
            dylu_H=dylu_H,
            dylu_timeout=dylu_timeout,
            heartbeat_timeout=heartbeat_timeout,
            should_quantize=should_quantize,
            grace_period=grace_period,
        )

    @torch.profiler.record_function("heloco.lookahead_snapshot")
    def _lookahead_snapshot(self) -> Dict[str, torch.Tensor]:
        """Compute θ̄ = θ − η·μ·m for each parameter (Eq. 5). Must hold self._lock."""
        param_to_hyper: Dict[int, Tuple[float, float]] = {}
        for group in self._outer_optimizer.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                param_to_hyper[id(p)] = (lr, mu)

        snapshot: Dict[str, torch.Tensor] = {}
        # Group by lr*mu so _foreach_sub can process all params in one C++ call.
        scale_groups: Dict[float, List[Tuple[str, torch.nn.Parameter, torch.Tensor]]] = defaultdict(list)

        for name, p in self._model.named_parameters():
            state = self._outer_optimizer.state.get(p)
            m = state["m"] if (state and "m" in state) else None
            lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
            if m is not None and mu > 0.0:
                scale_groups[lr * mu].append((name, p, m))
            else:
                snapshot[name] = p.data.clone().detach()  # momentum not yet seeded — send raw θ

        for scale, items in scale_groups.items():
            ps_f = [p.data.float() for _, p, _ in items]
            ms   = [m for _, _, m in items]
            for (name, p, _), la in zip(items, torch._foreach_sub(ps_f, ms, alpha=scale)):
                snapshot[name] = la.to(p.dtype).detach()

        return snapshot

    @torch.profiler.record_function("heloco.apply")
    def _heloco_apply(
        self,
        grads_list: List[Dict[str, torch.Tensor]],
        speeds: List[float],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Apply HeLoCo outer steps for one or more workers' pseudo-gradients.

        For the grace-period path, ``grads_list`` contains one dict per worker
        in arrival order; updates are applied sequentially (matching paper
        Algorithm 2: θ ← sync(θ, w.update) for each w). For the single-worker
        path ``grads_list`` has exactly one element.

        Each step: clone momentum (brief lock) → block_correct (no lock) →
        assign + step + zero_grad (lock). Momentum is re-cloned between
        sequential workers so each correction uses the updated momentum.

        Returns ``(lookahead_snapshot, pool_max_speed)``.
        """
        # Update DyLU pool once for all workers in this batch
        with self._lock:
            now = time.monotonic()
            for spd in speeds:
                if self._dylu_H > 0 and spd > 0:
                    self._worker_speeds.append((spd, now))
            cutoff = now - self._dylu_timeout
            self._worker_speeds = [
                (s, ts) for s, ts in self._worker_speeds if ts >= cutoff
            ]
            pool_max_speed = max(
                (s for s, _ in self._worker_speeds), default=0.0
            )

        # Apply each worker's update sequentially so block_correct for worker i+1
        # uses the momentum updated by worker i's step (paper Algorithm 2 ordering)
        for i, grads in enumerate(grads_list):
            # Clone current momentum outside the optimizer step (brief lock)
            with self._lock:
                mom_bufs: Dict[str, Optional[torch.Tensor]] = {}
                for name, p in self._model.named_parameters():
                    state = self._outer_optimizer.state.get(p)
                    m = state["m"] if (state and "m" in state) else None
                    mom_bufs[name] = m.clone() if m is not None else None

            # Block correction outside the lock
            corrected = block_correct(
                grads,
                mom_bufs,
                rho=self._rho,
                c_ok=self._c_ok,
                k_s=self._k_s,
                k_d=self._k_d,
                kappa=self._kappa,
                beta_max=self._beta_max,
                eps=self._eps,
            )

            # Apply corrected grad and step
            with self._lock:
                with torch.no_grad():
                    for name, p in self._model.named_parameters():
                        p.grad = corrected[name]

                self._outer_optimizer.step()
                self._outer_optimizer.zero_grad()

                # Compute final lookahead cache after the last step
                if i == len(grads_list) - 1:
                    self._lookahead_cache = self._lookahead_snapshot()

        return self._lookahead_cache, pool_max_speed

    @torch.profiler.record_function("heloco.forward")
    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        """
        Handle one worker sync session with HeLoCo modifications.

        Protocol (identical to AsyncDiLoCoServer — no wire changes):
          1. Worker → flag scalar: 0.0 = pull-only, 1.0 = full sync.
          2. Worker → speed scalar (DyLU).
          3. If full sync: Worker → pseudo-grads (one tensor per param).
          4. Server → look-ahead params θ̄ (one tensor per param).
          5. Server → new_steps scalar (DyLU recommendation).

        HeLoCo differences vs AsyncDiLoCoServer (server-side only):
          - Step 5 always sends θ̄ = θ − η·μ·m, not raw θ.
          - Full sync corrects pseudo-grads via block_correct() before
            applying the outer step.
        """
        # 1. Mode flag (worker→server)
        flag = torch.zeros(1)
        pg.broadcast_one(flag, root=1).wait()
        is_full_sync = flag[0].item() > 0.5

        # 2. Worker speed for DyLU (worker→server)
        speed_buf = torch.zeros(1)
        pg.broadcast_one(speed_buf, root=1).wait()
        worker_speed = speed_buf[0].item()

        _grad_bufs: Optional[Dict[str, torch.Tensor]] = None
        if is_full_sync:
            # 3. Receive pseudo-gradients
            pseudo_grads: Dict[str, torch.Tensor] = {}
            if not self._quantize:
                try:
                    _grad_bufs = self._grad_buf_q.get_nowait()
                except queue.Empty:
                    _grad_bufs = {
                        name: torch.zeros_like(p.data)
                        for name, p in self._model.named_parameters()
                    }
            for name, p in self._model.named_parameters():
                if self._quantize:
                    buf = torch.zeros(p.data.numel(), dtype=torch.float16)
                    pg.broadcast_one(buf, root=1).wait()
                    pseudo_grads[name] = buf.float().view_as(p.data)
                else:
                    pg.broadcast_one(_grad_bufs[name], root=1).wait()
                    pseudo_grads[name] = _grad_bufs[name]

            if self._grace_period > 0.0:
                batch, i_am_processor = self._grace_accumulate_and_wait(
                    pseudo_grads, worker_speed
                )
                # _grace_accumulate_and_wait clones grads; safe to return bufs now
                if _grad_bufs is not None:
                    self._grad_buf_q.put(_grad_bufs)
                    _grad_bufs = None

                if i_am_processor:
                    # Apply each worker's update sequentially (paper Algorithm 2)
                    batch.snapshot, batch.pool_max_speed = self._heloco_apply(
                        batch.grads_list, batch.speeds
                    )
                    self._grace_batch_publish(batch)

                snapshot = batch.snapshot
                if self._dylu_H > 0 and worker_speed > 0 and batch.pool_max_speed > 0:
                    new_steps = max(
                        1, int(worker_speed / batch.pool_max_speed * self._dylu_H)
                    )
                else:
                    new_steps = self._dylu_H
            else:
                snapshot, pool_max_speed = self._heloco_apply(
                    [pseudo_grads], [worker_speed]
                )
                # _heloco_apply is done with pseudo_grads; return bufs before send
                if _grad_bufs is not None:
                    self._grad_buf_q.put(_grad_bufs)
                    _grad_bufs = None
                if self._dylu_H > 0 and worker_speed > 0 and pool_max_speed > 0:
                    new_steps = max(
                        1, int(worker_speed / pool_max_speed * self._dylu_H)
                    )
                else:
                    new_steps = self._dylu_H
        else:  # pull-only: no outer step, just return cached look-ahead
            with self._lock:
                if self._lookahead_cache is None:
                    self._lookahead_cache = self._lookahead_snapshot()
                snapshot = self._lookahead_cache
            new_steps = self._dylu_H

        # 4. Send look-ahead params (server→worker)
        for name in self._param_names:
            buf = snapshot[name].half().flatten() if self._quantize else snapshot[name]
            pg.broadcast_one(buf, root=0).wait()
        
        # 5. Send DyLU recommended steps (server→worker)
        pg.broadcast_one(torch.tensor([float(new_steps)]), root=0).wait()


# Workers are standard AsyncDiLoCo — all HeLoCo logic lives on the server.
HeLoCoWorker = AsyncDiLoCo
