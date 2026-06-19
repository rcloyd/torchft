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

Reference: HeLoCo paper (see repo docs).
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, optim

from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer
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
    Algorithm 2: Tensor-block directional correction.

    For each parameter tensor block b, computes the unit vectors of the
    incoming pseudo-gradient (û_b) and the outer momentum (v̂_b), then
    applies a three-case cosine-based correction:

      cos_b ≥ c_ok   (aligned):      Δ̂_b = Δ_b                      (Eq. 9)
      cos_b < 0      (anti-aligned):  Δ̂_b = Δ_b − β_b·cos_b·‖Δ_b‖·v̂_b  (Eqs. 10-11)
      0 ≤ cos_b < c_ok (weak):        Δ̂_b = ‖Δ_b‖ · norm(ũ_mix)   (Eqs. 12-14)

    where β_b = clamp(k_s·|cos_b|, max=β_max) · conf_b   (conf AFTER clamp, Eq. 11)
    and   ũ_mix = (1−λ_b)·û_b + λ_b·v̂_b,  λ_b = clamp(k_d·(1−cos_b)·conf_b, 1)
    and   conf_b = ‖Δ_b‖ / (‖Δ_b‖ + κ·‖m_b‖ + ε)  (Eq. 15)

    If no momentum has been accumulated (None) or either block has negligible
    norm, the pseudo-gradient is passed through unchanged (only rho is applied).

    Args:
        pseudo_grads: Per-parameter pseudo-gradient tensors Δ_b = θ̄_s − θ_H.
        momentum_buffers: Per-parameter outer momentum m_t (None = not yet init).
        rho: Arrival weight ρ applied to every corrected block.
             Use 1/√K for K concurrent workers (paper §3.2).
        c_ok: Alignment acceptance threshold (default 0.2, paper Appendix B).
        k_s: Anti-aligned shrinkage strength (default 0.5).
        k_d: Weakly-aligned rotation strength (default 1.0).
        kappa: Momentum norm scale in confidence factor κ (default 3.0).
        beta_max: Maximum shrinkage coefficient (default 0.5).
        eps: Numerical floor for division stability (default 1e-8).

    Returns:
        Dict mapping parameter names to corrected and ρ-scaled tensors with
        the same dtype as the inputs.
    """
    corrected: Dict[str, torch.Tensor] = {}

    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)

        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue

        delta_f = delta.float()
        m_f = m.float()
        norm_d = delta_f.norm()
        norm_m = m_f.norm()

        if norm_d < eps or norm_m < eps:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue

        u_hat = delta_f / norm_d
        v_hat = m_f / norm_m

        cos_b = torch.dot(u_hat.flatten(), v_hat.flatten()).item()

        # Eq. 15: confidence scales down corrections when Δ is small vs m
        conf_b = (norm_d / (norm_d + kappa * norm_m + eps)).item()

        if cos_b >= c_ok:
            corrected_block = delta_f
        elif cos_b < 0:
            # Eqs. 10-11: β = clamp(k_s·|cos|, β_max) · conf  (conf AFTER clamp)
            beta_b = min(k_s * (-cos_b), beta_max) * conf_b
            corrected_block = delta_f - beta_b * cos_b * norm_d * v_hat
        else:
            lambda_b = min(k_d * (1.0 - cos_b) * conf_b, 1.0)
            u_mix = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
            norm_mix = u_mix.norm()
            if norm_mix > eps:
                corrected_block = norm_d * u_mix / norm_mix
            else:
                corrected_block = delta_f

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
        """
        # Set HeLoCo attrs before super().__init__ launches the server thread
        self._rho = rho
        self._c_ok = c_ok
        self._k_s = k_s
        self._k_d = k_d
        self._kappa = kappa
        self._beta_max = beta_max
        self._eps = eps
        super().__init__(
            model=model,
            outer_optimizer=outer_optimizer,
            port=port,
            store_port=store_port,
            dylu_H=dylu_H,
            dylu_timeout=dylu_timeout,
            heartbeat_timeout=heartbeat_timeout,
        )

    def _lookahead_snapshot(self) -> Dict[str, torch.Tensor]:
        """
        Compute θ̄ = θ − η·μ·m for each parameter (Eq. 5).

        Must be called while self._lock is held.
        Returns plain θ for any parameter whose momentum buffer has not
        been initialized yet (before the first full-sync outer step).
        Supports per-param-group lr/momentum values.
        """
        param_to_hyper: Dict[int, Tuple[float, float]] = {}
        for group in self._outer_optimizer.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                param_to_hyper[id(p)] = (lr, mu)

        snapshot: Dict[str, torch.Tensor] = {}
        for name, p in self._model.named_parameters():
            state = self._outer_optimizer.state.get(p)
            m = state["m"] if (state and "m" in state) else None
            lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
            if m is not None and mu > 0.0:
                lookahead = (p.data.float() - lr * mu * m).to(p.dtype)
            else:
                lookahead = p.data.clone()
            snapshot[name] = lookahead.detach()
        return snapshot

    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        """
        Handle one worker sync session with HeLoCo modifications.

        Protocol (identical to AsyncDiLoCoServer — no wire changes):
          1. Worker → flag scalar: 0.0 = pull-only, 1.0 = full sync.
          2. Worker → speed scalar (DyLU).
          3. Worker → worker_id: 16-byte UUID as uint8[16].
          4. If full sync: Worker → pseudo-grads (one tensor per param).
          5. Server → look-ahead params θ̄ (one tensor per param).
          6. Server → new_steps scalar (DyLU recommendation).

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

        # 3. Worker identity (worker→server)
        wid_buf = torch.zeros(16, dtype=torch.uint8)
        pg.broadcast_one(wid_buf, root=1).wait()
        worker_id = str(uuid.UUID(bytes=bytes(wid_buf.tolist())))

        if is_full_sync:
            # 4. Receive pseudo-gradients
            pseudo_grads: Dict[str, torch.Tensor] = {}
            for name, p in self._model.named_parameters():
                buf = torch.zeros_like(p.data)
                pg.broadcast_one(buf, root=1).wait()
                pseudo_grads[name] = buf

            with self._lock:
                now = time.monotonic()

                # DyLU speed tracking (Eq. 6)
                if self._dylu_H > 0 and worker_speed > 0:
                    self._worker_speeds[worker_id] = (worker_speed, now)
                cutoff = now - self._dylu_timeout
                active = {
                    wid: (spd, ts)
                    for wid, (spd, ts) in self._worker_speeds.items()
                    if ts >= cutoff
                }
                self._worker_speeds = active
                if self._dylu_H > 0 and worker_speed > 0 and active:
                    max_speed = max(spd for spd, _ in active.values())
                    new_steps = max(1, int(worker_speed / max_speed * self._dylu_H))
                else:
                    new_steps = self._dylu_H

                # Collect outer momentum m_t BEFORE the outer step
                mom_bufs: Dict[str, Optional[torch.Tensor]] = {}
                for name, p in self._model.named_parameters():
                    state = self._outer_optimizer.state.get(p)
                    mom_bufs[name] = state["m"] if (state and "m" in state) else None

                # Algorithm 2: block correction + ρ weighting
                corrected = block_correct(
                    pseudo_grads,
                    mom_bufs,
                    rho=self._rho,
                    c_ok=self._c_ok,
                    k_s=self._k_s,
                    k_d=self._k_d,
                    kappa=self._kappa,
                    beta_max=self._beta_max,
                    eps=self._eps,
                )

                with torch.no_grad():
                    for name, p in self._model.named_parameters():
                        p.grad = corrected[name]
                self._outer_optimizer.step()
                self._outer_optimizer.zero_grad()

                # Snapshot look-ahead θ̄_{t+1} for the worker's next window
                snapshot = self._lookahead_snapshot()
        else:
            # Pull-only: send look-ahead θ̄_r for worker initialization (Eq. 5)
            with self._lock:
                snapshot = self._lookahead_snapshot()
            new_steps = self._dylu_H

        # 5. Send look-ahead params (server→worker)
        for name in self._param_names:
            pg.broadcast_one(snapshot[name], root=0).wait()

        # 6. Send DyLU recommended steps (server→worker)
        pg.broadcast_one(torch.tensor([float(new_steps)]), root=0).wait()


# Workers are standard AsyncDiLoCo — all HeLoCo logic lives on the server.
HeLoCoWorker = AsyncDiLoCo
