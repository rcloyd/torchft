# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
SemiAsyncHeLoCo
======
SemiAsyncDiLoCo with a cosine-corrected outer optimizer (SemiAsyncHeLoCoOptimizer) and
optional look-ahead model dispatch.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, optim

from torchft.semi_async_diloco import SemiAsyncDiLoCo
from torchft.manager import Manager


class SemiAsyncHeLoCoOptimizer(optim.Optimizer):
    """MLA outer optimizer with cosine-based pseudo-gradient correction."""

    def __init__(
        self,
        params,
        lr: float = 0.7,
        momentum: float = 0.9,
        cos_ok: float = 0.2,
        k_dir: float = 1.0,
        conf_c: float = 3.0,
        k_shrink: float = 0.5,
        beta_max: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, cos_ok=cos_ok,
            k_dir=k_dir, conf_c=conf_c, k_shrink=k_shrink,
            beta_max=beta_max, eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _correct_delta(
        delta: torch.Tensor,
        m: torch.Tensor,
        norm_d: torch.Tensor,
        norm_m: torch.Tensor,
        conf: torch.Tensor,
        cos_ok: float,
        k_dir: float,
        k_shrink: float,
        beta_max: float,
        eps: float,
    ) -> torch.Tensor:
        cos = torch.dot(delta.flatten(), m.flatten()) / (norm_d * norm_m + eps)
        if cos >= cos_ok:
            return delta
        v_hat = m / (norm_m + eps)
        if cos < 0:
            # Shrink: remove the anti-momentum component (paper Eq. 10-11).
            # Δ̂ = Δ − β·cos·‖Δ‖·v̂  removes only the conflicting projection.
            beta = torch.clamp(k_shrink * (-cos), max=beta_max) * conf
            return delta - beta * cos * norm_d * v_hat
        # Rotate: blend unit vectors toward momentum, preserve ‖Δ‖ (paper Eq. 12-14).
        # λ = min{k_d·(1−cos)·conf, 1} — less aligned → larger λ → more rotation.
        lam = torch.clamp(k_dir * (1.0 - cos) * conf, max=1.0)
        u_mix = (1.0 - lam) * delta / (norm_d + eps) + lam * v_hat
        norm_mix = u_mix.norm()
        if norm_mix > eps:
            return u_mix / norm_mix * norm_d
        # u_mix collapsed to zero; keep delta unchanged
        return delta

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            cos_ok = group["cos_ok"]
            k_dir = group["k_dir"]
            conf_c = group["conf_c"]
            k_shrink = group["k_shrink"]
            beta_max = group["beta_max"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    state = self.state.get(p)
                    if state is not None and "m" in state:
                        state["m"].mul_(mu)
                    continue

                delta = p.grad.detach()
                state = self.state[p]
                if "m" not in state:
                    # Float32 momentum regardless of param dtype to avoid
                    # precision loss when grads are fp32 and params are bf16.
                    state["m"] = torch.zeros_like(
                        p, dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )
                m = state["m"]

                norm_d = delta.norm()
                norm_m = m.norm()

                # Confidence: correction fades when Δ is small vs momentum
                conf = norm_d / (norm_d + conf_c * norm_m + eps)

                if norm_m > eps and norm_d > eps:
                    delta = self._correct_delta(
                        delta, m, norm_d, norm_m, conf,
                        cos_ok, k_dir, k_shrink, beta_max, eps,
                    )

                # MLA look-ahead: update m first, then apply (Δ + μm_new).
                # Using m_new (not m_old) is the defining property of MLA —
                # the param step incorporates the just-updated momentum direction.
                m.mul_(mu).add_(delta, alpha=1.0 - mu)
                p.add_(-(delta + mu * m), alpha=lr)


class SemiAsyncHeLoCo(SemiAsyncDiLoCo):
    """
    SemiAsyncHeLoCo distributed training: SemiAsyncDiLoCo with a cosine-corrected outer
    optimizer and optional look-ahead model dispatch.

    The built-in outer optimizer corrects stale pseudo-gradients before the MLA
    update based on cosine alignment with the momentum buffer:

      - Aligned      (cos ≥ cos_ok):  use Δ as-is.
      - Moderate     (0 ≤ cos < cos_ok):  rotate Δ toward momentum, preserving ‖Δ‖.
      - Anti-aligned (cos < 0):  project out the anti-momentum component.

    When ``use_lookahead=True``, after each committed outer step the local model
    is initialised at ``outer − lr·μ·m`` so workers fine-tune from a predicted
    future position rather than the bare outer params.

    ``outer_optimizer`` must not be passed — it is built internally.
    """

    def __init__(
        self,
        manager: Manager,
        model_fragments: List[nn.Module],
        inner_optimizer: optim.Optimizer,
        sync_every: int,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        cos_ok: float = 0.2,
        k_dir: float = 1.0,
        conf_c: float = 3.0,
        k_shrink: float = 0.5,
        beta_max: float = 0.5,
        use_lookahead: bool = True,
        **kwargs: Any,
    ) -> None:
        # dict.fromkeys preserves insertion order while deduplicating — a set
        # comprehension would give non-deterministic parameter ordering.
        all_params = list(dict.fromkeys(p for m in model_fragments for p in m.parameters()))
        outer_optimizer = SemiAsyncHeLoCoOptimizer(
            all_params,
            lr=outer_lr,
            momentum=outer_momentum,
            cos_ok=cos_ok,
            k_dir=k_dir,
            conf_c=conf_c,
            k_shrink=k_shrink,
            beta_max=beta_max,
        )
        super().__init__(
            manager=manager,
            model_fragments=model_fragments,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            sync_every=sync_every,
            **kwargs,
        )
        self._use_lookahead = use_lookahead

    @property
    def outer_optimizer(self) -> SemiAsyncHeLoCoOptimizer:
        """The shared SemiAsyncHeLoCoOptimizer instance (same object across all fragments)."""
        return self._fragments[0]._outer_optimizer  # type: ignore[return-value]

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        did_apply_outer_step = super()._step_post_hook(_optim, _args, _kwargs)
        if self._use_lookahead and did_apply_outer_step:
            self._apply_lookahead()

    def _apply_lookahead(self) -> None:
        """
        Shift local params to the predicted future: outer − lr·μ·m.

        Each fragment may own a distinct outer optimizer, so look up the
        matching momentum buffer and per-group lr/mu from that fragment's
        optimizer when applying the shift.
        """
        with torch.no_grad():
            for fragment in self._fragments:
                outer_opt = fragment._outer_optimizer
                # Build a param → (lr, mu) map that respects per-group values.
                param_to_hyper: dict = {}
                for group in outer_opt.param_groups:
                    lr = group["lr"]
                    mu = group["momentum"]
                    for p in group["params"]:
                        param_to_hyper[id(p)] = (lr, mu)
                for p in fragment._model_fragment.parameters():
                    state = outer_opt.state.get(p)
                    if state and "m" in state:
                        lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
                        p.data.sub_(state["m"], alpha=lr * mu)
