# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import create_autospec, MagicMock

import torch
from torch import nn
from torch.distributed.distributed_c10d import Work

from torchft.semi_async_heloco import SemiAsyncHeLoCo, SemiAsyncHeLoCoOptimizer
from torchft.manager import Manager
from torchft.work import _DummyWork


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_manager() -> MagicMock:
    manager = create_autospec(Manager)
    manager.errored.return_value = None

    def mock_allreduce(tensor: torch.Tensor, should_quantize: bool = False) -> Work:
        return _DummyWork(tensor)

    manager.allreduce.side_effect = mock_allreduce
    return manager


class SemiAsyncHeLoCoOptimizerTest(TestCase):
    """Unit tests for SemiAsyncHeLoCoOptimizer correction logic."""

    def _make_opt(self, p: torch.Tensor, **kwargs) -> SemiAsyncHeLoCoOptimizer:
        defaults = dict(lr=0.1, momentum=0.9, cos_ok=0.2,
                        k_dir=1.0, conf_c=0.0, k_shrink=0.5, beta_max=0.5, eps=1e-8)
        defaults.update(kwargs)
        return SemiAsyncHeLoCoOptimizer([p], **defaults)

    def test_first_step_no_correction(self) -> None:
        """First step: momentum buffer is zero so no correction is applied."""
        p = torch.tensor([1.0, 0.0])
        opt = self._make_opt(p)
        p.grad = torch.tensor([-1.0, 0.0])  # anti-aligned, but m=0 → no correction
        p_before = p.data.clone()
        opt.step()
        # m was zero so no cosine correction; just the plain MLA first step
        # m_new = 0 + 0.1 * delta = 0.1 * [-1, 0]
        # p -= lr * (delta + mu * m_new) = 0.1*([-1,0] + 0.9*[-.1,0])
        m_new = (1 - 0.9) * torch.tensor([-1.0, 0.0])
        expected = p_before - 0.1 * (torch.tensor([-1.0, 0.0]) + 0.9 * m_new)
        torch.testing.assert_close(p.data, expected)

    def test_missing_grad_decays_existing_momentum(self) -> None:
        """A tensor with no grad should still have its momentum decayed on a shared outer step."""
        p_active = torch.tensor([0.0, 0.0])
        p_stale = torch.tensor([0.0, 0.0])
        opt = SemiAsyncHeLoCoOptimizer(
            [p_active, p_stale],
            lr=0.1,
            momentum=0.9,
            cos_ok=0.2,
            k_dir=1.0,
            conf_c=0.0,
            k_shrink=0.5,
            beta_max=0.5,
            eps=1e-8,
        )
        opt.state[p_active] = {"m": torch.tensor([0.0, 0.0])}
        opt.state[p_stale] = {"m": torch.tensor([2.0, -4.0])}

        p_active.grad = torch.tensor([1.0, 0.0])
        p_before = p_stale.data.clone()
        m_before = opt.state[p_stale]["m"].clone()

        opt.step()

        torch.testing.assert_close(opt.state[p_stale]["m"], 0.9 * m_before)
        torch.testing.assert_close(p_stale.data, p_before)

    def test_missing_grad_does_not_create_empty_state(self) -> None:
        """A skipped tensor with no prior state should remain absent from optimizer state."""
        p_active = torch.tensor([0.0, 0.0])
        p_skipped = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p_active)
        opt.param_groups[0]["params"].append(p_skipped)

        p_active.grad = torch.tensor([1.0, 0.0])
        p_skipped.grad = None

        opt.step()

        self.assertIn(p_active, opt.state)
        self.assertNotIn(p_skipped, opt.state)

    def test_aligned_no_correction(self) -> None:
        """Gradient well-aligned with momentum (cos ≥ cos_ok): passes through unchanged."""
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.2)
        # Pre-seed momentum in the same direction as gradient
        opt.state[p] = {"m": torch.tensor([1.0, 0.0])}
        p.grad = torch.tensor([2.0, 0.0])  # cos = 1.0 ≥ 0.2
        delta_before = p.grad.clone()
        opt.step()
        # delta should be unchanged; verify momentum updated with original delta
        m_after = opt.state[p]["m"]
        expected_m = 0.9 * torch.tensor([1.0, 0.0]) + 0.1 * delta_before
        torch.testing.assert_close(m_after, expected_m, atol=1e-6, rtol=0)

    def test_anti_aligned_shrinks_gradient(self) -> None:
        """Anti-aligned gradient (cos < 0): anti-momentum projection is reduced."""
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.2, k_shrink=0.5, beta_max=0.5, conf_c=0.0)
        opt.state[p] = {"m": torch.tensor([1.0, 0.0])}
        p.grad = torch.tensor([-1.0, 0.0])  # directly anti-aligned, cos = -1.0
        p_before = p.data.clone()
        opt.step()
        # conf_c=0 → conf=1.0; cos=-1 → beta = clamp(0.5*1.0, max=0.5)*1.0 = 0.5
        # paper formula: Δ̂ = Δ − β·cos·‖Δ‖·v̂ = [-1,0] - 0.5*(-1)*1*[1,0] = [-0.5, 0]
        # verify the gradient was shrunk: p moved less than if delta was uncorrected
        delta_uncorrected = torch.tensor([-1.0, 0.0])
        self.assertLess(
            (p.data - p_before).norm().item(),
            (delta_uncorrected * 0.1).norm().item(),
            "shrunk gradient should produce a smaller param update",
        )

    def test_moderate_misalignment_rotates_toward_momentum(self) -> None:
        """Moderately misaligned (0 ≤ cos < cos_ok): gradient rotated toward momentum."""
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        # m along x-axis, delta along y-axis → cos = 0 (perpendicular, in rotate zone)
        opt.state[p] = {"m": torch.tensor([1.0, 0.0])}
        p.grad = torch.tensor([0.0, 1.0])
        opt.step()
        # After rotation, the update should have gained an x-component (toward m)
        m_after = opt.state[p]["m"]
        self.assertGreater(
            m_after[0].item(), 0.0,
            "rotation toward momentum should give m a positive x-component",
        )

    def test_shrink_conf_applied_after_clamp(self) -> None:
        """conf scales beta AFTER the clamp, not before — catches the ordering bug."""
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.2, k_shrink=1.0, beta_max=0.3,
                             conf_c=10.0, eps=1e-8)
        # Large momentum buffer → conf is small
        opt.state[p] = {"m": torch.tensor([10.0, 0.0])}
        p.grad = torch.tensor([-1.0, 0.0])  # anti-aligned, cos=-1
        opt.step()
        m_after = opt.state[p]["m"]
        m_old = torch.tensor([10.0, 0.0])
        mu = 0.9
        delta_corr_recovered = (m_after - mu * m_old) / (1 - mu)
        norm_corr = delta_corr_recovered.norm().item()
        self.assertGreater(norm_corr, 0.995, "correct formula should shrink delta much less")
        self.assertLess(norm_corr, 1.001, "cannot exceed original norm")

    def test_rotation_lam_decreases_with_alignment(self) -> None:
        """λ = k_d*(1−cos)*conf: less aligned → larger λ → more rotation toward momentum."""
        p1 = torch.tensor([0.0, 0.0])
        opt1 = self._make_opt(p1, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        opt1.state[p1] = {"m": torch.tensor([1.0, 0.0])}
        p1.grad = torch.tensor([0.0, 1.0])  # cos=0, weakly aligned
        opt1.step()
        delta_corr1 = (opt1.state[p1]["m"] - 0.9 * torch.tensor([1.0, 0.0])) / 0.1

        p2 = torch.tensor([0.0, 0.0])
        opt2 = self._make_opt(p2, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        opt2.state[p2] = {"m": torch.tensor([1.0, 0.0])}
        p2.grad = torch.tensor([0.4, 0.9165])  # cos≈0.4, norm≈1, in rotate zone
        opt2.step()
        delta_corr2 = (opt2.state[p2]["m"] - 0.9 * torch.tensor([1.0, 0.0])) / 0.1

        self.assertGreater(
            delta_corr1[0].item(), delta_corr2[0].item(),
            "cos=0 grad should rotate more toward m than cos=0.4 grad",
        )

    def test_magnitude_preserved_after_rotation(self) -> None:
        """Rotation preserves the original pseudo-gradient magnitude."""
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        opt.state[p] = {"m": torch.tensor([1.0, 0.0])}
        delta = torch.tensor([0.0, 2.0])  # perpendicular to m, ‖delta‖=2
        p.grad = delta.clone()
        norm_before = delta.norm().item()
        opt.step()
        m_after = opt.state[p]["m"]
        delta_corr_recovered = (m_after - 0.9 * torch.tensor([1.0, 0.0])) / 0.1
        torch.testing.assert_close(
            delta_corr_recovered.norm(), torch.tensor(norm_before), atol=1e-4, rtol=0,
            msg="rotation must preserve pseudo-gradient magnitude",
        )


class SemiAsyncHeLoCoTest(TestCase):
    """Integration tests for the SemiAsyncHeLoCo class."""

    def test_semi_async_heloco_requires_async_quorum(self) -> None:
        """SemiAsyncHeLoCo inherits the async_quorum requirement from SemiAsyncDiLoCo."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = False
        with self.assertRaises(ValueError):
            SemiAsyncHeLoCo(manager, [model], inner_optimizer, sync_every=2)

    def test_semi_async_heloco_uses_semi_async_heloco_optimizer(self) -> None:
        """Each fragment's outer optimizer is a SemiAsyncHeLoCoOptimizer instance."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True
        semi_async_heloco = SemiAsyncHeLoCo(manager, [model], inner_optimizer, sync_every=2)
        for fragment in semi_async_heloco._fragments:
            self.assertIsInstance(fragment._outer_optimizer, SemiAsyncHeLoCoOptimizer)

    def test_semi_async_heloco_lookahead_uses_each_fragment_optimizer(self) -> None:
        """Lookahead must use the matching outer optimizer for each fragment."""
        model_a = SimpleModel()
        model_b = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            list(model_a.parameters()) + list(model_b.parameters()), lr=4e-4
        )
        manager = create_manager()
        manager._use_async_quorum = True

        semi_async_heloco = SemiAsyncHeLoCo(manager, [model_a, model_b], inner_optimizer, sync_every=2)
        fragment_a = semi_async_heloco._fragments[0]
        fragment_b = semi_async_heloco._fragments[1]

        fragment_b._outer_optimizer = SemiAsyncHeLoCoOptimizer(
            model_b.parameters(),
            lr=0.3,
            momentum=0.4,
            cos_ok=0.2,
            k_dir=1.0,
            conf_c=0.0,
            k_shrink=0.5,
            beta_max=0.5,
        )

        lr_a = fragment_a._outer_optimizer.param_groups[0]["lr"]
        mu_a = fragment_a._outer_optimizer.param_groups[0]["momentum"]
        lr_b = fragment_b._outer_optimizer.param_groups[0]["lr"]
        mu_b = fragment_b._outer_optimizer.param_groups[0]["momentum"]

        for p in model_a.parameters():
            fragment_a._outer_optimizer.state[p] = {"m": torch.full_like(p, 0.5)}
        for p in model_b.parameters():
            fragment_b._outer_optimizer.state[p] = {"m": torch.full_like(p, 1.5)}

        params_a_before = {name: p.data.clone() for name, p in model_a.named_parameters()}
        params_b_before = {name: p.data.clone() for name, p in model_b.named_parameters()}

        semi_async_heloco._apply_lookahead()

        for name, p in model_a.named_parameters():
            expected = params_a_before[name] - lr_a * mu_a * fragment_a._outer_optimizer.state[p]["m"]
            torch.testing.assert_close(p.data, expected)

        for name, p in model_b.named_parameters():
            expected = params_b_before[name] - lr_b * mu_b * fragment_b._outer_optimizer.state[p]["m"]
            torch.testing.assert_close(p.data, expected)

    def test_semi_async_heloco_first_window_does_not_apply_lookahead(self) -> None:
        """First window should not apply lookahead or build outer state."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with SemiAsyncHeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=True
        ) as semi_async_heloco:
            outer_opt = semi_async_heloco._fragments[0]._outer_optimizer
            for p in model.parameters():
                outer_opt.state[p] = {"m": torch.full_like(p, 0.5)}

            params_before = {name: p.data.clone() for name, p in model.named_parameters()}
            momentum_before = {p: outer_opt.state[p]["m"].clone() for p in model.parameters()}
            inp = torch.rand(2, 3)
            for _ in range(2):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            self.assertEqual(manager.start_quorum.call_count, 1)
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertTrue(semi_async_heloco._allreduce_launched)
            for p in model.parameters():
                torch.testing.assert_close(outer_opt.state[p]["m"], momentum_before[p])

            for name, p in model.named_parameters():
                torch.testing.assert_close(
                    p.data,
                    params_before[name],
                    msg=f"lookahead should not run on the first window for {name!r}",
                )

    def test_semi_async_heloco_lookahead_fires_via_hook(self) -> None:
        """_step_post_hook triggers look-ahead when a commit is detected."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True

        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with SemiAsyncHeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=True
        ) as semi_async_heloco:
            outer_opt = semi_async_heloco._fragments[0]._outer_optimizer
            inp = torch.rand(2, 3)

            # Two full windows — the second window commits an outer step
            for _ in range(4):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            self.assertTrue(len(outer_opt.state) > 0, "outer step must have built optimizer state")

            lr = outer_opt.param_groups[0]["lr"]
            mu = outer_opt.param_groups[0]["momentum"]
            fragment = semi_async_heloco._fragments[0]

            any_shifted = False
            for name, p in model.named_parameters():
                state = outer_opt.state.get(p)
                if state and "m" in state and state["m"].norm() > 1e-6:
                    expected = fragment.original_parameters[name].to(p.device) - lr * mu * state["m"]
                    torch.testing.assert_close(
                        p.data, expected,
                        msg=f"lookahead not applied via _step_post_hook for {name!r}",
                    )
                    any_shifted = True
            self.assertTrue(any_shifted, "at least one parameter should have been shifted by lookahead")

    def test_semi_async_heloco_no_lookahead(self) -> None:
        """With use_lookahead=False, _step_post_hook never shifts params after a commit."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True

        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with SemiAsyncHeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=False
        ) as semi_async_heloco:
            inp = torch.rand(2, 3)
            for _ in range(4):  # 2 full windows, including a commit
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            fragment = semi_async_heloco._fragments[0]
            for name, p in model.named_parameters():
                torch.testing.assert_close(
                    p.data,
                    fragment.original_parameters[name].to(p.device),
                    msg=f"{name!r}: params should equal outer with use_lookahead=False",
                )
