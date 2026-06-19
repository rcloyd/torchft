# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from unittest import TestCase

import torch
from torch import nn, optim

from torchft.heloco import HeLoCoOptimizer, HeLoCoServer, HeLoCoWorker, block_correct
from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer


def _make_worker_id() -> torch.Tensor:
    return torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8)


def _make_model(d: int = 4) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d))


# ---------------------------------------------------------------------------
# HeLoCoOptimizer
# ---------------------------------------------------------------------------

class TestHeLoCoOptimizer(TestCase):
    def test_update_rule_eqs_18_19(self) -> None:
        """Verify Eqs. 18-19: m_{t+1}=μm+(1-μ)G, θ_{t+1}=θ-η(G+μm_{t+1})."""
        lr, mu = 0.1, 0.9
        G = torch.tensor([2.0, -1.0])
        p = torch.nn.Parameter(torch.tensor([3.0, 5.0]))

        opt = HeLoCoOptimizer([p], lr=lr, momentum=mu)
        p.grad = G.clone()
        opt.step()

        m_expected = (1.0 - mu) * G           # Eq. 18, m_0=0
        p_expected = torch.tensor([3.0, 5.0]) - lr * (G + mu * m_expected)  # Eq. 19

        torch.testing.assert_close(opt.state[p]["m"], m_expected)
        torch.testing.assert_close(p.data, p_expected)

    def test_second_step_uses_updated_momentum(self) -> None:
        """Second step's momentum should build on the first step's m."""
        lr, mu = 0.1, 0.9
        G1 = torch.tensor([1.0])
        G2 = torch.tensor([3.0])
        p = torch.nn.Parameter(torch.tensor([0.0]))

        opt = HeLoCoOptimizer([p], lr=lr, momentum=mu)
        p.grad = G1.clone()
        opt.step()
        m1 = (1.0 - mu) * G1

        p.grad = G2.clone()
        opt.step()
        m2 = mu * m1 + (1.0 - mu) * G2

        torch.testing.assert_close(opt.state[p]["m"], m2, atol=1e-6, rtol=0)

    def test_none_grad_skipped(self) -> None:
        """Parameters with p.grad=None should be unchanged."""
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = HeLoCoOptimizer([p], lr=0.1, momentum=0.9)
        p_before = p.data.clone()
        opt.step()  # no grad set
        torch.testing.assert_close(p.data, p_before)

    def test_momentum_buffer_float32(self) -> None:
        """Momentum buffer should always be float32 regardless of param dtype."""
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        opt = HeLoCoOptimizer([p], lr=0.1, momentum=0.9)
        p.grad = torch.tensor([1.0, 1.0], dtype=torch.float64)
        opt.step()
        self.assertEqual(opt.state[p]["m"].dtype, torch.float32)

    def test_invalid_momentum_raises(self) -> None:
        """momentum must be in [0, 1)."""
        p = torch.nn.Parameter(torch.zeros(2))
        with self.assertRaises(ValueError):
            HeLoCoOptimizer([p], momentum=1.0)
        with self.assertRaises(ValueError):
            HeLoCoOptimizer([p], momentum=-0.1)


# ---------------------------------------------------------------------------
# block_correct
# ---------------------------------------------------------------------------

class TestBlockCorrect(TestCase):
    def _correct(self, delta, m, **kwargs) -> torch.Tensor:
        result = block_correct({"p": delta}, {"p": m}, **kwargs)
        return result["p"]

    def test_aligned_block_unchanged(self) -> None:
        """cos ≥ c_ok: pseudo-gradient passes through (×ρ only)."""
        delta = torch.tensor([2.0, 0.0])
        m = torch.tensor([1.0, 0.0])  # cos = 1.0
        out = self._correct(delta, m, rho=1.0, c_ok=0.2)
        torch.testing.assert_close(out, delta)

    def test_anti_aligned_exact_math(self) -> None:
        """
        Verify the anti-aligned correction formula and conf ordering.

        Critical: conf multiplies AFTER the clamp (Eq. 11), not before.
        β = clamp(k_s · |cos|, β_max) · conf
        NOT: clamp(k_s · |cos| · conf, β_max)
        """
        # Case 1: conf≈1 (kappa=0) — basic formula check
        # cos=-1, k_s=0.5, beta_max=0.5 → clamp(0.5·1, 0.5)·1 = 0.5
        # corrected = [-1,0] - 0.5·(-1)·1·[1,0] = [-0.5, 0]
        delta = torch.tensor([-1.0, 0.0])
        m = torch.tensor([1.0, 0.0])
        out = self._correct(delta, m, rho=1.0, c_ok=0.2, k_s=0.5, beta_max=0.5, kappa=0.0)
        torch.testing.assert_close(out, torch.tensor([-0.5, 0.0]), atol=1e-5, rtol=0)

        # Case 2: conf < 1, k_s·|cos| > beta_max — the two orderings diverge here.
        # delta=[-1,0], m=[2,0], kappa=1 → conf = 1/(1+1·2) = 1/3
        # k_s=1.0, cos=-1 → k_s·|cos| = 1.0 > beta_max=0.5 → clamp = 0.5
        # CORRECT: β = 0.5 · (1/3) ≈ 0.1667
        # WRONG:   β = clamp(1.0·(1/3), 0.5) = 0.333 (gives wrong result)
        delta2 = torch.tensor([-1.0, 0.0])
        m2 = torch.tensor([2.0, 0.0])
        conf2 = 1.0 / (1.0 + 1.0 * 2.0)              # kappa=1, norm_m=2
        beta2 = min(1.0 * 1.0, 0.5) * conf2           # clamp then scale
        expected2 = torch.tensor([-1.0 + beta2, 0.0])  # corrected[0] = -1 + beta
        out2 = self._correct(delta2, m2, rho=1.0, c_ok=0.2, k_s=1.0, beta_max=0.5, kappa=1.0)
        torch.testing.assert_close(out2, expected2, atol=1e-4, rtol=0)

    def test_weakly_aligned_rotated_toward_momentum(self) -> None:
        """0 ≤ cos < c_ok: block gains a component in the momentum direction."""
        delta = torch.tensor([0.0, 1.0])  # perpendicular to m
        m = torch.tensor([1.0, 0.0])     # cos = 0 → rotation zone
        out = self._correct(delta, m, rho=1.0, c_ok=0.5, k_d=1.0, kappa=0.0)
        self.assertGreater(out[0].item(), 0.0, "rotation should give x component toward m")

    def test_weakly_aligned_norm_preserved(self) -> None:
        """Rotation must preserve the original pseudo-gradient magnitude."""
        delta = torch.tensor([0.0, 2.0])  # ‖delta‖ = 2
        m = torch.tensor([1.0, 0.0])
        out = self._correct(delta, m, rho=1.0, c_ok=0.5, k_d=1.0, kappa=0.0)
        torch.testing.assert_close(out.norm(), torch.tensor(2.0), atol=1e-5, rtol=0)

    def test_none_momentum_passthrough(self) -> None:
        """None momentum → rho*delta with no correction."""
        delta = torch.tensor([-1.0, 0.0])
        result = block_correct({"p": delta}, {"p": None}, rho=0.7)
        torch.testing.assert_close(result["p"], 0.7 * delta)

    def test_tiny_delta_norm_passthrough(self) -> None:
        """‖delta‖ ≈ 0 → pass through without correction."""
        delta = torch.tensor([1e-10, 0.0])
        m = torch.tensor([1.0, 0.0])
        out = self._correct(delta, m, rho=1.0, eps=1e-8)
        torch.testing.assert_close(out, delta)

    def test_tiny_momentum_norm_passthrough(self) -> None:
        """‖m‖ ≈ 0 → pass through without correction."""
        delta = torch.tensor([1.0, 0.0])
        m = torch.tensor([1e-10, 0.0])
        out = self._correct(delta, m, rho=1.0, eps=1e-8)
        torch.testing.assert_close(out, delta)

    def test_rho_scales_all_cases(self) -> None:
        """rho scales the corrected output uniformly across all correction cases."""
        m = torch.tensor([1.0, 0.0])
        cases = {
            "aligned": torch.tensor([1.0, 0.0]),
            "anti":    torch.tensor([-1.0, 0.0]),
            "weak":    torch.tensor([0.0, 1.0]),
        }
        c_ok = 0.5
        for name, delta in cases.items():
            out1 = self._correct(delta, m, rho=1.0, c_ok=c_ok)
            out_r = self._correct(delta, m, rho=0.3, c_ok=c_ok)
            torch.testing.assert_close(out_r, 0.3 * out1, atol=1e-5, rtol=0,
                                       msg=f"rho scaling failed for {name} case")

    def test_confidence_scales_correction(self) -> None:
        """Larger kappa → smaller conf → less correction applied."""
        delta = torch.tensor([-1.0, 0.0])
        m = torch.tensor([1.0, 0.0])
        # kappa=0 → conf≈1 → more correction (x closer to 0)
        out_small = self._correct(delta, m, rho=1.0, kappa=0.0, k_s=0.5, beta_max=0.5)
        # kappa=100 → conf≈0 → less correction (x stays near -1)
        out_large = self._correct(delta, m, rho=1.0, kappa=100.0, k_s=0.5, beta_max=0.5)
        self.assertGreater(out_small[0].item(), out_large[0].item())

    def test_output_dtype_matches_input(self) -> None:
        """Output tensors should have the same dtype as the input pseudo-grads."""
        delta = torch.tensor([1.0, 0.0], dtype=torch.float16)
        m = torch.tensor([1.0, 0.0], dtype=torch.float32)
        result = block_correct({"p": delta}, {"p": m}, rho=1.0)
        self.assertEqual(result["p"].dtype, torch.float16)


# ---------------------------------------------------------------------------
# HeLoCoServer
# ---------------------------------------------------------------------------

class TestHeLoCoServer(TestCase):
    def _push_pull(
        self,
        server: HeLoCoServer,
        model: nn.Module,
        full_sync: bool = True,
        pseudo_grad_value: float = 1.0,
        speed: float = 1.0,
    ) -> dict:
        """Simulate one worker push-pull; returns received tensors keyed by param name."""
        addr = server.address()
        pg = HeLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1) if full_sync else torch.zeros(1), root=1).wait()
            pg.broadcast_one(torch.tensor([speed]), root=1).wait()
            pg.broadcast_one(_make_worker_id(), root=1).wait()

            if full_sync:
                for _, p in model.named_parameters():
                    pg.broadcast_one(torch.full_like(p.data, pseudo_grad_value), root=1).wait()

            received = {}
            for name, p in model.named_parameters():
                buf = torch.zeros_like(p.data)
                pg.broadcast_one(buf, root=0).wait()
                received[name] = buf.clone()

            steps_buf = torch.zeros(1)
            pg.broadcast_one(steps_buf, root=0).wait()
            received["__new_steps__"] = steps_buf.clone()
        finally:
            pg.shutdown()
        return received

    def test_pull_only_sends_theta_when_no_momentum(self) -> None:
        """Pull-only with no prior outer steps returns plain θ (m uninit → no shift)."""
        model = _make_model()
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(model, outer_opt, port=0)

        theta = {n: p.data.clone() for n, p in model.named_parameters()}
        received = self._push_pull(server, model, full_sync=False)

        for name, p in model.named_parameters():
            torch.testing.assert_close(received[name], theta[name])

    def test_pull_only_sends_lookahead(self) -> None:
        """Pull-only with pre-seeded momentum sends θ̄ = θ − η·μ·m, not θ."""
        model = _make_model()
        lr, mu = 0.1, 0.9
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=lr, momentum=mu)
        server = HeLoCoServer(model, outer_opt, port=0)

        m_val = 2.0
        for p in model.parameters():
            outer_opt.state[p] = {"m": torch.full_like(p, m_val, dtype=torch.float32)}

        theta = {n: p.data.clone() for n, p in model.named_parameters()}
        received = self._push_pull(server, model, full_sync=False)

        for name, p in model.named_parameters():
            expected = theta[name] - lr * mu * m_val
            torch.testing.assert_close(received[name], expected, atol=1e-5, rtol=0)

    def test_full_sync_updates_params(self) -> None:
        """Full sync applies an outer step so global params change."""
        model = _make_model()
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(model, outer_opt, port=0)

        initial = {n: p.data.clone() for n, p in model.named_parameters()}
        self._push_pull(server, model, full_sync=True, pseudo_grad_value=1.0)

        any_changed = any(
            not torch.equal(p.data, initial[name])
            for name, p in model.named_parameters()
        )
        self.assertTrue(any_changed)

    def test_full_sync_sends_lookahead_after_step(self) -> None:
        """After full sync the worker receives θ̄_{t+1} = θ_{t+1} − η·μ·m_{t+1}."""
        model = _make_model()
        lr, mu = 0.5, 0.8
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=lr, momentum=mu)
        server = HeLoCoServer(model, outer_opt, port=0)

        received = self._push_pull(server, model, full_sync=True, pseudo_grad_value=1.0)

        for name, p in model.named_parameters():
            state = outer_opt.state.get(p)
            self.assertIsNotNone(state)
            m = state["m"]
            expected = (p.data.float() - lr * mu * m).to(p.dtype)
            torch.testing.assert_close(received[name], expected, atol=1e-5, rtol=0)

    def test_block_correction_applied_on_full_sync(self) -> None:
        """
        With momentum=0, the outer update is purely p_new = p - lr*G, so the
        received params equal θ_{t+1} directly (look-ahead degenerates to θ).
        A strongly anti-aligned gradient gets shrunk by block correction, producing
        a smaller parameter change than the uncorrected baseline.

        Expected: G_raw=-1, m=+1, k_s=1, beta_max=0.9, kappa=0 → conf≈1
        → beta = clamp(1*1, 0.9)*1 = 0.9
        → G_corr = -1 - 0.9*(-1)*1*1 = -0.1
        → diff_corr = 0.01,  diff_no = 0.10
        """
        d = 1
        model_corr = nn.Sequential(nn.Linear(d, d, bias=False))
        model_no = nn.Sequential(nn.Linear(d, d, bias=False))
        with torch.no_grad():
            for p_c, p_n in zip(model_corr.parameters(), model_no.parameters()):
                p_n.data.copy_(p_c.data)

        lr = 0.1
        # mu=0: update is p_new = p - lr*G; look-ahead = θ (no shift).
        # This isolates correction from momentum-driven update noise.
        outer_opt_corr = HeLoCoOptimizer(model_corr.parameters(), lr=lr, momentum=0.0)
        # Pre-seed m=+1 for block correction; anti-aligned with incoming grad=-1
        for p in model_corr.parameters():
            outer_opt_corr.state[p] = {"m": torch.full_like(p, 1.0, dtype=torch.float32)}
        server_corr = HeLoCoServer(
            model_corr, outer_opt_corr, port=0,
            k_s=1.0, beta_max=0.9, kappa=0.0,
        )

        outer_opt_no = optim.SGD(model_no.parameters(), lr=lr)
        server_no = AsyncDiLoCoServer(model_no, outer_opt_no, port=0)

        initial = {n: p.data.clone() for n, p in model_corr.named_parameters()}
        received_corr = self._push_pull(server_corr, model_corr, pseudo_grad_value=-1.0)
        received_no = self._push_pull(server_no, model_no, pseudo_grad_value=-1.0)

        for name in [n for n, _ in model_corr.named_parameters()]:
            diff_corr = (received_corr[name] - initial[name]).abs().mean().item()
            diff_no = (received_no[name] - initial[name]).abs().mean().item()
            self.assertLess(diff_corr, diff_no,
                            msg=f"{name}: block correction should shrink the anti-aligned step")

    def test_dylu_inherited(self) -> None:
        """DyLU still works: first worker at speed v receives H steps back."""
        model = _make_model()
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(model, outer_opt, port=0, dylu_H=50)

        received = self._push_pull(server, model, full_sync=True, speed=10.0)
        self.assertEqual(int(received["__new_steps__"][0].item()), 50)

    def test_pull_only_returns_dylu_H(self) -> None:
        """Pull-only response echoes dylu_H as new_steps (different code path from full sync)."""
        model = _make_model()
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(model, outer_opt, port=0, dylu_H=42)

        received = self._push_pull(server, model, full_sync=False)
        self.assertEqual(int(received["__new_steps__"][0].item()), 42)

    def test_full_training_loop(self) -> None:
        """End-to-end: HeLoCoWorker trains with HeLoCoServer for one sync window."""
        d = 4
        global_model = _make_model(d)
        outer_opt = HeLoCoOptimizer(global_model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        sync_every = 3
        with HeLoCoWorker(addr, worker_model, inner_opt, sync_every=sync_every):
            x = torch.randn(2, d)
            y = torch.randint(0, d, (2,))
            criterion = nn.CrossEntropyLoss()
            for _ in range(sync_every):
                inner_opt.zero_grad()
                loss = criterion(worker_model(x), y)
                loss.backward()
                inner_opt.step()

        any_state = any("m" in state for state in outer_opt.state.values())
        self.assertTrue(any_state, "outer optimizer should have momentum after one sync")


# ---------------------------------------------------------------------------
# Integration: multiple workers
# ---------------------------------------------------------------------------

class TestHeLoCoMultiWorker(TestCase):
    def test_two_workers_sequential(self) -> None:
        """Two sequential workers trigger two outer steps; params and momentum accumulate."""
        d = 4
        model = _make_model(d)
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(model, outer_opt, port=0)
        addr = server.address()

        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        for _ in range(2):
            worker_model = _make_model(d)
            inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
            sync_every = 2
            with HeLoCoWorker(addr, worker_model, inner_opt, sync_every=sync_every):
                x = torch.randn(2, d)
                y = torch.randint(0, d, (2,))
                criterion = nn.CrossEntropyLoss()
                for _ in range(sync_every):
                    inner_opt.zero_grad()
                    loss = criterion(worker_model(x), y)
                    loss.backward()
                    inner_opt.step()

        any_changed = any(
            not torch.equal(p.data, params_before[name])
            for name, p in model.named_parameters()
        )
        self.assertTrue(any_changed)
        self.assertTrue(any("m" in s for s in outer_opt.state.values()))
