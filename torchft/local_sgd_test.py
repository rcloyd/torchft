# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock

import torch
from parameterized import parameterized
from torch import nn, optim, Tensor
from torch.distributed.distributed_c10d import Work
from torch.distributed.tensor import DTensor
from torchft.local_sgd import AsyncDiLoCo, DiLoCo, extract_local_tensor, HeLoCo, HeLoCoOptimizer, LocalSGD
from torchft.manager import Manager
from torchft.work import _DummyWork


def create_manager() -> MagicMock:
    """
    Creates a mock manager with some useful defaults for testing
    the optimizer's usage of the Manager
    """
    manager = create_autospec(Manager)

    manager.errored.return_value = None

    def mock_allreduce(tensor: torch.Tensor, should_quantize: bool = False) -> Work:
        return _DummyWork(tensor)

    manager.allreduce.side_effect = mock_allreduce

    return manager


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


def _params_dict(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data for name, p in m.named_parameters()}


def _copy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.clone().detach() for name, value in state_dict.items()}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.w2 = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w1.unsqueeze(0).T + self.w2.sum()


class LocalSGDTest(TestCase):
    def test_local_sgd_healthy(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_manager()
        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            self.assertEqual(local_sgd._local_step, 0)
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            self.assertEqual(local_sgd._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()
            self.assertEqual(manager.start_quorum.call_count, 1)

            manager.should_commit.return_value = True
            self.assertEqual(local_sgd._local_step, 0)
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, 4)

    def test_extract_local_tensor(self) -> None:
        regular_tensor = torch.rand(3, 3, requires_grad=True)
        regular_result = extract_local_tensor(regular_tensor)

        self.assertTrue(torch.equal(regular_result, regular_tensor))
        self.assertIsNone(regular_result.grad)
        self.assertNotEqual(id(regular_result), id(regular_tensor))
        local_tensor = torch.rand(3, 3, requires_grad=True)
        dtensor = MagicMock(spec=DTensor)
        dtensor.to_local.return_value = local_tensor
        dtensor_result = extract_local_tensor(dtensor)

        self.assertTrue(torch.equal(dtensor_result, local_tensor))
        self.assertIsNone(dtensor_result.grad)
        self.assertNotEqual(id(dtensor_result), id(local_tensor))
        dtensor.to_local.assert_called_once()

    def test_local_sgd_recovery(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_autospec(Manager)

        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            og_state_dict = _copy_state_dict(model.state_dict())

            inp = torch.rand(2, 3)

            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            # Check that the model's state dict has been updated
            for name, param in model.state_dict().items():
                # Ensure the parameter has changed
                self.assertFalse(
                    torch.equal(og_state_dict[name], param),
                    f"Parameter {name} did not change.",
                )
            self.assertEqual(local_sgd._local_step, 1)


class DiLoCoTest(TestCase):
    def test_diloco_healthy(self) -> None:
        model = SimpleModel()

        # Setup optimizers
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()
        manager._use_async_quorum = False
        with DiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as diloco:
            parameter_count = len(list(model.parameters()))
            initial_outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(initial_outer_opt_state["state"], {})

            self.assertEqual(diloco._local_step, 0)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(diloco._local_step, 1)
            manager.current_step.return_value = 0
            manager.should_commit.return_value = True
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 1)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, parameter_count)

            outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(len(outer_opt_state["state"]), parameter_count)

    @parameterized.expand(
        [
            ("bucketized_should_use_fewer_calls", True, True),
            ("non_bucketized_should_call_per_param", False, False),
        ]
    )
    def test_diloco_allreduce_call_efficiency(
        self,
        name: str,
        use_bucketization: bool,
        expect_fewer_calls: bool,
    ) -> None:
        model = SimpleModel()

        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        with DiLoCo(
            manager,
            [model],
            inner_optimizer,
            outer_optimizer,
            sync_every=2,
            use_bucketization=use_bucketization,
        ) as diloco:
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            manager.current_step.return_value = 0
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            allreduce_calls = manager.allreduce.call_count
            param_count = len([p for p in model.parameters() if p.requires_grad])

            if expect_fewer_calls:
                self.assertLess(int(allreduce_calls), int(param_count))
            else:
                self.assertEqual(int(allreduce_calls), int(param_count))

    def test_bucketization_correctness(self) -> None:
        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        manager = create_autospec(Manager)
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        # Define fake allreduce: multiplies buffer by 2
        def fake_allreduce(tensor: Tensor, should_quantize: bool) -> Work:
            tensor.mul_(2)
            return _DummyWork(tensor)

        manager.allreduce.side_effect = fake_allreduce

        diloco = DiLoCo(
            manager, [model], inner_opt, outer_opt, sync_every=2, use_bucketization=True
        )
        diloco._fragments[0].bucket_cap_mb = 10 * 1024 * 1024

        # Manually assign fake gradients
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        for g, (name, param) in zip(grads, model.named_parameters()):
            diloco._fragments[0]._grads[name] = g.clone()

        # Run only bucketized logic
        diloco._fragments[0]._average_grads()

        # The parameter gradients should not be set
        for param in model.parameters():
            self.assertEqual(param.grad, None)

        diloco._fragments[0]._set_grads()

        # Expect grads to have been doubled
        expected_grads = [g * 2 for g in grads]
        for param, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-8)

    def test_gradient_correctness(self) -> None:
        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        manager = create_autospec(Manager)
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        # Define fake allreduce: multiplies buffer by 2
        def fake_allreduce(tensor: Tensor, should_quantize: bool) -> Work:
            tensor.mul_(2)
            return _DummyWork(tensor)

        manager.allreduce.side_effect = fake_allreduce

        diloco = DiLoCo(manager, [model], inner_opt, outer_opt, sync_every=2)

        # save original parameters
        diloco._fragments[0].save_parameters()

        # change the model's parameters
        for p in model.parameters():
            p.data.add_(2)

        # calculate and set the gradients
        diloco._fragments[0]._save_grads()

        # calculate
        diloco._fragments[0]._average_grads()

        # The parameter gradients should not be set
        for param in model.parameters():
            self.assertEqual(param.grad, None)

        diloco._fragments[0]._set_grads()

        # we added 2 to the parameters, then multiplied the gradients by 2
        # so we should expect the model's gradient to be -4
        expected_grad = -4
        for param in model.parameters():
            assert param.grad is not None
            t = torch.empty_like(param.grad)
            t.fill_(expected_grad)
            torch.testing.assert_close(param.grad, t)


class AsyncDiLoCoTest(TestCase):
    def test_async_diloco_first_window(self) -> None:
        """First window: should_commit called once (to apply any pending late-joiner checkpoint), no outer step applied, allreduce launched."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as async_diloco:
            parameter_count = len(list(model.parameters()))
            inp = torch.rand(2, 3)

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(async_diloco._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(async_diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 1)
            # should_commit is called even on the first window so that any pending
            # checkpoint from a healing quorum is applied before pseudo-grads are
            # computed (late-joiner eavesdrop sync).
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, parameter_count)
            self.assertTrue(async_diloco._allreduce_launched)
            # No outer step was applied: outer optimizer must have no momentum state.
            self.assertEqual(outer_optimizer.state_dict()["state"], {})

    def test_async_diloco_healthy(self) -> None:
        """Two full windows: outer step applied at end of window 2."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )
        manager = create_manager()
        manager._use_async_quorum = True
        manager.should_commit.return_value = True
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as async_diloco:
            parameter_count = len(list(model.parameters()))
            self.assertEqual(outer_optimizer.state_dict()["state"], {})

            inp = torch.rand(2, 3)

            # Window 1
            for _ in range(2):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            self.assertEqual(async_diloco._local_step, 0)
            # start_quorum called once at window 1 boundary to cover its allreduce
            self.assertEqual(manager.start_quorum.call_count, 1)
            # should_commit called once on window 1 (else branch) to apply any
            # pending late-joiner checkpoint before computing pseudo-grads.
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, parameter_count)

            # Window 2, step 1 — triggers start_quorum (for window 2 allreduce)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(async_diloco._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 2)

            # Window 2, step 2 — triggers async_sync with should_commit
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(async_diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 2)
            self.assertEqual(manager.should_commit.call_count, 2)
            self.assertEqual(manager.allreduce.call_count, parameter_count * 2)
            torch.testing.assert_close(
                async_diloco._fragments[0].original_parameters, _params_dict(model)
            )
            self.assertEqual(len(outer_optimizer.state_dict()["state"]), parameter_count)

    def test_async_diloco_recovery(self) -> None:
        """Recovery: should_commit=False skips outer step and leaves original_parameters unchanged."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )
        manager = create_manager()
        manager._use_async_quorum = True
        manager.should_commit.return_value = False
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as async_diloco:
            initial_outer_params = {
                name: p.clone()
                for name, p in async_diloco._fragments[0].original_parameters.items()
            }

            inp = torch.rand(2, 3)

            # Two full windows
            for _ in range(4):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # Window 1 (else branch) + window 2 (if branch) = 2 calls
            self.assertEqual(manager.should_commit.call_count, 2)

            for name, param in async_diloco._fragments[0].original_parameters.items():
                torch.testing.assert_close(param.cpu(), initial_outer_params[name].cpu())

            self.assertEqual(outer_optimizer.state_dict()["state"], {})

    @parameterized.expand(
        [
            ("bucketized_should_use_fewer_calls", True, True),
            ("non_bucketized_should_call_per_param", False, False),
        ]
    )
    def test_async_diloco_allreduce_call_efficiency(
        self,
        name: str,
        use_bucketization: bool,
        expect_fewer_calls: bool,
    ) -> None:
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )
        manager = create_manager()
        manager._use_async_quorum = True
        manager.should_commit.return_value = True
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager,
            [model],
            inner_optimizer,
            outer_optimizer,
            sync_every=2,
            use_bucketization=use_bucketization,
        ):
            inp = torch.rand(2, 3)

            for _ in range(4):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            allreduce_calls = manager.allreduce.call_count
            param_count = len([p for p in model.parameters() if p.requires_grad])

            if expect_fewer_calls:
                self.assertLess(int(allreduce_calls), int(param_count * 2))
            else:
                self.assertEqual(int(allreduce_calls), int(param_count * 2))

    def test_async_diloco_gradient_correctness(self) -> None:
        """Pseudo-gradients computed at window 1 are averaged and applied at window 2."""
        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        def fake_allreduce(tensor: Tensor, should_quantize: bool = False) -> Work:
            tensor.mul_(2)
            return _DummyWork(tensor)

        manager = create_manager()
        manager._use_async_quorum = True
        manager.should_commit.return_value = True
        manager.allreduce.side_effect = fake_allreduce
        manager.current_step.return_value = 0

        async_diloco = AsyncDiLoCo(manager, [model], inner_opt, outer_opt, sync_every=2)

        initial_outer = {
            name: p.clone()
            for name, p in async_diloco._fragments[0].original_parameters.items()
        }

        # Shift local params by +2 so Δ = outer - local = -2
        # fake_allreduce doubles it → avg(Δ) = -4
        for p in model.parameters():
            p.data.add_(2)

        # Mirror the real call sequence: start_quorum before each sync boundary.
        manager.start_quorum()
        # Window 1 boundary: compute Δ=-2, launch allreduce (→ -4), reset local to outer
        async_diloco._fragments[0].async_sync()

        # Shift local again for window 2 pseudo-gradient (value doesn't matter for this assertion)
        for p in model.parameters():
            p.data.add_(2)

        manager.start_quorum()
        # Window 2 boundary: wait allreduce(-4), apply outer step, launch new allreduce
        async_diloco._fragments[0].async_sync()

        # outer(T) = outer(T-1) - lr * avg(Δ) = initial - 0.1 * (-4) = initial + 0.4
        for name, initial in initial_outer.items():
            updated = async_diloco._fragments[0].original_parameters[name]
            expected = initial + 0.4
            torch.testing.assert_close(updated.cpu(), expected.cpu(), rtol=1e-5, atol=1e-4)

    def test_async_diloco_non_blocking(self) -> None:
        """allreduce(T) stays in-flight throughout all inner steps of window T+1."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        outer_optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as async_diloco:
            fragment = async_diloco._fragments[0]
            inp = torch.rand(2, 3)

            # Window 1: two inner steps → boundary fires and launches allreduce(1)
            for _ in range(2):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # allreduce(1) must be in-flight — work submitted but not yet waited on
            self.assertGreater(
                len(fragment._allreduce_work),
                0,
                "allreduce work must be in-flight immediately after window 1 boundary",
            )

            # First inner step of window 2 — allreduce(1) still not waited on
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertGreater(
                len(fragment._allreduce_work),
                0,
                "allreduce work must remain in-flight during window 2 inner steps",
            )

    def test_async_diloco_late_joiner_checkpoint_applied(self) -> None:
        """Late-joiner: checkpoint applied by should_commit in the else branch gives zero pseudo-grads."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        outer_optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        with AsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as async_diloco:
            fragment = async_diloco._fragments[0]

            # Donor outer params — what the healing quorum would deliver to a late joiner
            donor_outer = {
                name: torch.full_like(p, 99.0)
                for name, p in model.named_parameters()
            }

            def apply_checkpoint() -> bool:
                # Simulate _apply_pending_state_dict: update original_parameters and model params
                for name, param in fragment.original_parameters.items():
                    param.data.copy_(donor_outer[name])
                for name, p in model.named_parameters():
                    p.data.copy_(donor_outer[name])
                return True

            manager.should_commit.side_effect = apply_checkpoint

            inp = torch.rand(2, 3)
            # Run first window from random initial params (simulates late joiner's first window)
            for _ in range(2):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # should_commit was called exactly once (in the else branch)
            self.assertEqual(manager.should_commit.call_count, 1)

            # original_parameters must reflect the donor's state
            for name, donor in donor_outer.items():
                torch.testing.assert_close(
                    fragment.original_parameters[name],
                    donor,
                    msg=f"original_parameters[{name!r}] must match donor checkpoint",
                )

            # Pseudo-grads must be ≈ 0: checkpoint set p.data = donor = original_parameters,
            # so new_grads = original_parameters - p.data = 0
            for name, grad in fragment._grads.items():
                torch.testing.assert_close(
                    grad,
                    torch.zeros_like(grad),
                    atol=1e-6,
                    rtol=0,
                    msg=f"late-joiner pseudo-grad for {name!r} should be zero",
                )


class HeLoCoOptimizerTest(TestCase):
    """Unit tests for HeLoCoOptimizer correction logic."""

    def _make_opt(self, p: torch.Tensor, **kwargs) -> HeLoCoOptimizer:
        defaults = dict(lr=0.1, momentum=0.9, cos_ok=0.2,
                        k_dir=1.0, conf_c=0.0, k_shrink=0.5, beta_max=0.5, eps=1e-8)
        defaults.update(kwargs)
        return HeLoCoOptimizer([p], **defaults)

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
        opt = HeLoCoOptimizer(
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
        # With k_shrink=1.0, cos=-1.0, beta_max=0.3, conf_c=0:
        #   raw = k_shrink * (-cos) = 1.0  →  clamp to 0.3  →  * conf=1.0  →  beta=0.3
        #   (wrong order: clamp(1.0 * conf, max=0.3) gives same 0.3 when conf=1)
        # With conf_c=10 and norm_m >> norm_d, conf becomes small:
        #   conf = 1/(1+10*10) ≈ 0.01  (norm_d=1, norm_m=10)
        #   correct: clamp(1.0, max=0.3) * 0.01 = 0.3 * 0.01 = 0.003
        #   wrong:   clamp(1.0 * 0.01, max=0.3) = 0.01  (different!)
        p = torch.tensor([0.0, 0.0])
        opt = self._make_opt(p, cos_ok=0.2, k_shrink=1.0, beta_max=0.3,
                             conf_c=10.0, eps=1e-8)
        # Large momentum buffer → conf is small
        opt.state[p] = {"m": torch.tensor([10.0, 0.0])}
        p.grad = torch.tensor([-1.0, 0.0])  # anti-aligned, cos=-1
        opt.step()
        # norm_d=1, norm_m=10 → conf = 1/(1+10*10) = 1/101 ≈ 0.0099
        # correct beta = clamp(1.0, 0, 0.3) * conf = 0.3 * 0.0099 ≈ 0.00297
        # wrong beta   = clamp(1.0 * conf, 0, 0.3) = clamp(0.0099, 0, 0.3) = 0.0099
        # The delta_corr magnitudes differ: (1-0.00297)≈0.997 vs (1-0.0099)≈0.990
        # Recover delta_corr from updated momentum
        m_after = opt.state[p]["m"]
        m_old = torch.tensor([10.0, 0.0])
        mu = 0.9
        # m_after = mu * m_old + (1-mu) * delta_corr
        delta_corr_recovered = (m_after - mu * m_old) / (1 - mu)
        norm_corr = delta_corr_recovered.norm().item()
        # Correct implementation: beta ≈ 0.003 → delta_corr ≈ -0.997
        # Wrong implementation:   beta ≈ 0.010 → delta_corr ≈ -0.990
        # The correct value is closer to 1.0
        self.assertGreater(norm_corr, 0.995, "correct formula should shrink delta much less")
        self.assertLess(norm_corr, 1.001, "cannot exceed original norm")

    def test_rotation_lam_decreases_with_alignment(self) -> None:
        """λ = k_d*(1−cos)*conf: less aligned → larger λ → more rotation toward momentum."""
        # p1: delta perpendicular to m (cos=0) → λ=1.0*1.0*1.0=1.0 → full rotation
        p1 = torch.tensor([0.0, 0.0])
        opt1 = self._make_opt(p1, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        opt1.state[p1] = {"m": torch.tensor([1.0, 0.0])}
        p1.grad = torch.tensor([0.0, 1.0])  # cos=0, weakly aligned
        opt1.step()
        delta_corr1 = (opt1.state[p1]["m"] - 0.9 * torch.tensor([1.0, 0.0])) / 0.1

        # p2: delta at cos≈0.4 with m (better aligned) → λ=1.0*0.6*1.0=0.6 → partial rotation
        p2 = torch.tensor([0.0, 0.0])
        opt2 = self._make_opt(p2, cos_ok=0.5, k_dir=1.0, conf_c=0.0)
        opt2.state[p2] = {"m": torch.tensor([1.0, 0.0])}
        p2.grad = torch.tensor([0.4, 0.9165])  # cos≈0.4, norm≈1, in rotate zone
        opt2.step()
        delta_corr2 = (opt2.state[p2]["m"] - 0.9 * torch.tensor([1.0, 0.0])) / 0.1

        # Weakly aligned (cos=0) should gain more x-component (toward m) than cos=0.4
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
        # Record norm of delta before step; after correction norm should be same
        norm_before = delta.norm().item()
        p_before = p.data.clone()
        opt.step()
        # The correction is applied before the MLA update; we can't directly
        # observe delta_corr, but the p update magnitude should reflect ‖delta‖=2
        # (not more, not less from the rotation itself)
        # Verify by checking that the update is in the rotated direction with same magnitude
        m_after = opt.state[p]["m"]
        # m_new = mu*m_old + (1-mu)*delta_corr; norm of delta_corr should equal norm_before
        # ‖delta_corr‖ = norm_before → ‖m_new - mu*m_old‖ / (1-mu) ≈ norm_before
        delta_corr_recovered = (m_after - 0.9 * torch.tensor([1.0, 0.0])) / 0.1
        torch.testing.assert_close(
            delta_corr_recovered.norm(), torch.tensor(norm_before), atol=1e-4, rtol=0,
            msg="rotation must preserve pseudo-gradient magnitude",
        )


class HeLoCoTest(TestCase):
    """Integration tests for the HeLoCo class."""

    def test_heloco_requires_async_quorum(self) -> None:
        """HeLoCo inherits the async_quorum requirement from AsyncDiLoCo."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = False
        with self.assertRaises(ValueError):
            HeLoCo(manager, [model], inner_optimizer, sync_every=2)

    def test_heloco_uses_heloco_optimizer(self) -> None:
        """Each fragment's outer optimizer is a HeLoCoOptimizer instance."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True
        heloco = HeLoCo(manager, [model], inner_optimizer, sync_every=2)
        for fragment in heloco._fragments:
            self.assertIsInstance(fragment._outer_optimizer, HeLoCoOptimizer)

    def test_heloco_lookahead_uses_each_fragment_optimizer(self) -> None:
        """Lookahead must use the matching outer optimizer for each fragment."""
        model_a = SimpleModel()
        model_b = SimpleModel()
        inner_optimizer = torch.optim.AdamW(
            list(model_a.parameters()) + list(model_b.parameters()), lr=4e-4
        )
        manager = create_manager()
        manager._use_async_quorum = True

        heloco = HeLoCo(manager, [model_a, model_b], inner_optimizer, sync_every=2)
        fragment_a = heloco._fragments[0]
        fragment_b = heloco._fragments[1]

        fragment_b._outer_optimizer = HeLoCoOptimizer(
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

        heloco._apply_lookahead()

        for name, p in model_a.named_parameters():
            expected = params_a_before[name] - lr_a * mu_a * fragment_a._outer_optimizer.state[p]["m"]
            torch.testing.assert_close(p.data, expected)

        for name, p in model_b.named_parameters():
            expected = params_b_before[name] - lr_b * mu_b * fragment_b._outer_optimizer.state[p]["m"]
            torch.testing.assert_close(p.data, expected)

    def test_heloco_first_window_does_not_apply_lookahead(self) -> None:
        """First window should not apply lookahead or build outer state."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with HeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=True
        ) as heloco:
            outer_opt = heloco._fragments[0]._outer_optimizer
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
            self.assertTrue(heloco._allreduce_launched)
            for p in model.parameters():
                torch.testing.assert_close(outer_opt.state[p]["m"], momentum_before[p])

            for name, p in model.named_parameters():
                torch.testing.assert_close(
                    p.data,
                    params_before[name],
                    msg=f"lookahead should not run on the first window for {name!r}",
                )

    def test_heloco_lookahead_fires_via_hook(self) -> None:
        """_step_post_hook triggers look-ahead when a commit is detected."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True

        # Simulate current_step incrementing on each should_commit() == True call,
        # matching what the real manager does.
        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with HeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=True
        ) as heloco:
            outer_opt = heloco._fragments[0]._outer_optimizer
            inp = torch.rand(2, 3)

            # Two full windows — the second window commits an outer step
            for _ in range(4):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # Outer step was applied; optimizer should have momentum state
            self.assertTrue(len(outer_opt.state) > 0, "outer step must have built optimizer state")

            lr = outer_opt.param_groups[0]["lr"]
            mu = outer_opt.param_groups[0]["momentum"]
            fragment = heloco._fragments[0]

            # With lookahead: p.data = original_parameters - lr*mu*m  ≠  original_parameters
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

    def test_heloco_no_lookahead(self) -> None:
        """With use_lookahead=False, _step_post_hook never shifts params after a commit."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        manager = create_manager()
        manager._use_async_quorum = True

        step_val = [0]
        manager.current_step.side_effect = lambda: step_val[0]
        manager.should_commit.side_effect = lambda: (step_val.__setitem__(0, step_val[0] + 1), True)[1]

        with HeLoCo(
            manager, [model], inner_optimizer, sync_every=2, use_lookahead=False
        ) as heloco:
            inp = torch.rand(2, 3)
            for _ in range(4):  # 2 full windows, including a commit
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            fragment = heloco._fragments[0]
            # Without lookahead, local params must equal original_parameters (outer) exactly
            for name, p in model.named_parameters():
                torch.testing.assert_close(
                    p.data,
                    fragment.original_parameters[name].to(p.device),
                    msg=f"{name!r}: params should equal outer with use_lookahead=False",
                )
