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
from torchft.local_sgd import AsyncDiLoCo, DiLoCo, extract_local_tensor, LocalSGD
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

    def test_async_diloco_requires_async_quorum(self) -> None:
        """AsyncDiLoCo must raise if the manager does not use async quorum."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        outer_optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        manager = create_manager()
        manager._use_async_quorum = False

        with self.assertRaises(ValueError):
            AsyncDiLoCo(manager, [model], inner_optimizer, outer_optimizer, sync_every=2)

    def test_async_diloco_multi_window(self) -> None:
        """Three full windows: verify pipeline stays correct across multiple commits."""
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
            inp = torch.rand(2, 3)

            # Three full windows = 6 steps
            for _ in range(6):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # start_quorum: once at window 1 boundary + once at step 1 of windows 2 and 3
            self.assertEqual(manager.start_quorum.call_count, 3)
            # should_commit called at every boundary: window 1 (else branch for late-joiner
            # checkpoint application) and windows 2 and 3 (if branch, normal outer step).
            self.assertEqual(manager.should_commit.call_count, 3)
            # allreduce called once per window
            self.assertEqual(manager.allreduce.call_count, parameter_count * 3)
            self.assertEqual(async_diloco._local_step, 0)

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
