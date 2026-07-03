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
from torch import nn, Tensor
from torch.distributed.distributed_c10d import Work

from torchft.semi_async_diloco import SemiAsyncDiLoCo
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


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.w2 = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w1.unsqueeze(0).T + self.w2.sum()


def _params_dict(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data for name, p in m.named_parameters()}


def create_manager() -> MagicMock:
    manager = create_autospec(Manager)
    manager.errored.return_value = None
    # instance attribute — not captured by create_autospec's class spec
    manager._rank0_synchronization_only = False

    def mock_allreduce(tensor: torch.Tensor, should_quantize: bool = False) -> Work:
        return _DummyWork(tensor)

    manager.allreduce.side_effect = mock_allreduce
    return manager


class SemiAsyncDiLoCoTest(TestCase):
    def test_semi_async_diloco_first_window(self) -> None:
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

        with SemiAsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as semi_async_diloco:
            parameter_count = len(list(model.parameters()))
            inp = torch.rand(2, 3)

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(semi_async_diloco._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(semi_async_diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 1)
            # should_commit is called even on the first window so that any pending
            # checkpoint from a healing quorum is applied before pseudo-grads are
            # computed (late-joiner eavesdrop sync).
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, parameter_count)
            self.assertTrue(semi_async_diloco._allreduce_launched)
            # No outer step was applied: outer optimizer must have no momentum state.
            self.assertEqual(outer_optimizer.state_dict()["state"], {})

    def test_semi_async_diloco_healthy(self) -> None:
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

        with SemiAsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as semi_async_diloco:
            parameter_count = len(list(model.parameters()))
            self.assertEqual(outer_optimizer.state_dict()["state"], {})

            inp = torch.rand(2, 3)

            # Window 1
            for _ in range(2):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            self.assertEqual(semi_async_diloco._local_step, 0)
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

            self.assertEqual(semi_async_diloco._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 2)

            # Window 2, step 2 — triggers async_sync with should_commit
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(semi_async_diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 2)
            self.assertEqual(manager.should_commit.call_count, 2)
            self.assertEqual(manager.allreduce.call_count, parameter_count * 2)
            torch.testing.assert_close(
                semi_async_diloco._fragments[0].original_parameters, _params_dict(model)
            )
            self.assertEqual(len(outer_optimizer.state_dict()["state"]), parameter_count)

    def test_semi_async_diloco_recovery(self) -> None:
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

        with SemiAsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as semi_async_diloco:
            initial_outer_params = {
                name: p.clone()
                for name, p in semi_async_diloco._fragments[0].original_parameters.items()
            }

            inp = torch.rand(2, 3)

            # Two full windows
            for _ in range(4):
                loss = model(inp).mean()
                loss.backward()
                inner_optimizer.step()

            # Window 1 (else branch) + window 2 (if branch) = 2 calls
            self.assertEqual(manager.should_commit.call_count, 2)

            for name, param in semi_async_diloco._fragments[0].original_parameters.items():
                torch.testing.assert_close(param.cpu(), initial_outer_params[name].cpu())

            self.assertEqual(outer_optimizer.state_dict()["state"], {})

    @parameterized.expand(
        [
            ("bucketized_should_use_fewer_calls", True, True),
            ("non_bucketized_should_call_per_param", False, False),
        ]
    )
    def test_semi_async_diloco_allreduce_call_efficiency(
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

        with SemiAsyncDiLoCo(
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

    def test_semi_async_diloco_gradient_correctness(self) -> None:
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

        semi_async_diloco = SemiAsyncDiLoCo(manager, [model], inner_opt, outer_opt, sync_every=2)

        initial_outer = {
            name: p.clone()
            for name, p in semi_async_diloco._fragments[0].original_parameters.items()
        }

        # Shift local params by +2 so Δ = outer - local = -2
        # fake_allreduce doubles it → avg(Δ) = -4
        for p in model.parameters():
            p.data.add_(2)

        # Mirror the real call sequence: start_quorum before each sync boundary.
        manager.start_quorum()
        # Window 1 boundary: compute Δ=-2, launch allreduce (→ -4), reset local to outer
        semi_async_diloco._fragments[0].async_sync()

        # Shift local again for window 2 pseudo-gradient (value doesn't matter for this assertion)
        for p in model.parameters():
            p.data.add_(2)

        manager.start_quorum()
        # Window 2 boundary: wait allreduce(-4), apply outer step, launch new allreduce
        semi_async_diloco._fragments[0].async_sync()

        # outer(T) = outer(T-1) - lr * avg(Δ) = initial - 0.1 * (-4) = initial + 0.4
        for name, initial in initial_outer.items():
            updated = semi_async_diloco._fragments[0].original_parameters[name]
            expected = initial + 0.4
            torch.testing.assert_close(updated.cpu(), expected.cpu(), rtol=1e-5, atol=1e-4)

    def test_semi_async_diloco_non_blocking(self) -> None:
        """allreduce(T) stays in-flight throughout all inner steps of window T+1."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        outer_optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        with SemiAsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as semi_async_diloco:
            fragment = semi_async_diloco._fragments[0]
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

    def test_semi_async_diloco_late_joiner_checkpoint_applied(self) -> None:
        """Late-joiner: checkpoint applied by should_commit in the else branch gives zero pseudo-grads."""
        model = SimpleModel()
        inner_optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
        outer_optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
        manager = create_manager()
        manager._use_async_quorum = True
        manager.current_step.return_value = 0

        with SemiAsyncDiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as semi_async_diloco:
            fragment = semi_async_diloco._fragments[0]

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
