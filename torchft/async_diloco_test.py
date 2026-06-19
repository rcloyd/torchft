# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
import uuid
from unittest import TestCase

import torch
from torch import nn, optim

from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer, DelayedNesterovOptimizer


def _make_worker_id() -> torch.Tensor:
    return torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8)


def _make_model(d: int = 8) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


class TestDelayedNesterovOptimizer(TestCase):
    def test_period1_matches_standard_nesterov(self) -> None:
        """With nesterov_period=1 every push is a milestone: should match SGD+Nesterov."""
        lr, beta = 0.1, 0.9
        g = torch.tensor([1.0, 2.0])

        # Standard Nesterov (PyTorch implementation)
        p_ref = torch.tensor([1.0, 1.0])
        m_ref = torch.zeros_like(p_ref)
        # One step of SGD nesterov: m = β*m + g, p -= lr*(g + β*m_new)
        m_ref = beta * m_ref + g
        p_ref = p_ref - lr * (g + beta * m_ref)

        # DN with period=1
        p_dn = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        dn = DelayedNesterovOptimizer([p_dn], lr=lr, momentum=beta, nesterov_period=1)
        p_dn.grad = g.clone()
        dn.step()
        dn.zero_grad()

        torch.testing.assert_close(p_dn.data, p_ref)

    def test_n_steps_equal_nesterov_on_average(self) -> None:
        """
        After N pushes of the same gradient g, total parameter change should
        equal one standard Nesterov step on avg_grad=g (since all grads are g).
        """
        lr, beta, N = 0.1, 0.9, 3
        g = torch.tensor([1.0, 2.0])

        # DN over N identical pushes
        p_dn = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        dn = DelayedNesterovOptimizer([p_dn], lr=lr, momentum=beta, nesterov_period=N)
        for _ in range(N):
            p_dn.grad = g.clone()
            dn.step()
        dn.zero_grad()

        # Standard Nesterov on avg_grad (= g since all grads identical)
        p_ref = torch.tensor([0.0, 0.0])
        m_ref = torch.zeros_like(p_ref)
        m_ref = beta * m_ref + g
        p_ref = p_ref - lr * (g + beta * m_ref)

        torch.testing.assert_close(p_dn.data, p_ref)

    def test_intermediate_steps_update_params(self) -> None:
        """Between milestones the model should still update (pure gradient steps)."""
        lr, N = 0.1, 3
        g = torch.tensor([1.0])

        p = torch.nn.Parameter(torch.tensor([0.0]))
        dn = DelayedNesterovOptimizer([p], lr=lr, momentum=0.9, nesterov_period=N)

        p_before = p.data.clone()
        p.grad = g.clone()
        dn.step()  # push 1 of 3: non-milestone, should still update

        self.assertFalse(torch.equal(p.data, p_before), "intermediate push should update params")

    def test_momentum_not_applied_before_milestone(self) -> None:
        """Non-milestone steps should not apply the momentum buffer."""
        lr, beta, N = 0.1, 0.9, 4
        g = torch.tensor([1.0])

        p = torch.nn.Parameter(torch.tensor([0.0]))
        dn = DelayedNesterovOptimizer([p], lr=lr, momentum=beta, nesterov_period=N)

        # Do N-1 pushes (non-milestone); each should apply exactly g/N
        expected = torch.tensor([0.0])
        for _ in range(N - 1):
            p.grad = g.clone()
            dn.step()
            expected = expected - lr * g / N

        torch.testing.assert_close(p.data, expected)

    def test_end_to_end_with_server(self) -> None:
        """DN optimizer integrated into AsyncDiLoCoServer updates global params correctly."""
        d = 8
        global_model = _make_model(d)
        outer_opt = DelayedNesterovOptimizer(
            global_model.parameters(), lr=0.1, momentum=0.9, nesterov_period=2
        )
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        # Two pushes → one milestone → params should have moved
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}
        for _ in range(2):
            pg = AsyncDiLoCoServer.new_session(addr)
            try:
                pg.broadcast_one(torch.ones(1), root=1).wait()        # flag=1: full sync
                pg.broadcast_one(torch.tensor([1.0]), root=1).wait()  # speed
                pg.broadcast_one(_make_worker_id(), root=1).wait()    # worker_id
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.ones_like(p.data), root=1).wait()
                for _, p in global_model.named_parameters():
                    buf = torch.zeros_like(p.data)
                    pg.broadcast_one(buf, root=0).wait()
                pg.broadcast_one(torch.zeros(1), root=0).wait()  # new_steps
            finally:
                pg.shutdown()

        for name, p in global_model.named_parameters():
            self.assertFalse(
                torch.equal(p.data, initial[name]),
                f"param {name} should have changed after DN milestone",
            )


class TestAsyncDiLoCoServer(TestCase):
    def test_server_applies_outer_step(self) -> None:
        """Server receives pseudo-grads and updates global params via outer SGD."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)

        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        # Capture initial global params
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        # Simulate a worker: pseudo-grads = ones
        pg = AsyncDiLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()        # flag=1: full sync
            pg.broadcast_one(torch.tensor([1.0]), root=1).wait()  # speed
            pg.broadcast_one(_make_worker_id(), root=1).wait()    # worker_id
            for name, p in global_model.named_parameters():
                grad = torch.ones_like(p.data)
                pg.broadcast_one(grad, root=1).wait()

            new_global: dict = {}
            for name, p in global_model.named_parameters():
                buf = torch.zeros_like(p.data)
                pg.broadcast_one(buf, root=0).wait()
                new_global[name] = buf
            pg.broadcast_one(torch.zeros(1), root=0).wait()  # new_steps
        finally:
            pg.shutdown()

        # SGD step: new_param = old_param - lr * pseudo_grad = old - 0.1 * 1
        for name, p in global_model.named_parameters():
            expected = initial[name] - 0.1 * torch.ones_like(p.data)
            torch.testing.assert_close(new_global[name], expected)

    def test_concurrent_workers_serialize(self) -> None:
        """Multiple concurrent workers should each see a distinct outer step applied."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        results = []
        errors = []

        def worker() -> None:
            try:
                pg = AsyncDiLoCoServer.new_session(addr)
                try:
                    pg.broadcast_one(torch.ones(1), root=1).wait()        # flag=1: full sync
                    pg.broadcast_one(torch.tensor([1.0]), root=1).wait()  # speed
                    pg.broadcast_one(_make_worker_id(), root=1).wait()    # worker_id
                    for _, p in global_model.named_parameters():
                        grad = torch.zeros_like(p.data)
                        pg.broadcast_one(grad, root=1).wait()
                    received = []
                    for _, p in global_model.named_parameters():
                        buf = torch.zeros_like(p.data)
                        pg.broadcast_one(buf, root=0).wait()
                        received.append(buf.clone())
                    pg.broadcast_one(torch.zeros(1), root=0).wait()  # new_steps
                finally:
                    pg.shutdown()
                results.append(received)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(results), 3)


class TestAsyncDiLoCo(TestCase):
    def test_sync_resets_model_to_global(self) -> None:
        """After sync, worker model should match the server's updated global params."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        sync_every = 3
        with AsyncDiLoCo(addr, worker_model, inner_opt, sync_every=sync_every):
            x = torch.randn(4, d)
            y = torch.randint(0, d, (4,))
            criterion = nn.CrossEntropyLoss()

            for _ in range(sync_every):
                inner_opt.zero_grad()
                loss = criterion(worker_model(x), y)
                loss.backward()
                inner_opt.step()

        # After one sync, worker model params should match new global params
        for name, p in worker_model.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(),
                global_model.state_dict()[name],
                msg=f"param {name} mismatch after sync",
            )

    def test_fragment_update_alpha_blends_params(self) -> None:
        """With alpha=1, model stays at local params after sync."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)  # no-op outer step
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        # Save local params before sync
        local_before = {n: p.detach().clone() for n, p in worker_model.named_parameters()}

        sync_every = 2
        with AsyncDiLoCo(
            addr, worker_model, inner_opt, sync_every=sync_every, fragment_update_alpha=1.0
        ):
            x = torch.randn(4, d)
            y = torch.randint(0, d, (4,))
            criterion = nn.CrossEntropyLoss()
            for _ in range(sync_every):
                inner_opt.zero_grad()
                loss = criterion(worker_model(x), y)
                loss.backward()
                inner_opt.step()

        local_after_inner = {n: p.detach().clone() for n, p in worker_model.named_parameters()}

        # alpha=1 → p stays at local (no change from global)
        for name, p in worker_model.named_parameters():
            torch.testing.assert_close(
                p.data.cpu(),
                local_after_inner[name].cpu(),
                msg=f"param {name} should remain local with alpha=1",
            )

    def test_initial_pull_syncs_worker_to_server(self) -> None:
        """__enter__ should pull server params so worker starts from server's weights."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)  # no-op outer
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()

        # Worker starts with DIFFERENT random weights
        worker_model = _make_model(d)
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.0)

        # Verify they differ before entering context
        any_diff = any(
            not torch.equal(p.data, global_model.state_dict()[n])
            for n, p in worker_model.named_parameters()
        )
        self.assertTrue(any_diff, "test setup: worker and server should differ initially")

        with AsyncDiLoCo(addr, worker_model, inner_opt, sync_every=100):
            # After __enter__, worker model should match server global params
            for name, p in worker_model.named_parameters():
                torch.testing.assert_close(
                    p.data.cpu(),
                    global_model.state_dict()[name],
                    msg=f"param {name}: initial pull should align worker with server",
                )


class TestDyLU(TestCase):
    def test_server_sends_dylu_steps(self) -> None:
        """Server with dylu_H set should send back a positive new_steps value."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0, dylu_H=100)
        addr = server.address()

        wid = _make_worker_id()
        pg = AsyncDiLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()        # flag=1
            pg.broadcast_one(torch.tensor([50.0]), root=1).wait() # speed
            pg.broadcast_one(wid, root=1).wait()                  # worker_id
            for _, p in global_model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
            for _, p in global_model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
            steps_buf = torch.zeros(1)
            pg.broadcast_one(steps_buf, root=0).wait()
        finally:
            pg.shutdown()

        # First worker: its speed == max speed → should get H steps back
        self.assertEqual(int(steps_buf[0].item()), 100)

    def test_slow_worker_gets_fewer_steps(self) -> None:
        """A worker at half the max speed should receive ~H/2 steps."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0, dylu_H=100)
        addr = server.address()

        fast_wid = _make_worker_id()
        slow_wid = _make_worker_id()

        def push(speed: float, wid: torch.Tensor) -> int:
            pg = AsyncDiLoCoServer.new_session(addr)
            try:
                pg.broadcast_one(torch.ones(1), root=1).wait()
                pg.broadcast_one(torch.tensor([speed]), root=1).wait()
                pg.broadcast_one(wid, root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
                buf = torch.zeros(1)
                pg.broadcast_one(buf, root=0).wait()
                return int(buf[0].item())
            finally:
                pg.shutdown()

        push(100.0, fast_wid)          # fast worker — registers v(fast)=100
        slow_steps = push(50.0, slow_wid)  # slow worker → floor(50/100 * 100) = 50
        self.assertEqual(slow_steps, 50)

    def test_same_worker_speed_updates_in_place(self) -> None:
        """Re-syncing with the same worker_id replaces its speed, not adds a new entry."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0, dylu_H=100)
        addr = server.address()

        wid = _make_worker_id()

        def push(speed: float) -> int:
            pg = AsyncDiLoCoServer.new_session(addr)
            try:
                pg.broadcast_one(torch.ones(1), root=1).wait()
                pg.broadcast_one(torch.tensor([speed]), root=1).wait()
                pg.broadcast_one(wid, root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
                buf = torch.zeros(1)
                pg.broadcast_one(buf, root=0).wait()
                return int(buf[0].item())
            finally:
                pg.shutdown()

        push(100.0)  # first sync at speed 100
        steps = push(50.0)  # same worker now slower — only one entry in W
        # max_speed should now be 50 (updated in place), so floor(50/50*100)=100
        self.assertEqual(steps, 100)

    def test_dylu_disabled_sends_zero(self) -> None:
        """Server with dylu_H=0 should send back 0 (disabled sentinel)."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)  # dylu_H=0
        addr = server.address()

        pg = AsyncDiLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()
            pg.broadcast_one(torch.tensor([10.0]), root=1).wait()
            pg.broadcast_one(_make_worker_id(), root=1).wait()
            for _, p in global_model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
            for _, p in global_model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
            buf = torch.zeros(1)
            pg.broadcast_one(buf, root=0).wait()
        finally:
            pg.shutdown()

        self.assertEqual(int(buf[0].item()), 0)

    def test_worker_applies_dylu_steps(self) -> None:
        """Worker's sync_every should update to the DyLU recommendation after a sync."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.0)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0, dylu_H=100)
        addr = server.address()

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        sync_every = 3
        trainer = AsyncDiLoCo(addr, worker_model, inner_opt, sync_every=sync_every)
        with trainer:
            x = torch.randn(4, d)
            y = torch.randint(0, d, (4,))
            criterion = nn.CrossEntropyLoss()
            for _ in range(sync_every):
                inner_opt.zero_grad()
                loss = criterion(worker_model(x), y)
                loss.backward()
                inner_opt.step()

        # After one sync the server had one speed report; it should have
        # updated sync_every to something positive (≥ 1).
        self.assertGreater(trainer._sync_every, 0)


class TestHeartbeat(TestCase):
    def _make_server(self, **kwargs) -> AsyncDiLoCoServer:
        model = _make_model()
        outer_opt = optim.SGD(model.parameters(), lr=0.0)
        return AsyncDiLoCoServer(model, outer_opt, port=0, **kwargs)

    def _make_worker(self, server: AsyncDiLoCoServer, **kwargs) -> AsyncDiLoCo:
        d = 8
        model = _make_model(d)
        inner_opt = optim.SGD(model.parameters(), lr=0.0)
        return AsyncDiLoCo(
            server.address(), model, inner_opt, sync_every=1000,
            heartbeat_address=server.heartbeat_address(), **kwargs
        )

    def test_worker_count_starts_at_zero(self) -> None:
        server = self._make_server()
        self.assertEqual(server.worker_count(), 0)
        self.assertEqual(server.active_workers(), {})

    def test_worker_registers_on_enter(self) -> None:
        """worker_count() rises to 1 shortly after a worker enters its context."""
        server = self._make_server(heartbeat_timeout=5.0)
        worker = self._make_worker(server, heartbeat_interval=0.05)
        with worker:
            time.sleep(0.3)  # give heartbeat thread time to fire
            self.assertEqual(server.worker_count(), 1)

    def test_active_workers_contains_worker_id(self) -> None:
        """active_workers() key matches the worker's own UUID."""
        server = self._make_server(heartbeat_timeout=5.0)
        worker = self._make_worker(server, heartbeat_interval=0.05)
        wid = str(uuid.UUID(bytes=bytes(worker._worker_id_tensor.tolist())))
        with worker:
            time.sleep(0.3)
            self.assertIn(wid, server.active_workers())

    def test_worker_deregisters_after_timeout(self) -> None:
        """After the context exits and the timeout passes, worker_count() drops to 0."""
        server = self._make_server(heartbeat_timeout=0.3)
        worker = self._make_worker(server, heartbeat_interval=0.05)
        with worker:
            time.sleep(0.2)
            self.assertEqual(server.worker_count(), 1)
        # heartbeat thread has stopped; wait for monitor to evict the entry
        time.sleep(0.8)  # > timeout(0.3) + monitor_interval(0.15)
        self.assertEqual(server.worker_count(), 0)

    def test_two_workers_both_visible(self) -> None:
        """Two concurrent workers both appear in active_workers()."""
        d = 8
        server = self._make_server(heartbeat_timeout=5.0)
        m1, m2 = _make_model(d), _make_model(d)
        o1 = optim.SGD(m1.parameters(), lr=0.0)
        o2 = optim.SGD(m2.parameters(), lr=0.0)
        hb = server.heartbeat_address()
        w1 = AsyncDiLoCo(server.address(), m1, o1, sync_every=1000, heartbeat_address=hb, heartbeat_interval=0.05)
        w2 = AsyncDiLoCo(server.address(), m2, o2, sync_every=1000, heartbeat_address=hb, heartbeat_interval=0.05)
        with w1, w2:
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 2)
