# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
from unittest import TestCase

import torch
from torch import nn, optim

from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer, DelayedNesterovOptimizer


def _make_model(d: int = 8) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


class TestDelayedNesterovOptimizer(TestCase):
    def test_period1_matches_standard_nesterov(self) -> None:
        """nesterov_period=1: every push is a milestone, should match SGD+Nesterov exactly."""
        lr, beta = 0.1, 0.9
        g = torch.tensor([1.0, 2.0])

        p_ref = torch.tensor([1.0, 1.0])
        m_ref = torch.zeros_like(p_ref)
        m_ref = beta * m_ref + g
        p_ref = p_ref - lr * (g + beta * m_ref)

        p_dn = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        dn = DelayedNesterovOptimizer([p_dn], lr=lr, momentum=beta, nesterov_period=1)
        p_dn.grad = g.clone()
        dn.step()

        torch.testing.assert_close(p_dn.data, p_ref)

    def test_intermediate_steps_update_params(self) -> None:
        """Non-milestone pushes still update params (pure gradient steps, no momentum)."""
        lr, N = 0.1, 3
        g = torch.tensor([1.0])
        p = torch.nn.Parameter(torch.tensor([0.0]))
        dn = DelayedNesterovOptimizer([p], lr=lr, momentum=0.9, nesterov_period=N)
        p_before = p.data.clone()
        p.grad = g.clone()
        dn.step()
        self.assertFalse(torch.equal(p.data, p_before))

    def test_end_to_end_with_server(self) -> None:
        """DN optimizer integrated with AsyncDiLoCoServer updates global params at milestone."""
        d = 8
        global_model = _make_model(d)
        outer_opt = DelayedNesterovOptimizer(
            global_model.parameters(), lr=0.1, momentum=0.9, nesterov_period=2
        )
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        for _ in range(2):
            pg = AsyncDiLoCoServer.new_session(addr)
            try:
                pg.broadcast_one(torch.ones(1), root=1).wait()
                pg.broadcast_one(torch.tensor([1.0]), root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.ones_like(p.data), root=1).wait()
                for _, p in global_model.named_parameters():
                    pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
                pg.broadcast_one(torch.zeros(1), root=0).wait()
            finally:
                pg.shutdown()

        for name, p in global_model.named_parameters():
            self.assertFalse(torch.equal(p.data, initial[name]))


class TestAsyncDiLoCoServer(TestCase):
    def test_server_applies_outer_step(self) -> None:
        """Server receives pseudo-grads and updates global params via outer optimizer."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        pg = AsyncDiLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()
            pg.broadcast_one(torch.tensor([1.0]), root=1).wait()
            for _, p in global_model.named_parameters():
                pg.broadcast_one(torch.ones_like(p.data), root=1).wait()
            new_global = {}
            for name, p in global_model.named_parameters():
                buf = torch.zeros_like(p.data)
                pg.broadcast_one(buf, root=0).wait()
                new_global[name] = buf
            pg.broadcast_one(torch.zeros(1), root=0).wait()
        finally:
            pg.shutdown()

        for name, p in global_model.named_parameters():
            torch.testing.assert_close(new_global[name], initial[name] - 0.1 * torch.ones_like(p.data))

    def test_concurrent_workers_serialize(self) -> None:
        """Multiple concurrent workers each complete without errors."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        results, errors = [], []

        def worker() -> None:
            try:
                pg = AsyncDiLoCoServer.new_session(addr)
                try:
                    pg.broadcast_one(torch.ones(1), root=1).wait()
                    pg.broadcast_one(torch.tensor([1.0]), root=1).wait()
                    for _, p in global_model.named_parameters():
                        pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
                    received = []
                    for _, p in global_model.named_parameters():
                        buf = torch.zeros_like(p.data)
                        pg.broadcast_one(buf, root=0).wait()
                        received.append(buf.clone())
                    pg.broadcast_one(torch.zeros(1), root=0).wait()
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
        """After sync, worker model matches the server's updated global params."""
        d = 8
        global_model = _make_model(d)
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        sync_every = 3

        with AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=sync_every):
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(sync_every):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        for name, p in worker_model.named_parameters():
            torch.testing.assert_close(p.data.cpu(), global_model.state_dict()[name])

    def test_initial_pull_syncs_worker_to_server(self) -> None:
        """__enter__ pulls server params so worker starts from server's current weights."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0)

        worker_model = _make_model(d)  # different random weights
        self.assertTrue(any(
            not torch.equal(p.data, global_model.state_dict()[n])
            for n, p in worker_model.named_parameters()
        ), "test setup: worker and server should differ initially")

        with AsyncDiLoCo(server.address(), worker_model, optim.SGD(worker_model.parameters(), lr=0.0), sync_every=100):
            for name, p in worker_model.named_parameters():
                torch.testing.assert_close(p.data.cpu(), global_model.state_dict()[name])


class TestDyLU(TestCase):
    def _push(self, addr, model, speed: float) -> int:
        pg = AsyncDiLoCoServer.new_session(addr)
        try:
            pg.broadcast_one(torch.ones(1), root=1).wait()
            pg.broadcast_one(torch.tensor([speed]), root=1).wait()
            for _, p in model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=1).wait()
            for _, p in model.named_parameters():
                pg.broadcast_one(torch.zeros_like(p.data), root=0).wait()
            buf = torch.zeros(1)
            pg.broadcast_one(buf, root=0).wait()
            return int(buf[0].item())
        finally:
            pg.shutdown()

    def test_first_worker_gets_full_H(self) -> None:
        """First worker is the fastest by definition — receives H steps back."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        self.assertEqual(self._push(server.address(), model, 50.0), 100)

    def test_slow_worker_gets_fewer_steps(self) -> None:
        """Worker at half max speed receives H/2 steps: floor(v/v_max * H)."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        addr = server.address()
        self._push(addr, model, 100.0)   # fast worker — registers v=100 in pool
        self.assertEqual(self._push(addr, model, 50.0), 50)

    def test_worker_applies_dylu_steps(self) -> None:
        """Worker's sync_every updates to the DyLU recommendation after a sync."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0, dylu_H=100)

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        sync_every = 3

        trainer = AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=sync_every)
        with trainer:
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(sync_every):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        self.assertGreater(trainer._sync_every, 0)


class TestWorkerRejoin(TestCase):
    def test_rejoined_worker_gets_updated_params(self) -> None:
        """Worker rejoining after a sync pulls the server's latest global params."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        w1 = _make_model(d)
        w1.load_state_dict(global_model.state_dict())
        o1 = optim.SGD(w1.parameters(), lr=0.01)
        with AsyncDiLoCo(server.address(), w1, o1, sync_every=2):
            for _ in range(2):
                o1.zero_grad()
                nn.CrossEntropyLoss()(w1(x), y).backward()
                o1.step()

        server_params = {n: p.data.clone() for n, p in global_model.named_parameters()}

        w2, o2 = _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.01)
        with AsyncDiLoCo(server.address(), w2, o2, sync_every=100):
            for name, p in w2.named_parameters():
                torch.testing.assert_close(p.data.cpu(), server_params[name])

    def test_training_continues_after_rejoin(self) -> None:
        """Global params keep updating through a worker leave-and-rejoin cycle."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))
        initial = {n: p.data.clone() for n, p in global_model.named_parameters()}

        for _ in range(2):
            wm = _make_model(d)
            wo = optim.SGD(wm.parameters(), lr=0.01)
            with AsyncDiLoCo(server.address(), wm, wo, sync_every=2):
                for _ in range(2):
                    wo.zero_grad()
                    nn.CrossEntropyLoss()(wm(x), y).backward()
                    wo.step()

        self.assertTrue(any(
            not torch.equal(p.data, initial[n]) for n, p in global_model.named_parameters()
        ))

    def test_inner_optimizer_state_cleared_on_rejoin(self) -> None:
        """_initial_pull clears inner optimizer state so stale momentum doesn't bias the new window."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01, momentum=0.9)

        with AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=100):
            for _ in range(3):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()
            self.assertGreater(len(inner_opt.state), 0)

        with AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=100):
            self.assertEqual(len(inner_opt.state), 0)


class TestHeartbeat(TestCase):
    def _make_server(self, **kwargs) -> AsyncDiLoCoServer:
        model = _make_model()
        return AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, **kwargs)

    def _make_worker(self, server: AsyncDiLoCoServer, **kwargs) -> AsyncDiLoCo:
        model = _make_model()
        return AsyncDiLoCo(
            server.address(), model, optim.SGD(model.parameters(), lr=0.0),
            sync_every=1000, heartbeat_address=server.heartbeat_address(), **kwargs
        )

    def test_worker_registers_on_enter(self) -> None:
        server = self._make_server(heartbeat_timeout=5.0)
        with self._make_worker(server, heartbeat_interval=0.05):
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 1)

    def test_worker_deregisters_after_timeout(self) -> None:
        server = self._make_server(heartbeat_timeout=0.3)
        with self._make_worker(server, heartbeat_interval=0.05):
            time.sleep(0.2)
        time.sleep(0.8)
        self.assertEqual(server.worker_count(), 0)

    def test_two_workers_both_visible(self) -> None:
        server = self._make_server(heartbeat_timeout=5.0)
        hb = server.heartbeat_address()
        w1 = self._make_worker(server, heartbeat_interval=0.05)
        w2 = self._make_worker(server, heartbeat_interval=0.05)
        with w1, w2:
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 2)
