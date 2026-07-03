# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import multiprocessing
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn, optim

from torchft.async_diloco import (
    AsyncDiLoCo,
    AsyncDiLoCoServer,
    DelayedNesterovOptimizer,
    _dequantize_int8,
    _GraceBatch,
    _quantize_int8,
)


def _make_model(d: int = 8) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


def _total_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def push_pull(
    addr: str,
    model: nn.Module,
    full_sync: bool = True,
    speed: float = 1.0,
    grad_value: float = 1.0,
    baseline_revision: int = 0,
    quantize: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int, int, bool]:
    """One raw-protocol sync against an AsyncDiLoCoServer (or subclass): a
    single HTTP POST with a JSON header line plus raw tensor bytes.

    Returns ``(params, new_steps, revision, applied)`` where params is the
    unflattened server response keyed by parameter name.
    """
    total = _total_numel(model)
    header: Dict[str, object] = {
        "flag": 1 if full_sync else 0,
        "speed": speed,
        "baseline_revision": baseline_revision,
    }
    body = b""
    if full_sync:
        flat = torch.full((total,), grad_value)
        header["numel"] = total
        if quantize:
            numels = [p.numel() for _, p in model.named_parameters()]
            q, scales = _quantize_int8(flat, numels)
            header["dtype"] = "int8"
            body = scales.numpy().tobytes() + q.numpy().tobytes()
        else:
            header["dtype"] = "float32"
            body = flat.numpy().tobytes()

    request = urllib.request.Request(
        addr,
        data=(json.dumps(header) + "\n").encode() + body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        resp_header = json.loads(resp.readline())
        numel = int(resp_header["numel"])
        flat_params = torch.frombuffer(
            bytearray(resp.read(numel * 4)), dtype=torch.float32
        )
    assert flat_params.numel() == numel

    params: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        params[name] = flat_params[offset : offset + n].view(p.shape).clone()
        offset += n
    return (
        params,
        int(resp_header["new_steps"]),
        int(resp_header["revision"]),
        bool(resp_header["applied"]),
    )


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
            push_pull(addr, global_model, grad_value=1.0)

        for name, p in global_model.named_parameters():
            self.assertFalse(torch.equal(p.data, initial[name]))


class TestAsyncDiLoCoServer(TestCase):
    def test_server_applies_outer_step(self) -> None:
        """Server receives pseudo-grads and updates global params via outer optimizer."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        initial = {n: p.detach().clone() for n, p in global_model.named_parameters()}

        new_global, _, revision, applied = push_pull(
            server.address(), global_model, grad_value=1.0
        )

        self.assertTrue(applied)
        self.assertEqual(revision, 1)
        for name, p in global_model.named_parameters():
            torch.testing.assert_close(
                new_global[name], initial[name] - 0.1 * torch.ones_like(p.data)
            )

    def test_concurrent_workers_serialize(self) -> None:
        """Multiple concurrent workers each complete without errors."""
        global_model = _make_model()
        outer_opt = optim.SGD(global_model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(global_model, outer_opt, port=0)
        addr = server.address()
        results, errors = [], []

        def worker() -> None:
            try:
                params, _, _, _ = push_pull(addr, global_model, grad_value=0.0)
                results.append(params)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(results), 3)

    def test_revision_increments_per_push(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        _, _, rev1, applied1 = push_pull(server.address(), model)
        _, _, rev2, applied2 = push_pull(server.address(), model)
        self.assertTrue(applied1 and applied2)
        self.assertEqual((rev1, rev2), (1, 2))

    def test_session_cap_returns_503(self) -> None:
        """R7: syncs beyond max_sessions get 503 instead of exhausting
        server threads (monitoring endpoints stay available)."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            max_sessions=0,  # every sync is over capacity
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            push_pull(server.address(), model)
        self.assertEqual(ctx.exception.code, 503)
        # /status is unaffected by the sync session cap.
        with urllib.request.urlopen(server.status_address()) as resp:
            self.assertEqual(resp.status, 200)

    def test_stale_baseline_rejected(self) -> None:
        """A push whose baseline revision is ahead of the server (checkpoint-restore
        scenario) must be rejected without touching the global params."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}

        params, _, revision, applied = push_pull(
            server.address(), model, baseline_revision=5
        )

        self.assertFalse(applied)
        self.assertEqual(revision, 0)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.data, initial[name])
            torch.testing.assert_close(params[name], initial[name])


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

    def test_worker_ids_unique_per_instance(self) -> None:
        """Worker heartbeat ids must be unique per instance (not a module counter)."""
        d = 8
        w1 = AsyncDiLoCo("http://unused", _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.0), sync_every=10)
        w2 = AsyncDiLoCo("http://unused", _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.0), sync_every=10)
        self.assertNotEqual(w1._worker_id, w2._worker_id)

    def test_quantized_upload_download_stays_fp32(self) -> None:
        """R5: quantization applies to the upload only; the pulled params must
        equal the server's authoritative fp32 params exactly."""
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0,
            should_quantize=True,
        )
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)

        with AsyncDiLoCo(
            server.address(), worker_model, inner_opt, sync_every=2,
            should_quantize=True,
        ):
            x, y = torch.randn(4, d), torch.randint(0, d, (4,))
            for _ in range(2):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        for name, p in worker_model.named_parameters():
            self.assertTrue(
                torch.equal(p.data.cpu(), global_model.state_dict()[name]),
                f"download was degraded for {name!r}",
            )

    def test_single_worker_matches_sync_diloco(self) -> None:
        """Convergence parity: with one worker, AsyncDiLoCo must reproduce
        hand-rolled synchronous DiLoCo exactly (inner state persists across
        windows, outer SGD on pseudo-gradients)."""
        torch.manual_seed(0)
        d, H, windows = 4, 3, 3
        outer_lr, inner_lr, momentum = 0.5, 0.05, 0.9
        data = [(torch.randn(8, d), torch.randn(8, d)) for _ in range(H * windows)]

        ref_global = nn.Linear(d, d)
        init = {k: v.clone() for k, v in ref_global.state_dict().items()}

        # Reference: synchronous DiLoCo, inner optimizer state persists.
        ref_local = nn.Linear(d, d)
        ref_local.load_state_dict(ref_global.state_dict())
        ref_inner = optim.SGD(ref_local.parameters(), lr=inner_lr, momentum=momentum)
        it = iter(data)
        for _ in range(windows):
            for _ in range(H):
                x, y = next(it)
                ref_inner.zero_grad()
                ((ref_local(x) - y) ** 2).mean().backward()
                ref_inner.step()
            with torch.no_grad():
                for gp, lp in zip(ref_global.parameters(), ref_local.parameters()):
                    gp.data.sub_(outer_lr * (gp.data - lp.data))
                for gp, lp in zip(ref_global.parameters(), ref_local.parameters()):
                    lp.data.copy_(gp.data)

        # AsyncDiLoCo, single worker, same init and data.
        srv_model = nn.Linear(d, d)
        wrk_model = nn.Linear(d, d)
        srv_model.load_state_dict(init)
        wrk_model.load_state_dict(init)
        server = AsyncDiLoCoServer(
            srv_model, optim.SGD(srv_model.parameters(), lr=outer_lr), port=0
        )
        inner = optim.SGD(wrk_model.parameters(), lr=inner_lr, momentum=momentum)
        it = iter(data)
        with AsyncDiLoCo(server.address(), wrk_model, inner, sync_every=H):
            for _ in range(windows * H):
                x, y = next(it)
                inner.zero_grad()
                ((wrk_model(x) - y) ** 2).mean().backward()
                inner.step()

        for gp, sp in zip(ref_global.parameters(), srv_model.parameters()):
            torch.testing.assert_close(sp.data, gp.data, atol=1e-6, rtol=1e-5)


class TestInt8Quantization(TestCase):
    """R5: upload-only blockwise symmetric int8 quantization."""

    def test_roundtrip_error_bound(self) -> None:
        """Per-element error is bounded by max|block|/254 within each block."""
        torch.manual_seed(0)
        numels = [64, 1, 300, 17]
        chunks = [torch.randn(n) * scale for n, scale in zip(numels, [1.0, 100.0, 1e-4, 3.0])]
        flat = torch.cat(chunks)

        q, scales = _quantize_int8(flat, numels)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(len(scales), len(numels))
        out = _dequantize_int8(q, scales, numels)

        offset = 0
        for n, chunk in zip(numels, chunks):
            bound = chunk.abs().max().item() / 254.0 + 1e-7
            err = (out[offset : offset + n] - chunk).abs().max().item()
            self.assertLessEqual(err, bound)
            offset += n

    def test_zero_block_roundtrips_exactly(self) -> None:
        flat = torch.zeros(10)
        q, scales = _quantize_int8(flat, [4, 6])
        torch.testing.assert_close(_dequantize_int8(q, scales, [4, 6]), flat)

    def test_constant_block_roundtrips_exactly(self) -> None:
        """A constant block maps to ±127 exactly — no quantization error."""
        flat = torch.cat([torch.full((8,), 3.5), torch.full((5,), -0.25)])
        q, scales = _quantize_int8(flat, [8, 5])
        torch.testing.assert_close(_dequantize_int8(q, scales, [8, 5]), flat)

    def test_server_applies_quantized_push(self) -> None:
        """An int8 push with constant blocks is applied exactly like fp32."""
        model = _make_model()
        ref_model = _make_model()
        ref_model.load_state_dict(model.state_dict())
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            should_quantize=True,
        )
        ref_server = AsyncDiLoCoServer(
            ref_model, optim.SGD(ref_model.parameters(), lr=0.1), port=0
        )

        params_q, _, _, applied = push_pull(
            server.address(), model, grad_value=1.0, quantize=True
        )
        params_ref, _, _, _ = push_pull(
            ref_server.address(), ref_model, grad_value=1.0
        )

        self.assertTrue(applied)
        for name in params_q:
            torch.testing.assert_close(params_q[name], params_ref[name])


class TestInnerOptimizerState(TestCase):
    """B5: DiLoCo persists inner optimizer state across windows by default;
    clearing is an opt-in deviation."""

    def _run_windows(self, reset_inner_state: bool) -> optim.Optimizer:
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(global_model, optim.SGD(global_model.parameters(), lr=0.0), port=0)
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01, momentum=0.9)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        with AsyncDiLoCo(
            server.address(), worker_model, inner_opt, sync_every=2,
            reset_inner_state=reset_inner_state,
        ):
            for _ in range(2):  # exactly one full window ending in a sync
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()
        return inner_opt

    def test_state_persists_across_windows_by_default(self) -> None:
        inner_opt = self._run_windows(reset_inner_state=False)
        self.assertGreater(len(inner_opt.state), 0)

    def test_reset_inner_state_opt_in(self) -> None:
        inner_opt = self._run_windows(reset_inner_state=True)
        self.assertEqual(len(inner_opt.state), 0)


class TestDyLU(TestCase):
    def _push(self, addr: str, model: nn.Module, speed: float) -> int:
        _, new_steps, _, _ = push_pull(addr, model, speed=speed, grad_value=0.0)
        return new_steps

    def test_first_worker_gets_full_H(self) -> None:
        """First worker is the fastest by definition — receives H steps back."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        self.assertEqual(self._push(server.address(), model, 50.0), 100)

    def test_slow_worker_gets_fewer_steps(self) -> None:
        """Worker at half the reference speed receives H/2 steps."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        addr = server.address()
        self._push(addr, model, 100.0)   # fast worker — registers v=100 in pool
        self.assertEqual(self._push(addr, model, 50.0), 50)

    def test_outlier_does_not_shrink_everyone(self) -> None:
        """R10: with a large pool, one outlier speed must not become the
        reference — the percentile excludes it."""
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0, dylu_H=100)
        addr = server.address()
        for _ in range(15):
            self._push(addr, model, 100.0)
        self._push(addr, model, 1000.0)  # one mis-measured outlier window
        # Reference is p90 of the pool (=100), not the 1000 outlier, so a
        # normal worker keeps its full window instead of dropping to 10.
        self.assertEqual(self._push(addr, model, 100.0), 100)

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


class TestGracePeriod(TestCase):
    def _make_server(self, grace_period: float) -> Tuple[AsyncDiLoCoServer, nn.Module]:
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            grace_period=grace_period,
        )
        return server, model

    def _grads(self, model: nn.Module, value: float) -> Dict[str, torch.Tensor]:
        return {n: torch.full_like(p, value) for n, p in model.named_parameters()}

    def test_two_workers_share_one_batch(self) -> None:
        """Two pushes inside the grace window are applied as one batch: both
        workers receive the same post-batch snapshot and revision."""
        server, model = self._make_server(grace_period=0.5)
        addr = server.address()
        results, errors = [], []

        def worker() -> None:
            try:
                results.append(push_pull(addr, model, grad_value=1.0))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(results), 2)
        (p1, _, rev1, ap1), (p2, _, rev2, ap2) = results
        self.assertTrue(ap1 and ap2)
        self.assertEqual(rev1, rev2)
        self.assertEqual(rev1, 2)  # one outer step per worker in the batch
        for name in p1:
            torch.testing.assert_close(p1[name], p2[name])

    def test_batch_detached_at_claim_time(self) -> None:
        """B2: once a processor is elected the batch is closed — a late
        arrival opens a fresh batch instead of racing the processor."""
        server, model = self._make_server(grace_period=0.05)

        batch, is_processor = server._grace_accumulate_and_wait(
            self._grads(model, 1.0), 1.0
        )
        self.assertTrue(is_processor)
        # Batch was detached at claim time: late arrivals can't join it.
        self.assertIsNone(server._grace_batch)

        late_batch, late_is_processor = server._grace_accumulate_and_wait(
            self._grads(model, 2.0), 1.0
        )
        self.assertTrue(late_is_processor)
        self.assertIsNot(late_batch, batch)
        self.assertEqual(len(batch.grads_list), 1)
        self.assertEqual(len(late_batch.grads_list), 1)

    def test_processor_failure_publishes_error_to_waiters(self) -> None:
        """B2: if the elected processor dies, waiters get an error promptly
        instead of spinning until the transport timeout."""
        server, model = self._make_server(grace_period=30.0)  # deadline far out
        outcome: Dict[str, object] = {}

        def waiter() -> None:
            batch, is_processor = server._grace_accumulate_and_wait(
                self._grads(model, 1.0), 1.0
            )
            outcome["is_processor"] = is_processor
            outcome["error"] = batch.error

        t = threading.Thread(target=waiter)
        t.start()

        # Wait for the waiter to open the batch, then claim it on its behalf
        # (simulating another session's processor) and publish a failure.
        deadline = time.monotonic() + 5
        while server._grace_batch is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(server._grace_batch)
        with server._grace_cond:
            batch = server._grace_batch
            batch.claimed = True
            server._grace_batch = None
            server._grace_cond.notify_all()

        server._grace_batch_publish(batch, error="RuntimeError: boom")

        t.join(timeout=5)  # well under the 30s deadline
        self.assertFalse(t.is_alive(), "waiter hung after processor failure")
        self.assertEqual(outcome["is_processor"], False)
        self.assertEqual(outcome["error"], "RuntimeError: boom")

    def test_failed_batch_returns_http_error(self) -> None:
        """A sync whose grace batch failed must get a fast HTTP 500 so the
        worker drops the push and resyncs — no hanging until a transport
        timeout."""
        server, model = self._make_server(grace_period=0.05)
        addr = server.address()

        with patch.object(
            server, "_apply_one", side_effect=RuntimeError("optimizer exploded")
        ):
            start = time.monotonic()
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                push_pull(addr, model, grad_value=1.0)
            self.assertEqual(ctx.exception.code, 500)
            self.assertLess(time.monotonic() - start, 30)


class TestWorkerResilience(TestCase):
    """B3/R11: a PS outage must not kill the fleet."""

    def test_worker_survives_server_outage_and_resyncs(self) -> None:
        d = 8
        global_model = _make_model(d)
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1), port=0
        )
        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        x, y = torch.randn(4, d), torch.randint(0, d, (4,))

        def step() -> None:
            inner_opt.zero_grad()
            nn.CrossEntropyLoss()(worker_model(x), y).backward()
            inner_opt.step()

        trainer = AsyncDiLoCo(server.address(), worker_model, inner_opt, sync_every=2)
        with trainer:
            good_addr = trainer._server_address
            # Simulate a PS outage: nothing listens here.
            trainer._server_address = "http://127.0.0.1:9/sync"

            for _ in range(2):
                step()  # boundary sync fails — must NOT raise
            self.assertTrue(trainer._pending_resync)

            for _ in range(2):
                step()  # training continues while the server is down
            self.assertTrue(trainer._pending_resync)

            # Server comes back.
            trainer._server_address = good_addr
            trainer._resync_at = 0.0
            for _ in range(2):
                step()  # boundary triggers the pull-only resync
            self.assertFalse(trainer._pending_resync)

            # Worker re-baselined to the server's authoritative params.
            for name, p in worker_model.named_parameters():
                torch.testing.assert_close(
                    p.data.cpu(), global_model.state_dict()[name]
                )


class TestCheckpoint(TestCase):
    def test_checkpoint_written_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "server.ckpt")
            model = _make_model()
            server = AsyncDiLoCoServer(
                model, optim.SGD(model.parameters(), lr=0.1), port=0,
                checkpoint_path=path, checkpoint_every=1,
            )
            push_pull(server.address(), model, grad_value=1.0)
            self.assertTrue(os.path.exists(path))
            expected = {n: p.detach().clone() for n, p in model.named_parameters()}

            # A fresh server restores model + revision from the checkpoint.
            model2 = _make_model()
            server2 = AsyncDiLoCoServer(
                model2, optim.SGD(model2.parameters(), lr=0.1), port=0,
                checkpoint_path=path,
            )
            self.assertEqual(server2._revision, 1)
            for name, p in model2.named_parameters():
                torch.testing.assert_close(p.data, expected[name])

            # And serves the restored params (with the restored revision).
            params, _, revision, _ = push_pull(
                server2.address(), model2, full_sync=False
            )
            self.assertEqual(revision, 1)
            for name in params:
                torch.testing.assert_close(params[name], expected[name])

    def test_save_checkpoint_atomic_no_tmp_left(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "server.ckpt")
            model = _make_model()
            server = AsyncDiLoCoServer(
                model, optim.SGD(model.parameters(), lr=0.1), port=0
            )
            server.save_checkpoint(path)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))


class TestAdvertiseHost(TestCase):
    """B4: all worker-facing addresses must honor advertise_host so
    multi-host deployments aren't at the mercy of socket.gethostname()."""

    def _make_server(self, **kwargs) -> AsyncDiLoCoServer:
        model = _make_model()
        return AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.0), port=0, **kwargs
        )

    def test_addresses_use_advertise_host(self) -> None:
        server = self._make_server(advertise_host="ps.example.com")
        self.assertIn("//ps.example.com:", server.address())
        self.assertIn("//ps.example.com:", server.heartbeat_address())
        self.assertIn("//ps.example.com:", server.status_address())

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"TORCHFT_PS_ADVERTISE_HOST": "10.1.2.3"}):
            server = self._make_server()
        self.assertIn("//10.1.2.3:", server.address())
        self.assertIn("//10.1.2.3:", server.heartbeat_address())

    def test_explicit_arg_beats_env(self) -> None:
        with patch.dict(os.environ, {"TORCHFT_PS_ADVERTISE_HOST": "10.1.2.3"}):
            server = self._make_server(advertise_host="ps.example.com")
        self.assertIn("//ps.example.com:", server.address())

    def test_advertised_sessions_work_end_to_end(self) -> None:
        """A worker syncing via an advertised (non-gethostname) address must
        complete a full push/pull."""
        model = _make_model()
        server = AsyncDiLoCoServer(
            model, optim.SGD(model.parameters(), lr=0.1), port=0,
            advertise_host="localhost",
        )
        self.assertIn("//localhost:", server.address())
        _, _, revision, applied = push_pull(server.address(), model)
        self.assertTrue(applied)
        self.assertEqual(revision, 1)


class TestStatusEndpoint(TestCase):
    def test_status_reports_progress(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.1), port=0)
        push_pull(server.address(), model)

        with urllib.request.urlopen(server.status_address()) as resp:
            status = json.loads(resp.read())

        self.assertEqual(status["revision"], 1)
        self.assertEqual(status["applied_pushes"], 1)
        self.assertEqual(status["worker_count"], 0)
        self.assertIsNotNone(status["last_outer_step_time"])


class TestShutdown(TestCase):
    def test_shutdown_stops_server(self) -> None:
        model = _make_model()
        server = AsyncDiLoCoServer(model, optim.SGD(model.parameters(), lr=0.0), port=0)
        addr = server.address()
        server.shutdown()
        self.assertTrue(server._shutdown_event.is_set())
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(addr, timeout=5)


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
        w1 = self._make_worker(server, heartbeat_interval=0.05)
        w2 = self._make_worker(server, heartbeat_interval=0.05)
        with w1, w2:
            time.sleep(0.3)
            self.assertEqual(server.worker_count(), 2)

    def test_single_port_for_all_endpoints(self) -> None:
        """R6: one port to open/advertise — /sync, /heartbeat and /status all
        live on the same server."""
        server = self._make_server()
        sync_port = server.address().rsplit(":", 1)[1].split("/")[0]
        hb_port = server.heartbeat_address().rsplit(":", 1)[1].split("/")[0]
        status_port = server.status_address().rsplit(":", 1)[1].split("/")[0]
        self.assertEqual(sync_port, hb_port)
        self.assertEqual(hb_port, status_port)


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


def _mp_worker_main(addr: str, hb_addr: str, d: int, queue) -> None:
    """Real multi-process worker: separate interpreter, real sockets."""
    try:
        import torch
        from torch import nn, optim

        from torchft.async_diloco import AsyncDiLoCo

        model = nn.Sequential(nn.Linear(d, d))
        inner = optim.SGD(model.parameters(), lr=0.01)
        with AsyncDiLoCo(
            addr, model, inner, sync_every=2,
            heartbeat_address=hb_addr, heartbeat_interval=0.05,
        ) as trainer:
            x, y = torch.randn(4, d), torch.randn(4, d)
            for _ in range(4):
                inner.zero_grad()
                ((model(x) - y) ** 2).mean().backward()
                inner.step()
            # Stay alive briefly so the parent can observe both heartbeats.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                time.sleep(0.05)
        queue.put(("ok", trainer._worker_id))
    except Exception as e:
        queue.put(("error", repr(e)))


class TestMultiProcess(TestCase):
    """Real multi-process test over actual sockets: two separate worker
    processes must register distinct heartbeat ids (B1) and sync against a
    server addressed by hostname (B4)."""

    def test_two_processes_sync_and_register(self) -> None:
        d = 4
        global_model = nn.Sequential(nn.Linear(d, d))
        server = AsyncDiLoCoServer(
            global_model, optim.SGD(global_model.parameters(), lr=0.1),
            port=0, advertise_host="localhost",
        )

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_mp_worker_main,
                args=(server.address(), server.heartbeat_address(), d, queue),
            )
            for _ in range(2)
        ]
        for p in procs:
            p.start()
        try:
            # Both processes must appear as distinct active workers.
            deadline = time.monotonic() + 60
            while server.worker_count() < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertEqual(server.worker_count(), 2)

            results = [queue.get(timeout=60) for _ in range(2)]
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()

        statuses = [r[0] for r in results]
        self.assertEqual(statuses, ["ok", "ok"], f"worker failures: {results}")
        worker_ids = {r[1] for r in results}
        self.assertEqual(len(worker_ids), 2, "worker ids collided across processes")
        # Both workers pushed at least once.
        self.assertGreaterEqual(server._revision, 2)
