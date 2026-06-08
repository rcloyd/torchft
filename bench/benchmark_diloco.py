# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark DiLoCo vs AsyncDiLoCo on the torchtitan Llama3 debugmodel or 1B model.

Two GPUs per island: LOCAL_RANK 0 owns the torchft Manager and DiLoCo/AsyncDiLoCo
wrapper. LOCAL_RANK 1 participates in DDP gradient sync only. At each boundary
LOCAL_RANK 0 broadcasts the outer-updated params to LOCAL_RANK 1 so both stay
in sync before the next window begins.

Usage:
  torchft_lighthouse --min_replicas 2 --bind 127.0.0.1:29511 --join_timeout_ms 10000

  TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=0 MODE=diloco \
    MODEL=1B USE_NCCL=True \
    torchrun --standalone --nproc_per_node=2 benchmark_diloco.py

  TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=1 MODE=diloco \
    MODEL=1B USE_NCCL=True SLOW_MS=50 SLOW_REPLICA=1 \
    torchrun --standalone --nproc_per_node=2 benchmark_diloco.py

Or use run_benchmark.sh to run everything automatically.
"""

import sys
# Local torchft (has AsyncDiLoCo) must come before any venv torchft
sys.path.insert(0, "/home/rileycloyd/torchft")
# torchtitan lives only in the symphony-learn venv
sys.path.insert(1, "/home/rileycloyd/symphony-learn/.venv/lib/python3.10/site-packages")
# User-installed packages (e.g. matplotlib 3.10 compatible with NumPy 2)
# must come before the stale system matplotlib in /usr/lib/python3/dist-packages
sys.path.insert(2, "/home/rileycloyd/.local/lib/python3.10/site-packages")

import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path

# Read LOCAL_RANK / LOCAL_WORLD early — torchrun sets these before exec
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))

# CUDA_VISIBLE_DEVICES is set by run_benchmark.sh before torchrun so that each
# island gets its own contiguous block of physical GPUs. LOCAL_RANK then selects
# within that block. Do not override it here.
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

MODE = os.environ.get("MODE", "diloco")          # "diloco", "async_diloco", or "heloco"
MODEL = os.environ.get("MODEL", "tiny")          # "tiny", "debugmodel", or "1B"
USE_NCCL = os.getenv("USE_NCCL", "False") == "True"
SLOW_MS = int(os.getenv("SLOW_MS", "0"))
SLOW_REPLICA = int(os.getenv("SLOW_REPLICA", "-1"))
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "20"))
NUM_OUTER_STEPS = int(os.getenv("NUM_OUTER_STEPS", "30"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
SEED = int(os.getenv("SEED", "42"))
_BENCH_DIR = Path(__file__).parent
DATA_PATH = os.getenv("DATA_PATH", str(_BENCH_DIR / "data" / "tinyshakespeare.txt"))
NUM_REPLICAS = int(os.getenv("NUM_REPLICAS", "2"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

try:
    from torchtitan.models.llama3 import model_registry
except ImportError:
    model_registry = None  # type: ignore[assignment]

from torchft import Manager, ProcessGroupGloo, ProcessGroupNCCL
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.local_sgd import AsyncDiLoCo, DiLoCo, HeLoCo

logging.basicConfig(level=logging.INFO)

_MODEL_SPEC = None
VOCAB_SIZE: int = 256  # set dynamically in make_model()


class _TinyModel(nn.Module):
    """~50K param byte-level model: embedding + 2-layer MLP over flattened context."""
    def __init__(self, vocab: int = 256, d: int = 128, ctx: int = 8):
        super().__init__()
        self.ctx = ctx
        self.embed = nn.Embedding(vocab, d)
        self.net = nn.Sequential(
            nn.Linear(d * ctx, d * 4), nn.GELU(),
            nn.Linear(d * 4, d * 2), nn.GELU(),
            nn.Linear(d * 2, vocab),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) — predict each position from its ctx previous tokens
        B, T = x.shape
        emb = self.embed(x)                            # (B, T, d)
        # build context windows via unfold
        padded = torch.cat([torch.zeros(B, self.ctx - 1, emb.shape[-1],
                                        device=x.device), emb], dim=1)
        windows = padded.unfold(1, self.ctx, 1)        # (B, T, d, ctx)
        h = windows.reshape(B, T, -1)                  # (B, T, d*ctx)
        return self.net(h)                             # (B, T, vocab)


# ── Dataset ───────────────────────────────────────────────────────────────────
_DATA: torch.Tensor | None = None


def _load_data() -> torch.Tensor:
    global _DATA
    if _DATA is None:
        raw = open(DATA_PATH, "rb").read()
        _DATA = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    return _DATA


def make_model(device: torch.device) -> nn.Module:
    global _MODEL_SPEC, VOCAB_SIZE
    torch.manual_seed(SEED)
    if MODEL == "tiny":
        VOCAB_SIZE = 256
        return _TinyModel(vocab=256).to(device)
    if model_registry is None:
        raise RuntimeError("torchtitan is required for non-tiny models; install it or set MODEL=tiny")
    _MODEL_SPEC = model_registry(MODEL)
    model = _MODEL_SPEC.model.build().to(device)
    VOCAB_SIZE = _MODEL_SPEC.model.vocab_size
    return model


_batch_offset: int = 0


def generate_batch(device: torch.device):
    global _batch_offset
    data = _load_data()
    n = len(data)
    # Stagger starting position by replica so each island sees different data
    base = (REPLICA_GROUP_ID * (n // NUM_REPLICAS) + _batch_offset) % (n - SEQ_LEN - 1)
    xs, ys = [], []
    for i in range(BATCH_SIZE):
        start = (base + i * SEQ_LEN) % (n - SEQ_LEN - 1)
        chunk = data[start : start + SEQ_LEN + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    _batch_offset = (_batch_offset + BATCH_SIZE * SEQ_LEN) % (n - SEQ_LEN - 1)
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y


@record
def main() -> None:
    # ── Intra-island DDP setup ────────────────────────────────────────────────
    using_ddp = LOCAL_WORLD > 1
    if using_ddp:
        # Use Gloo for intra-island DDP gradient sync. Using NCCL here would
        # deadlock with the torchft inter-island NCCL communicator setup
        # (_configure_pg) since both NCCL inits compete for the same rendezvous.
        dist.init_process_group(backend="gloo")

    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Only LOCAL_RANK 0 writes metrics and owns the torchft Manager
    is_primary = LOCAL_RANK == 0
    benchmark_outdir = Path(os.environ.get("BENCHMARK_OUTDIR", str(_BENCH_DIR / "output" / "benchmark")))

    if is_primary:
        output_dir = benchmark_outdir / MODE / f"replica-{REPLICA_GROUP_ID}"
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(f"{output_dir}/tensorboard", max_queue=1000)

    m = make_model(device)
    num_params = sum(p.numel() for p in m.parameters())

    if is_primary:
        print(
            f"[replica-{REPLICA_GROUP_ID} rank-{LOCAL_RANK}] {MODE} | "
            f"model={MODEL} params={num_params:,} | "
            f"data={DATA_PATH} | "
            f"gpus_per_island={LOCAL_WORLD} | "
            f"slow_ms={SLOW_MS if REPLICA_GROUP_ID == SLOW_REPLICA else 0}"
        )

    # Wrap in DDP for intra-island gradient averaging
    if using_ddp:
        ddp_m = DDP(m, device_ids=[LOCAL_RANK])
    else:
        ddp_m = m

    inner_optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    # Outer lr: HeLoCo/MLA use 0.7 (paper default); diloco uses 0.7 with Nesterov;
    # async_diloco uses 0.07. Apply sqrt(K)/K per-update weighting for all async
    # modes (async DiLoCo paper, Table 3 / Appendix A.5).
    _ASYNC_MODES = ("async_diloco", "heloco")
    _HELOCO_LR = float(os.getenv("HELOCO_LR", "0.7"))
    _SGD_LR = float(os.getenv("SGD_LR", "0.7" if MODE == "diloco" else "0.07"))
    _outer_lr = _HELOCO_LR if MODE == "heloco" else _SGD_LR
    if MODE in _ASYNC_MODES:
        _outer_lr *= math.sqrt(NUM_REPLICAS) / NUM_REPLICAS  # = 1/sqrt(K)

    # diloco/async_diloco use explicit outer SGD (Nesterov for diloco per the paper).
    # mla/heloco_no_la/heloco build HeLoCoOptimizer internally via HeLoCo.
    _uses_heloco_opt = MODE == "heloco"
    outer_optimizer = (
        None if _uses_heloco_opt
        else torch.optim.SGD(
            m.parameters(), lr=_outer_lr, momentum=0.9,
            nesterov=True,
        )
    )
    outer_optimizer_ref = {"optimizer": outer_optimizer}

    # ── torchft Manager and DiLoCo — LOCAL_RANK 0 only ───────────────────────
    if is_primary:
        pg = (
            ProcessGroupNCCL(timeout=timedelta(seconds=300))
            if torch.cuda.is_available() and USE_NCCL
            else ProcessGroupGloo(timeout=timedelta(seconds=300))
        )
        transport = HTTPTransport(timeout=timedelta(seconds=300), num_chunks=0)
        # In async modes the manager calls disallow_checkpoint() immediately
        # after each should_commit(), which holds the HTTP write-lock until
        # the NEXT quorum's send_checkpoint(). For a fast model the next
        # quorum starts in <1 ms — far shorter than the time a recovering
        # replica needs to initiate the HTTP fetch — causing a 300-second
        # deadlock. The checkpoint URL already encodes the step number, so
        # the server rejects stale reads regardless; the lock is redundant
        # for this benchmark where both replicas start from the same seed.
        if MODE in _ASYNC_MODES:
            transport.disallow_checkpoint = lambda: None  # type: ignore[method-assign]

        def load_state_dict(state_dict):
            m.load_state_dict(state_dict["model"])
            inner_optimizer.load_state_dict(state_dict["inner_optim"])
            outer_opt = outer_optimizer_ref["optimizer"]
            if outer_opt is not None and "outer_optim" in state_dict:
                outer_opt.load_state_dict(state_dict["outer_optim"])

        def state_dict():
            payload = {
                "model": m.state_dict(),
                "inner_optim": inner_optimizer.state_dict(),
            }
            outer_opt = outer_optimizer_ref["optimizer"]
            if outer_opt is not None:
                payload["outer_optim"] = outer_opt.state_dict()
            return payload

        manager = Manager(
            pg=pg,
            use_async_quorum=(MODE in _ASYNC_MODES),
            min_replica_size=1,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=f"benchmark_{MODE}_{REPLICA_GROUP_ID}",
            timeout=timedelta(seconds=300),
            quorum_timeout=timedelta(seconds=300),
            checkpoint_transport=transport,
        )

        if MODE == "diloco":
            context_cls = DiLoCo
            context_kwargs = dict(
                manager=manager,
                model_fragments=[m],
                inner_optimizer=inner_optimizer,
                outer_optimizer=outer_optimizer,
                sync_every=SYNC_EVERY,
                backup_device=device,
                use_bucketization=True,
                fragment_sync_delay=0,
            )
        elif MODE == "async_diloco":
            context_cls = AsyncDiLoCo
            context_kwargs = dict(
                manager=manager,
                model_fragments=[m],
                inner_optimizer=inner_optimizer,
                outer_optimizer=outer_optimizer,
                sync_every=SYNC_EVERY,
                backup_device=device,
                use_bucketization=True,
            )
        else:  # heloco
            context_cls = HeLoCo
            context_kwargs = dict(
                manager=manager,
                model_fragments=[m],
                inner_optimizer=inner_optimizer,
                outer_lr=_outer_lr,
                sync_every=SYNC_EVERY,
                backup_device=device,
                use_bucketization=True,
            )

        records = {
            "mode": MODE,
            "model": MODEL,
            "replica_id": REPLICA_GROUP_ID,
            "gpus_per_island": LOCAL_WORLD,
            "slow_ms": SLOW_MS if REPLICA_GROUP_ID == SLOW_REPLICA else 0,
            "sync_every": SYNC_EVERY,
            "num_outer_steps": NUM_OUTER_STEPS,
            "num_params": num_params,
            "steps": [],
            "outer_steps": [],
        }

    global_inner_step = 0
    experiment_t0 = time.perf_counter()

    def run_training(diloco_ctx=None):
        nonlocal global_inner_step

        for outer_step in range(NUM_OUTER_STEPS):
            outer_t0 = time.perf_counter()

            for inner_step in range(SYNC_EVERY):
                x, y = generate_batch(device)

                inner_optimizer.zero_grad()
                # Forward through DDP model so gradients are averaged across island ranks
                logits = ddp_m(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                loss.backward()

                # Artificial delay on LOCAL_RANK 0 of the designated slow replica.
                # Simulates slow compute: delays when this island reaches the allreduce,
                # forcing the fast island to wait at the barrier.
                if (
                    REPLICA_GROUP_ID == SLOW_REPLICA
                    and LOCAL_RANK == 0
                    and SLOW_MS > 0
                ):
                    time.sleep(SLOW_MS / 1000.0)

                step_t0 = time.perf_counter()
                inner_optimizer.step()  # DiLoCo boundary hook fires here on last step
                step_t1 = time.perf_counter()

                is_boundary = inner_step == SYNC_EVERY - 1

                # After the boundary, LOCAL_RANK 0 has applied the outer step and
                # reset the model to outer params. Broadcast to keep island in sync.
                if is_boundary and using_ddp:
                    for p in m.parameters():
                        dist.broadcast(p.data, src=0)

                if is_primary:
                    step_ms = (step_t1 - step_t0) * 1000
                    loss_val = loss.item()
                    wall_time = step_t1 - experiment_t0

                    records["steps"].append({
                        "outer_step": outer_step,
                        "inner_step": inner_step,
                        "global_inner_step": global_inner_step,
                        "loss": loss_val,
                        "step_ms": step_ms,
                        "wall_time_s": wall_time,
                        "is_boundary": is_boundary,
                    })
                    writer.add_scalar("loss/inner", loss_val, global_inner_step)
                    writer.add_scalar("timing/step_ms", step_ms, global_inner_step)
                    if is_boundary:
                        writer.add_scalar("timing/boundary_step_ms", step_ms, outer_step)

                global_inner_step += 1

            if is_primary:
                outer_ms = (time.perf_counter() - outer_t0) * 1000
                loss_val = records["steps"][-1]["loss"]
                records["outer_steps"].append({
                    "outer_step": outer_step,
                    "outer_wall_ms": outer_ms,
                    "wall_time_s": time.perf_counter() - experiment_t0,
                })
                writer.add_scalar("timing/outer_step_ms", outer_ms, outer_step)
                print(
                    f"[replica-{REPLICA_GROUP_ID}] outer={outer_step} "
                    f"loss={loss_val:.4f} outer_ms={outer_ms:.0f}"
                )

    if is_primary:
        with context_cls(**context_kwargs) as ctx:
            if _uses_heloco_opt:
                outer_optimizer_ref["optimizer"] = ctx._fragments[0]._outer_optimizer
            run_training()
        writer.flush()
        out_path = benchmark_outdir / MODE / f"replica-{REPLICA_GROUP_ID}" / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[replica-{REPLICA_GROUP_ID}] metrics written to {out_path}")
    else:
        # LOCAL_RANK 1+: train with DDP, wait for broadcasts at boundaries
        run_training()

    if using_ddp:
        dist.destroy_process_group()


# ── Plotting (used when invoked as: python3 benchmark_diloco.py --plot <files>) ──

def _label(r: dict) -> str:
    slow = r["slow_ms"]
    mode = r["mode"].replace("_", " ").title()
    return f"{mode} (slow={slow}ms)" if slow else mode


def _boundary_steps(r: dict) -> list:
    return [s for s in r["steps"] if s["is_boundary"]]


def _plot_loss_vs_walltime(records: list, out: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        wall = [s["wall_time_s"] for s in r["steps"]]
        loss = [s["loss"] for s in r["steps"]]
        ax.plot(wall, loss, label=_label(r), alpha=0.85)
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Wall Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_loss_vs_step(records: list, out: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        xs = [s["global_inner_step"] for s in r["steps"]]
        ys = [s["loss"] for s in r["steps"]]
        ax.plot(xs, ys, label=_label(r), alpha=0.85)
    ax.set_xlabel("Inner step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Inner Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_outer_step_time(records: list, out: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        xs = [o["outer_step"] for o in r["outer_steps"]]
        ys = [o["outer_wall_ms"] for o in r["outer_steps"]]
        ax.plot(xs, ys, label=_label(r), marker="o", markersize=3, alpha=0.85)
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Outer Step Wall Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_boundary_step_ms(records: list, out: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        durations = [s["step_ms"] for s in _boundary_steps(r)]
        ax.hist(durations, bins=20, alpha=0.6, label=_label(r))
    ax.set_xlabel("Last-step duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Boundary Step Duration — DiLoCo blocks, AsyncDiLoCo overlaps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _write_summary(records: list, out: str) -> None:
    import numpy as np

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def _p95(values: list[float]) -> float:
        return float(np.percentile(values, 95)) if values else float("nan")

    lines = []
    for r in records:
        steps = r["steps"]
        outer = r["outer_steps"]
        bsteps = _boundary_steps(r)
        total_wall = outer[-1]["wall_time_s"] if outer else float("nan")
        inner_ms = [s["step_ms"] for s in steps if not s["is_boundary"]]
        boundary_ms = [s["step_ms"] for s in bsteps]
        outer_ms = [o["outer_wall_ms"] for o in outer]
        final_loss = steps[-1]["loss"] if steps else float("nan")
        lines += [
            f"=== {_label(r)} ===",
            f"  Total wall time          : {total_wall:.1f}s",
            f"  Final loss               : {final_loss:.4f}",
            f"  Inner step (non-boundary): mean={_mean(inner_ms):.1f}ms  "
            f"p95={_p95(inner_ms):.1f}ms",
            f"  Boundary step            : mean={_mean(boundary_ms):.1f}ms  "
            f"p95={_p95(boundary_ms):.1f}ms",
            f"  Outer step               : mean={_mean(outer_ms):.1f}ms  "
            f"p95={_p95(outer_ms):.1f}ms",
            f"  Boundary overhead vs inner: "
            f"{(_mean(boundary_ms) - _mean(inner_ms)) if inner_ms and boundary_ms else float('nan'):+.1f}ms/step",
            "",
        ]
    text = "\n".join(lines)
    Path(out).write_text(text)
    print(text)
    print(f"saved {out}")


def plot_main() -> None:
    """
    Generate comparison plots from saved metrics files.

    Usage:
      python3 benchmark_diloco.py --plot <metrics1.json> [metrics2.json ...]
    """
    paths = sys.argv[2:]
    if not paths:
        print("usage: benchmark_diloco.py --plot <metrics1.json> [metrics2.json ...]")
        sys.exit(1)
    records = [json.load(open(p)) for p in paths]
    out_base = Path(os.environ.get("BENCHMARK_OUTDIR", str(_BENCH_DIR / "output" / "benchmark")))
    out_dir = out_base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_loss_vs_walltime(records, str(out_dir / "loss_vs_walltime.png"))
    _plot_loss_vs_step(records, str(out_dir / "loss_vs_step.png"))
    _plot_outer_step_time(records, str(out_dir / "outer_step_time.png"))
    _plot_boundary_step_ms(records, str(out_dir / "boundary_step_ms.png"))
    _write_summary(records, str(out_dir / "summary.txt"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot_main()
    else:
        main()
