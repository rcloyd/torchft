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

import json
import logging
import os
import time
from datetime import timedelta

# Read LOCAL_RANK / LOCAL_WORLD early — torchrun sets these before exec
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))

# CUDA_VISIBLE_DEVICES is set by run_benchmark.sh before torchrun so that each
# island gets its own contiguous block of physical GPUs. LOCAL_RANK then selects
# within that block. Do not override it here.
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

MODE = os.environ.get("MODE", "diloco")          # "diloco" or "async_diloco"
MODEL = os.environ.get("MODEL", "tiny")          # "tiny", "debugmodel", or "1B"
USE_NCCL = os.getenv("USE_NCCL", "False") == "True"
SLOW_MS = int(os.getenv("SLOW_MS", "0"))
SLOW_REPLICA = int(os.getenv("SLOW_REPLICA", "-1"))
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "20"))
NUM_OUTER_STEPS = int(os.getenv("NUM_OUTER_STEPS", "30"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
SEED = int(os.getenv("SEED", "42"))
DATA_PATH = os.getenv("DATA_PATH", "data/tinyshakespeare.txt")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from torchtitan.models.llama3 import model_registry

from torchft import Manager, ProcessGroupGloo, ProcessGroupNCCL
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.local_sgd import AsyncDiLoCo, DiLoCo

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
    base = (REPLICA_GROUP_ID * (n // 2) + _batch_offset) % (n - SEQ_LEN - 1)
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

    if is_primary:
        output_dir = f"output/benchmark/{MODE}/replica-{REPLICA_GROUP_ID}"
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(f"{output_dir}/tensorboard", max_queue=1000)

    m = make_model(device)
    num_params = sum(p.numel() for p in m.parameters())

    if is_primary:
        print(
            f"[replica-{REPLICA_GROUP_ID} rank-{LOCAL_RANK}] {MODE} | "
            f"model={MODEL} params={num_params:,} | "
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
    outer_optimizer = torch.optim.SGD(
        m.parameters(), lr=0.01, momentum=0.9, nesterov=False
    )

    # ── torchft Manager and DiLoCo — LOCAL_RANK 0 only ───────────────────────
    if is_primary:
        pg = (
            ProcessGroupNCCL(timeout=timedelta(seconds=300))
            if torch.cuda.is_available() and USE_NCCL
            else ProcessGroupGloo(timeout=timedelta(seconds=300))
        )
        transport = HTTPTransport(timeout=timedelta(seconds=300), num_chunks=0)

        def load_state_dict(state_dict):
            m.load_state_dict(state_dict["model"])
            inner_optimizer.load_state_dict(state_dict["inner_optim"])
            outer_optimizer.load_state_dict(state_dict["outer_optim"])

        def state_dict():
            return {
                "model": m.state_dict(),
                "inner_optim": inner_optimizer.state_dict(),
                "outer_optim": outer_optimizer.state_dict(),
            }

        manager = Manager(
            pg=pg,
            use_async_quorum=(MODE == "async_diloco"),
            min_replica_size=1,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=f"benchmark_{MODE}_{REPLICA_GROUP_ID}",
            timeout=timedelta(seconds=300),
            quorum_timeout=timedelta(seconds=300),
            checkpoint_transport=transport,
        )

        context_cls = DiLoCo if MODE == "diloco" else AsyncDiLoCo
        context_kwargs = dict(
            manager=manager,
            model_fragments=[m],
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            sync_every=SYNC_EVERY,
            backup_device=device,
            use_bucketization=True,
        )
        # Half-window overlap for DiLoCo: tuned for fast hardware, exposed by stragglers
        if MODE == "diloco":
            context_kwargs["fragment_sync_delay"] = 0

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
        with context_cls(**context_kwargs):
            run_training()
        writer.flush()
        out_path = f"output/benchmark/{MODE}/replica-{REPLICA_GROUP_ID}/metrics.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[replica-{REPLICA_GROUP_ID}] metrics written to {out_path}")
    else:
        # LOCAL_RANK 1+: train with DDP, wait for broadcasts at boundaries
        run_training()

    if using_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
