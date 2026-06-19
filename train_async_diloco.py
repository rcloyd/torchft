# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os
import random
import time

REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "CUDA_VISIBLE_DEVICES", str(REPLICA_GROUP_ID % 4)
)

import torch
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from torchft.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer, DelayedNesterovOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    OUTPUT_SUBDIR = os.environ.get("OUTPUT_SUBDIR", "async")
    SYNC_EVERY = int(os.environ.get("SYNC_EVERY", 100))
    N_LAYERS = int(os.environ.get("N_LAYERS", 4))
    D_HID = int(os.environ.get("D_HID", 128))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
    MAX_STEPS = int(os.environ.get("MAX_STEPS", 500))
    DN_PERIOD = int(os.environ.get("DN_PERIOD", 4))
    WORKER_DELAY_MS = float(os.environ.get("WORKER_DELAY_MS", 0))
    WORKER_DELAY_RANDOM_MIN = float(os.environ.get("WORKER_DELAY_RANDOM_MIN", -1))
    WORKER_DELAY_RANDOM_MAX = float(os.environ.get("WORKER_DELAY_RANDOM_MAX", -1))
    use_random_delay = WORKER_DELAY_RANDOM_MIN >= 0 and WORKER_DELAY_RANDOM_MAX >= 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int = 100000, feature_dim: int = 128, num_classes: int = 10):
            self.size = size
            self.feature_dim = feature_dim
            self.num_classes = num_classes

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int):
            features = torch.rand(self.feature_dim)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return features, label

    class MLPModule(nn.Module):
        def __init__(self, d_hid: int, n_layers: int):
            super().__init__()
            layers = []
            for _ in range(n_layers):
                layers += [nn.Linear(d_hid, d_hid), nn.ReLU()]
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # Server setup: start inline if no address provided, otherwise connect to existing.
    server_addr = os.environ.get("ASYNC_DILOCO_SERVER_ADDR", "")
    server = None
    if not server_addr:
        global_model = MLPModule(D_HID, N_LAYERS)
        outer_optimizer: optim.Optimizer = DelayedNesterovOptimizer(
            global_model.parameters(), lr=0.7, momentum=0.9, nesterov_period=DN_PERIOD,
        )
        server = AsyncDiLoCoServer(global_model, outer_optimizer, port=0)
        server_addr = server.address()
        logger.info(f"AsyncDiLoCoServer started at {server_addr}")

    if os.environ.get("SERVER_ONLY", "0") == "1":
        logger.info("SERVER_ONLY=1: set ASYNC_DILOCO_SERVER_ADDR=%s in workers", server_addr)
        while True:
            time.sleep(3600)

    m = MLPModule(D_HID, N_LAYERS).to(device)
    if server is not None:
        m.load_state_dict(server._model.state_dict())

    inner_optimizer: optim.Optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    criterion = nn.CrossEntropyLoss()

    trainset = DummyDataset(size=100000, feature_dim=D_HID)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True
    )

    output_folder = f"output/{OUTPUT_SUBDIR}/replica-{REPLICA_GROUP_ID}"
    os.makedirs(output_folder, exist_ok=True)
    writer = SummaryWriter(f"{output_folder}/tensorboard", max_queue=1000)

    csv_path = f"{output_folder}/metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "step_duration_ms", "is_sync", "wall_time"])

    num_params = sum(p.numel() for p in m.parameters())
    if use_random_delay:
        delay_desc = f"random {WORKER_DELAY_RANDOM_MIN:.0f}–{WORKER_DELAY_RANDOM_MAX:.0f} ms/step"
    else:
        delay_desc = f"{WORKER_DELAY_MS:.0f} ms/step"
    logger.info(f"Worker {REPLICA_GROUP_ID}: {num_params:,} params, delay={delay_desc}")

    # Barrier: wait for all benchmark workers to reach this point before any
    # worker enters the AsyncDiLoCo context and calls _initial_pull.  Without
    # this, fast-starting workers can do several outer syncs before slow-starting
    # workers connect, causing them to receive a partially-trained server model.
    BARRIER_DIR     = os.environ.get("BARRIER_DIR", "")
    BARRIER_WORKERS = int(os.environ.get("BARRIER_WORKERS", "0"))
    if BARRIER_DIR and BARRIER_WORKERS > 0:
        os.makedirs(BARRIER_DIR, exist_ok=True)
        open(os.path.join(BARRIER_DIR, f"ready_{REPLICA_GROUP_ID}"), "w").close()
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            if all(
                os.path.exists(os.path.join(BARRIER_DIR, f"ready_{i}"))
                for i in range(BARRIER_WORKERS)
            ):
                break
            time.sleep(0.05)

    run_start = time.perf_counter()
    step = 0

    with AsyncDiLoCo(
        server_address=server_addr,
        model=m,
        inner_optimizer=inner_optimizer,
        sync_every=SYNC_EVERY,
    ):
        while step < MAX_STEPS:
            for inputs, labels in trainloader:
                if step >= MAX_STEPS:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                inner_optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()

                is_sync = ((step + 1) % SYNC_EVERY == 0)

                if use_random_delay:
                    delay_ms = random.uniform(WORKER_DELAY_RANDOM_MIN, WORKER_DELAY_RANDOM_MAX)
                else:
                    delay_ms = WORKER_DELAY_MS

                t0 = time.perf_counter()
                inner_optimizer.step()
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                step_ms = (time.perf_counter() - t0) * 1000.0

                wall_time = time.perf_counter() - run_start
                loss_val = loss.item()

                writer.add_scalar("loss", loss_val, step)
                writer.add_scalar("step_duration_ms", step_ms, step)
                csv_writer.writerow(
                    [step, f"{loss_val:.6f}", f"{step_ms:.2f}", int(is_sync), f"{wall_time:.3f}"]
                )

                if step % 100 == 0:
                    logger.info(
                        f"[worker {REPLICA_GROUP_ID}] step={step} "
                        f"loss={loss_val:.4f} step_ms={step_ms:.1f}"
                    )
                step += 1

    csv_file.close()
    writer.flush()
    logger.info(f"Worker {REPLICA_GROUP_ID} finished after {step} steps.")


if __name__ == "__main__":
    main()
