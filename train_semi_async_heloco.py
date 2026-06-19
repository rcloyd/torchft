# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from datetime import timedelta

REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(REPLICA_GROUP_ID % 4)
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

USE_NCCL = os.getenv("USE_NCCL", "False") == "True"

import torch
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
from torchft import (
    Manager,
    ProcessGroupGloo,
    ProcessGroupNCCL,
)
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.semi_async_heloco import SemiAsyncHeLoCo

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))
    RUN = int(os.environ.get("RUN", 0))

    output_folder = f"output/replica-{REPLICA_GROUP_ID}"

    writer = SummaryWriter(f"{output_folder}/tensorboard", max_queue=1000)

    # SemiAsyncHeLoCo builds its outer optimizer (SemiAsyncHeLoCoOptimizer) internally.
    # We reference it via semi_async_heloco.outer_optimizer after entering the context,
    # using a mutable holder so the state_dict closures can see it.
    outer_optimizer_holder: dict = {"opt": None}

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        inner_optimizer.load_state_dict(state_dict["inner_optim"])
        if outer_optimizer_holder["opt"] is not None and "outer_optim" in state_dict:
            outer_optimizer_holder["opt"].load_state_dict(state_dict["outer_optim"])

    def state_dict():
        payload = {
            "model": m.state_dict(),
            "inner_optim": inner_optimizer.state_dict(),
        }
        if outer_optimizer_holder["opt"] is not None:
            payload["outer_optim"] = outer_optimizer_holder["opt"].state_dict()
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pg = (
        ProcessGroupNCCL(timeout=timedelta(seconds=10))
        if torch.cuda.is_available() and USE_NCCL
        else ProcessGroupGloo(timeout=timedelta(seconds=10))
    )

    transport = HTTPTransport(
        timeout=timedelta(seconds=10),
        num_chunks=0,
    )

    manager = Manager(
        pg=pg,
        use_async_quorum=True,  # required for SemiAsyncHeLoCo (extends SemiAsyncDiLoCo)
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_semi_async_heloco_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
    )

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10000, feature_dim=128, num_classes=10):
            self.size = size
            self.feature_dim = feature_dim
            self.num_classes = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            features = torch.rand(self.feature_dim)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return features, label

    class MLPModule(torch.nn.Module):
        def __init__(self, d_hid: int, n_layers: int):
            super().__init__()
            layers = []
            for _ in range(n_layers):
                layers += [torch.nn.Linear(d_hid, d_hid), torch.nn.ReLU()]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    n_layers = int(os.environ.get("N_LAYERS", 4))
    d_hid = int(os.environ.get("D_HID", 4096))

    m = MLPModule(d_hid, n_layers).to(device)

    batch_size = int(os.environ.get("BATCH_SIZE", 256))
    trainset = DummyDataset(size=100000, feature_dim=d_hid)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=2, shuffle=True
    )

    inner_optimizer: optim.Optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    # Scale outer lr by 1/sqrt(K) so K async updates match one averaged update.
    outer_lr = float(os.environ.get("OUTER_LR", 0.7)) / math.sqrt(NUM_REPLICA_GROUPS)

    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    def trace_handler(p):
        dir = f"{output_folder}/profiles"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        p.export_chrome_trace(f"{dir}/step-{p.step_num}.json")

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=100, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
    )

    tensorboard_key_prefix = f"Run:{RUN}"
    prof.start()
    with SemiAsyncHeLoCo(
        manager,
        [m],
        inner_optimizer,
        sync_every=int(os.environ.get("SYNC_EVERY", 100)),
        outer_lr=outer_lr,
        outer_momentum=float(os.environ.get("OUTER_MOMENTUM", 0.9)),
        cos_ok=float(os.environ.get("COS_OK", 0.2)),
        k_dir=float(os.environ.get("K_DIR", 1.0)),
        conf_c=float(os.environ.get("CONF_C", 3.0)),
        k_shrink=float(os.environ.get("K_SHRINK", 0.5)),
        beta_max=float(os.environ.get("BETA_MAX", 0.5)),
        use_lookahead=os.getenv("USE_LOOKAHEAD", "True") == "True",
        backup_device=device,
        use_bucketization=True,
        bucket_cap_mb=25,
    ) as semi_async_heloco:
        outer_optimizer_holder["opt"] = semi_async_heloco.outer_optimizer
        while True:
            for i, (inputs, labels) in enumerate(trainloader):
                prof.step()

                inputs = inputs.to(device)
                labels = labels.to(device)

                inner_optimizer.zero_grad()

                out = m(inputs)
                loss = criterion(out, labels)

                writer.add_scalar(f"{tensorboard_key_prefix}/loss", loss, i)

                loss.backward()

                inner_optimizer.step()

                writer.add_scalar(
                    f"{tensorboard_key_prefix}/num_participants",
                    manager.num_participants(),
                    i,
                )
                writer.add_scalar(
                    f"{tensorboard_key_prefix}/current_step", manager.current_step(), i
                )
                if manager.current_step() % 100 == 0:
                    print(f"[{manager.current_step()}] loss = {loss.item():.4f}")

                if manager.current_step() >= 15:
                    prof.stop()
                    writer.flush()
                    exit()


if __name__ == "__main__":
    main()
