# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare DiLoCo vs AsyncDiLoCo benchmark results.

Usage:
  python3 plot_diloco_comparison.py \
    output/benchmark/diloco/replica-0/metrics.json \
    output/benchmark/async_diloco/replica-0/metrics.json

Produces:
  - loss_vs_walltime.png   : loss curves aligned on wall clock (shows real throughput)
  - loss_vs_step.png       : loss curves aligned on outer step (shows convergence)
  - outer_step_time.png    : time per outer step
  - boundary_step_ms.png   : distribution of last-step durations (allreduce blocking cost)
  - summary.txt            : numeric summary
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def label(r: dict) -> str:
    slow = r["slow_ms"]
    mode = r["mode"].replace("_", " ").title()
    return f"{mode} (slow={slow}ms)" if slow else mode


def boundary_steps(r: dict) -> list[dict]:
    return [s for s in r["steps"] if s["is_boundary"]]


def plot_loss_vs_walltime(records: list[dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        steps = r["steps"]
        wall = [s["wall_time_s"] for s in steps]
        loss = [s["loss"] for s in steps]
        ax.plot(wall, loss, label=label(r), alpha=0.85)
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Wall Time — DiLoCo vs AsyncDiLoCo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_loss_vs_step(records: list[dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        steps = r["steps"]
        xs = [s["global_inner_step"] for s in steps]
        ys = [s["loss"] for s in steps]
        ax.plot(xs, ys, label=label(r), alpha=0.85)
    ax.set_xlabel("Inner step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss vs Inner Step — DiLoCo vs AsyncDiLoCo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_outer_step_time(records: list[dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        xs = [o["outer_step"] for o in r["outer_steps"]]
        ys = [o["outer_wall_ms"] for o in r["outer_steps"]]
        ax.plot(xs, ys, label=label(r), marker="o", markersize=3, alpha=0.85)
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Outer Step Wall Time — DiLoCo vs AsyncDiLoCo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_boundary_step_ms(records: list[dict], out: str) -> None:
    """Histogram of optimizer-step durations at the boundary.

    For DiLoCo (fragment_sync_delay=0), the last inner step includes the full
    allreduce wait — so its distribution should be much wider/slower.
    For AsyncDiLoCo, the allreduce was in-flight during inner steps, so the
    boundary step is close to a normal inner step.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in records:
        durations = [s["step_ms"] for s in boundary_steps(r)]
        ax.hist(durations, bins=20, alpha=0.6, label=label(r))
    ax.set_xlabel("Last-step duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Boundary Step Duration — DiLoCo blocks, AsyncDiLoCo overlaps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def write_summary(records: list[dict], out: str) -> None:
    lines = []
    for r in records:
        steps = r["steps"]
        outer = r["outer_steps"]
        bsteps = boundary_steps(r)

        total_wall = outer[-1]["wall_time_s"] if outer else float("nan")
        inner_ms = [s["step_ms"] for s in steps if not s["is_boundary"]]
        boundary_ms = [s["step_ms"] for s in bsteps]
        outer_ms = [o["outer_wall_ms"] for o in outer]
        final_loss = steps[-1]["loss"] if steps else float("nan")

        lines += [
            f"=== {label(r)} ===",
            f"  Total wall time          : {total_wall:.1f}s",
            f"  Final loss               : {final_loss:.4f}",
            f"  Inner step (non-boundary): mean={np.mean(inner_ms):.1f}ms  "
            f"p95={np.percentile(inner_ms, 95):.1f}ms",
            f"  Boundary step            : mean={np.mean(boundary_ms):.1f}ms  "
            f"p95={np.percentile(boundary_ms, 95):.1f}ms",
            f"  Outer step               : mean={np.mean(outer_ms):.1f}ms  "
            f"p95={np.percentile(outer_ms, 95):.1f}ms",
            f"  Boundary overhead vs inner: "
            f"+{np.mean(boundary_ms) - np.mean(inner_ms):.1f}ms/step",
            "",
        ]
    text = "\n".join(lines)
    Path(out).write_text(text)
    print(text)
    print(f"saved {out}")


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: plot_diloco_comparison.py <metrics1.json> <metrics2.json> [...]")
        sys.exit(1)

    records = [load(p) for p in sys.argv[1:]]
    out_dir = Path("output/benchmark/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_vs_walltime(records, str(out_dir / "loss_vs_walltime.png"))
    plot_loss_vs_step(records, str(out_dir / "loss_vs_step.png"))
    plot_outer_step_time(records, str(out_dir / "outer_step_time.png"))
    plot_boundary_step_ms(records, str(out_dir / "boundary_step_ms.png"))
    write_summary(records, str(out_dir / "summary.txt"))


if __name__ == "__main__":
    main()
