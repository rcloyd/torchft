"""
Analyze a torchft async DiLoCo profile trace to check if allreduce overlaps
with inner training steps.

Usage:
    python3 analyze_overlap.py output/replica-0/profiles/step-350.json
"""

import json
import sys
from collections import Counter


def analyze(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    events = [e for e in data["traceEvents"] if "ts" in e and "dur" in e]

    # ── Use should_commit as a reliable async_sync boundary marker ──────────
    # should_commit is called exactly once per async_sync, has short duration,
    # and doesn't span profiler schedule transitions.
    boundary_evs = [
        e for e in events
        if e.get("name") == "torchft::manager::should_commit"
        and e.get("pid") != 0
    ]
    boundary_evs.sort(key=lambda e: e["ts"])

    if not boundary_evs:
        print(f"{path}: no should_commit events found (no async_sync in trace).")
        return

    # Pick main pid
    main_pid = Counter(e["pid"] for e in boundary_evs).most_common(1)[0][0]
    boundary_evs = [e for e in boundary_evs if e["pid"] == main_pid]
    BASE = boundary_evs[0]["ts"]

    # ── GPU kernel events ────────────────────────────────────────────────────
    nccl_evs = [
        e for e in events
        if "nccl" in str(e.get("name", "")).lower() and e.get("pid") == 0
    ]
    train_evs = [
        e for e in events
        if any(k in str(e.get("name", "")).lower() for k in ["sgemm", "gemm", "conv", "matmul"])
        and e.get("pid") == 0
    ]

    print(f"Trace:  {path}")
    print(f"Boundaries (should_commit events): {len(boundary_evs)}")
    print(f"NCCL GPU kernels total: {len(nccl_evs)}")
    print(f"Training GPU kernels (gemm): {len(train_evs)}")
    print()

    # ── Per-window analysis ──────────────────────────────────────────────────
    # A "window" is the gap between two consecutive should_commit events.
    # The allreduce for window N is launched AFTER should_commit(N) and should
    # run during the gap before should_commit(N+1).
    overlap_windows = 0
    for i in range(len(boundary_evs) - 1):
        b_end   = boundary_evs[i]["ts"] + boundary_evs[i]["dur"]   # after commit N
        b_next  = boundary_evs[i + 1]["ts"]                         # start of commit N+1
        gap_ms  = (b_next - b_end)

        nccl_in_gap = [e for e in nccl_evs if b_end <= e["ts"] < b_next]
        train_in_gap = [e for e in train_evs if b_end <= e["ts"] < b_next]

        nccl_total_ms = sum(e["dur"] for e in nccl_in_gap)
        train_total_ms = sum(e["dur"] for e in train_in_gap)

        rel_end  = (b_end  - BASE)
        rel_next = (b_next - BASE)

        print(f"  Gap {i+1}: {rel_end:.0f} → {rel_next:.0f} ms  ({gap_ms:.0f} ms)")
        print(f"    Training kernels in gap: {len(train_in_gap)}  ({train_total_ms:.0f} ms)")
        print(f"    NCCL kernels in gap:     {len(nccl_in_gap)}  ({nccl_total_ms:.0f} ms)", end="")

        if nccl_in_gap and train_in_gap:
            # Check actual interleaving
            nccl_start = min(e["ts"] for e in nccl_in_gap) - BASE
            nccl_end   = max(e["ts"] + e["dur"] for e in nccl_in_gap) - BASE
            train_start = min(e["ts"] for e in train_in_gap) - BASE
            train_end   = max(e["ts"] + e["dur"] for e in train_in_gap) - BASE
            overlap = min(nccl_end, train_end) - max(nccl_start, train_start)
            if overlap > 0:
                print(f"  ← OVERLAP CONFIRMED ({overlap:.0f} ms concurrent)")
                overlap_windows += 1
            else:
                print(f"  (sequential, no concurrent overlap)")
        else:
            print()
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = max(len(boundary_evs) - 1, 1)
    print("Summary:")
    print(f"  NCCL + training overlap in {overlap_windows}/{total} inter-boundary gaps")
    if overlap_windows == total:
        print("  ✓ Async allreduce overlapping with inner training steps")
    elif overlap_windows > 0:
        print("  ~ Partial overlap")
    else:
        print("  ✗ No overlap detected")
        print()
        # Diagnose why
        all_nccl = [e for e in nccl_evs]
        if not all_nccl:
            print("  Diagnosis: zero NCCL kernels in trace — allreduce may be CPU-based (Gloo)")
            print("  Try: USE_NCCL=True to use GPU-to-GPU NCCL allreduce")
        else:
            first_nccl = min(e["ts"] for e in all_nccl) - BASE
            last_nccl  = max(e["ts"] + e["dur"] for e in all_nccl) - BASE
            print(f"  Diagnosis: NCCL kernels exist ({len(all_nccl)} total) but run {first_nccl:.0f}–{last_nccl:.0f} ms")
            print("  They may complete before inner steps start — try a larger model or more workers")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "output/replica-0/profiles/step-350.json",
        "output/replica-1/profiles/step-350.json",
    ]
    for path in paths:
        analyze(path)
        print("=" * 60)
