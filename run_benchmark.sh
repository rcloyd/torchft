#!/usr/bin/env bash
# Run DiLoCo vs AsyncDiLoCo benchmark and generate comparison plots.
#
# Usage:
#   chmod +x run_benchmark.sh && ./run_benchmark.sh
#
# Tunable knobs:
#   SLOW_MS=50            ms of artificial delay per inner step on replica 1
#   SYNC_EVERY=20         inner steps per outer step
#   NUM_OUTER_STEPS=30    how many outer steps to train for
#   USE_NCCL=True         use NCCL (recommended for GPU runs)
#   MODEL=1B              model size: "debugmodel" or "1B"
#   GPUS_PER_ISLAND=2     GPUs per island (2 = DDP within island)
#   LH_PORT=29511         lighthouse port

set -euo pipefail

SLOW_MS=${SLOW_MS:-0}
SYNC_EVERY=${SYNC_EVERY:-20}
NUM_OUTER_STEPS=${NUM_OUTER_STEPS:-100}
USE_NCCL=${USE_NCCL:-True}
MODEL=${MODEL:-debugmodel}
GPUS_PER_ISLAND=${GPUS_PER_ISLAND:-1}
LH_PORT=${LH_PORT:-29511}
LH_ADDR="http://127.0.0.1:${LH_PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_mode() {
    local mode=$1
    log "=== Starting mode: $mode (model=$MODEL gpus_per_island=$GPUS_PER_ISLAND) ==="

    log "Starting lighthouse on port $LH_PORT..."
    torchft_lighthouse \
        --min_replicas 2 \
        --bind "127.0.0.1:${LH_PORT}" \
        --join_timeout_ms 300000 \
        &
    LH_PID=$!
    sleep 2

    # Build comma-separated GPU lists for each island.
    # Island 0 → GPUs 0..(GPUS_PER_ISLAND-1); Island 1 → GPUs GPUS_PER_ISLAND..2*GPUS_PER_ISLAND-1
    R0_GPUS=$(seq -s, 0 $((GPUS_PER_ISLAND-1)))
    R1_GPUS=$(seq -s, $GPUS_PER_ISLAND $((GPUS_PER_ISLAND*2-1)))

    # Replica 0: fast island
    log "Starting replica 0 (fast, GPUs ${R0_GPUS})..."
    CUDA_VISIBLE_DEVICES=$R0_GPUS \
    TORCHFT_LIGHTHOUSE=$LH_ADDR \
    REPLICA_GROUP_ID=0 \
    MODE=$mode \
    MODEL=$MODEL \
    USE_NCCL=$USE_NCCL \
    SYNC_EVERY=$SYNC_EVERY \
    NUM_OUTER_STEPS=$NUM_OUTER_STEPS \
    SLOW_MS=$SLOW_MS \
    SLOW_REPLICA=1 \
        torchrun --standalone --nproc_per_node=$GPUS_PER_ISLAND benchmark_diloco.py \
        > "output/benchmark/${mode}/replica-0/run.log" 2>&1 \
        &
    R0_PID=$!

    # Replica 1: slow island
    log "Starting replica 1 (slow +${SLOW_MS}ms, GPUs ${R1_GPUS})..."
    CUDA_VISIBLE_DEVICES=$R1_GPUS \
    TORCHFT_LIGHTHOUSE=$LH_ADDR \
    REPLICA_GROUP_ID=1 \
    MODE=$mode \
    MODEL=$MODEL \
    USE_NCCL=$USE_NCCL \
    SYNC_EVERY=$SYNC_EVERY \
    NUM_OUTER_STEPS=$NUM_OUTER_STEPS \
    SLOW_MS=$SLOW_MS \
    SLOW_REPLICA=1 \
        torchrun --standalone --nproc_per_node=$GPUS_PER_ISLAND benchmark_diloco.py \
        > "output/benchmark/${mode}/replica-1/run.log" 2>&1 \
        &
    R1_PID=$!

    log "Waiting for replicas to finish..."
    if ! wait $R0_PID; then
        log "ERROR: replica 0 failed — check output/benchmark/${mode}/replica-0/run.log"
        kill $LH_PID $R1_PID 2>/dev/null || true
        exit 1
    fi
    if ! wait $R1_PID; then
        log "ERROR: replica 1 failed — check output/benchmark/${mode}/replica-1/run.log"
        kill $LH_PID 2>/dev/null || true
        exit 1
    fi

    log "Shutting down lighthouse..."
    kill $LH_PID 2>/dev/null || true
    wait $LH_PID 2>/dev/null || true

    log "=== $mode complete ==="
}

mkdir -p output/benchmark/diloco/replica-{0,1}
mkdir -p output/benchmark/async_diloco/replica-{0,1}
mkdir -p output/benchmark/comparison

run_mode diloco
sleep 3
run_mode async_diloco

log "Generating comparison plots..."
python3 plot_diloco_comparison.py \
    output/benchmark/diloco/replica-0/metrics.json \
    output/benchmark/async_diloco/replica-0/metrics.json

log "Done. Results in output/benchmark/comparison/"
log "  loss_vs_walltime.png  — real throughput comparison"
log "  loss_vs_step.png      — convergence comparison"
log "  outer_step_time.png   — per-window wall time"
log "  boundary_step_ms.png  — allreduce blocking cost distribution"
log "  summary.txt           — numeric summary"