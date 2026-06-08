#!/usr/bin/env bash
# Run DiLoCo vs AsyncDiLoCo vs HeLoCo benchmark with 4 replicas and generate comparison plots.
#
# Usage:
#   chmod +x run_benchmark.sh && ./run_benchmark.sh
#
# Tunable knobs:
#   NUM_REPLICAS=4        number of replica islands (default: 4, needs 4 GPUs)
#   SLOW_MS=10            ms added per inner step on slow replicas (default: 10 ≈ 1 outer step lag)
#   SYNC_EVERY=10         inner steps per outer step
#   NUM_OUTER_STEPS=30    how many outer steps to train for
#   USE_NCCL=True         use NCCL (recommended for GPU runs)
#   MODEL=debugmodel      model: "tiny" (~750K params, byte-level), "debugmodel" (~6M params), "1B"
#   DATA_PATH=...         training data path (default: data/gutenberg_corpus.txt)
#   SGD_LR=0.07           outer SGD lr for diloco/async_diloco (before sqrt(K)/K scaling)
#   HELOCO_LR=0.7         outer lr passed to HeLoCoOptimizer (before sqrt(K)/K scaling)
#   GPUS_PER_ISLAND=1     GPUs per island
#   LH_PORT=29511         lighthouse port

set -euo pipefail

NUM_REPLICAS=${NUM_REPLICAS:-4}
SLOW_MS=${SLOW_MS:-10}
SYNC_EVERY=${SYNC_EVERY:-10}
NUM_OUTER_STEPS=${NUM_OUTER_STEPS:-100}
USE_NCCL=${USE_NCCL:-False}
MODEL=${MODEL:-debugmodel}
BATCH_SIZE=${BATCH_SIZE:-4}
GPUS_PER_ISLAND=${GPUS_PER_ISLAND:-1}
DATA_PATH=${DATA_PATH:-data/gutenberg_corpus.txt}
LH_PORT=${LH_PORT:-29511}
LH_ADDR="http://127.0.0.1:${LH_PORT}"
BENCHMARK_OUTDIR=${BENCHMARK_OUTDIR:-output/benchmark}

# Slow-replica delay schedule (ms per inner step) for replicas 1..N-1.
# Replica 0 is always fast (0ms). Replica 1 = SLOW_MS, 2 = 2*SLOW_MS, 3 = 3*SLOW_MS.
# This matches the paper's heterogeneous worker setup (varying staleness across workers).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_mode() {
    local mode=$1
    log "=== Starting mode: $mode (model=$MODEL replicas=$NUM_REPLICAS gpus_per_island=$GPUS_PER_ISLAND) ==="
    log "    data=$DATA_PATH slow_ms_per_replica=0,${SLOW_MS},$(( SLOW_MS * 2 )),$(( SLOW_MS * 3 ))"

    log "Starting lighthouse on port $LH_PORT..."
    torchft_lighthouse \
        --min_replicas "$NUM_REPLICAS" \
        --bind "127.0.0.1:${LH_PORT}" \
        --join_timeout_ms 300000 \
        &
    LH_PID=$!
    sleep 2

    local pids=()

    for replica_id in $(seq 0 $((NUM_REPLICAS - 1))); do
        # Each replica gets its own contiguous GPU block
        local gpu_start=$(( replica_id * GPUS_PER_ISLAND ))
        local gpu_end=$(( gpu_start + GPUS_PER_ISLAND - 1 ))
        local gpu_list
        gpu_list=$(seq -s, "$gpu_start" "$gpu_end")

        # Slow delay increases linearly per replica: 0, SLOW_MS, 2*SLOW_MS, 3*SLOW_MS
        local this_slow=$(( replica_id * SLOW_MS ))

        mkdir -p "${BENCHMARK_OUTDIR}/${mode}/replica-${replica_id}"

        log "Starting replica ${replica_id} (GPUs ${gpu_list}, slow_ms=${this_slow})..."
        CUDA_VISIBLE_DEVICES=$gpu_list \
        TORCHFT_LIGHTHOUSE=$LH_ADDR \
        REPLICA_GROUP_ID=$replica_id \
        MODE=$mode \
        MODEL=$MODEL \
        USE_NCCL=$USE_NCCL \
        SYNC_EVERY=$SYNC_EVERY \
        NUM_OUTER_STEPS=$NUM_OUTER_STEPS \
        BATCH_SIZE=$BATCH_SIZE \
        SLOW_MS=$this_slow \
        SLOW_REPLICA=$replica_id \
        NUM_REPLICAS=$NUM_REPLICAS \
        DATA_PATH=$DATA_PATH \
        BENCHMARK_OUTDIR=$BENCHMARK_OUTDIR \
            torchrun --standalone --nproc_per_node=$GPUS_PER_ISLAND \
                     --master_port=$(( 29600 + replica_id )) \
                     benchmark_diloco.py \
            > "${BENCHMARK_OUTDIR}/${mode}/replica-${replica_id}/run.log" 2>&1 \
            &
        pids+=($!)
    done

    log "Waiting for all ${NUM_REPLICAS} replicas to finish..."
    local failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            log "ERROR: replica $i failed — check ${BENCHMARK_OUTDIR}/${mode}/replica-${i}/run.log"
            failed=1
        fi
    done

    log "Shutting down lighthouse..."
    kill $LH_PID 2>/dev/null || true
    wait $LH_PID 2>/dev/null || true

    if [[ $failed -ne 0 ]]; then
        log "=== $mode FAILED ==="
        exit 1
    fi
    log "=== $mode complete ==="
}

# Create output directories for all replicas
for mode in diloco async_diloco heloco; do
    for r in $(seq 0 $((NUM_REPLICAS - 1))); do
        mkdir -p "${BENCHMARK_OUTDIR}/${mode}/replica-${r}"
    done
done
mkdir -p "${BENCHMARK_OUTDIR}/comparison"

run_mode diloco
sleep 3
run_mode async_diloco
sleep 3
run_mode heloco

log "Generating comparison plots..."
python3 benchmark_diloco.py --plot \
    "${BENCHMARK_OUTDIR}/diloco/replica-0/metrics.json" \
    "${BENCHMARK_OUTDIR}/async_diloco/replica-0/metrics.json" \
    "${BENCHMARK_OUTDIR}/heloco/replica-0/metrics.json"

log "Done. Results in ${BENCHMARK_OUTDIR}/comparison/"
log "  loss_vs_walltime.png  — real throughput comparison"
log "  loss_vs_step.png      — convergence comparison"
log "  outer_step_time.png   — per-window wall time"
log "  boundary_step_ms.png  — allreduce blocking cost distribution"
log "  summary.txt           — numeric summary"
