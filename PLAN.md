# update _StreamingDiLoCoFragment

combine prepare_sync and perform_sync into a new function that has the following order:
1. perform inner steps on outer(T-1) to compute local(T)
2. compute gradient(T) = outer(T-1) - local(T)
   store gradient(T) seperately since gradient(T-1) is still being sent to other workers

wait on allreduce(T-1) during inner steps:
1. send, recieve, and average gradient(T-1) from all workers
2. update outer(T) = outer(T-1) - avg{gradient(T-1)}
   so outer(T) = outer(T-1) - avg{outer(T-2) - local(T-1)}

when both tasks finish:
1. launch allreduce(T) with gradient(T)
2. reset local to outer(T) and begin inner steps

# create class AsyncDiLoCo

mostly the same structure as class DiLoCo
- require async quorum true instead of false
- remove partial overlap parameter fragment_sync_delay since it has the entire window 
- call start_quorum early in each window to run asynchronously
- handle worker joining between windows or during a window when allreduce has already been launched
- discard that outer step, perform resync if outer values are desynced
- update _step_post_hook to call the new sync method that we add to _StreamingDiLoCoFragment

# created train_asyncdiloco.py

- based on train_diloco.py

# tests

local_sgd_test.py:

- **test_async_diloco_first_window**: should_commit is called once (to apply any pending late-joiner checkpoint) and allreduce(T) is launched, but no outer step fires — satisfies the requirement that no outer step fires until a prior averaged gradient is available, while still enabling the late-joiner eavesdrop sync.
- **test_async_diloco_healthy**: allreduce(T-1) is waited on, outer(T) = outer(T-1) − avg{grad(T-1)} is applied, and allreduce(T) is launched — satisfies the full async pipeline requirement across two windows.
- **test_async_diloco_recovery**: when should_commit is False the outer step is skipped and outer parameters stay unchanged — satisfies the fault-tolerance requirement to discard the outer step on a failed commit.
- **test_async_diloco_allreduce_call_efficiency**: bucketization coalesces per-parameter allreduce calls into fewer round trips — satisfies the communication efficiency requirement.
- **test_async_diloco_gradient_correctness**: outer(T) = outer(T-1) − lr × avg{outer(T-1) − local(T)} is numerically correct — satisfies the requirement that the outer update uses the properly averaged pseudo-gradient.
- **test_async_diloco_requires_async_quorum**: construction fails when the manager is not in async quorum mode — satisfies the requirement that AsyncDiLoCo enforces async_quorum=True.
- **test_async_diloco_multi_window**: quorum, commit, and allreduce each fire at the correct cadence across three windows — satisfies the pipelining invariant that the async pipeline stays consistent beyond the initial two-window ramp-up.
- **test_async_diloco_non_blocking**: allreduce(T) remains in-flight (work queue non-empty) throughout all inner steps of window T+1 — satisfies the requirement that allreduce is non-blocking and deferred until the next boundary.
- **test_async_diloco_late_joiner_checkpoint_applied**: when should_commit() applies a checkpoint in the else branch, original_parameters reflects the donor's state and the pseudo-gradient is zero — satisfies the late-joiner eavesdrop requirement.

python3 -m pytest torchft/local_sgd_test.py -q -k AsyncDiLoCo
python3 -m pytest torchft/local_sgd_test.py -q

# run lighthouse, island0, island1

torchft_lighthouse --min_replicas 2 --bind 127.0.0.1:29511 --join_timeout_ms 10000

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=0 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=1 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=2 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

- get profiler results /home/rileycloyd/torchft/output/replica-0/profiles/step-350.json

python3 analyze_overlap.py output/replica-0/profiles/step-350.json output/replica-1/profiles/step-350.json
python3 analyze_overlap.py output/replica-0/profiles/step-350.json output/replica-2/profiles/step-350.json

- verify that allreduce runs async with inner steps
