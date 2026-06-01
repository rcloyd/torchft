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

python3 -m pytest torchft/local_sgd_test.py -q -k AsyncDiLoCo
python3 -m pytest torchft/local_sgd_test.py -q

run lighthouse, island0, island1:

torchft_lighthouse --min_replicas 2 --bind 127.0.0.1:29511 --join_timeout_ms 10000

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=0 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=1 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

TORCHFT_LIGHTHOUSE=http://127.0.0.1:29511 REPLICA_GROUP_ID=2 USE_NCCL=True \
torchrun --standalone --nproc_per_node=1 train_asyncdiloco.py

- get profiler results in /home/rileycloyd/torchft/output/replica-0/profiles/step-350.json

python3 analyze_overlap.py output/replica-0/profiles/step-350.json output/replica-1/profiles/step-350.json
python3 analyze_overlap.py output/replica-0/profiles/step-350.json output/replica-2/profiles/step-350.json

- verify that allreduce runs async with inner steps

# performance

script that can compare the total runtime for diloco vs async diloco. 
we need to use the debug llama3 model from torchtitan. 
we also need to compare the training curves. 
we should also artificially slow one of the islands to simulate different pc specs


# todo

meet with brandon?


after finishing async diloco, create another class which inherits from async deloco for heloco


https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl


see if we can get RL Training with TorchTitan to work with torch FT -> diloco/async/heloco

