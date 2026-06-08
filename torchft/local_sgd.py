# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
LocalSGD
=========
This module implements a fault tolerant version of LocalSGD and related methods.
"""

import logging
import math
import os
from contextlib import nullcontext
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Callable

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
from torch.distributed.distributed_c10d import Work
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle
from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)

USE_BUCKETIZATION_ENV: str = "TORCHFT_USE_BUCKETIZATION"


def extract_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Returns a cloned version of the input tensor. If the input tensor is a DTensor,
    it extracts and clones its local representation.
    """
    new_tensor = None
    if isinstance(t, DTensor):
        new_tensor = t.to_local().clone()
    else:
        new_tensor = t.clone()
    new_tensor.grad = None
    return new_tensor


class LocalSGD:
    """
    LocalSGD is a context manager that
    implements the algorithm described in https://arxiv.org/pdf/1805.09767

    This will synchronize the model parameters periodically in a fault tolerant
    way using a torchft Manager. The allreduce on the parameters will happen
    every sync_every steps after the optimizer.step call.

    The torchft quorum is computed at the beginning of ``sync_every`` steps. If
    any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
    discarded and a new quorum will be computed on the next step.

    If running in async mode, on a joining worker the first ``sync_every`` steps
    will discarded as the model will be recovering during that period. When
    using sync mode, the checkpoint will be restored prior to the first step.
    """

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model: The model to wrap.
            optimizer: The optimizer used by the model.
            sync_every: How often to sync the model weights.
        """
        super().__init__()
        self._manager = manager
        self._model = model
        self._local_optimizer = optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"

        self._hooks: List[RemovableHandle] = []

    def __enter__(self) -> "LocalSGD":
        self._hooks.append(
            self._local_optimizer.register_step_pre_hook(self._step_pre_hook)
        )
        # Add optimizer hook which increments the local step counter and syncs if necessary
        self._hooks.append(
            self._local_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Handle any cleanup or error handling here
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _step_pre_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        # The checkpoint may transfer model parameters, so we need to make access to it thread safe
        self._manager.disallow_state_dict_read()

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        self._manager.allow_state_dict_read()

        self._local_step += 1
        if self._local_step >= self._sync_every:
            self.sync()

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the manager.
        """
        self._manager.start_quorum()
        self._perform_sync()
        self._local_step = 0

    def _perform_sync(self) -> None:
        """
        Performs the synchronization of the model weights across the manager.
        """
        averaged_parameters = self._average()
        if self._manager.should_commit():
            # Update the model parameters with the averaged values
            for param, avg_param in zip(self._model.parameters(), averaged_parameters):
                if isinstance(param, DTensor):
                    # we averaged the local version of the tensor so need to copy it back as a DTensor
                    param.data.copy_(
                        DTensor.from_local(
                            avg_param,
                            param.device_mesh,
                            param.placements,
                            shape=param.shape,
                            stride=param.stride(),
                        )
                    )
                else:
                    param.data.copy_(avg_param)

    def _average(self) -> list[torch.Tensor]:
        """
        Averages the model parameters across the manager and returns the averaged parameters.
        """
        works = []
        averaged_parameters = []
        for p in self._model.parameters():
            # Create a new tensor to store the averaged parameter
            avg_param = extract_local_tensor(p)
            works.append(self._manager.allreduce(avg_param))
            averaged_parameters.append(avg_param)
        for work in works:
            work.wait()
        return averaged_parameters


class _StreamingDiLoCoFragment:
    bucket_cap_mb: int = 1 * 1024 * 1024 * 1024
    use_bucketization: bool = False

    def __init__(
        self,
        manager: Manager,
        model_fragment: nn.Module,
        fragment_id: int,
        fragment_sync_offset: int,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
        fragment_sync_delay: int = 0,
        fragment_update_alpha: float = 0.0,
    ) -> None:
        if fragment_sync_offset > sync_every:
            raise ValueError("Fragment must be synced once before `sync_every` steps")

        self._fragment_id = fragment_id
        self._manager = manager
        self._model_fragment = model_fragment
        self._fragment_sync_offset = fragment_sync_offset
        self._local_optimizer = inner_optimizer
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._fragment_sync_delay = fragment_sync_delay
        self._fragment_update_alpha = fragment_update_alpha

        self._outer_optimizer = outer_optimizer

        # Stores pending all reduce
        self._allreduce_work: list[Work] = []
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # Recorded on `_stream` to wait for allreduce to finish
        self._stop_event: Optional[torch.cuda.Event] = None

        if bucket_cap_mb is not None:
            self.bucket_cap_mb = int(bucket_cap_mb * 1024 * 1024)

        if os.getenv(USE_BUCKETIZATION_ENV, "False") == "True":
            self.use_bucketization = True
        else:
            self.use_bucketization = use_bucketization

        self.should_quantize = should_quantize

        self._grads: Dict[str, torch.Tensor] = {}

        # Used to save global parameters so that they can be restored in case
        # commit fails
        self.original_parameters: Dict[str, torch.Tensor] = {}

        # Used to mix the local and global parameters
        self._local_parameters: Dict[str, torch.Tensor] = {}

        for name, p in self._model_fragment.named_parameters():
            if isinstance(p, DTensor):
                p = extract_local_tensor(p.data)

            backup_device = self._backup_device or torch.device("cpu")
            t = torch.empty(*tuple(p.shape), dtype=p.dtype, device=backup_device)
            if (
                self._pin_memory
                and t.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self.original_parameters[name] = t
        
        if self._manager._rank0_synchronization_only:
            self._gloo_group_pg = self._manager._gloo_group_pg

    def register_state_dict_fn(self) -> None:
        """
        Register state dict functions for this fragment with the manager.
        This allows for saving and loading the original_parameters during checkpointing and recovery.

        Args:
            manager: The manager to register with
            fragment_id: Optional identifier for this fragment, used in the key
        """
        if self._manager._rank0_synchronization_only:
            self._register_state_dict_fn_rank0()
        else:
            self._register_state_dict_fn_all_ranks()
    
    def _register_state_dict_fn_all_ranks(self) -> None:

        # Generate a unique key for this fragment based on the model fragment's name or provided ID
        fragment_key = f"StreamingDiLoCoFragment_{self._fragment_id}"

        # Define load function for this fragment
        def load_fn(state_dict: Dict[str, Dict[str, torch.Tensor]]) -> None:
            for name, param in state_dict["original_parameters"].items():
                if name in self.original_parameters:
                    self.original_parameters[name].copy_(param)

            self._outer_optimizer.load_state_dict(state_dict["outer_optimizer"])

        # Define save function for this fragment
        def save_fn() -> Dict[str, Dict[str, torch.Tensor]]:
            return {
                "outer_optimizer": self._outer_optimizer.state_dict(),
                "original_parameters": {
                    name: extract_local_tensor(param)
                    for name, param in self.original_parameters.items()
                },
            }

        # Register the functions with the manager
        self._manager.register_state_dict_fn(fragment_key, load_fn, save_fn)

    def _register_state_dict_fn_rank0(self) -> None:
        # Generate a unique key for this fragment based on the model fragment's name or provided ID
        fragment_key = f"StreamingDiLoCoFragment_{self._fragment_id}"

        state_dict_options = StateDictOptions(full_state_dict=True)

        # Define load function for this fragment
        def load_fn(state_dict: Dict[str, Dict[str, torch.Tensor]]) -> None:
            for name, global_tensor in state_dict["original_parameters"].items():
                if name in self.original_parameters:
                    width = self.original_parameters[name].shape[0]
                    start = self._manager._group_rank * width
                    end = start + width
                    self.original_parameters[name].copy_(global_tensor[start:end, ...])
            
            # call step() to initialize the outer optimizer correctly before setting its state dict
            self._outer_optimizer.step()
            set_optimizer_state_dict(
                self._model_fragment,
                self._outer_optimizer,
                state_dict["outer_optimizer"],
                options=state_dict_options
            )

        # Define save function for this fragment
        def save_fn() -> Dict[str, Dict[str, torch.Tensor]]:
            if self._manager._group_rank == 0:
                state_dict = {}
                state_dict["outer_optimizer"] = get_optimizer_state_dict(
                    self._model_fragment,
                    self._outer_optimizer,
                    options=state_dict_options
                )
                state_dict["original_parameters"] = {}
                for name, local_tensor in self.original_parameters.items():
                    local_tensors = [torch.empty_like(local_tensor) for _ in range(self._manager._group_world_size)]
                    dist.gather(local_tensor, local_tensors, dst=0, group=self._gloo_group_pg)
                    state_dict["original_parameters"][name] = torch.cat(local_tensors)
                return state_dict
            else:
                # call get_optimizer_state_dict to gather outer optimizer state dict at rank 0
                get_optimizer_state_dict(
                    self._model_fragment,
                    self._outer_optimizer,
                    options=state_dict_options
                )
                for name, local_tensor in self.original_parameters.items():
                    dist.gather(local_tensor, None, dst=0, group=self._gloo_group_pg)
                return {}
            

        # Register the functions with the manager
        self._manager.register_state_dict_fn(fragment_key, load_fn, save_fn)

    @torch.profiler.record_function("torchft::local_sgd::save_parameters")
    def save_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model_fragment.named_parameters():
                param_to_local = extract_local_tensor(p.data)
                self.original_parameters[name].copy_(param_to_local, non_blocking=True)

    def _save_local_parameters(self) -> None:
        """
        Saves a copy of the model's parameters.
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                self._local_parameters[name] = extract_local_tensor(p.data)

    @torch.profiler.record_function("torchft::local_sgd::restore_parameters")
    def restore_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model_fragment.named_parameters():
                if isinstance(p, DTensor):
                    # we averaged the local version of the tensor so need to copy it back as a DTensor
                    p.data.copy_(
                        DTensor.from_local(
                            self.original_parameters[name],
                            p.device_mesh,
                            p.placements,
                            shape=p.shape,
                            stride=p.stride(),
                        ),
                        non_blocking=False,
                    )
                else:
                    p.data.copy_(self.original_parameters[name], non_blocking=False)

    def _save_grads(self) -> None:
        """
        Saves pseudo-gradients of the parameters
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                if isinstance(p, DTensor):
                    local_param = p.to_local()
                else:
                    local_param = p
                pseudogradient = (
                    self.original_parameters[name].to(p.device) - local_param
                )
                self._grads[name] = pseudogradient

    def _set_grads(self) -> None:
        """
        Sets the gradients of the model fragment from the allreduce result
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                # avoid copying the gradient, it should be on the same device
                if isinstance(p, DTensor):
                    p.grad = DTensor.from_local(
                        self._grads[name],
                        p.device_mesh,
                        p.placements,
                        shape=p.shape,
                        stride=p.stride(),
                    )
                else:
                    p.grad = self._grads[name]

                # No longer needed
                del self._grads[name]

    def _clear_local_parameters(self) -> None:
        """
        Clears the saved copy of the model's parameters
        """
        self._local_parameters = {}

    def _merge_parameters(self) -> None:
        """
        Merges the local and global parameters.
        """
        for name, p in self._model_fragment.named_parameters():
            # we averaged the local version of the tensor so need to copy it back as a DTensor
            if isinstance(p, DTensor):
                p.data.lerp_(
                    DTensor.from_local(
                        self._local_parameters[name],
                        p.device_mesh,
                        p.placements,
                        shape=p.shape,
                        stride=p.stride(),
                    ),
                    self._fragment_update_alpha,
                )
            else:
                p.data.lerp_(self._local_parameters[name], self._fragment_update_alpha)

    @torch.profiler.record_function("torchft::local_sgd::wait")
    def wait(self) -> None:
        """
        Waits for the previously scheduled allreduce to finish
        """
        if len(self._allreduce_work) == 0:
            return

        if self._stream is not None:
            assert self._stop_event is not None
            self._stop_event.synchronize()
            self._stop_event = None

        self._allreduce_work = []

    @torch.profiler.record_function("torchft::local_sgd::prepare_sync")
    def prepare_sync(self) -> None:
        """
        Calculate the pseugradient, average them across the manager group and starts
        allreduce on the pseudo-gradients but doesn't wait for it to finish.
        """
        self._save_grads()

        assert len(self._allreduce_work) == 0

        # Make sure tensors are available to `_stream`
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())

        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            self._average_grads()

    @torch.profiler.record_function("torchft::local_sgd::perform_sync")
    def perform_sync(self) -> bool:
        """
        Overrides the sync method to wait for the scheduled allreduce to finish and
        steps using the outer optimizer.
        """
        # Waiting for an allreduce before it has been sent is currently not supported.
        assert len(self._allreduce_work) > 0

        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            for work in self._allreduce_work:
                work.wait()

            if self._stream is not None:
                self._stop_event = torch.cuda.Event()
                self._stop_event.record()

        self.wait()
        # scatter pseudogradients locally from rank 0
        if self._manager._rank0_synchronization_only:
            self._manager.scatter_grads()
        # save the parameters so they can be used for merging
        self._save_local_parameters()
        # Restore the parameters back to the previous state
        self.restore_parameters()

        # For large values of `fragment_sync_delay`, this call can be
        # a problem.
        #
        # This can return success even if the allreduce failed. Because
        # the process group could have been reconfigured while the
        # allreduce was inflight. The inflight allreduce may or may
        # not have been aborted.
        #
        # We can track errors per allreduce to
        # let the commit fail here. But this has the downside of
        # reconfiguring the pg too many times resulting in
        # more aborts and more commit failures.
        should_commit = self._manager.should_commit()

        if should_commit:
            # Use the outer optimizer to update the model parameters
            self._set_grads()
            self._outer_optimizer.step()
            self.save_parameters()
            self._merge_parameters()
        self._outer_optimizer.zero_grad()

        # free up memory
        self._clear_local_parameters()

        return should_commit

    def _wait_allreduce(self) -> None:
        # work.wait() blocks CPU until allreduce completes and fires the division
        # callback synchronously (future is already done, so .then() runs inline).
        # Launched H steps ago so these waits should be near-instant.
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                for work in self._allreduce_work:
                    work.wait()
            torch.cuda.current_stream().wait_stream(self._stream)
        else:
            for work in self._allreduce_work:
                work.wait()
        self._allreduce_work = []

    def _launch_allreduce(self) -> None:
        # Enqueue allreduce(T) on _stream to run concurrently with inner steps.
        # We do not call block_current_stream() here: that path calls
        # _set_future_callback() before the future is done, causing deferred
        # .then() callbacks to capture a stale managed_fut reference (closure bug).
        # work.wait() in _wait_allreduce is safe because by then the future is
        # already complete and .then() fires synchronously.
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            self._average_grads()

    @torch.profiler.record_function("torchft::local_sgd::async_sync")
    def async_sync(self) -> bool:
        """
        Performs the async DiLoCo boundary sync.

        Computes the pseudo-gradient for the current window into a temp buffer,
        waits for the previous window's allreduce to finish, applies the outer
        step, resets local params to the new outer params, then launches the
        current window's allreduce in the background for the next boundary to
        wait on.
        """
        should_commit = False
        did_apply_outer_step = False
        new_grads: Dict[str, torch.Tensor] = {}

        if self._allreduce_work:
            # Compute pseudo-gradients for window T while allreduce(T-1) is still
            # running, overlapping CPU work with the in-flight communication.
            with torch.no_grad():
                for name, p in self._model_fragment.named_parameters():
                    local_param = p.to_local() if isinstance(p, DTensor) else p
                    new_grads[name] = (
                        self.original_parameters[name].to(p.device) - local_param
                    )

            self._wait_allreduce()
            self._save_local_parameters()
            self.restore_parameters()
            should_commit = self._manager.should_commit()
            if should_commit:
                self._set_grads()
                self._outer_optimizer.step()
                self.save_parameters()
                self._merge_parameters()
                did_apply_outer_step = True
            self._outer_optimizer.zero_grad()
            self._clear_local_parameters()
        else:
            # First window for this worker. For a late joiner, should_commit()
            # applies the pending checkpoint (updating original_parameters and
            # model params to the donor's post-outer-step state) before we
            # compute pseudo-grads or restore. For initial startup this is a
            # no-op on the healing path.
            should_commit = self._manager.should_commit()
            # Compute pseudo-grads after the checkpoint may have updated
            # original_parameters, but before restore_parameters() overwrites
            # p.data. For a late joiner: original_parameters = outer(T) and
            # p.data = outer(T) (set by checkpoint load), so new_grads = 0
            # (no real inner steps yet). For initial startup: p.data = local(1)
            # from the first window's inner steps, giving a real pseudo-grad.
            with torch.no_grad():
                for name, p in self._model_fragment.named_parameters():
                    local_param = p.to_local() if isinstance(p, DTensor) else p
                    new_grads[name] = (
                        self.original_parameters[name].to(p.device) - local_param
                    )
            self.restore_parameters()
            self._outer_optimizer.zero_grad()

        self._grads = new_grads
        self._launch_allreduce()
        return did_apply_outer_step

    def _average_grads(self) -> None:
        """
        Efficiently averages gradients across the group using either:
        - Per-parameter allreduce (old behavior)
        - Bucketized allreduce (new behavior)
        """
        if self.use_bucketization:
            self._allreduce_bucketized()
        else:
            self._allreduce_per_param()

    def _allreduce_per_param(self) -> None:
        """Performs allreduce on each gradient tensor separately (original method)."""
        for name, p in self._model_fragment.named_parameters():
            # Perform allreduce on the pseudogradients
            work = self._manager.allreduce(
                self._grads[name], should_quantize=self.should_quantize
            )

            self._allreduce_work.append(work)

    def _bucketize_and_allreduce(
        self,
        tensors: List[torch.Tensor],
        bucket_size_bytes: int,
    ) -> None:
        """
        Applies allreduce on a list of tensors using bucketization.

        Args:
            tensors: List of torch tensors (e.g., gradients).
            bucket_size_bytes: Max size of each bucket in bytes.
        """
        if not tensors:
            return

        total_size = sum(t.numel() for t in tensors)
        dtype, device = tensors[0].dtype, tensors[0].device

        offset = 0
        flat_index = 0
        while offset < total_size:
            chunk_size = min(
                bucket_size_bytes // tensors[0].element_size(), total_size - offset
            )
            flat_buffer: torch.Tensor = torch.zeros(
                chunk_size, dtype=dtype, device=device
            )

            pack_offset: int = 0
            bucket_tensors: list[Tuple[torch.Tensor, int, int]] = []
            for t in tensors[flat_index:]:
                numel = t.numel()
                if pack_offset + numel > chunk_size:
                    break
                flat_buffer[pack_offset : pack_offset + numel].copy_(t.view(-1))
                bucket_tensors.append((t, pack_offset, numel))
                pack_offset += numel
                flat_index += 1

            work = self._manager.allreduce(
                flat_buffer, should_quantize=self.should_quantize
            )

            def callback(
                fut: torch.futures.Future[list[torch.Tensor]],
            ) -> list[torch.Tensor]:
                nonlocal bucket_tensors, flat_buffer
                for t, pack_offset, numel in bucket_tensors:
                    t.copy_(flat_buffer[pack_offset : pack_offset + numel].view_as(t))

                return []

            fut = work.get_future()
            fut = fut.then(callback)

            self._allreduce_work.append(work)

            offset += chunk_size

    def _allreduce_bucketized(self) -> None:
        """
        Averages gradients using bucketized allreduce with a fixed buffer.
        """
        grads = list(self._grads.values())
        assert len(grads) > 0, "No gradients to allreduce"
        self._bucketize_and_allreduce(
            grads,
            bucket_size_bytes=self.bucket_cap_mb,
        )


class DiLoCo:
    """
    DiLoCo implements distributed optimization by averaging and synchronizing
    pseudogradients (delta of the previous global weight and current local weights).

    The class implements a more general version of DiLoco, Streaming DiLoCo,
    which synchronizes fragments of pseudogradients at different steps.

    This algorithm requires a backup copy of the
    weights. By default these are stored in CPU memory. If any error occurs
    during the DiLoCo step, the step will be discarded and the model
    parameters will reset back to the last time DiLoCo synchronized.

    DiLoCo paper: https://arxiv.org/pdf/2311.08105
    Streaming DiLoCo paper: https://arxiv.org/pdf/2501.18512
    """

    def __init__(
        self,
        manager: Manager,
        model_fragments: List[nn.Module],
        inner_optimizer: optim.Optimizer,
        # TODO: this is for backward compatibility
        outer_optimizer: optim.Optimizer | list[optim.Optimizer],
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
        fragment_sync_delay: int = 0,
        fragment_update_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model_fragments: The fragments of the model to wrap.
            inner_optimizer: The optimizer used for the local parameters every step.
            outer_optimizer: The optimizer used for the global parameters updated every "sync_every" steps.
            sync_every: How often to update the model weights.
            backup_device: The device to store the backup weights on. If None, the backup weights will be on CPU.
            pin_memory: Whether to pin the memory for the backup weights (only for CPU device).
            should_quantize: Whether to quantize the gradients before allreduce.
            fragment_sync_delay: Controls the number of inner steps to wait before blocking on a fragment's
                                 synchronization. This is the "tao" parameter in the Streaming DiLoCo paper.
            fragment_update_alpha: Determines how to mix the local and global optimized parameters
        """

        if isinstance(outer_optimizer, list):
            assert len(outer_optimizer) == len(model_fragments), (
                "The number of outer optimizers must match the number of model fragments"
            )

        if manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the manager is initialized with use_async_quorum=False"
            )

        if sync_every < len(model_fragments):
            raise ValueError("Only 1 fragment can be syncrhonized at a time")

        if sync_every % len(model_fragments) != 0:
            raise ValueError("sync_every must divide the number of fragments")

        self._sync_every: int = sync_every // len(model_fragments)
        if fragment_sync_delay >= self._sync_every:
            raise ValueError(
                "Fragment must be synced before it is reduced another time"
            )

        if fragment_update_alpha < 0 or fragment_update_alpha > 1:
            raise ValueError("fragment_update_alpha must be between 0 and 1")

        super().__init__()
        self._manager = manager

        # The number of training iterations performed.
        # Used to synchronize which fragment to send across all
        # replicas
        self._local_step = 0

        self._fragment_sync_delay = fragment_sync_delay

        self._hooks: List[RemovableHandle] = []

        self._local_optimizer = inner_optimizer

        self._fragments: List[_StreamingDiLoCoFragment] = [
            _StreamingDiLoCoFragment(
                manager,
                model_fragment,
                i,
                math.floor((sync_every / len(model_fragments)) * (i + 1)),
                inner_optimizer,
                (
                    outer_optimizer[i]
                    if isinstance(outer_optimizer, list)
                    else outer_optimizer
                ),
                sync_every,
                backup_device,
                pin_memory,
                use_bucketization,
                bucket_cap_mb,
                should_quantize,
                fragment_sync_delay,
                fragment_update_alpha,
            )
            for i, model_fragment in enumerate(model_fragments)
        ]

        # This is to make sure we adhere to the assumptions made by the
        # `_StreamingDiLoCoFragment` about the fragment sync schedule.
        assert fragment_sync_delay < sync_every // len(model_fragments)

        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._save_parameters()
        self._register_state_dict_fn()

    def _register_state_dict_fn(self) -> None:
        for fragment in self._fragments:
            fragment.register_state_dict_fn()

    def _save_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.save_parameters()

    def _restore_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.restore_parameters()

    def __enter__(self) -> "DiLoCo":
        self._hooks.append(
            self._local_optimizer.register_step_pre_hook(self._step_pre_hook)
        )
        # Add optimizer hook which increments the local step counter and syncs if necessary
        self._hooks.append(
            self._local_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def _step_pre_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        # The checkpoint may transfer model parameters, so we need to make access to it thread safe
        self._manager.disallow_state_dict_read()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Handle any cleanup or error handling here
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _wait(self) -> None:
        """
        Waits for allreduce to finish on all fragments
        """
        for fragment in self._fragments:
            fragment.wait()

    def _current_fragment(self) -> int:
        """
        Determines which fragment to prepare/sync based on the current step.
        """
        step = self._manager.current_step()
        return step % len(self._fragments)

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        self._manager.allow_state_dict_read()

        # We need to make sure all nodes send the same fragments in order.
        # This is to avoid deadlocking e.g.
        #
        # 1. Step 1 - Node A sends fragment 1
        # 2. Step 1 - Node B sends fragment 2
        # 3. Step 2 - Node A waits for fragment 1
        # 4. Step 2 - Node B waits for fragment 2
        #
        # Both of them will fail because Node A didn't send fragment 2
        # and Node B didn't send fragment 1.
        self._local_step += 1

        if self._local_step == self._sync_every - self._fragment_sync_delay:
            # Time to prepare a fragment
            #
            # Some replicas will get the same copy of the model, implying batches
            # can be overrepresented.
            self._manager.start_quorum()
            fragment = self._current_fragment()
            logger.info(f"Preparing fragment={fragment} step={self._local_step}")
            self._fragments[fragment].prepare_sync()

        if self._local_step < self._sync_every:
            return

        if self._local_step == self._sync_every:
            # Time to sync a fragment
            fragment = self._current_fragment()
            logger.info(
                f"Syncing fragment={fragment} step={self._local_step} manager_step={self._manager.current_step()}"
            )
            self._fragments[fragment].perform_sync()

            # If the allreduce truly failed, we'll keep retrying this fragment.
            # We reset the parameters upon failure. We'll skip over some data
            # but we won't over train before syncing.

            self._local_step = 0
            return

        assert False, (
            f"{self._local_step=} should never be greater than {self._sync_every=}"
        )


class AsyncDiLoCo:
    """
    AsyncDiLoCo implements async distributed optimization where the outer
    allreduce for window T runs concurrently with all inner steps of window T+1,
    maximizing communication/compute overlap.

    Unlike DiLoCo, which overlaps the allreduce with only the last
    ``fragment_sync_delay`` inner steps of the same window, AsyncDiLoCo
    creates a permanent 1-step pipeline: the allreduce from window T is in
    flight for the full duration of window T+1 and is applied at the T+1
    boundary before the T+2 window begins.

    The pseudogradient Δt = outer(T-1) - local(T) is computed from outer
    params that are one outer step behind when applied — this is the accepted
    approximation described in the async DiLoCo literature.

    This requires the manager to be initialized with ``use_async_quorum=True``
    so the quorum check runs in a background thread during the window and is
    ready by the time ``should_commit()`` is called at the boundary.

    If any error occurs during a window, ``should_commit()`` returns False,
    the outer step is skipped, and local params are reset to the last committed
    outer state. One outer step is wasted for a recovering worker, matching the
    behavior of DiLoCo.
    """

    def __init__(
        self,
        manager: Manager,
        model_fragments: List[nn.Module],
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer | list[optim.Optimizer],
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
        fragment_update_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            manager: The manager to use. Must be initialized with use_async_quorum=True.
            model_fragments: The fragments of the model to wrap.
            inner_optimizer: The optimizer used for the local parameters every step.
            outer_optimizer: The optimizer used for the global parameters updated every sync_every steps.
            sync_every: How often to update the model weights.
            backup_device: The device to store the backup weights on. Defaults to CPU.
            pin_memory: Whether to pin the memory for the backup weights (CPU only).
            use_bucketization: Whether to use bucketized allreduce.
            bucket_cap_mb: Max bucket size in MB for bucketized allreduce.
            should_quantize: Whether to quantize the gradients before allreduce.
            fragment_update_alpha: Determines how to mix the local and global optimized parameters.
        """
        if isinstance(outer_optimizer, list):
            assert len(outer_optimizer) == len(model_fragments), (
                "The number of outer optimizers must match the number of model fragments"
            )

        if not manager._use_async_quorum:
            raise ValueError(
                "AsyncDiLoCo requires async quorum to be enabled. "
                "Ensure that the manager is initialized with use_async_quorum=True"
            )

        if sync_every < len(model_fragments):
            raise ValueError("Only 1 fragment can be synchronized at a time")

        if sync_every % len(model_fragments) != 0:
            raise ValueError("sync_every must divide the number of fragments")

        if fragment_update_alpha < 0 or fragment_update_alpha > 1:
            raise ValueError("fragment_update_alpha must be between 0 and 1")

        super().__init__()
        self._manager = manager
        self._local_step = 0
        self._sync_every: int = sync_every // len(model_fragments)
        self._hooks: List[RemovableHandle] = []
        self._local_optimizer = inner_optimizer

        self._fragments: List[_StreamingDiLoCoFragment] = [
            _StreamingDiLoCoFragment(
                manager,
                model_fragment,
                i,
                math.floor((sync_every / len(model_fragments)) * (i + 1)),
                inner_optimizer,
                (
                    outer_optimizer[i]
                    if isinstance(outer_optimizer, list)
                    else outer_optimizer
                ),
                sync_every,
                backup_device,
                pin_memory,
                use_bucketization,
                bucket_cap_mb,
                should_quantize,
                fragment_sync_delay=0,
                fragment_update_alpha=fragment_update_alpha,
            )
            for i, model_fragment in enumerate(model_fragments)
        ]

        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._save_parameters()
        self._register_state_dict_fn()

        # Quorum is only started once there is a prior allreduce in flight to
        # wait on. The first window has no prior allreduce so quorum is
        # deferred until window 2.
        self._allreduce_launched = False

    def _register_state_dict_fn(self) -> None:
        for fragment in self._fragments:
            fragment.register_state_dict_fn()

    def _save_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.save_parameters()

    def _restore_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.restore_parameters()

    def __enter__(self) -> "AsyncDiLoCo":
        self._hooks.append(
            self._local_optimizer.register_step_pre_hook(self._step_pre_hook)
        )
        # Add optimizer hook which increments the local step counter and syncs if necessary
        self._hooks.append(
            self._local_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def _step_pre_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        # The checkpoint may transfer model parameters, so we need to make access to it thread safe
        self._manager.disallow_state_dict_read()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Handle any cleanup or error handling here
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _wait(self) -> None:
        """
        Waits for allreduce to finish on all fragments
        """
        for fragment in self._fragments:
            fragment.wait()

    def _current_fragment(self) -> int:
        """
        Determines which fragment to prepare/sync based on the current step.
        """
        step = self._manager.current_step()
        return step % len(self._fragments)

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> bool:
        self._manager.allow_state_dict_read()

        self._local_step += 1

        # Fire quorum check early so the background thread has the full window
        # to complete before should_commit() is called at step sync_every.
        # Skip the first window — there is no prior allreduce in flight so
        # there is nothing to commit and no paired should_commit() call.
        if self._local_step == 1 and self._allreduce_launched:
            self._manager.start_quorum()

        if self._local_step < self._sync_every:
            return False

        if self._local_step == self._sync_every:
            fragment = self._current_fragment()
            logger.info(
                f"AsyncDiLoCo syncing fragment={fragment} step={self._local_step} "
                f"manager_step={self._manager.current_step()}"
            )
            if not self._allreduce_launched:
                # First window: start_quorum was never called early, do it now.
                self._manager.start_quorum()
            did_apply_outer_step = self._fragments[fragment].async_sync()
            self._allreduce_launched = True
            self._local_step = 0
            return did_apply_outer_step

        assert False, (
            f"{self._local_step=} should never be greater than {self._sync_every=}"
        )


class HeLoCoOptimizer(optim.Optimizer):
    """MLA outer optimizer with cosine-based pseudo-gradient correction."""

    def __init__(
        self,
        params,
        lr: float = 0.7,
        momentum: float = 0.9,
        cos_ok: float = 0.2,
        k_dir: float = 1.0,
        conf_c: float = 3.0,
        k_shrink: float = 0.5,
        beta_max: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, cos_ok=cos_ok,
            k_dir=k_dir, conf_c=conf_c, k_shrink=k_shrink,
            beta_max=beta_max, eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _correct_delta(
        delta: torch.Tensor,
        m: torch.Tensor,
        norm_d: torch.Tensor,
        norm_m: torch.Tensor,
        conf: torch.Tensor,
        cos_ok: float,
        k_dir: float,
        k_shrink: float,
        beta_max: float,
        eps: float,
    ) -> torch.Tensor:
        cos = torch.dot(delta.flatten(), m.flatten()) / (norm_d * norm_m + eps)
        if cos >= cos_ok:
            return delta
        v_hat = m / (norm_m + eps)
        if cos < 0:
            # Shrink: remove the anti-momentum component (paper Eq. 10-11).
            # Δ̂ = Δ − β·cos·‖Δ‖·v̂  removes only the conflicting projection.
            beta = torch.clamp(k_shrink * (-cos), max=beta_max) * conf
            return delta - beta * cos * norm_d * v_hat
        # Rotate: blend unit vectors toward momentum, preserve ‖Δ‖ (paper Eq. 12-14).
        # λ = min{k_d·(1−cos)·conf, 1} — less aligned → larger λ → more rotation.
        lam = torch.clamp(k_dir * (1.0 - cos) * conf, max=1.0)
        u_mix = (1.0 - lam) * delta / (norm_d + eps) + lam * v_hat
        norm_mix = u_mix.norm()
        if norm_mix > eps:
            return u_mix / norm_mix * norm_d
        # u_mix collapsed to zero; keep delta unchanged
        return delta

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            cos_ok = group["cos_ok"]
            k_dir = group["k_dir"]
            conf_c = group["conf_c"]
            k_shrink = group["k_shrink"]
            beta_max = group["beta_max"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    state = self.state.get(p)
                    if state is not None and "m" in state:
                        state["m"].mul_(mu)
                    continue

                delta = p.grad.detach()
                state = self.state[p]
                if "m" not in state:
                    # Float32 momentum regardless of param dtype to avoid
                    # precision loss when grads are fp32 and params are bf16.
                    state["m"] = torch.zeros_like(
                        p, dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )
                m = state["m"]

                norm_d = delta.norm()
                norm_m = m.norm()

                # Confidence: correction fades when Δ is small vs momentum
                conf = norm_d / (norm_d + conf_c * norm_m + eps)

                if norm_m > eps and norm_d > eps:
                    delta = self._correct_delta(
                        delta, m, norm_d, norm_m, conf,
                        cos_ok, k_dir, k_shrink, beta_max, eps,
                    )

                # MLA look-ahead: update m first, then apply (Δ + μm_new).
                # Using m_new (not m_old) is the defining property of MLA —
                # the param step incorporates the just-updated momentum direction.
                m.mul_(mu).add_(delta, alpha=1.0 - mu)
                p.add_(-(delta + mu * m), alpha=lr)


class HeLoCo(AsyncDiLoCo):
    """
    HeLoCo distributed training: AsyncDiLoCo with a cosine-corrected outer
    optimizer and optional look-ahead model dispatch.

    The built-in outer optimizer corrects stale pseudo-gradients before the MLA
    update based on cosine alignment with the momentum buffer:

      - Aligned      (cos ≥ cos_ok):  use Δ as-is.
      - Moderate     (0 ≤ cos < cos_ok):  rotate Δ toward momentum, preserving ‖Δ‖.
      - Anti-aligned (cos < 0):  project out the anti-momentum component.

    When ``use_lookahead=True``, after each committed outer step the local model
    is initialised at ``outer − lr·μ·m`` so workers fine-tune from a predicted
    future position rather than the bare outer params.

    ``outer_optimizer`` must not be passed — it is built internally.
    """

    def __init__(
        self,
        manager: Manager,
        model_fragments: List[nn.Module],
        inner_optimizer: optim.Optimizer,
        sync_every: int,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        cos_ok: float = 0.2,
        k_dir: float = 1.0,
        conf_c: float = 3.0,
        k_shrink: float = 0.5,
        beta_max: float = 0.5,
        use_lookahead: bool = True,
        **kwargs: Any,
    ) -> None:
        # dict.fromkeys preserves insertion order while deduplicating — a set
        # comprehension would give non-deterministic parameter ordering.
        all_params = list(dict.fromkeys(p for m in model_fragments for p in m.parameters()))
        outer_optimizer = HeLoCoOptimizer(
            all_params,
            lr=outer_lr,
            momentum=outer_momentum,
            cos_ok=cos_ok,
            k_dir=k_dir,
            conf_c=conf_c,
            k_shrink=k_shrink,
            beta_max=beta_max,
        )
        super().__init__(
            manager=manager,
            model_fragments=model_fragments,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            sync_every=sync_every,
            **kwargs,
        )
        self._use_lookahead = use_lookahead

    @property
    def outer_optimizer(self) -> HeLoCoOptimizer:
        """The shared HeLoCoOptimizer instance (same object across all fragments)."""
        return self._fragments[0]._outer_optimizer  # type: ignore[return-value]

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        did_apply_outer_step = super()._step_post_hook(_optim, _args, _kwargs)
        if self._use_lookahead and did_apply_outer_step:
            self._apply_lookahead()

    def _apply_lookahead(self) -> None:
        """
        Shift local params to the predicted future: outer − lr·μ·m.

        Each fragment may own a distinct outer optimizer, so look up the
        matching momentum buffer and per-group lr/mu from that fragment's
        optimizer when applying the shift.
        """
        with torch.no_grad():
            for fragment in self._fragments:
                outer_opt = fragment._outer_optimizer
                # Build a param → (lr, mu) map that respects per-group values.
                param_to_hyper: dict = {}
                for group in outer_opt.param_groups:
                    lr = group["lr"]
                    mu = group["momentum"]
                    for p in group["params"]:
                        param_to_hyper[id(p)] = (lr, mu)
                for p in fragment._model_fragment.parameters():
                    state = outer_opt.state.get(p)
                    if state and "m" in state:
                        lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
                        p.data.sub_(state["m"], alpha=lr * mu)

