# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
SemiAsyncDiLoCo
===========
Async distributed optimization where the outer allreduce for window T runs
concurrently with all inner steps of window T+1.
"""

import logging
import math
from contextlib import nullcontext
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn, optim
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle

from torchft.local_sgd import _StreamingDiLoCoFragment
from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)


class _SemiAsyncDiLoCoFragment(_StreamingDiLoCoFragment):

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

            # wait on allreduce
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

        # launch allreduce
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

        return did_apply_outer_step


class SemiAsyncDiLoCo:
    """
    SemiAsyncDiLoCo implements async distributed optimization where the outer
    allreduce for window T runs concurrently with all inner steps of window T+1,
    maximizing communication/compute overlap.

    Unlike DiLoCo, which overlaps the allreduce with only the last
    ``fragment_sync_delay`` inner steps of the same window, SemiAsyncDiLoCo
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
                "SemiAsyncDiLoCo requires async quorum to be enabled. "
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

        self._fragments: List[_SemiAsyncDiLoCoFragment] = [
            _SemiAsyncDiLoCoFragment(
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

    def __enter__(self) -> "SemiAsyncDiLoCo":
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
        # torch ignores post-hook return values; the bool is for
        # SemiAsyncHeLoCo._step_post_hook, which calls this via super() and
        # applies its lookahead shift only when an outer step was committed.
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
                f"SemiAsyncDiLoCo syncing fragment={fragment} step={self._local_step} "
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
