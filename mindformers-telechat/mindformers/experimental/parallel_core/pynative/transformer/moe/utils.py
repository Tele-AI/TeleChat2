# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""define token sort and unsort"""
import math
import sys

import mindspore as ms
from mindspore import mint, nn, ops
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_and_context_parallel_group,
    get_tensor_and_context_parallel_world_size,
    get_context_parallel_group,
    get_context_parallel_world_size
)


class SelfDefinedGather(nn.Cell):
    """SelfDefinedGather"""
    def __init__(self, indices):
        """init"""
        super(SelfDefinedGather, self).__init__()
        self.indices = indices

    def construct(self, input_):
        """forward process"""
        indices = self.indices
        if indices.max() >= input_.shape[0]:
            raise ValueError(f"expect indices.max() < input_.shape[0], but got {indices} and {input_.shape}")
        output = ops.gather(input_, self.indices, 0)
        return output

    # pylint: disable=W0613
    def bprop(self, input_, output, dout):
        """bprop process"""
        indices = self.indices
        if indices.max() >= dout.shape[0]:
            raise ValueError(f"expect indices.max() < dout.shape[0], but got {indices} and {dout.shape}")
        dout = ops.gather(dout, ops.argsort(indices), 0)
        return (dout,)


def token_sort(tokens, indices, topk: int = 1, num_out_token: int = None, padded_mode: bool = False):
    """define token_sort"""
    if num_out_token is not None:
        raise NotImplementedError("num_out_token not implemented.")

    if padded_mode:
        raise NotImplementedError("padded_mode not implemented.")

    if topk > 1:
        if indices.shape[1] != topk:
            raise ValueError(f"expect indices.shape[1] == topk, but got {indices.shape[1]} and {topk}")
    flatten_indices = indices.reshape(-1)
    sorted_indices = ops.argsort(flatten_indices)

    sorted_tokens = mint.index_select(tokens, 0, sorted_indices // topk)

    return sorted_tokens, sorted_indices


def token_unsort(sorted_tokens, sorted_indices, probs: ms.Tensor = None, topk: int = 1, padded_mode: bool = False,
                 restore_shape: ops.Size = None):
    """define token_unsort"""
    if padded_mode:
        raise NotImplementedError("padded_mode not implemented.")
    if restore_shape is not None:
        raise NotImplementedError("restore_shape not implemented.")

    if topk > 1:
        if probs is None:
            raise ValueError("probs should not be None.")
        if probs.shape[0] != sorted_tokens.shape[0] // topk:
            raise ValueError(f"Expected probs.shape[0] == sorted_tokens.shape[0] // topk, " +
                             f"but got {probs.shape[0]} and {sorted_tokens.shape[0] // topk}")
    if probs is not None:
        if probs.shape[0] != sorted_tokens.shape[0] // topk:
            raise ValueError(f"expect probs.shape[0] == {sorted_tokens.shape[0] // topk}, "
                             f"but got {probs.shape[0]}.")

        if probs.shape[1] != topk:
            raise ValueError(f"Expected probs.shape[1] == {topk}, but got {probs.shape[1]}.")
    unsorted_indices = ops.argsort(sorted_indices)

    unsorted_tokens = mint.index_select(sorted_tokens, 0, unsorted_indices)
    unsorted_tokens = unsorted_tokens.reshape(-1, topk, sorted_tokens.shape[-1])

    if probs is not None:
        unsorted_tokens = unsorted_tokens * probs.unsqueeze(-1)

    unsorted_tokens = unsorted_tokens.sum(axis=1)
    return unsorted_tokens


def switch_load_balancing_loss_func(probs, tokens_per_expert, topk, moe_aux_loss_coeff, sequence_partition_group=None):
    """switch load balancing loss"""
    num_sub_sequence = 1
    if sequence_partition_group == "tp-cp":
        num_sub_sequence = get_tensor_and_context_parallel_world_size()
        tokens_per_expert = comm_func.all_reduce(tokens_per_expert, group=get_tensor_and_context_parallel_group())[0]
    elif sequence_partition_group == "cp":
        num_sub_sequence = get_context_parallel_world_size()
        tokens_per_expert = comm_func.all_reduce(tokens_per_expert, group=get_context_parallel_group())[0]

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    aggregated_probs_per_expert = mint.sum(probs, dim=0)
    aux_loss = mint.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss


class MoEAuxLossAutoScaler(nn.Cell):
    r"""
    Moe Aux loss auto scaler.

    Scale the aux loss in backward.

    """
    main_loss_backward_scale = ms.Tensor(1.0)

    # pylint: disable=W0613
    def construct(self, output, aux_loss):
        return output

    # pylint: disable=W0613
    def bprop(self, output, aux_loss, out, dout):
        scale_aux_loss_grad = mint.ones_like(aux_loss) * MoEAuxLossAutoScaler.main_loss_backward_scale
        return dout, scale_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale):
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def get_capacity(num_tokens, num_experts, capacity_factor, min_capacity=None):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


# pylint: disable=C0111
def topk_softmax_with_capacity(logits, topk, capacity_factor=None, pad_to_capacity=False,
                               drop_policy="probs"):
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [num_tokens, num_experts], got {logits.ndim}.")
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]

    scores, top_indices = mint.topk(logits, k=topk, dim=1)
    probs = mint.nn.functional.softmax(scores, dim=-1, dtype=ms.float32)
    probs = ops.cast(probs, logits.dtype)

    # pylint: disable=R1705
    if capacity_factor is None:
        tokens_per_expert = ops.bincount(top_indices.view(-1), minlength=num_experts)
        return probs, top_indices, tokens_per_expert
    else:
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk,
            num_experts=num_experts,
            capacity_factor=capacity_factor
        )

        topk_masked_gates = mint.zeros_like(logits).scatter(1, top_indices, probs)
        topk_mask = mint.zeros_like(logits).scatter(1, top_indices, 1)

        if drop_policy == "probs":
            capacity_probs, capacity_indices = mint.topk(topk_masked_gates, k=expert_capacity,
                                                         dim=0, sorted=False)
            capacity_mask = mint.zeros_like(logits).scatter(0, capacity_indices, 1)
        elif drop_policy == "position":
            _, capacity_indices = mint.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = mint.zeros_like(logits).scatter(0, capacity_indices, 1)
            capacity_probs = mint.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}.")

        if pad_to_capacity:
            final_probs, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
            tokens_per_expert_before_capacity = mint.sum(topk_mask, dim=0)
        else:
            final_mask = mint.logical_and(topk_mask, capacity_mask)
            drop_mask = mint.logical_not(final_mask)
            exceed_mask = mint.gather(drop_mask, 1, top_indices)
            final_probs = probs * mint.logical_not(exceed_mask)
            final_indices = top_indices.masked_fill(exceed_mask, sys.maxsize)
            tokens_per_expert_before_capacity = mint.sum(topk_mask, dim=0)
        return final_probs, final_indices, tokens_per_expert_before_capacity


def z_loss_func(logits, z_loss_coeff):
    z_loss = mint.mean(mint.square(ops.logsumexp(logits, axis=-1))) * z_loss_coeff
    return z_loss
