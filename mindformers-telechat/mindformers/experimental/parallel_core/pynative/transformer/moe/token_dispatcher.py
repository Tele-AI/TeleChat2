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
"""token dispatcher"""
from typing import List

import mindspore as ms
import mindspore.ops as ops

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    all_to_all_hp2sp,
    AllGatherFromTensorParallelRegion,
    AllToAll,
    AllToAllSP2HP,
    GatherFromTensorAndExpertParallelRegion,
    ReduceScatterToTensorParallelRegion,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.mappings import (
    gather_along_first_dim_expert_parallel
)

from .utils import token_sort, token_unsort


class MoEAlltoAllTokenDispatcher():
    """
    AlltoAll Token Dispatcher, perform a dp <-> ep permutation.

    Args:
        num_local_experts (int): how many local experts on this rank.
        local_expert_indices (List[int]): Indices of local experts on this rank.
        config (TransformerConfig): Configuration object for the transformer model.

    Raises:
        ValueError: If `num_local_experts` is not larger than 0.
        ValueError: If the length of `local_expert_indices` is not equal to `num_local_experts`.
    """
    def __init__(self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig):
        self.config = config
        self.moe_config = config.moe_config
        self.parallel_config = self.config.parallel_config
        self.hidden_shape = None
        self.hidden_size = self.config.hidden_size
        self.num_local_experts = num_local_experts
        if self.num_local_experts <= 0:
            raise ValueError("expect num_local_experts > 0")

        self.router_topk = self.moe_config.moe_router_topk
        self.add_bias = self.config.add_bias_linear
        self.en = self.moe_config.num_experts
        self.ep = self.parallel_config.expert_model_parallel_size
        self.use_self_defined_alltoall = self.moe_config.use_self_defined_alltoall
        self.local_expert_indices = local_expert_indices
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError(f"expect len(self.local_expert_indices) == {self.num_local_experts}, "
                             f"but got {len(self.local_expert_indices)}")

        expert_ids_per_ep_rank = [i % self.num_local_experts for i in range(self.en)]
        self.expert_ids_per_ep_rank = ms.Tensor(expert_ids_per_ep_rank, dtype=ms.int32)

        self.probs = None
        self.output_shape = None
        self.input_shape = None
        self.local_input_splits = None
        self.local_output_splits = None
        self.dp_group_input_splits = None
        self.dp_group_output_splits = None
        self.dp_group_input_tokens_local_experts_indices = None

        self.ep_group = get_expert_model_parallel_group()
        self.rank_id = get_expert_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.gather_from_mp = GatherFromTensorAndExpertParallelRegion()
        self.sp2hp = AllToAllSP2HP()
        self.gather_from_tp = AllGatherFromTensorParallelRegion()
        self.scatter_to_tp = ReduceScatterToTensorParallelRegion()

    def preprocess(self, indices):
        """
        This function will calculate the input and output splits for alltoall communitcation.

        Args:
            indices (ms.Tensor): indicates every token was dispatched to which expert.

        Outputs:
            count_tokens_per_local_expert (ms.Tensor): Tensor containing the number of tokens assigned to local expert.

        For example:
            assuming data_parallel = expert_parallel = 2, num_experts = 4,
            there are two sample on each rank before dispatch:
                rank0: hidden_states: ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜"]
                       indices:       [ 3,    1,    3,    2,    2,    0,    1,    2,    1,   2 ]
                rank1: hidden_states: ["白", "日", "依", "山", "尽", "黄", "河", "入", "海", "流"]
                       indices:       [ 1,    2,    2,    0,    0,    2,    3,    3,    1,   0 ]

            1. `count_local_tokens_per_expert` should be:
                rank0: [1, 3, 4, 2]
                rank1: [3, 2, 3, 2]

            2. `dp_group_input_splits` should be:
                rank0: [4, 6]

                rank1: [5, 5]

            3. `dp_group_output_splits` should be:
                rank0: [[4],
                        [6]]

                rank1: [[5],
                        [5]]

        """

        ep, en = self.ep, self.en

        # count_local_tokens_per_expert: [en]
        count_local_tokens_per_expert = ops.histc(indices, bins=en, min=0, max=en)

        if ep > 1:
            # cal input splits
            local_input_splits = count_local_tokens_per_expert.reshape(ep, self.num_local_experts)
            local_input_splits = local_input_splits.sum(axis=1).to(ms.int32)
            self.local_input_splits = local_input_splits.asnumpy().tolist()
            dp_group_input_splits = gather_along_first_dim_expert_parallel(local_input_splits)

            self.dp_group_input_splits = dp_group_input_splits.reshape(-1, ep)
            # cal output splits
            count_dp_group_tokens_per_expert = gather_along_first_dim_expert_parallel(count_local_tokens_per_expert)
            # count_dp_group_tokens_per_expert: [ep, en]
            count_dp_group_tokens_per_expert = count_dp_group_tokens_per_expert.reshape(ep, en)
            # count_dp_group_tokens_per_local_expert: [ep, num_local_experts]
            count_dp_group_tokens_per_local_expert = count_dp_group_tokens_per_expert[:, self.local_expert_indices]
            self.local_output_splits = count_dp_group_tokens_per_local_expert.sum(axis=-1).asnumpy().tolist()
            self.dp_group_output_splits = self.dp_group_input_splits.T

            self.output_shape = (
                self.dp_group_input_splits.sum(axis=0)[self.rank_id].tolist(), self.hidden_size // self.tp_size
            )
            self.input_shape = (
                self.dp_group_output_splits.sum(axis=0)[self.rank_id].tolist(), self.hidden_size // self.tp_size
            )
            # count_tokens_per_local_expert: [num_local_experts]
            count_tokens_per_local_expert = count_dp_group_tokens_per_local_expert.sum(axis=0)

        else: # ep <= 1
            count_dp_group_tokens_per_local_expert = count_local_tokens_per_expert.reshape(-1, en)
            count_tokens_per_local_expert = count_local_tokens_per_expert

        # total num of tokens in this rank
        self.count_dp_group_tokens_local_expert = count_dp_group_tokens_per_local_expert.sum()

        self.dp_group_input_tokens_local_experts_indices = []
        if self.count_dp_group_tokens_local_expert > 0 and self.num_local_experts > 1:
            self.dp_group_input_tokens_local_experts_indices = ops.repeat_interleave(
                self.expert_ids_per_ep_rank, count_dp_group_tokens_per_local_expert.flatten())

        self.alltoall = AllToAll(self.ep_group, self.output_shape, self.input_shape,
                                 self.local_output_splits, self.local_input_splits, self.use_self_defined_alltoall)
        self.alltoall_reverse = AllToAll(self.ep_group, self.input_shape, self.output_shape, self.local_input_splits,
                                         self.local_output_splits, self.use_self_defined_alltoall)

        return count_tokens_per_local_expert

    def token_permutation(self, hidden_states, probs, indices):
        """
        Performs dp -> ep permutation.

        Args:
            hidden_states (ms.Tensor): hidden_states.
            probs (ms.Tensor): probs of hidden_states.
            indices (ms.Tensor): indicates every token was dispatched to which expert.

        Outputs:
            dp_group_input_tokens (ms.Tensor): permuted tokens.
            count_tokens_per_local_expert (ms.Tensor): Tensor containing the number of tokens assigned to local expert.

        For example:
            assuming data_parallel = expert_parallel = 2, num_experts = 4,
            there are two sample on each rank before dispatch:
                rank0: hidden_states: ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜"]
                       indices:       [ 3,    1,    3,    2,    2,    0,    1,    2,    1,   2 ]
                rank1: hidden_states: ["白", "日", "依", "山", "尽", "黄", "河", "入", "海", "流"]
                       indices:       [ 1,    2,    2,    0,    0,    2,    3,    3,    1,   0 ]

            1. `sorted_local_input_tokens` should be a tensor sorted by indices:
                rank0: sorted_tokens: ["疑", "前", "是", "上", "月", "光", "地", "霜", "床", "明"]
                       sorted_indices:[ 0,    1,    1,    1,    2,    2,    2,    2,    3,   3 ]
                rank1: sorted_tokens: ["山", "近", "流", "白", "海", "日", "依", "黄", "河", "入"]
                       sorted_indices:[ 0,    0,    0,    1,    1,    2,    2,    2,    3,   3 ]

            2. `dp_group_input_tokens` should be dispatched tensor:
                rank0: dp_group_input_tokens:         ["疑", "山", "近", "流", "前", "是", "上", "白", "海"]
                       dp_group_input_tokens_indices: [ 0,    0,    0,    0,    1,    1,    1,    1,   1 ]
                       count_tokens_per_local_expert: [4, 5]
                rank1: dp_group_input_tokens:         ["月", "光", "地", "霜", "日", "依", "黄", "床", "明", "河", "入"]
                       dp_group_input_tokens_indices: [ 2,    2,    2,    2,    2,    2,    2,    3,    3,    3,   3 ]
                       count_tokens_per_local_expert: [7, 4]

        """
        self.hidden_shape = hidden_states.shape
        self.hidden_dtype = hidden_states.dtype
        self.probs = probs
        if probs.ndim != 2:
            raise ValueError(f"expect `probs` is 2d tensor, but got shape {probs.shape}")
        if indices.ndim != 2:
            raise ValueError(f"expect `indices` is 2d tensor, but got shape {indices.shape}")

        # count_tokens_per_local_expert: [num_local_experts]
        count_tokens_per_local_expert = self.preprocess(indices)

        hidden_states = hidden_states.reshape(-1, self.hidden_shape[-1])

        if self.tp_size > 1:
            # hidden_states: [seq_len*bs/tp, hidden_size] -> [seq_len*bs, hidden_size/tp]
            hidden_states = self.sp2hp(hidden_states)
        sorted_local_input_tokens, self.local_input_sorted_map = token_sort(hidden_states,
                                                                            indices,
                                                                            topk=self.router_topk)
        dp_group_input_tokens = self.alltoall(sorted_local_input_tokens)
        if self.num_local_experts > 1 and self.count_dp_group_tokens_local_expert > 0:
            dp_group_input_tokens, self.dp_group_input_sorted_map = token_sort(
                dp_group_input_tokens, self.dp_group_input_tokens_local_experts_indices)

        if self.tp_size > 1:
            dp_group_input_tokens = self.gather_from_tp(dp_group_input_tokens)

        return dp_group_input_tokens, count_tokens_per_local_expert

    def token_unpermutation(self, hidden_states, bias=None):
        """
        Performs dp <- ep permutation, reverse process of token_permutation.

        Args:
            hidden_states (ms.Tensor): local experts output.
            bias (ms.Tensor): bias tensor, not implemented.

        Outputs:
            output (ms.Tensor): unpermuted tokens.
        """

        if bias is not None:
            raise ValueError("Bias is not supported in AlltoAllDispatcher")

        if self.tp_size > 1:
            hidden_states = self.scatter_to_tp(hidden_states)

        if self.num_local_experts > 1 and self.count_dp_group_tokens_local_expert > 0:
            hidden_states = token_unsort(hidden_states,
                                         self.dp_group_input_sorted_map)
        hidden_states = self.alltoall_reverse(hidden_states)
        output = token_unsort(hidden_states,
                              self.local_input_sorted_map,
                              probs=self.probs,
                              topk=self.router_topk)
        if self.tp_size > 1:
            output = all_to_all_hp2sp(output)

        return output, None
