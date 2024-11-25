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
"""moe layer"""
import mindspore as ms

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_tensor_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.transformer.module import Module

from .experts import GroupedMLP, SequentialMLP
from .router import TopKRouter
from .token_dispatcher import MoEAlltoAllTokenDispatcher


class MoELayer(Module):
    """
    expert layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        submodules: reserve arguments, not used now.
        layer_number: reserve arguments, not used now.

    Inputs:
        - **hidden_states** (Tensor) - The input hidden states of the local experts.

    Outputs:
        Tuple of 2 Tensors.

        - **output** (Tensor) - The output of the local experts
        - **mlp_bias** (Tensor) - Not used now.

    Raises:
        ValueError: if `ep_world_size` is less than or equal to 0.
        ValueError: if `num_experts % ep_world_size` is not equal to 0.
        ValueError: if the elements of `local_expert_indices` is larger than or equal to `num_experts`.
        ValueError: if `moe_config.moe_token_dispatcher_type` is not "alltoall"
        ValueError: if `self.training` is true and `get_tensor_model_parallel_world_size()` is larger than 1,
            and `self.sp` is not true
    """
    # pylint: disable=C0103
    def __init__(self, config: TransformerConfig, submodules=None, layer_number: int = None):
        super(MoELayer, self).__init__()
        self.submodules = submodules
        self.layer_number = layer_number

        moe_config = config.moe_config
        ep_world_size = get_expert_model_parallel_world_size()
        num_experts = moe_config.num_experts
        rank_id = get_expert_model_parallel_rank()

        self.tp = config.parallel_config.tensor_model_parallel_size
        self.sp = config.parallel_config.sequence_parallel

        if ep_world_size <= 0:
            raise ValueError(f"Expect expert parallel size > 0, but got {ep_world_size}")
        if num_experts % ep_world_size != 0:
            raise ValueError(f"Expect num_experts % ep_world_size == 0, but got {num_experts} and {ep_world_size}")

        num_local_experts = num_experts // ep_world_size
        local_expert_indices = [rank_id * num_local_experts + i for i in range(num_local_experts)]

        for x in local_expert_indices:
            if x >= num_experts:
                raise ValueError(f"expect all local expert indices < expert num, but got {local_expert_indices}")

        self.router = TopKRouter(config=config)

        if moe_config.moe_grouped_gemm:
            self.experts = GroupedMLP(num_local_experts, config)
        else:
            self.experts = SequentialMLP(num_local_experts, config)

        if moe_config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                num_local_experts=num_local_experts,
                local_expert_indices=local_expert_indices,
                config=config
                )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {moe_config.moe_token_dispatcher_type}")

    def construct(self, hidden_states: ms.Tensor):
        """moe layer forward"""
        if self.training and get_tensor_model_parallel_world_size() > 1 and not self.sp:
            raise ValueError(
                "During training, if tensor parallelism > 1 and not use sequence parallelism, "
                "would result in low performance in MoE."
            )
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores, indices = self.router(hidden_states)

        dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        expert_output, _ = self.experts(dispatched_input, tokens_per_expert)

        output, _ = self.token_dispatcher.token_unpermutation(expert_output, bias=None)
        output = output.reshape(hidden_shape)
        return output, None
