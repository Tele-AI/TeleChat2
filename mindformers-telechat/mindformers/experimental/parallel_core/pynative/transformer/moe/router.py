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
"""moe router"""
import mindspore as ms
from mindspore import mint, nn, ops, Tensor

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_and_context_parallel_world_size,
    get_tensor_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromSequenceParallelRegion
from mindformers.modules.layers import Linear

from .utils import MoEAuxLossAutoScaler, switch_load_balancing_loss_func, topk_softmax_with_capacity, z_loss_func


class TopKRouter(nn.Cell):
    """
    TopK router.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.

    Inputs:
        - **input** (Tensor) - input tensor.

    Outputs:
        Tuple of 2 Tensors.

        - **scores** (Tensor) - The probabilities tensor after load balancing.
        - **indices** (Tensor) - The indices tensor after top-k selection.

    Raises:
        NotImplementedError: If `self.routing_type` is equal to "sinkhorn".
        ValueError: If `self.routing_type` is not equal to "none".
    """
    def __init__(self, config: TransformerConfig):
        super(TopKRouter, self).__init__()
        self.config = config
        self.moe_config = config.moe_config
        self.param_init_dtype = config.params_dtype
        self.compute_dtype = config.compute_dtype
        self.gating = Linear(config.hidden_size,
                             self.moe_config.num_experts,
                             has_bias=False,
                             param_init_type=self.param_init_dtype,
                             compute_dtype=self.compute_dtype)
        self.topk = self.moe_config.moe_router_topk
        self.routing_type = self.moe_config.moe_router_load_balancing_type
        self.num_experts = self.moe_config.num_experts
        self.gather_from_sp = GatherFromSequenceParallelRegion(need_to_swapaxes=False)

        self.moe_aux_loss_auto_scaler = MoEAuxLossAutoScaler()

    # pylint: disable=W0622
    def construct(self, input: ms.Tensor):
        """forward process"""
        ### add noise ###
        if self.moe_config.moe_input_jitter_eps is not None:
            eps = self.moe_config.moe_input_jitter_eps
            self.input_jitter = ops.uniform(input.shape, Tensor(1.0 - eps), Tensor(1.0 + eps))
            input = input * self.input_jitter
        ### add noise ###

        logits = self.gating(input)

        ### routint process ###
        # [b*s/tp, experts_num]
        logits = logits.reshape(-1, self.moe_config.num_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)  # [b*s/tp, experts_num]

        if get_tensor_model_parallel_world_size() > 1 and self.moe_config.moe_token_dispatcher_type == "alltoall":
            # [b*s, experts_num]
            logits = self.gather_from_sp(logits)

        if self.routing_type == "sinkhorn":
            raise NotImplementedError("sinkhorn not implemented.")
        if self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            scores, indices, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.moe_config.moe_expert_capacity_factor,
                pad_to_capacity=self.moe_config.moe_pad_expert_input_to_capacity,
                drop_policy=self.moe_config.moe_token_drop_policy
            )
            indices = ops.cast(indices, ms.int32)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        ### routint process ###

        return scores, indices

    def aux_loss_load_balancing(self, logits):
        """
        Apply aux loss balancing to the logits tensor.

        Args:
            logits (Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            Tuple of Tensor

            - **probs** (Tensor) - The probabilities tensor after load balancing.
            - **indices** (Tensor) - The indices tensor after top-k selection.
        """
        probs, indices, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.moe_config.moe_expert_capacity_factor,
            pad_to_capacity=self.moe_config.moe_pad_expert_input_to_capacity,
            drop_policy=self.moe_config.moe_token_drop_policy
        )
        if self.training:
            scores = mint.nn.functional.softmax(logits, dim=-1, dtype=ms.float32)
            probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
        return probs, ops.cast(indices, ms.int32)

    def apply_load_balancing_loss(self, probs, num_local_tokens_per_expert, activation):
        """
        Applies auxiliary loss to the MoE layer.

        Args:
            probs (Tensor): The probs output by the router for each token, shape: [num_tokens, num_experts].
            num_local_tokens_per_expert (Tensor): The number of tokens per expert, shape: [num_experts].
            activation (Tensor): The activation tensor to attach the gradient function to.

        Returns:
            Tensor, the activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.moe_config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.moe_config.moe_token_dispatcher_type == "alltoall":
            sequence_partition_group = "cp"
            moe_aux_loss_coeff /= get_tensor_model_parallel_world_size()

        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group
        )
        activation = self.moe_aux_loss_auto_scaler(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """
        Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper `ST-MoE paper <https://arxiv.org/pdf/2202.08906.pdf>`_ for details.

        Args:
            logits (Tensor): The logits of the router.

        Returns:
            Tensor, the logits after applying the z-loss.
        """
        if self.moe_config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.moe_config.moe_z_loss_coeff
                / get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = self.moe_aux_loss_auto_scaler(logits, z_loss)
        return logits
