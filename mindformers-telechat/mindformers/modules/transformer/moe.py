# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Note: Mixture of Expert (MoE) structure. This is an experimental interface that is subject to change or deletion.
"""
from __future__ import absolute_import
from __future__ import division

import math
import copy
import numpy as np
import mindspore.ops as ops

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops.operations._sequence_ops import TensorToScalar
import mindspore.common.dtype as mstype
import mindspore.communication.management as D

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.modules.transformer.op_parallel_config import default_moeparallel_config, MoEParallelConfig
from mindformers.version_control import check_valid_moefinalizerouting_op
from mindformers.modules.transformer.moe_utils import ZLoss
from mindformers.tools.utils import get_predict_run_mode

__all__ = [
    "MoEConfig"]

dtype_map = {
    'float16': mstype.float32,
    'float32': mstype.float32,
    'bfloat16': mstype.bfloat16
}


def _check_aux_loss_config(aux_loss_types, aux_loss_factors):
    """
        Check if aux_loss_types and aux_loss_factors are valid.
        Args:
            aux_loss_types: list of auxiliary loss types
            aux_loss_factors: list of auxiliary loss factors

        Returns:
            aux_loss_config (dict): dict of auxiliary loss types and factors.
    """
    supported_loss_types = ["expert", "device", "comm"]

    if aux_loss_types is None:
        aux_loss_types = []
        aux_loss_factors = []
    else:
        if not (isinstance(aux_loss_types, list) and isinstance(aux_loss_factors, list)):
            raise ValueError(f"Auxiliary loss types and factors should be list, bug got {aux_loss_types} and "
                             f"{aux_loss_factors}")
    if set(aux_loss_types) - set(supported_loss_types):
        raise ValueError(f"Auxiliary loss types in {supported_loss_types} only supported, but got {aux_loss_types}")
    if aux_loss_factors is None:
        raise ValueError(f"Got auxiliary loss types {aux_loss_types}, but corresponding loss factors are not set.")

    return aux_loss_types, aux_loss_factors


class MoEConfig:
    r"""
        The configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int): The number of experts employed. Default: 1
            capacity_factor (float): The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float): The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int): The number of experts is chosen by each token and it should not be larger
                than expert_num. Default: 1.
            expert_group_size (int): The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            group_wise_a2a (bool): Whether to enable group-wise alltoall communication, which can reduce communication
                time by converting part of intercommunication into intra communication. Default: False. This parameter
                is effective only when model parallel > 1 and data_parallel equal to expert parallel.
            comp_comm_parallel (bool): Whether to enable ffn compute and communication parallel, which can reduce pure
                communicattion time by splitting and overlapping compute and communication. Default: False.
            comp_comm_parallel_degree (int): The split number of compute and communication. The larger the numbers,
                the more overlap there will be but will consume more memory. Default: 2. This parameter is effective
                only when comp_comm_parallel enable.
            routing_policy (str): The routing policy to use in MoE layer. Default: TopkRouterV1.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import MoEConfig
            >>> moe_config = MoEConfig(expert_num=4, capacity_factor=5.0, aux_loss_factor=0.05, num_experts_chosen=1,
            ...                        expert_group_size=64, group_wise_a2a=True, comp_comm_parallel=False,
            ...                        comp_comm_parallel_degree=2, routing_policy="TopkRouterV2")
    """

    def __init__(self, expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1,
                 expert_group_size=None, group_wise_a2a=False, comp_comm_parallel=False, comp_comm_parallel_degree=2,
                 save_token_distribution=False, cur_layer=0, enable_cold_hot_expert=False, update_step=10000,
                 hot_expert_num=0, cold_token_percent=1.0, moe_module_name="", routing_policy="TopkRouterV1",
                 norm_topk_prob=True, enable_sdrop=False, use_fused_ops_topkrouter=False, router_dense_type="float32",
                 shared_expert_num=0, use_shared_expert_gating=False, max_router_load=128 * 1024,
                 topk_method="greedy", topk_group=None, n_group=None,
                 first_k_dense_replace=True, moe_intermediate_size=1407, routed_scaling_factor=1.0,
                 aux_loss_types=None, aux_loss_factors=None, z_loss_factor=0.):
        Validator.check_positive_int(expert_num, "expert_num")
        Validator.check_positive_float(aux_loss_factor, "aux_loss_factor")
        Validator.check_positive_int(num_experts_chosen, "num_experts_chosen")
        Validator.check_bool(group_wise_a2a, "group_wise_a2a")
        Validator.check_bool(comp_comm_parallel, "comp_comm_parallel")
        Validator.check_positive_int(comp_comm_parallel_degree, "comp_comm_parallel_degree")
        Validator.check_bool(save_token_distribution, "save_token_distribution")
        Validator.check_non_negative_int(cur_layer, "cur_layer")
        Validator.check_bool(enable_cold_hot_expert, "enable_cold_hot_expert")
        Validator.check_positive_int(update_step, "update_step")
        Validator.check_non_negative_int(hot_expert_num, "hot_expert_num")
        Validator.check_non_negative_float(cold_token_percent, "cold_token_percent")
        Validator.check_string(router_dense_type, ["float16", "float32", "bfloat16"], "router_dense_type")
        if expert_group_size is not None:
            Validator.check_positive_int(expert_group_size, "expert_group_size")
        if aux_loss_factor >= 1.0:
            raise ValueError(f"'aux_loss_factor' must be less than 1.0, "
                             f"but got {aux_loss_factor}.")
        if num_experts_chosen > expert_num:
            raise ValueError(f"'num_experts_chosen' must not be larger than 'expert_num', "
                             f"but got {num_experts_chosen}.")
        if hot_expert_num > expert_num:
            raise ValueError(f"'hot_expert_num' must not be larger than 'expert_num', "
                             f"but got {hot_expert_num}.")
        if cold_token_percent > 1.0 or cold_token_percent <= 0.0:
            raise ValueError(f"'cold_token_percent' must be in the range (0.0, 1.0], "
                             f"but got {cold_token_percent}.")

        self.expert_num = expert_num
        self.capacity_factor = capacity_factor
        self.aux_loss_factor = aux_loss_factor
        self.num_experts_chosen = num_experts_chosen
        self.expert_group_size = expert_group_size
        self.group_wise_a2a = group_wise_a2a
        self.comp_comm_parallel = comp_comm_parallel
        self.comp_comm_parallel_degree = comp_comm_parallel_degree
        self.save_token_distribution = save_token_distribution
        self.cur_layer = cur_layer
        self.enable_cold_hot_expert = enable_cold_hot_expert
        self.update_step = update_step
        self.hot_expert_num = hot_expert_num
        self.cold_token_percent = cold_token_percent
        self.moe_module_name = moe_module_name
        self.routing_policy = routing_policy
        self.norm_topk_prob = norm_topk_prob
        self.enable_sdrop = enable_sdrop
        self.use_fused_ops_topkrouter = use_fused_ops_topkrouter
        self.router_dense_type = dtype_map.get(router_dense_type)
        self.shared_expert_num = shared_expert_num
        self.use_shared_expert_gating = use_shared_expert_gating
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_intermediate_size = moe_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.aux_loss_types, self.aux_loss_factors = _check_aux_loss_config(aux_loss_types, aux_loss_factors)
        self.z_loss_factor = z_loss_factor
        self.max_router_load = max_router_load

    def __eq__(self, other) -> bool:
        return isinstance(other, MoEConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        """
        Compare the configuration dictionary of the current object with the default configuration dictionary,
        identify the differences between the two, and store these differences in a new dictionary called res-dict
        """
        config_dict = self.to_dict()
        default_dict = MoEConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


default_moe_config = MoEConfig()


def _check_moe_config(moe_config=None, parallel_config=None):
    """
        check if MoE with right configuration.
    """
    if not isinstance(moe_config, MoEConfig):
        raise TypeError(f"'moe_config' must be an instance of MoEConfig, but got {type(moe_config).__name__}.")
    use_moe = (moe_config.expert_num > 1)
    if use_moe is False:
        return
    if moe_config.expert_num % parallel_config.expert_parallel != 0:
        raise ValueError(f"When using MoE, the 'expert_num' in {type(moe_config).__name__} must be a multiple "
                         f"of 'expert_parallel' value in {type(parallel_config).__name__}, but got "
                         f"{moe_config.expert_num} for 'expert_num' and {parallel_config.expert_parallel} for "
                         f"'expert_parallel'.")

    device_num = D.get_group_size()
    if device_num % parallel_config.expert_parallel != 0:
        raise ValueError(f"device_num: {device_num} must be a multiple of expert_parallel: "
                         f"{parallel_config.expert_parallel}.")
    if parallel_config.data_parallel % parallel_config.expert_parallel != 0:
        raise ValueError(f"data parallel: {parallel_config.data_parallel} must be a multiple of "
                         f"expert_parallel: {parallel_config.expert_parallel} when using MoE.")
    if parallel_config.data_parallel * parallel_config.model_parallel > device_num:
        raise ValueError(f"The product of the data parallel: {parallel_config.data_parallel} and "
                         f"model parallel: {parallel_config.model_parallel} "
                         f"should be less than device_num: {device_num}.")


@constexpr
def calculate_expert_capacity(k, tokens_per_group, capacity_factor, expert_dim):
    return math.ceil(k * tokens_per_group * capacity_factor / expert_dim)


@constexpr
def calculate_expert_capacity_v2(k, tokens_per_group, capacity_factor, expert_dim, mp):
    raw_capacity = math.ceil(k * tokens_per_group * capacity_factor / expert_dim)
    if tokens_per_group < raw_capacity:
        raw_capacity = tokens_per_group
    if raw_capacity % mp > 0:
        raw_capacity = raw_capacity + mp - (raw_capacity % mp)
    return raw_capacity


class MoE(Cell):
    """
    The mixture of experts (MoE) implementation. The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.

    Args:
        hidden_size (int): The dimension of the inputs.
        ffn_hidden_size (int): The intermediate hidden size.
        dropout_rate (float): The dropout rate for the second linear's output.
        hidden_act (str): The activation of the internal feedforward layer. Supports 'relu',
                         'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                         'hsigmoid', 'logsigmoid' and so on. Default: gelu.
        param_init_type (dtype.Number): The parameter initialization type. Can be dtype.float32 or dtype.float16.
        moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig with
            default values. Please see `MoEConfig`.
        parallel_config(MoEParallelConfig): The parallel config for MoE, see `MoEParallelConfig`.
            Default `default_moeparallel_config`, an instance of `MoEParallelConfig` with default args.

    Inputs:
        - **x** (Tensor) - should be `[batch, seq_length, hidden_size]`. Float tensor.

    Outputs:
        Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.
    """

    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config):
        super(MoE, self).__init__()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.hidden_size = hidden_size
            self.expert_dim = moe_config.expert_num
            self.capacity_factor = moe_config.capacity_factor
            self.aux_loss_factor = moe_config.aux_loss_factor
            self.num_experts_chosen = moe_config.num_experts_chosen
            self.expert_group_size = moe_config.expert_group_size
            self.dp_group = parallel_config.data_parallel
            self.dp = parallel_config.data_parallel
            self.ep = parallel_config.expert_parallel
            self.mp = parallel_config.model_parallel
            self.comp_comm_parallel = moe_config.comp_comm_parallel
            self.comp_comm_parallel_degree = moe_config.comp_comm_parallel_degree
            self.group_wise_a2a = moe_config.group_wise_a2a
            if not (self.mp > 1 and self.dp == self.ep):
                self.group_wise_a2a = False
            from mindformers.modules.transformer import FeedForward

            self.ffn = FeedForward(hidden_size=hidden_size,
                                   ffn_hidden_size=ffn_hidden_size,
                                   dropout_rate=dropout_rate,
                                   hidden_act=hidden_act,
                                   expert_num=self.expert_dim,
                                   expert_group_size=self.expert_group_size,
                                   param_init_type=param_init_type,
                                   parallel_config=parallel_config)
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.transpose_2dim = P.Transpose().shard(((self.dp, 1),))
            self.transpose_3dim = P.Transpose().shard(((self.dp, 1, 1),))
            self.transpose_4dim = P.Transpose().shard(((1, self.dp, 1, 1),))
            self.transpose_4dim_dp = P.Transpose().shard(((1, 1, self.dp, 1),))
            self.batch_mm = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
            self.batch_mm2 = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
            self.mul = P.Mul()
            self.router = Router(d_model=hidden_size, moe_config=moe_config, routing_policy=None,
                                 training=True, parallel_config=parallel_config)
            self.cast = P.Cast()
            self.concat = P.Concat(3).shard(tuple((self.dp, 1, 1, 1) for _ in range(self.comp_comm_parallel_degree)))
            self.concat_dp = P.Concat(2).shard(((1, self.dp, 1, 1), (1, self.dp, 1, 1)))
            self.split = P.Split(axis=2, output_num=self.comp_comm_parallel_degree).shard(((1, self.dp, 1, 1),))
            self.stride_slice = P.StridedSlice().shard(((self.dp, 1, 1, 1),))
            self.stride_slice_dp = P.StridedSlice().shard(((1, self.dp, 1, 1),))
            self.stride_slice_ep = P.StridedSlice().shard(((self.ep, 1, 1, 1),))
            self.stride_slice_dp_mp = P.StridedSlice().shard(((1, self.dp, self.mp, 1),))
            self.stride_slice_ep_mp = P.StridedSlice().shard(((self.ep, 1, self.mp, 1),))
        else:
            self.hidden_size = hidden_size
            self.expert_dim = moe_config.expert_num
            self.capacity_factor = moe_config.capacity_factor
            self.aux_loss_factor = moe_config.aux_loss_factor
            self.num_experts_chosen = moe_config.num_experts_chosen
            self.dp_group = parallel_config.data_parallel
            self.dp = parallel_config.data_parallel
            self.ep = parallel_config.expert_parallel
            self.mp = parallel_config.model_parallel
            self.comp_comm_parallel = moe_config.comp_comm_parallel
            self.comp_comm_parallel_degree = moe_config.comp_comm_parallel_degree
            self.group_wise_a2a = moe_config.group_wise_a2a
            if not (self.mp > 1 and self.dp == self.ep):
                self.group_wise_a2a = False
            from mindformers.modules.transformer import FeedForward

            self.ffn = FeedForward(hidden_size=hidden_size,
                                   ffn_hidden_size=ffn_hidden_size,
                                   dropout_rate=dropout_rate,
                                   hidden_act=hidden_act,
                                   expert_num=self.expert_dim,
                                   param_init_type=param_init_type,
                                   parallel_config=parallel_config)
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.transpose_2dim = P.Transpose().shard(((self.dp, 1),))
            self.transpose_3dim = P.Transpose().shard(((self.dp, 1, 1),))
            self.transpose_4dim = P.Transpose().shard(((1, self.dp, 1, 1),))
            self.transpose_4dim_dp = P.Transpose().shard(((1, 1, self.dp, 1),))
            self.batch_mm = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
            self.batch_mm2 = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
            self.mul = P.Mul().shard(((), ()))
            self.router = Router(d_model=hidden_size, moe_config=moe_config, routing_policy="TopkRouterV1",
                                 training=True, parallel_config=parallel_config)
            self.cast = P.Cast()
            self.concat = P.Concat(3).shard(tuple((self.dp, 1, 1, 1) for _ in range(self.comp_comm_parallel_degree)))
            self.concat_dp = P.Concat(2).shard(((1, self.dp, 1, 1), (1, self.dp, 1, 1)))
            self.split = P.Split(axis=2, output_num=self.comp_comm_parallel_degree).shard(((1, self.dp, 1, 1),))
            self.stride_slice = P.StridedSlice().shard(((self.dp, 1, 1, 1),))
            self.stride_slice_dp = P.StridedSlice().shard(((1, self.dp, 1, 1),))
            self.stride_slice_ep = P.StridedSlice().shard(((self.ep, 1, 1, 1),))
            self.stride_slice_dp_mp = P.StridedSlice().shard(((1, self.dp, self.mp, 1),))
            self.stride_slice_ep_mp = P.StridedSlice().shard(((self.ep, 1, self.mp, 1),))
            self.enable_cold_hot_expert = moe_config.enable_cold_hot_expert
            if self.enable_cold_hot_expert:
                self.cur_layer = moe_config.cur_layer
                self.hot_expert_num = moe_config.hot_expert_num
                self.update_step = moe_config.update_step
                self.cold_token_percent = moe_config.cold_token_percent
                self.hot_expert_index = Parameter(
                    initializer(Tensor([[i for i in range(self.hot_expert_num)]], mstype.int32),
                                (1, self.hot_expert_num,), mstype.int32),
                    name="hot_expert_index" + str(self.cur_layer),
                    requires_grad=False, parallel_optimizer=False)
                self.cold_expert_index = Parameter(
                    initializer(Tensor([[i for i in range(self.hot_expert_num, self.expert_dim)]], mstype.int32),
                                (1, self.expert_dim - self.hot_expert_num,), mstype.int32),
                    name="cold_expert_index" + str(self.cur_layer),
                    requires_grad=False, parallel_optimizer=False)
                mlp_parallel_config = MoEParallelConfig(data_parallel=self.dp,
                                                        model_parallel=self.mp,
                                                        expert_parallel=1)
                self.mlp = FeedForward(hidden_size=hidden_size,
                                       ffn_hidden_size=ffn_hidden_size,
                                       dropout_rate=dropout_rate,
                                       hidden_act=hidden_act,
                                       expert_num=self.hot_expert_num,
                                       param_init_type=param_init_type,
                                       parallel_config=mlp_parallel_config)
                self.gather = P.Gather(0).shard(((1, 1, self.dp, 1), (1,)))
                self.gather2 = P.Gather(0).shard(((self.dp, 1, 1, 1), (1,)))
                self.concat0 = P.Concat(0).shard(((1,), (1,)))
                self.concat1 = P.Concat(1).shard(((self.dp, 1, 1, 1), (self.dp, 1, 1, 1)))
                self.concat2 = P.Concat(2).shard(((self.dp, 1, 1, 1), (self.dp, 1, 1, 1)))
                self.zeros = P.Zeros()
                self.transpose_1dim_dp = P.Transpose().shard(((self.dp, 1, 1, 1),))
                self.equal = P.Equal().shard(((1, 1), (1, 1)))
                self.equal2 = P.Equal().shard(((1,), ()))
                self.reduce_any = P.ReduceAny().shard(((1, 1),))
                self.topk = P.TopK().shard(((1,),))

    def ffn_forward(self, expert_input, capacity):
        """
        Computing the FFN.
        """
        pad_size = 0
        if self.group_wise_a2a:
            # If capacity can't div by mp, pad for mp shard.
            if capacity % self.mp != 0:
                pad_size = self.mp - (capacity % self.mp)
            if pad_size != 0:
                capacity += pad_size
                pad_tensor = self.stride_slice_dp(expert_input, (0, 0, 0, 0),
                                                  (self.expert_dim, self.dp_group, pad_size, self.hidden_size),
                                                  (1, 1, 1, 1))
                expert_input = self.concat_dp((expert_input, pad_tensor))
            # capacity shard by mp
            expert_input = self.stride_slice_dp_mp(expert_input, (0, 0, 0, 0),
                                                   (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                   (1, 1, 1, 1))
            # group-wise alltoall
            expert_input = self.stride_slice_ep_mp(expert_input, (0, 0, 0, 0),
                                                   (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                   (1, 1, 1, 1))
            # allgather
            expert_input = self.stride_slice_ep(expert_input, (0, 0, 0, 0),
                                                (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                (1, 1, 1, 1))

        expert_input = self.reshape(expert_input, (self.expert_dim * self.dp_group * capacity,
                                                   self.hidden_size))
        # expert_output's shape: (self.expert_dim, self.dp_group*expert_capacity, self.hidden_size)
        expert_output = self.ffn(expert_input)
        expert_output = self.reshape(expert_output, (self.expert_dim, self.dp_group,
                                                     capacity, self.hidden_size))

        if self.group_wise_a2a:
            # capacity shard by mp
            expert_output = self.stride_slice_ep_mp(expert_output, (0, 0, 0, 0),
                                                    (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                    (1, 1, 1, 1))
            # group-wise alltoall
            expert_output = self.stride_slice_dp_mp(expert_output, (0, 0, 0, 0),
                                                    (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                    (1, 1, 1, 1))
            # allgather
            expert_output = self.stride_slice_dp(expert_output, (0, 0, 0, 0),
                                                 (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                 (1, 1, 1, 1))
            # Slice capacity back to org shape.
            if pad_size != 0:
                capacity -= pad_size
                expert_output = self.stride_slice_dp(expert_output, (0, 0, 0, 0),
                                                     (self.expert_dim, self.dp_group, capacity, self.hidden_size),
                                                     (1, 1, 1, 1))
        if self.enable_cold_hot_expert:
            # expert_output's shape: (self.dp_group, self.expert_dim, expert_capacity, self.hidden_size)
            expert_output = self.transpose_4dim(expert_output, (1, 0, 2, 3))
        else:
            # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
            expert_output = self.transpose_4dim(expert_output, (1, 3, 0, 2))
        return expert_output

    def ffn_parallel_forward(self, expert_input, capacity):
        """
        Split and overlap FFN compute and communication.
        """
        # Pad capacity for comp_comm_parallel_degree split.
        pad_size = 0
        if capacity % self.comp_comm_parallel_degree != 0:
            pad_size = self.comp_comm_parallel_degree - (capacity % self.comp_comm_parallel_degree)
            capacity += pad_size
            pad_tensor = self.stride_slice_dp(expert_input, (0, 0, 0, 0),
                                              (self.expert_dim, self.dp_group, pad_size, self.hidden_size),
                                              (1, 1, 1, 1))
            expert_input = self.concat_dp((expert_input, pad_tensor))

        sub_capacity = capacity // self.comp_comm_parallel_degree
        output_list = []
        for sub_expert_input in self.split(expert_input):
            sub_expert_output = self.ffn_forward(sub_expert_input, sub_capacity)
            output_list.append(sub_expert_output)
        expert_output = self.concat(output_list)

        # Slice capacity back to org shape.
        if pad_size != 0:
            capacity -= pad_size
            expert_output = self.stride_slice(expert_output, (0, 0, 0, 0),
                                              (self.dp_group, self.hidden_size, self.expert_dim, capacity),
                                              (1, 1, 1, 1))
        return expert_output

    def construct(self, input_tensor):
        """forward process"""
        input_shape = F.shape(input_tensor)
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))
        bs_and_dmodel = self.shape(input_tensor)
        tokens_per_group = bs_and_dmodel[0] // self.dp_group
        input_tensor = self.reshape(input_tensor, (self.dp_group, tokens_per_group, self.hidden_size))

        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group,
                                                    self.capacity_factor, self.expert_dim)
        # dispatch_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        # combine_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        dispatch_tensor, combine_tensor, aux_loss = self.router(input_tensor)

        # after transpose, input_tensor's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        input_tensor = self.transpose_3dim(input_tensor, (0, 2, 1))
        dispatch_tensor = self.reshape(dispatch_tensor, (self.dp_group, tokens_per_group,
                                                         self.expert_dim * expert_capacity))
        dispatch_tensor = self.cast(dispatch_tensor, F.dtype(input_tensor))
        # expert_input's shape: (self.dp_group, self.hidden_size, self.expert_dim * expert_capacity)
        expert_input = self.batch_mm(input_tensor, dispatch_tensor)
        expert_input = self.reshape(expert_input, (self.dp_group, self.hidden_size, self.expert_dim,
                                                   expert_capacity))
        # The following four ops are to implement transpose(expert_input, (2, 0, 3, 1)), for that a single transpose
        # has bad performance
        expert_input = self.reshape(expert_input, (self.dp_group * self.hidden_size,
                                                   self.expert_dim * expert_capacity))
        expert_input = self.transpose_2dim(expert_input, (1, 0))
        expert_input = self.reshape(expert_input, (self.expert_dim, expert_capacity, self.dp_group,
                                                   self.hidden_size))
        if self.enable_cold_hot_expert:
            hot_expert_index = self.hot_expert_index.value().copy()[0]
            cold_expert_index = self.cold_expert_index.value().copy()[0]

            hot_expert_input = self.gather(expert_input, hot_expert_index, 0)
            cold_expert_input = expert_input
            cold_expert_capacity = int(expert_capacity * self.cold_token_percent)
            hot_expert_input = self.transpose_4dim_dp(hot_expert_input, (2, 0, 1, 3))
            cold_expert_input = self.transpose_4dim_dp(cold_expert_input, (0, 2, 1, 3))
            cold_expert_input = self.stride_slice_dp(
                cold_expert_input, (0, 0, 0, 0),
                (self.expert_dim, self.dp_group, cold_expert_capacity, self.hidden_size),
                (1, 1, 1, 1))
            # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
            if self.comp_comm_parallel:
                cold_expert_output = self.ffn_parallel_forward(cold_expert_input, cold_expert_capacity)
            else:
                cold_expert_output = self.ffn_forward(cold_expert_input, cold_expert_capacity)

            hot_expert_input = self.reshape(hot_expert_input,
                                            (self.hot_expert_num * self.dp_group * expert_capacity, self.hidden_size))
            hot_expert_output = self.mlp(hot_expert_input)

            hot_expert_output = self.reshape(hot_expert_output,
                                             (self.dp_group, self.hot_expert_num, expert_capacity, self.hidden_size))

            cold_expert_output = self.gather2(cold_expert_output, cold_expert_index, 1)
            if self.cold_token_percent < 1.0:
                zeros = self.zeros((self.dp_group, self.expert_dim - self.hot_expert_num,
                                    expert_capacity - cold_expert_capacity, self.hidden_size), mstype.float16)
                cold_expert_output = self.concat2((cold_expert_output, zeros))

            expert_output = self.concat1((hot_expert_output, cold_expert_output))
            expert_index = self.concat0((hot_expert_index, cold_expert_index))
            _, expert_gather_index = self.reshape(expert_index, (1, -1)).topk(self.expert_dim, largest=False)
            expert_gather_index = self.reshape(expert_gather_index, (-1,))
            expert_output = self.gather2(expert_output, expert_gather_index, 1)
            # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
            expert_output = self.transpose_1dim_dp(expert_output, (0, 3, 1, 2))
        else:
            # expert_input's shape: (self.expert_dim, self.dp_group, expert_capacity, self.hidden_size)
            expert_input = self.transpose_4dim_dp(expert_input, (0, 2, 1, 3))
            # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
            if self.comp_comm_parallel:
                expert_output = self.ffn_parallel_forward(expert_input, expert_capacity)
            else:
                expert_output = self.ffn_forward(expert_input, expert_capacity)

        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size,
                                                     self.expert_dim * expert_capacity))
        combine_tensor = self.reshape(combine_tensor, (self.dp_group, tokens_per_group,
                                                       self.expert_dim * expert_capacity))
        # combine_tensor's shape: (self.dp_group, self.expert_dim * expert_capacity, tokens_per_group)
        combine_tensor = self.transpose_3dim(combine_tensor, (0, 2, 1))
        combine_tensor = self.cast(combine_tensor, F.dtype(expert_output))

        # combined_output's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        combined_output = self.batch_mm2(expert_output, combine_tensor)
        # combined_output's shape: (self.dp_group, tokens_per_group, self.hidden_size)
        combined_output = self.transpose_3dim(combined_output, (0, 2, 1))
        combined_output = self.reshape(combined_output, (bs_and_dmodel[0], bs_and_dmodel[1]))
        combined_output = self.reshape(combined_output, input_shape)

        aux_loss = self.mul(self.aux_loss_factor, aux_loss)
        return combined_output, aux_loss


class MoEV2(Cell):
    """
    The mixture of experts (MoE) implementation. The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.
    This is a common interface, which allows any ffn class and any Router algorithm(implemented in V2 form).

    Args:
        hidden_size (int): The dimension of the inputs.
        ffn_hidden_size (int): The intermediate hidden size.
        dropout_rate (float): The dropout rate for the second linear's output.
        hidden_act (str): The activation of the internal feedforward layer. Supports 'relu',
                         'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                         'hsigmoid', 'logsigmoid' and so on. Default: gelu.
        param_init_type (dtype.Number): The parameter initialization type. Can be dtype.float32 or dtype.float16.
        moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig with
            default values. Please see `MoEConfig`.
        parallel_config(MoEParallelConfig): The parallel config for MoE, see `MoEParallelConfig`.
            Default `default_moeparallel_config`, an instance of `MoEParallelConfig` with default args.
        return_extra_loss (bool): whether to return extra regularization loss. Default False.

    Inputs:
        - **x** (Tensor) - should be `[batch, seq_length, hidden_size]`. Float tensor.

    Outputs:
        Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.
    """

    def __init__(self,
                 ffn,
                 dim,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config,
                 return_extra_loss=False):
        super(MoEV2, self).__init__()
        self.hidden_size = dim
        self.expert_dim = moe_config.expert_num
        self.return_extra_loss = return_extra_loss
        self.capacity_factor = moe_config.capacity_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.dp_group = parallel_config.data_parallel
        self.dp = parallel_config.data_parallel
        self.ep = parallel_config.expert_parallel
        self.mp = parallel_config.model_parallel
        self.group_wise_a2a = moe_config.group_wise_a2a if self.mp > 1 else False
        self.add_loss = P.Add()
        self.dp_moe = self.dp // self.ep
        self.dp_range = Tensor(np.arange(self.dp_group).reshape(-1, 1), mstype.int32)  # (dp, 1) = [[0],[1],[2]...[dp]]

        self.ffn = ffn
        Validator.check_string(moe_config.routing_policy, ["TopkRouterV2"], "routing_policy")
        self.router = Router(d_model=self.hidden_size,
                             moe_config=moe_config,
                             routing_policy=moe_config.routing_policy,
                             training=(not get_predict_run_mode()),
                             parallel_config=parallel_config)

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.transpose_4dim_dp1 = P.Transpose().shard(((1, self.dp, 1, 1),))
        self.transpose_4dim_dp0 = P.Transpose().shard(((self.dp, 1, 1, 1),))
        self.transpose_5dim_ep2 = P.Transpose().shard(((self.dp_moe, 1, self.ep, 1, 1),))
        self.concat_dp = P.Concat(2).shard(((1, self.dp, 1, 1), (1, self.dp, 1, 1)))
        self.stride_slice = P.StridedSlice().shard(((self.dp, 1, 1, 1),))
        self.stride_slice_dp = P.StridedSlice().shard(((1, self.dp, 1, 1),))
        self.stride_slice_ep = P.StridedSlice().shard(((self.ep, 1, 1, 1),))
        self.stride_slice_dp_mp = P.StridedSlice().shard(((1, self.dp, self.mp, 1),))
        self.stride_slice_ep_mp = P.StridedSlice().shard(((self.ep, 1, self.mp, 1),))

        # outer_dp groupwiseall2all
        self.stride_slice_outer_dp_mp = P.StridedSlice().shard(((self.dp_moe, 1, self.ep, self.mp, 1),))
        self.stride_slice_outer_ep_mp = P.StridedSlice().shard(((self.dp_moe, self.ep, 1, self.mp, 1),))
        self.stride_slice_outer_ep = P.StridedSlice().shard(((self.dp_moe, self.ep, 1, 1, 1),))
        self.stride_slice_outer_dp = P.StridedSlice().shard(((self.dp_moe, 1, self.ep, 1, 1),))
        self.transpose_5dim_ep1 = P.Transpose().shard(((self.dp_moe, self.ep, 1, 1, 1),))

    def ffn_forward(self, expert_input, capacity):
        """
        Computing the FFN.
        """
        if self.group_wise_a2a:
            # (dp_moe, ep, E, n, h) <-- (dp, E, n, h)
            expert_input = self.reshape(expert_input,
                                        (self.dp_moe, self.ep, self.expert_dim, -1, self.hidden_size))
            # dp_moe <==> outer_dp <==> dp // ep
            # (dp_moe, E, ep, n, h) <-- (dp_moe, ep, E, n, h)
            expert_input = self.transpose_5dim_ep1(expert_input, (0, 2, 1, 3, 4))
            # capacity shard by mp
            expert_input = self.stride_slice_outer_dp_mp(expert_input, (0, 0, 0, 0, 0),
                                                         (self.dp_moe, self.expert_dim,
                                                          self.ep, capacity, self.hidden_size),
                                                         (1, 1, 1, 1, 1))
            # group-wise alltoall
            expert_input = self.stride_slice_outer_ep_mp(expert_input, (0, 0, 0, 0, 0),
                                                         (self.dp_moe, self.expert_dim,
                                                          self.ep, capacity, self.hidden_size),
                                                         (1, 1, 1, 1, 1))
            # allgather
            expert_input = self.stride_slice_outer_ep(expert_input, (0, 0, 0, 0, 0),
                                                      (self.dp_moe, self.expert_dim,
                                                       self.ep, capacity, self.hidden_size),
                                                      (1, 1, 1, 1, 1))
        else:
            # (dp_moe, ep, E, n, h) <-- (dp, E, n, h)
            expert_input = self.reshape(expert_input,
                                        (self.dp_moe, self.ep, self.expert_dim, -1, self.hidden_size))
            # (dp_moe, E, ep, n, h) <-- (dp_moe, ep, E, n, h)
            expert_input = self.transpose_5dim_ep2(expert_input, (0, 2, 1, 3, 4))

        expert_input = self.reshape(expert_input, (-1, self.hidden_size))
        expert_output = self.ffn(expert_input)

        if self.group_wise_a2a:
            expert_output = self.reshape(expert_output,
                                         (self.dp_moe, self.expert_dim, self.ep, -1, self.hidden_size))
            # dp_moe <==> outer_dp <==> dp // ep
            # capacity shard by mp
            expert_output = self.stride_slice_outer_ep_mp(expert_output, (0, 0, 0, 0, 0),
                                                          (self.dp_moe, self.expert_dim,
                                                           self.ep, capacity, self.hidden_size),
                                                          (1, 1, 1, 1, 1))
            # group-wise alltoall
            expert_output = self.stride_slice_outer_dp_mp(expert_output, (0, 0, 0, 0, 0),
                                                          (self.dp_moe, self.expert_dim,
                                                           self.ep, capacity, self.hidden_size),
                                                          (1, 1, 1, 1, 1))
            # allgather
            expert_output = self.stride_slice_outer_dp(expert_output, (0, 0, 0, 0, 0),
                                                       (self.dp_moe, self.expert_dim,
                                                        self.ep, capacity, self.hidden_size),
                                                       (1, 1, 1, 1, 1))
            expert_output = self.transpose_5dim_ep2(expert_output, (0, 2, 1, 3, 4))
        else:
            expert_output = self.reshape(expert_output,
                                         (self.dp_moe, self.expert_dim, self.ep, -1, self.hidden_size))
            # (dp_moe, ep, E, n, h) <-- (dp_moe, E, ep, n, h)
            expert_output = self.transpose_5dim_ep2(expert_output, (0, 2, 1, 3, 4))
        expert_output = self.reshape(expert_output, (self.dp, self.expert_dim, -1, self.hidden_size))
        return expert_output

    def construct(self, input_tensor, extra_loss=0.):
        """forward process"""
        input_tensor_shape = self.shape(input_tensor)
        input_tensor = self.reshape(input_tensor, (self.dp_group, -1, self.hidden_size))  # (dp, N, h) <-- (B*S, h)

        # calculate router, we do not use router_aux_loss right now
        # (dp, E, n)int32, (dp, N, k)int32, (dp, N, k)fp16, (1,) <-- (dp, N, h),
        # where 0<= dispatch_index < 1+N, 0<= combine_index <E*(1+n)
        dispatch_policy, combine_policy, router_coeff, router_aux_loss = self.router(input_tensor)

        # dispatch
        expert_capacity = dispatch_policy.shape[-1]
        # (dp, E, n, h) <-- (dp, N, h), (dp, E, n)
        expert_input = self.router.router.dispatch(input_tensor, dispatch_policy)
        # ffn, (E, dp, n, h) <-- (E, dp, n, h)
        expert_output = self.ffn_forward(expert_input, expert_capacity)
        # combine, (dp, N, k, h) <-- (dp, E*(1+n), h), (dp, N, k)
        output_tensor = self.router.router.combine(expert_output, combine_policy, router_coeff)
        output_tensor = self.reshape(output_tensor, input_tensor_shape)  # (B*S, h) <-- (dp, N, h)
        if self.return_extra_loss:
            final_extra_loss = self.add_loss(extra_loss, router_aux_loss)
            return output_tensor, final_extra_loss
        return output_tensor  # (dp, N, h)


class Router(Cell):
    r"""
        A router backbone used to calculate logits of each token, which should be cascaded by router implementations
        mapping tokens to experts.
        when moe_config.num_experts_chosen = 1, use top1 routing;
        when moe_config.num_experts_chosen > 1, use topk routing

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            routing_policy: The policy of mapping tokens to experts. Default: topkRouter
            training (bool): The value indicating whether is in training phase.
            parallel_config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 routing_policy=None,
                 training=True,
                 parallel_config=None):
        super(Router, self).__init__()
        dp = parallel_config.data_parallel
        self.d_model = d_model
        self.moe_config = moe_config
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.training = training
        self.routing_policy = routing_policy
        self.noisy_policy = None  # candidate: ["jitter", "rsample", "None"]
        self.noisy_epsilon = 1e-2
        self.noise = Tensor(np.random.uniform(1 - self.noisy_epsilon, 1 + self.noisy_epsilon, (d_model,)))

        self.dense = Dense(in_channels=self.d_model, out_channels=self.expert_dim,
                           has_bias=False, dtype=moe_config.router_dense_type)
        self.dense.matmul.shard(((dp, 1), (1, 1)))
        self.mul = P.Mul()
        self.cast = P.Cast()

        if self.routing_policy == "TopkRouterV1":
            self.router = TopkRouter(d_model=d_model, moe_config=moe_config, training=training,
                                     parallel_config=parallel_config)
        elif self.routing_policy == "TopkRouterV2":
            self.router = TopkRouterV2(d_model=d_model, moe_config=moe_config, training=training,
                                       parallel_config=parallel_config)
        else:
            self.router = routing_policy

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.mul.shard(((dp, 1, 1), (dp,)))

    def construct(self, input_tensor):
        input_tensor = self.cast(input_tensor, self.moe_config.router_dense_type)
        if self.noisy_policy == "jitter" and self.training:
            # Here, we temporarily implement the multiplicative jitter this way,
            # for the lack of UniforReal parallel operator.
            input_tensor = self.mul(input_tensor, self.noise)

        router_logits = self.dense(input_tensor)
        return self.router(router_logits)


class TopkRouter(Cell):
    r"""
        A router implementation which maps each tokens to the topk expert.

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            training (bool): The value indicating whether is in training phase.
            config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(1)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 training=True,
                 parallel_config=None):
        super(TopkRouter, self).__init__()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            dp = parallel_config.data_parallel
            self.d_model = d_model
            self.expert_dim = moe_config.expert_num
            self.capacity_factor = moe_config.capacity_factor
            self.training = training
            self.dp_group = dp
            self.noisy_policy = None
            self.cast = P.Cast()
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.softmax = P.Softmax(axis=-1)
            self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False)
            self.num_experts_chosen = moe_config.num_experts_chosen
            self.onehot = P.OneHot()
            self.onehot2 = P.OneHot()
            self.onehot3 = P.OneHot()
            self.on_value = Tensor(1.0, mstype.float32)
            self.off_value = Tensor(0.0, mstype.float32)

            self.reduce_mean = P.ReduceMean(keep_dims=False)
            self.reduce_mean2 = P.ReduceMean(keep_dims=False)
            self.reduce_mean3 = P.ReduceMean(keep_dims=False)
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.mul3 = P.Mul()
            self.mul4 = P.Mul()
            self.mul5 = P.Mul()
            self.mul6 = P.Mul()
            self.mul7 = P.Mul()
            self.mul8 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul9 = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.not_equal = P.NotEqual()
            self.div1 = P.RealDiv()
            self.div2 = P.RealDiv()
            self.add = P.Add()
            self.add1 = P.Add()
            self.add2 = P.Add()
            self.add3 = P.Add()
            self.add4 = P.Add()
            self.sub = P.Sub()

            self.cumsum = P.CumSum(exclusive=True)
            self.less = P.Less()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.reduce_sum_keep = P.ReduceSum(keep_dims=True)
            self.reduce_sum_keep2 = P.ReduceSum(keep_dims=True)
            self.expand = P.ExpandDims()
            self.expand2 = P.ExpandDims()
            self.add_scala = P.Add()
            self.init_loss = Tensor(0.0, mstype.float32)
        else:
            dp = parallel_config.data_parallel
            self.d_model = d_model
            self.expert_dim = moe_config.expert_num
            self.capacity_factor = moe_config.capacity_factor
            self.save_token_distribution = moe_config.save_token_distribution
            self.enable_cold_hot_expert = moe_config.enable_cold_hot_expert
            self.training = training
            self.dp_group = dp
            self.noisy_policy = None
            self.cast = P.Cast()
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.softmax = P.Softmax(axis=-1).shard(((dp, 1, 1,),))
            self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False).shard(((dp, 1, 1),))
            self.num_experts_chosen = moe_config.num_experts_chosen
            self.onehot = P.OneHot().shard(((dp, 1, 1), (), ()))
            self.onehot2 = P.OneHot().shard(((dp, 1, 1), (), ()))
            self.onehot3 = P.OneHot().shard(((dp, 1, 1, 1), (), ()))
            self.on_value = Tensor(1.0, mstype.float32)
            self.off_value = Tensor(0.0, mstype.float32)

            self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_mean2 = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_mean3 = P.ReduceMean(keep_dims=False).shard(((dp, 1),))
            self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul2 = P.Mul().shard(((), ()))
            self.mul3 = P.Mul().shard(((), ()))
            self.mul4 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul5 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul6 = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul7 = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul8 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul9 = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.not_equal = P.NotEqual().shard(((dp, 1, 1, 1), ()))
            self.div1 = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
            self.div2 = P.RealDiv().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
            self.add1 = P.Add().shard(((dp, 1, 1), ()))
            self.add2 = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.add3 = P.Add().shard(((dp, 1), (dp, 1)))
            self.add4 = P.Add().shard(((dp, 1, 1, 1), ()))
            self.sub = P.Sub().shard(((), (dp, 1, 1)))

            self.cumsum = P.CumSum(exclusive=True).shard(((dp, 1, 1),))
            self.less = P.Less().shard(((dp, 1, 1), ()))
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_sum_keep = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
            self.reduce_sum_keep2 = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1, 1),))
            self.expand = P.ExpandDims().shard(((dp, 1),))
            self.expand2 = P.ExpandDims().shard(((dp, 1, 1),))
            self.add_scala = P.Add().shard(((), ()))
            self.init_loss = Tensor(0.0, mstype.float32)
            if self.save_token_distribution:
                self.cur_layer = moe_config.cur_layer
                self.tensor_summary = P.TensorSummary()
            if self.enable_cold_hot_expert:
                self.cur_layer = moe_config.cur_layer
                self.cumsum_value = Parameter(initializer('zeros', (self.expert_dim,), mstype.int32),
                                              name="cumsum_value" + str(self.cur_layer), requires_grad=False,
                                              parallel_optimizer=False)
                self.assign = P.Assign().shard(((1,), (1,)))

    def construct(self, router_logits):
        """forward process"""
        router_logits_shape = self.shape(router_logits)
        router_logits = self.reshape(router_logits, (-1, router_logits_shape[-1]))
        logits_shape = self.shape(router_logits)
        tokens_per_group = logits_shape[0] // self.dp_group
        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group, self.capacity_factor,
                                                    self.expert_dim)
        router_logits = self.reshape(router_logits, (self.dp_group, tokens_per_group, self.expert_dim))

        accum_expert_mask = 0
        accum_expert_gate = 0
        loss = self.init_loss
        mask_count = 0
        accum_combine_tensor = 0
        # Probabilities for each token of what expert is should be sent to
        router_prob = self.softmax(router_logits)

        for expert_chosen_index in range(self.num_experts_chosen):
            # for each token, set the router_prob of the selected experts to zero
            router_prob = self.mul4(router_prob, self.sub(self.on_value, accum_expert_mask))
            # shape is : (dp_group, tokens_per_group)
            expert_index, expert_gate = self.argmax(router_prob)
            # expert_mask's shape: (dp_group, tokens_per_group, self.expert_dim)
            expert_mask = self.onehot(expert_index, self.expert_dim, self.on_value, self.off_value)
            # renormalize the rest prob to be of sum 1
            router_prob_normal = self.div1(router_prob, self.add1(self.reduce_sum_keep(router_prob, -1), 1e-9))

            # the balance loss is computed at each routing step
            loss = self.add_scala(loss, self._auxiliary_loss(expert_mask, router_prob_normal))

            output = self._maskout_overflowed_tokens(expert_mask, expert_capacity, expert_gate,
                                                     mask_count, expert_chosen_index)
            expert_mask, expert_gate, expert_mask_flat, position_in_expert = output[0], output[1], output[2], output[3]
            accum_expert_mask = self.add(accum_expert_mask, expert_mask)
            accum_expert_gate = self.add3(accum_expert_gate, expert_gate)
            mask_count = self.add(mask_count, self.reduce_sum_keep(expert_mask, 1))

            # combine_tensor's shape: (dp_group, tokens_per_group)
            combine_tensor = self.mul7(expert_gate, expert_mask_flat)
            # combine_tensor's shape: (dp_group, tokens_per_group, self.expert_dim)
            combine_tensor = self.mul8(self.expand(combine_tensor, -1),
                                       self.onehot2(expert_index, self.expert_dim, self.on_value, self.off_value))
            # combine_tensor's shape: (dp_group, tokens_per_group, self.expert_dim, self.expert_capacity)
            combine_tensor = self.mul9(self.expand2(combine_tensor, -1),
                                       self.onehot3(self.cast(position_in_expert, mstype.int32), expert_capacity,
                                                    self.on_value, self.off_value))
            accum_combine_tensor = self.add2(accum_combine_tensor, combine_tensor)

        # expert weights normalization
        combine_tensor_sum = self.reduce_sum_keep2(self.reduce_sum_keep2(accum_combine_tensor, -1), -2)
        accum_combine_tensor = self.div2(accum_combine_tensor, self.add4(combine_tensor_sum, 1e-9))
        # dispatch_tensor is of boolean type. Here, using NotEqual instead of Cast, for that 'Cast to bool' has
        # bad performance
        dispatch_tensor = self.not_equal(accum_combine_tensor, 0.0)
        return dispatch_tensor, accum_combine_tensor, loss

    def _auxiliary_loss(self, expert_mask, router_prob):
        """
        Computing the load balance loss.
        """
        # density_1's shape: (dp_group, self.expert_dim)
        density_1 = self.reduce_mean(expert_mask, 1)
        # density_1_proxy's shape: (dp_group, self.expert_dim)
        density_1_proxy = self.reduce_mean2(router_prob, 1)
        loss = self.mul(density_1, density_1_proxy)
        loss = self.reduce_mean3(loss)
        loss = self.mul3(self.mul2(loss, self.expert_dim), self.expert_dim)
        return loss

    def _maskout_overflowed_tokens(self, expert_mask, expert_capacity, expert_gate, last_num, expert_chosen_index):
        """
        Keeping only the tokens that fit within expert_capacity.
        """
        cumsum = self.cumsum(expert_mask, 1)
        if expert_chosen_index > 0:
            cumsum = self.add(cumsum, last_num)
        if self.save_token_distribution:
            record_name = 'layer-' + str(self.cur_layer)
            self.tensor_summary(record_name, cumsum[0][-1])
        if self.enable_cold_hot_expert:
            cumsum_int_value = self.cast(cumsum[0][-1], mstype.int32)
            self.assign(self.cumsum_value, cumsum_int_value)
        # position_in_expert's shape: (dp_group, tokens_per_group, self.expert_dim)
        position_in_expert = self.mul4(cumsum, expert_mask)
        less_result = self.less(position_in_expert, expert_capacity)
        # expert_mask's shape: (dp_group, tokens_per_group, self.expert_dim)
        expert_mask = self.mul5(less_result, expert_mask)
        # expert_mask_flat's shape: (dp_group, tokens_per_group)
        expert_mask_flat = self.reduce_sum(expert_mask, -1)

        # Mask out the experts that have overflowed the expert_capacity.
        # expert_gate's shape: (dp_group, tokens_per_group)
        expert_gate = self.mul6(expert_gate, expert_mask_flat)
        output = (expert_mask, expert_gate, expert_mask_flat, position_in_expert)
        return output


# pylint: disable=C0330
class TopkRouterV2(Cell):
    r"""
        A router implementation which maps each tokens to the topk expert.

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            training (bool): The value indicating whether is in training phase.
            config: The parallel-related configuration.
        Inputs:
            - **router_logits** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group,
            expert\_dim)`.(dp, N, expert_dim)

        Outputs:
            - **dispatch_index** (Tensor) - Tensor of shape :math:`(data\_parallel, expert\_dim, expert\_capacity)`,
            - **combine_index** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group, k)`,
            - **router_coeff** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group, k)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 training=True,
                 parallel_config=None):
        super(TopkRouterV2, self).__init__()

        dp = parallel_config.data_parallel
        self.mp = parallel_config.model_parallel
        self.ep = parallel_config.expert_parallel
        self.d_model = d_model
        self.moe_config = moe_config
        self.expert_dim = moe_config.expert_num
        self.egroup_size = moe_config.expert_group_size
        self.capacity_factor = moe_config.capacity_factor
        self.save_token_distribution = moe_config.save_token_distribution
        self.training = training
        self.dp_group = dp
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.range = Tensor(np.tile(np.arange(moe_config.max_router_load) + 1,
                                    (self.num_experts_chosen, 1)), mstype.float32)

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.softmax = P.Softmax(axis=-1).shard(((dp, 1, 1,),))
        self.topk = P.TopK().shard(((dp, 1, 1),))
        self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False).shard(((dp, 1, 1),))
        self.onehot_2d = P.OneHot().shard(((dp, 1, 1), (), ()))
        self.onehot_3d = P.OneHot().shard(((dp, 1, 1, 1), (), ()))
        self.cumsum = P.CumSum(exclusive=False).shard(((dp, 1, 1),))
        self.mul_2d_1d = P.Mul().shard(((dp, 1), ()))
        self.mul_2d = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul_3d = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul_4d = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.add_2d = P.Add().shard(((dp, 1), (dp, 1)))
        self.add_3d = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.less = P.Less().shard(((dp, 1, 1), ()))
        self.gt = P.Greater().shard(((dp, 1), ()))
        self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, 1, 1),))
        self.transpose_3d = P.Transpose().shard(((dp, 1, 1),))
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))
        self.slice = P.StridedSlice().shard(((dp, 1, 1),))
        self.slice_range = P.StridedSlice().shard(((1, 1),))
        self.bmm_range = P.BatchMatMul().shard(((1, 1), (dp, 1, 1, 1)))
        self.add_eps = P.Add().shard(((dp, 1, 1), ()))
        self.reduce_sum_keep = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
        self.div_3d = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.concat_3d = P.Concat(1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.zeros = Tensor(np.zeros((dp, self.expert_dim, 1, d_model)), mstype.float16)
        self.zeros_3d = Tensor(np.zeros((dp, 1, d_model)), mstype.float16)
        self.dispatch_gather = P.Gather(batch_dims=1).shard(((dp, 1, 1), (dp, 1, 1),))
        self.concat = P.Concat(2).shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.combine_gather = P.Gather(batch_dims=1).shard(((dp, 1, 1), (dp, 1, 1),))
        self.mul_router_coeff = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.sum_router_coeff = P.ReduceSum(keep_dims=False).shard(((dp, 1, 1, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1, 1), ()))

        # sort indexing
        self.range2 = Tensor(np.tile(np.arange(moe_config.max_router_load),
                                     (self.expert_dim, 1)), mstype.float32)
        self.add_one = P.Add().shard(((dp, 1, 1), ()))
        self.add_range = P.Add().shard(((1, 1, 1), ()))
        self.sub_range = P.Sub().shard(((), (dp, 1, 1)))
        self.mul_range = P.Mul().shard(((dp, 1, 1), (1, 1, 1)))
        self.sort_range = P.Sort().shard(((dp, 1, 1),))
        self.mod = P.Mod().shard(((dp, 1, 1), ()))
        self.print = P.Print()

        # group limited greedy
        self.tile = P.Tile()

        # auxiliary loss
        self.z_loss_func = ZLoss(parallel_config=parallel_config)
        self.z_loss_coeff = Tensor(moe_config.z_loss_factor, mstype.float32)
        # for auxiliary balancing loss
        self.mul = P.Mul().shard(((), ()))
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.reduce_mean_2d = P.ReduceMean(keep_dims=False).shard(((dp, 1),))
        self.add_scalar = P.Add()
        # auxiliary loss config
        self.aux_loss_config = dict(zip(set(moe_config.aux_loss_types), moe_config.aux_loss_factors))

        # dynamic capacity
        self.on_value_int = Tensor(1, mstype.int32)
        self.off_value_int = Tensor(0, mstype.int32)
        self.mod = P.Mod().shard(((dp, 1, 1), ()))
        self.add = P.Add()
        self.sub = P.Sub()
        self.mod_expert = P.Mod()
        self.tensor2scalar = TensorToScalar()

        # topkrouter
        if self.moe_config.use_fused_ops_topkrouter:
            # pylint: disable=W0212
            self.topkrouter = P._inner_ops.TopKRouter().shard(((dp, 1, 1),))

    def dispatch(self, input_tensor, dispatch_index):
        r"""
            Implementing dispatch operation.
            Inputs:
                - **input_tensor** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group,
                hidden\_size)`.(dp, N, h),
                - **dispatch_index** (Tensor) - Tensor of shape :math:`(data\_parallel, expert\_num,
                expert\_capacity)`.(dp, E, n).

            Outputs:
                - **expert_input** (Tensor) - Tensor of shape :math:`(data\_parallel, expert\_num,
                expert\_capacity, hidden\_size)`.(dp, E, n, h).
        """
        input_tensor_padded = self.concat_3d(
            (self.cast(self.zeros_3d, F.dtype(input_tensor)), input_tensor))  # (dp, 1+N, h) <-- (dp, N, h)
        # (dp, E, n, h) <-- (dp, N, h), (dp, E, n)
        expert_input = self.dispatch_gather(input_tensor_padded, dispatch_index, 1)
        return expert_input

    def combine(self, expert_output, combine_index, router_coeff):
        r"""
            Implementing combine operation.
            Inputs:
                - **expert_output** (Tensor) - Tensor of shape :math:`(data\_parallel, expert\_num,
                expert\_capacity, hidden\_size)`.(dp, E, n, h),
                - **combine_index** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group,
                num\_experts\_chosen)`.(dp, N, k),
                - **router_coeff** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group,
                num\_experts\_chosen)`.(dp, N, k).

            Outputs:
                - **output_tensor** (Tensor) - Tensor of shape :math:`(data\_parallel, tokens\_per\_group,
                hidden\_size)`.(dp, N, h).
        """
        expert_output = self.concat(
            (self.cast(self.zeros, F.dtype(expert_output)), expert_output))  # (dp, E, 1+n, h) <-- (dp, E, n, h)
        expert_output = self.reshape(
            expert_output, (expert_output.shape[0],
                            expert_output.shape[1] * expert_output.shape[2],
                            expert_output.shape[3]))  # (dp, E*(1+n), h) <-- (dp, E, 1+n, h)
        output_tensor = self.combine_gather(
            expert_output, combine_index, 1)  # (dp, N, k, h) <-- (dp, E*(1+n), h), (dp, N, k)
        router_coeff = self.cast(router_coeff, F.dtype(expert_output))
        # (dp, N, k, h) <-- (dp, N, k, h) (dp, N, k, 1)
        output_tensor = self.mul_router_coeff(
            output_tensor,
            self.reshape(router_coeff, (router_coeff.shape[0], router_coeff.shape[1], router_coeff.shape[2], 1)))
        output_tensor = self.sum_router_coeff(output_tensor, 2)  # reduce sum # (dp, N, h) <-- (dp, N, k, h)
        return output_tensor

    def construct(self, router_logits):
        """
        Calculate dispatch_policy, combine_policy, router_coeff.
        """
        z_loss = self.z_loss_func(router_logits, self.z_loss_coeff)
        extra_loss = z_loss

        router_prob = self.softmax(router_logits)  # (dp, N, expert_dim)fp32 <-- (dp, N, expert_dim)fp32
        if self.moe_config.topk_method == "group_limited_greedy":
            # (dp, N, n_group)fp32 <-- (dp, N, expert_dim)fp32
            group_scores = self.reshape(
                router_prob, (router_prob.shape[0], router_prob.shape[1], self.moe_config.n_group, -1)).max(axis=-1)
            top_values, _ = self.topk(group_scores, self.moe_config.topk_group)  # (dp, N, top_k)
            group_mask = self.cast(
                group_scores.ge(self.tile(top_values.min(axis=-1, keepdims=True),
                                          (1, 1, self.moe_config.n_group))), mstype.float32)
            score_mask = self.reshape(
                group_mask, (group_mask.shape[0], group_mask.shape[1], group_mask.shape[2], 1))  # (dp, N,  n_group, 1)
            score_mask = score_mask.repeat(self.moe_config.expert_num // self.moe_config.n_group, axis=-1)
            score_mask = self.reshape(
                score_mask, (score_mask.shape[0], score_mask.shape[1], -1))  # (dp, N, n_routed_experts)
            tmp_scores = ops.masked_fill(router_prob, ~score_mask.bool(), 0.0)
            expert_gate, expert_index = self.topk(tmp_scores, self.num_experts_chosen)
        else:
            # in default, normal topk will be used
            expert_gate, expert_index = self.topk(router_prob, self.num_experts_chosen)

        if self.aux_loss_config.get("expert", 0):
            expert_load_loss = self._expert_load_balancing(router_prob, expert_index,
                                                           self.aux_loss_config.get("expert"))
            extra_loss = self.add_scalar(extra_loss, expert_load_loss)
        if self.aux_loss_config.get("device", 0):
            device_load_loss = self._device_load_balancing(router_prob, expert_index,
                                                           self.aux_loss_config.get("device"))
            extra_loss = self.add_scalar(extra_loss, device_load_loss)
        if self.aux_loss_config.get("comm", 0):
            comm_load_loss = self._comm_load_balancing(router_prob, expert_index,
                                                       self.aux_loss_config.get("comm"))
            extra_loss = self.add_scalar(extra_loss, comm_load_loss)

        if self.moe_config.enable_sdrop:
            if self.moe_config.use_fused_ops_topkrouter:
                # (dp, E, n)int32, (dp, N, k), (dp, N, k) <-- (dp, N, k), (dp, N, k)
                dispatch_idx, combine_idx, router_coeff = self._maskout_overflowed_tokens_use_topkrouter(expert_index,
                                                                                                         expert_gate)
            else:
                # (dp, E, n)int32, (dp, N, k), (dp, N, k) <-- (dp, N, k), (dp, N, k)
                dispatch_idx, combine_idx, router_coeff = self._maskout_overflowed_tokens_sort_sdrop(expert_index,
                                                                                                     expert_gate)
        else:
            if self.moe_config.use_fused_ops_topkrouter:
                # (dp, E, n)int32, (dp, N, k), (dp, N, k) <-- (dp, N, k), (dp, N, k)
                dispatch_idx, combine_idx, router_coeff = self._maskout_overflowed_tokens_use_topkrouter(expert_index,
                                                                                                         expert_gate)
            else:
                # (dp, E, n)int32, (dp, N, k), (dp, N, k) <-- (dp, N, k), (dp, N, k)
                dispatch_idx, combine_idx, router_coeff = self._maskout_overflowed_tokens_sort_kdrop(expert_index,
                                                                                                     expert_gate)
        return dispatch_idx, combine_idx, router_coeff, extra_loss  # (dp, E, n)int32, (dp, N, k), (dp, N, k)

    def _maskout_overflowed_tokens_sort_kdrop(self, expert_index, expert_gate):
        """
        Keeping only the tokens that fit within expert_capacity.
        # if tokens_per_group>10: self.print("range_kn", range_kn)
        """
        k = self.num_experts_chosen
        tokens_per_group = self.shape(expert_index)[1]
        kn = k * tokens_per_group  # this n refers to N
        if self.capacity_factor > 0:
            expert_capacity = calculate_expert_capacity_v2(self.num_experts_chosen, tokens_per_group,
                                                           self.capacity_factor, self.expert_dim, self.mp)

        else:
            expert_capacity = self._calculate_expert_capacity_dynamic(expert_index)
        # calculate combine_index from cumsum
        expert_index = self.reshape(self.transpose_3d(expert_index, (0, 2, 1)),
                                    (expert_index.shape[0], -1))  # (dp, kN) <-- (dp, N, k) account for topk priority
        expert_mask = self.onehot_2d(expert_index,
                                     self.expert_dim,
                                     self.on_value,
                                     self.off_value)  # (dp, kN, E)fp32 <-- (dp, kN)int32
        position_in_expert = self.mul_3d(
            self.cumsum(expert_mask, 1), expert_mask)  # (dp, kN, E)fp16 <-- (dp, kN, E)fp32, (dp, kN, E)fp32

        # (dp, kN, E)fp32 <-- (dp, kN, E)fp32, (dp, kN, E)bool, where 0<=position_in_expert<(1+n)
        position_in_expert = self.mul_3d(position_in_expert, self.less(position_in_expert, expert_capacity + 1))
        position_in_expert_2d = self.reduce_sum(position_in_expert, -1)  # (dp, kN)fp32 <-- (dp, kN, E)fp32
        # (dp, kN)fp32 <-- (dp, kN)fp32, (dp, kN)fp32 where 0<= combine_index <E*(1+n),
        # combine_index = expert_id *(1+n) + position_in_expert_2d
        combine_index = self.add_2d(
            self.mul_2d_1d(expert_index, expert_capacity + 1),
            position_in_expert_2d)
        combine_index = self.transpose_3d(
            self.reshape(combine_index, (combine_index.shape[0], k, tokens_per_group)),
            (0, 2, 1))  # (dp, N, k) <-- (dp, kN) account for topk priority
        within_capacity = self.cast(self.gt(position_in_expert_2d, 0), mstype.float32)  # (dp, kN)bool

        # calculate dispatch_index from position_in_expert_onehot
        safe_kn = 2 * kn  # factor=2 for safety
        range_kn = self.slice_range(self.range2, (0, 0),
                                    (self.expert_dim, kn),
                                    (1, 1)).reshape(1, self.expert_dim, kn)  # (1, E, kN) fp32 <-- (E, 131072)
        select = self.transpose_3d(expert_mask, (0, 2, 1))  # (dp, E, kN) fp32 <-- (dp, kN, E) fp32
        dispatch_index_raw = self.add_3d(
            self.mul_range(select, range_kn),
            self.mul_range(self.sub_range(1, select),
                           self.add_range(range_kn, safe_kn)))  # (dp, E, kN) <-- (dp, E, kN) fp32
        dispatch_index, _ = self.sort_range(dispatch_index_raw)  # (dp, E, k) <-- (dp, E, kN）
        dispatch_index = self.slice(dispatch_index, (0, 0, 0),
                                    (dispatch_index.shape[0], dispatch_index.shape[1], expert_capacity),
                                    (1, 1, 1))  # (dp, E, n) <-- (dp, E, kN) fp32
        is_safe = self.less(dispatch_index, safe_kn)  # (dp, E, n) bool
        dispatch_index = self.add_one(self.mod(dispatch_index, tokens_per_group), 1)  # (dp, E, n) fp32
        dispatch_index = self.mul_3d(dispatch_index, is_safe)  # (dp, E, n) fp32

        dispatch_index = self.cast(dispatch_index, mstype.int32)
        combine_index = self.cast(combine_index, mstype.int32)
        router_coeff_raw = self.mul_3d(
            expert_gate,
            self.transpose_3d(
                self.reshape(within_capacity, (within_capacity.shape[0], k, tokens_per_group)),
                (0, 2, 1)))  # apply within_capacity (dp, N, k) <-- (dp, N, k), (dp, N, k) <--  (dp, kN)
        if self.num_experts_chosen > 1 and self.moe_config.norm_topk_prob:
            router_coeff = self._normalize(router_coeff_raw)  # (dp, N, k) <-- (dp, N, k)
        else:
            router_coeff = ops.mul(self.moe_config.routed_scaling_factor, router_coeff_raw)  # (dp, N, k) <-- (dp, N, k)
        return dispatch_index, combine_index, router_coeff  # (dp, E, n), (dp, N, k), (dp, N, k)

    def _maskout_overflowed_tokens_sort_sdrop(self, expert_index, expert_gate):
        """
        Keeping only the tokens that fit within expert_capacity.
        # if tokens_per_group>10: self.print("range_kn", range_kn)
        """
        k = self.num_experts_chosen
        tokens_per_group = self.shape(expert_index)[1]
        kn = k * tokens_per_group  # this n refers to N
        if self.capacity_factor > 0:
            expert_capacity = calculate_expert_capacity_v2(self.num_experts_chosen, tokens_per_group,
                                                           self.capacity_factor, self.expert_dim, self.mp)

        else:
            expert_capacity = self._calculate_expert_capacity_dynamic(expert_index)
        # calculate combine_index from cumsum
        expert_index = self.reshape(expert_index,
                                    (expert_index.shape[0], -1))  # (dp, Nk) <-- (dp, N, k) account for topk priority
        expert_mask = self.onehot_2d(expert_index,
                                     self.expert_dim,
                                     self.on_value,
                                     self.off_value)  # (dp, Nk, E)fp32 <-- (dp, Nk)int32
        position_in_expert = self.mul_3d(self.cumsum(expert_mask, 1),
                                         expert_mask)  # (dp, Nk, E)fp16 <-- (dp, Nk, E)fp32, (dp, Nk, E)fp32
        # (dp, Nk, E)fp32 <-- (dp, Nk, E)fp32, (dp, Nk, E)bool, where 0<=position_in_expert<(1+n)
        position_in_expert = self.mul_3d(position_in_expert, self.less(position_in_expert, expert_capacity + 1))
        position_in_expert_2d = self.reduce_sum(position_in_expert, -1)  # (dp, Nk)fp32 <-- (dp, Nk, E)fp32
        # (dp, Nk)fp32 <-- (dp, Nk)fp32, (dp, Nk)fp32 where 0<= combine_index <E*(1+n),
        # combine_index = expert_id *(1+n) + position_in_expert_2d
        combine_index = self.add_2d(self.mul_2d_1d(expert_index, expert_capacity + 1), position_in_expert_2d)
        combine_index = self.reshape(
            combine_index,
            (combine_index.shape[0], tokens_per_group, k))  # (dp, N, k) <-- (dp, Nk) account for topk priority
        within_capacity = self.cast(self.gt(position_in_expert_2d, 0), mstype.float32)  # (dp, Nk)bool

        # calculate dispatch_index from position_in_expert_onehot
        safe_kn = 2 * kn  # factor=2 for safety
        range_kn = self.slice_range(self.range2,
                                    (0, 0),
                                    (self.expert_dim, kn),
                                    (1, 1)).reshape(1, self.expert_dim, kn)  # (1, E, kN) fp32 <-- (E, 131072)
        select = self.transpose_3d(expert_mask, (0, 2, 1))  # (dp, E, Nk) fp32 <-- (dp, Nk, E) fp32
        dispatch_index_raw = self.add_3d(
            self.mul_range(select, range_kn),
            self.mul_range(
                self.sub_range(1, select),
                self.add_range(range_kn, safe_kn)))  # (dp, E, kN) <-- (dp, E, kN) fp32
        dispatch_index, _ = self.sort_range(dispatch_index_raw)  # (dp, E, k) <-- (dp, E, kN）
        dispatch_index = self.slice(dispatch_index,
                                    (0, 0, 0),
                                    (dispatch_index.shape[0], dispatch_index.shape[1], expert_capacity),
                                    (1, 1, 1))  # (dp, E, n) <-- (dp, E, kN) fp32
        is_safe = self.less(dispatch_index, safe_kn)  # (dp, E, n) bool
        dispatch_index = self.add_one(ops.floor_divide(dispatch_index, k), 1)  # (dp, E, n) fp32
        dispatch_index = self.mul_3d(dispatch_index, is_safe)  # (dp, E, n) fp32

        dispatch_index = self.cast(dispatch_index, mstype.int32)
        combine_index = self.cast(combine_index, mstype.int32)
        within_capacity = self.reshape(within_capacity, (within_capacity.shape[0], tokens_per_group, k))
        router_coeff_raw = self.mul_3d(expert_gate, within_capacity)
        if self.num_experts_chosen > 1 and self.moe_config.norm_topk_prob:
            router_coeff = self._normalize(router_coeff_raw)  # (dp, N, k) <-- (dp, N, k)
        else:
            router_coeff = ops.mul(self.moe_config.routed_scaling_factor, router_coeff_raw)  # (dp, N, k) <-- (dp, N, k)
        return dispatch_index, combine_index, router_coeff  # (dp, E, n), (dp, N, k), (dp, N, k)

    def _maskout_overflowed_tokens_use_topkrouter(self, expert_index, expert_gate):
        """
        Using TopKRouter calculate dispatch_policy and combine_policy.
        """
        if self.capacity_factor > 0:
            tokens_per_group = self.shape(expert_index)[1]
            expert_capacity = calculate_expert_capacity_v2(self.num_experts_chosen, tokens_per_group,
                                                           self.capacity_factor, self.expert_dim, self.mp)
        else:
            expert_capacity = self._calculate_expert_capacity_dynamic(expert_index)
        if self.moe_config.enable_sdrop:
            dispatch_index, combine_index = self.topkrouter(expert_index, expert_capacity, self.expert_dim)
        else:
            dispatch_index, combine_index = self.topkrouter(expert_index, expert_capacity, self.expert_dim, 1)
        within_capacity = self.mod(combine_index, expert_capacity + 1)
        within_capacity = self.not_equal(self.cast(within_capacity, mstype.int32), 0)
        expert_gate = self.mul_3d(within_capacity, expert_gate)
        if self.num_experts_chosen > 1 and self.moe_config.norm_topk_prob:
            router_coeff = self._normalize(expert_gate)
        else:
            router_coeff = ops.mul(self.moe_config.routed_scaling_factor, expert_gate)
        return dispatch_index, combine_index, router_coeff

    def _normalize(self, router_coeff_raw):
        router_coeff_sum = self.reduce_sum_keep(router_coeff_raw, 2)  # (dp, N, 1) <-- (dp, N, k)
        router_coeff = self.div_3d(router_coeff_raw,
                                   self.add_eps(router_coeff_sum, 1e-9))  # (dp, N, k) <-- (dp, N, k) (dp, N, 1)
        return router_coeff  # (dp, N, k)

    def _calculate_expert_capacity_dynamic(self, expert_index):
        """
        Calculate dynamic capacity.
        """
        expert_index = self.reshape(expert_index, (self.dp_group, -1))
        expert_mask = self.onehot_2d(expert_index,
                                     self.expert_dim,
                                     self.on_value_int,
                                     self.off_value_int)  # (dp, kN, E) <- (dp, kN)
        expert_mask = self.reduce_sum(expert_mask, 1)  # (dp, E) <- (dp, kN, E)
        expert_capacity = expert_mask.max()  # (1, ) <- (dp, E)
        expert_capacity = self.cast(expert_capacity, mstype.int64)
        expert_capacity_scalar = self.tensor2scalar(expert_capacity)
        expert_capacity_scalar = (expert_capacity_scalar // self.mp + 1) * self.mp
        return expert_capacity_scalar

    def _expert_load_balancing(self, scores, top_indices, alpha):
        """Expert level load balance loss, which regularizes the load from local batch data on each
        expert to be balanced.
        Please refer to DeepSeek-V2:
        A Strong, Economical, and Efficient Mixture-of-Experts Language Model, https://arxiv.org/abs/2405.04434
        """
        pi = self.reduce_mean(scores, 1)  # (dp, E)<- (dp, N, E), 1/N * \sum_t^T s_{i,t}

        top_indices = self.reshape(self.transpose_3d(top_indices, (0, 2, 1)),
                                   (top_indices.shape[0], -1))  # (dp, kN) <-- (dp, N, k) account for topk priority
        mask = self.onehot_2d(top_indices,
                              self.expert_dim,
                              self.on_value,
                              self.off_value)  # (dp, kN, E)fp32 <-- (dp, kN)int32
        fi = self.reduce_mean(mask, 1)  # (dp, E) <- (dp, kN, E), 1/(kN) * \sum_t^T 1(token t selects expert i)

        expert_load_loss = self.mul(self.reduce_mean_2d(self.mul_2d(pi, fi)),
                                    alpha * self.expert_dim ** 2)  # alpha*E \sum_i^E (f_i * P_i)
        return expert_load_loss

    def _device_load_balancing(self, scores, top_indices, alpha):
        """
        Device level load balance loss, which regularizes the load from local batch data on each
        expert group to be balanced.
        Please refer to DeepSeek-V2:
        A Strong, Economical, and Efficient Mixture-of-Experts Language Model, https://arxiv.org/abs/2405.04434
        """
        scores = self.reshape(self.transpose_3d(scores, (0, 2, 1)),
                              (scores.shape[0], self.egroup_size, -1))  # (dp, egs, E/egs*N) <- (dp, N, E)
        pi = self.reduce_mean(
            scores, -1)  # (dp, egs) <- (dp, egs, E/egs*N), 1/(E/egs*N) * \sum_{j in E'} \sum_t^N s_{j, t}

        top_indices = self.reshape(self.transpose_3d(top_indices, (0, 2, 1)),
                                   (top_indices.shape[0], -1))  # (dp, kN) <- (dp, N, k)
        mask = self.onehot_2d(top_indices,
                              self.expert_dim,
                              self.on_value,
                              self.off_value)  # (dp, kN, E)fp32 <- (dp, kN)int32
        mask = self.reshape(
            self.transpose_3d(mask, (0, 2, 1)),
            (mask.shape[0], self.egroup_size, -1))  # (dp, egs, E/egs*kN) <- (dp, kN, E)
        # (dp, egs) <- (dp, egs, E/egs*kN), 1/(E/egs*kN) * \sum_{j in E'} \sum_t^N 1(token t selects expert j)
        fi = self.reduce_mean(mask, -1)

        device_load_loss = self.mul(
            self.reduce_mean_2d(
                self.mul_2d(pi, fi)), alpha * self.expert_dim ** 2)  # \alpha * (E/ep)*E \sum_i^{ep} (fi * Pi)

        return device_load_loss

    def _comm_load_balancing(self, scores, top_indices, alpha):
        """
        Communication level load balance loss, which regularizes the load from local batch data on each
        device to be balanced.
        Please refer to DeepSeek-V2:
        A Strong, Economical, and Efficient Mixture-of-Experts Language Model, https://arxiv.org/abs/2405.04434
        """
        scores = self.reshape(
            self.transpose_3d(scores, (0, 2, 1)), (scores.shape[0], self.ep, -1))  # (dp, ep, E/ep*N) <- (dp, N, E)
        pi = self.reduce_mean(scores, -1)  # (dp, ep) <- (dp, ep, E/ep*N), 1/(E/ep*N) * \sum_{j in E'} \sum_t^N s_{j, t}

        top_indices = self.reshape(
            self.transpose_3d(top_indices, (0, 2, 1)), (top_indices.shape[0], -1))  # (dp, kN) <- (dp, N, k)
        mask = self.onehot_2d(top_indices,
                              self.expert_dim,
                              self.on_value,
                              self.off_value)  # (dp, kN, E)fp32 <- (dp, Nk)int32
        mask = self.reshape(
            self.transpose_3d(mask, (0, 2, 1)), (mask.shape[0], self.ep, -1))  # (dp, ep, E/ep*kN) <- (dp, kN, E)
        # (dp, ep) <- (dp, ep, E/dp*kN) ,  1/(E/ep) * \sum_{j in E'} \sum_{t}^N 1(token t sent to device j)
        fi = self.reduce_mean(mask, -1)
        # \alpha * (E/ep)**2 * ep / dp / N \sum_i^{ep} (fi * Pi)
        comm_load_loss = self.mul(self.reduce_mean_2d(self.mul_2d(pi, fi)), alpha * self.expert_dim ** 2)

        return comm_load_loss


class MoEInfer(Cell):
    r"""
        MoEInfer. Routing each tokens to the topk expert and calculating the final output.

        Args:
            ffn (Cell): The FeedForward Module.
            dim (int): The hidden size of each token.
            moe_config (MoEConfig): The configuration of MoE (Mixture of Expert).
            parallel_config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].

        Outputs:
            - **output_tensor** (Tensor) - should be `[batch, seq_length, hidden_size].
    """

    def __init__(self,
                 ffn,
                 dim,
                 moe_config,
                 parallel_config):
        super(MoEInfer, self).__init__()
        self.hidden_size = dim
        self.expert_dim = moe_config.expert_num
        self.topk_norm_prob = moe_config.norm_topk_prob
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.moe_config = moe_config

        self.ffn = ffn
        self.router = Router(d_model=self.hidden_size, moe_config=moe_config, routing_policy=None,
                             training=True, parallel_config=parallel_config)
        self.gating = self.router.dense
        self.noise = 1e-9

        self.zeros = ops.Zeros()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.mod = P.Mod().shard(((1,), ()))
        self.topk = P.TopK().shard(((1, 1),))
        self.softmax = P.Softmax().shard(((1, 1),))
        self.expand_dims = P.ExpandDims().shard(((1,),))
        self.transpose_2d = P.Transpose().shard(((1, 1),))
        self.sort = P.Sort().shard(((1,),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.onehot = P.OneHot().shard(((1, 1), (), ()))
        self.cumsum = P.CumSum(exclusive=False).shard(((1,),))

        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)
        if check_valid_moefinalizerouting_op():
            from mindspore.ops.auto_generate import MoeFinalizeRouting
            self.moe_finalize_routing = MoeFinalizeRouting().shard(((1, 1), (1, 1), (1, 1), (1, 1), (1,), (1, 1)))

    def tensor_sort(self, input_tensor, expert_ids):
        """dispatch and get unsort map for routing"""
        expert_shape = expert_ids.shape
        transposed_index = self.transpose_2d(expert_ids, (1, 0))  # (N, k) -> (k, N)
        reshaped_index = self.reshape(transposed_index, (-1,))  # (k, N) -> (kN)
        _, sort_map = self.sort(reshaped_index)

        inter_map = self.mod(sort_map, expert_shape[0])
        output_tensor = self.gather(input_tensor, inter_map, 0)
        expert_mask = self.onehot(reshaped_index, self.expert_dim, self.on_value, self.off_value)
        expert_cnt = ops.sum(expert_mask, 0)
        group_list = self.cast(self.cumsum(expert_cnt, 0), mstype.int64)

        _, unsort_map = self.sort(sort_map)
        unsort_map = self.cast(unsort_map, mstype.int32)
        return output_tensor, group_list, unsort_map

    def tensor_moe_finalize_routing(self, input_tensor, expert_weight, expert_index, unsort_map):
        """calculate the final output by multiplying FeedForward's output and experts' weight in MoeFinalizeRouting"""
        input_shape = input_tensor.shape  # (2N, h)
        x1 = self.zeros((input_shape[0] // self.num_experts_chosen, input_shape[-1]), input_tensor.dtype)
        x2 = None
        bias = self.zeros((self.expert_dim, input_shape[-1]), input_tensor.dtype)
        expert_weight = self.cast(expert_weight, input_tensor.dtype)
        output_tensor = self.moe_finalize_routing(input_tensor, x1, x2, bias, expert_weight, unsort_map, expert_index)
        return output_tensor

    def construct(self, input_tensor):
        """forward process"""
        input_tensor_shape = self.shape(input_tensor)  # (B, S, H)
        input_dtype = input_tensor.dtype
        input_tensor = self.reshape(
            input_tensor, (-1, self.hidden_size))  # (bs, seq/1, h) -> (bs*seq, h) : use N replace bs*seq

        # gating + topk + softmax
        gating_logits = self.gating(input_tensor.astype(mstype.float32))  # (N, h) * (h, E) -> (bs*seq, E)
        routing_weights = self.softmax(gating_logits.astype(mstype.float32))  # (N, E) -> (N, E)
        expert_val, expert_index = self.topk(routing_weights, self.num_experts_chosen)  # (N, E) -> (N, 2), (N, 2)

        if self.moe_config.norm_topk_prob and self.num_experts_chosen > 1:
            expert_val = self.cast(expert_val, mstype.float32)
            expert_weight = expert_val / (self.expand_dims(ops.sum(expert_val, -1), -1) + 1e-9)
        else:
            expert_weight = ops.mul(self.moe_config.routed_scaling_factor, expert_val)

        expert_weight = self.cast(expert_weight, input_dtype)

        sorted_input_tensor, group_list, unsort_map = self.tensor_sort(input_tensor, expert_index)

        # moeffn
        expert_output = self.ffn(sorted_input_tensor, group_list)  # (N, h) (N, 2) -> (N, 2, h)

        moe_output = self.tensor_moe_finalize_routing(
            expert_output, expert_weight, expert_index, unsort_map)  # -> (N, h)

        output_tensor = self.reshape(moe_output, input_tensor_shape)  # (N, h) -> (bs, seq, h)
        return output_tensor
