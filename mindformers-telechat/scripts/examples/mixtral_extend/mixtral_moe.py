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
"""
Note: Mixture of Expert (MoE) structure. This is an experimental interface that is subject to change or deletion.
"""
from __future__ import absolute_import
from __future__ import division

import copy
import numpy as np
import mindspore.ops as ops

from mindspore.common.tensor import Tensor
from mindspore.ops.operations._sequence_ops import TensorToScalar
import mindspore.common.dtype as mstype

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.modules.transformer.op_parallel_config import default_moeparallel_config
from mindformers.modules.transformer.moe_utils import ZLoss
from mindformers.modules.transformer.moe import calculate_expert_capacity_v2, TopkRouter, MoEConfig
from mindformers.tools.utils import get_predict_run_mode

default_moe_config = MoEConfig()


class MixtralMoEV2(Cell):
    """
    The mixture of experts (MoE) implementation. The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.
    This is a common interface, which allows any ffn class and any MixtralRouter algorithm(implemented in V2 form).

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
        super(MixtralMoEV2, self).__init__()
        self.hidden_size = dim
        self.expert_dim = moe_config.expert_num
        self.return_extra_loss = return_extra_loss
        self.capacity_factor = moe_config.capacity_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.dp_group = parallel_config.data_parallel * parallel_config.context_parallel
        self.dp = parallel_config.data_parallel * parallel_config.context_parallel
        self.ep = parallel_config.expert_parallel
        self.cp = parallel_config.context_parallel
        self.mp = parallel_config.model_parallel
        self.group_wise_a2a = moe_config.group_wise_a2a if self.mp > 1 else False
        self.add_loss = P.Add()
        self.dp_moe = self.dp // self.ep
        self.dp_range = Tensor(np.arange(self.dp_group).reshape(-1, 1), mstype.int32)  # (dp, 1) = [[0],[1],[2]...[dp]]

        self.ffn = ffn
        Validator.check_string(moe_config.routing_policy, ["TopkRouterV2"], "routing_policy")
        self.ffn_forward = self._ffn_forward
        router_parallel_config = copy.deepcopy(parallel_config)
        self.use_seq_parallel = parallel_config.use_seq_parallel
        if self.use_seq_parallel:
            self.ffn_forward = self._ffn_forward_sq
            router_parallel_config.data_parallel = self.dp * self.mp
            router_parallel_config.context_parallel = 1
            router_parallel_config.model_parallel = 1
            self.dp_group *= self.mp
        self.router = MixtralRouter(d_model=self.hidden_size,
                                    moe_config=moe_config,
                                    routing_policy=moe_config.routing_policy,
                                    training=(not get_predict_run_mode()),
                                    parallel_config=router_parallel_config)

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

        self.transpose_mp1 = P.Transpose().shard(((self.dp, self.mp, 1, 1, 1),))
        self.transpose_mp2 = P.Transpose().shard(((self.dp, 1, self.mp, 1, 1),))
        self.stride_slice_ata1 = P.StridedSlice().shard(((self.dp_moe, self.ep, 1, self.mp, 1),))
        self.stride_slice_ata2 = P.StridedSlice().shard(((self.dp_moe, 1, self.ep, self.mp, 1),))
        self.stride_slice_allgather = P.StridedSlice().shard(((self.dp_moe, 1, self.ep, 1, 1),))
        self.transpose_all2all = P.Transpose().shard(((1, self.ep, 1, 1),))

    def _ffn_forward_sq(self, expert_input, capacity):
        """
        use sp Computing the FFN.
        """
        expert_shape = (self.dp_moe, self.ep, self.expert_dim, self.mp * capacity, self.hidden_size)

        # all2all
        expert_input = self.reshape(expert_input, (self.dp, self.mp, self.expert_dim, capacity, self.hidden_size))
        expert_input = self.transpose_mp1(expert_input, (0, 2, 1, 3, 4))  # (dp,mp,E,C/mp,h) -> (dp,E,mp,C/mp,h)
        expert_input = self.reshape(expert_input, expert_shape)
        expert_input = self.stride_slice_ata1(expert_input, (0, 0, 0, 0, 0), expert_shape, (1, 1, 1, 1, 1))
        # (outdp,dp',E,mp,C/mp,h) -> (outdp,dp'*ep,E/ep,mp,C/mp,h)
        expert_input = self.stride_slice_ata2(expert_input, (0, 0, 0, 0, 0), expert_shape, (1, 1, 1, 1, 1))
        expert_input = self.stride_slice_allgather(expert_input, (0, 0, 0, 0, 0), expert_shape, (1, 1, 1, 1, 1))

        # ffns
        expert_input = self.reshape(expert_input, (self.dp, -1, self.hidden_size))
        expert_output = self.ffn(expert_input)
        expert_output = self.reshape(expert_output, expert_shape)

        # all2all
        expert_output = self.stride_slice_ata2(expert_output, (0, 0, 0, 0, 0), expert_shape, (1, 1, 1, 1, 1))
        expert_output = self.stride_slice_ata1(expert_output, (0, 0, 0, 0, 0), expert_shape, (1, 1, 1, 1, 1))
        expert_output = self.reshape(expert_output, (self.dp, self.expert_dim, self.mp, capacity, self.hidden_size))
        expert_output = self.transpose_mp2(expert_output, (0, 2, 1, 3, 4))
        expert_output = self.reshape(expert_output, (self.dp * self.mp, self.expert_dim, -1, self.hidden_size))
        return expert_output


    def _ffn_forward(self, expert_input, capacity):
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


class MixtralRouter(Cell):
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
        super(MixtralRouter, self).__init__()
        dp = parallel_config.data_parallel * parallel_config.context_parallel
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
            self.router = MixtralTopkRouterV2(d_model=d_model, moe_config=moe_config, training=training,
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


# pylint: disable=C0330
class MixtralTopkRouterV2(Cell):
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
        super(MixtralTopkRouterV2, self).__init__()

        dp = parallel_config.data_parallel * parallel_config.context_parallel
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
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.reduce_mean_2d = P.ReduceMean(keep_dims=False).shard(((dp, 1),))
        self.add_scalar = P.Add()
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
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
            scores, m - 1)  # (dp, egs) <- (dp, egs, E/egs*N), 1/(E/egs*N) * \sum_{j in E'} \sum_t^N s_{j, t}

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
