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
"""LLaMA Model Layers' APIs."""

import copy
import mindspore as ms
from mindspore.common.parameter import Parameter
from mindspore import nn, Layout
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.common.initializer import initializer, Normal
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindformers.version_control import check_valid_big_kernel
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, \
    _valid_value_checks
from mindformers.tools.logger import _LogActionOnce
from mindformers.version_control import check_rmsnorm_big_kernel_valid
from mindformers.modules.transformer.moe import MoEV2, MoEInfer
from mindformers.modules.transformer.moev3 import MoEV3
from mindformers.tools.utils import get_predict_run_mode


class LlamaSiLU(Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = silu(x).
    """

    def __init__(self):
        super().__init__()
        if check_valid_big_kernel():
            # pylint: disable=W0212
            self.silu = P._inner_ops.SiLU()
            self.self_define = False
        else:
            self.sigmoid = P.Sigmoid()
            self.mul = P.Mul()
            self.silu = self._self_silu
            self.self_define = True

    def _self_silu(self, x):
        return self.mul(x, self.sigmoid(x))

    def construct(self, x):
        return self.silu(x)

    def shard(self, strategy):
        if self.self_define:
            self.sigmoid.shard(strategy)
            self.mul.shard((strategy[0], strategy[0]))
        else:
            self.silu.shard(strategy)

    def activation_shard(self, strategy, use_gmm=False):
        # activation_shard is the api called by moe [dp_group, expert_dim, capacity, ffn_hidden]
        if hasattr(strategy, "expert_parallel"):
            if use_gmm:
                moe_strategy = ((strategy.data_parallel, strategy.model_parallel),)
            else:
                moe_strategy = ((strategy.data_parallel, strategy.expert_parallel, 1, strategy.model_parallel),)
            self.shard(moe_strategy)


class LlamaEmbedding(Cell):
    """
    Embedding Layer.

    Args:
            - **vocab_size** (int): Size of the dictionary of embeddings.
            - **embedding_size** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **parallel_config** (TransformerOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.
            - **init_method_std** (float): The sigma value when using normal type to initialize Linear. Default `0.01`
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    @_LogActionOnce(m_logger=logger, key='Embedding',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(vocab_table_size=Validator.check_positive_int,
                                embedding_size=Validator.check_positive_int)
    def __init__(self, vocab_table_size, embedding_size, init_method_std=0.01, param_init_type=mstype.float32,
                 param_init='normal', parallel_optimizer=False, rmsnorm_compute_2d=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        if param_init == "normal":
            param_init = Normal(sigma=init_method_std, mean=0.0)
            logger.info(f"Embedding use init method: sigma={init_method_std}, mean=0.0")
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        self.gather = P.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        if self.rmsnorm_compute_2d:
            input_ids = input_ids.reshape(-1)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output

    def shard(self, parallel_config):
        """sharding for embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if parallel_config.vocab_emb_dp:
            if not self.rmsnorm_compute_2d:
                self.gather.shard(((1, 1), (dp, cp)))
                logger.info(f"Using {dp*cp} data parallel for the embedding lookup.")
            else:
                self.gather.shard(((1, 1), (dp * cp,)))
                logger.info(f"Using {dp * cp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % (mp * cp) != 0:
                logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s"
                               "model_parallel: %s * context_parallel: %s.",
                               self.vocab_table_size, mp, cp)
                logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
                if not self.rmsnorm_compute_2d:
                    self.gather.shard(((1, 1), (dp, cp)))
                else:
                    self.gather.shard(((1, 1), (dp * cp,)))
            else:
                if not self.rmsnorm_compute_2d:
                    self.gather.shard(((mp * cp, 1), (dp, 1)))
                    logger.info(f"Using {dp} data parallel, {cp} context parallel and {mp} "
                                f"model parallel for the embedding lookup.")
                else:
                    self.gather.shard(((1, 1), (dp,)))
                    logger.info(f"Using {dp} data parallel for the embedding lookup.")


class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
            fused_kernel (bool): whether to use fused kernel. Default True.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32, fused_kernel=True):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.compute_type), parallel_optimizer=False)

        if fused_kernel and check_rmsnorm_big_kernel_valid():
            self.norm = P.RmsNorm(eps)
            self.rms_norm = self._rms_norm
            self.self_define = False
            self.cast = P.Cast()
            self.rcast = P.Cast()
            is_predict_mode = get_predict_run_mode()
            if not is_predict_mode:
                self.cast.recompute()
        else:
            self.cast = P.Cast()
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.square = P.Square()
            self.mean = P.ReduceMean(keep_dims=True)
            self.add = P.Add()
            self.rsqrt = P.Rsqrt()
            self.rms_norm = self._self_norm
            self.self_define = True

    def _self_norm(self, x):
        original_type = x.dtype
        h = self.cast(x, self.compute_type)
        norm_factor = self.square(h)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(h, norm_factor)
        output = self.mul2(self.cast(output, original_type), self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)

    def shard(self, strategy_in):
        """Parallel strategy configuratiuon interface."""
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            if not self.self_define:
                self.norm.shard((strategy_in, (1,)))
        else:
            if self.self_define:
                self.square.shard((strategy_in,))
                self.mean.shard((strategy_in,))
                self.rsqrt.shard((strategy_in,))
                self.add.shard((strategy_in, ()))
                self.mul.shard((strategy_in, strategy_in))
                self.mul2.shard((strategy_in, (1,)))
            else:
                self.norm.shard((strategy_in, (1,)))

    def shard_layout(self, strategy_in: Layout, strategy_gamma: Layout):
        """Parallel layout configuratiuon interface."""
        if self.self_define:
            raise ValueError("Layout shard for self_define rmsnorm is not support yet.")
        self.norm.shard((strategy_in, strategy_gamma))


class LlamaFeedForward(Cell):
    r"""
    LLaMA FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 intermediate_size=None,
                 hidden_dim=None,
                 expert_num=1,
                 multiple_of=256,
                 hidden_act=LlamaSiLU,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 ffn_concat=False,
                 parallel_config=default_dpmp_config,
                 moe_config=None,
                 init_method_std=0.01,
                 rmsnorm_compute_2d=False,
                 use_3d_tensor_parallel=False,
                 tp_x=1,
                 tp_y=1,
                 tp_z=1,
                 use_fused_swiglu=False):
        super().__init__()

        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")

        self.use_fused_swiglu = use_fused_swiglu
        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            if ffn_dim_multiplier is not None:
                hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * \
                         ((hidden_dim + multiple_of - 1) // multiple_of)
        if moe_config is not None:
            self.use_allgather_dispatcher = moe_config.use_allgather_dispatcher
            self.mp_moe_flag = moe_config.expert_model_parallel is not None
            self.mp_moe = moe_config.expert_model_parallel if moe_config.expert_model_parallel is not None \
                else parallel_config.model_parallel
        else:
            self.use_allgather_dispatcher = False
            self.mp_moe_flag = False

        if expert_num > 1:
            cp = parallel_config.context_parallel
            ep = parallel_config.expert_parallel
            mp = parallel_config.model_parallel
            if self.use_allgather_dispatcher:
                dp_moe = parallel_config.data_parallel * cp
            else:
                dp_moe = parallel_config.data_parallel * cp // ep
            if self.mp_moe_flag:
                dp_moe = parallel_config.data_parallel * cp * mp // (ep * self.mp_moe)
                mp = self.mp_moe
        else:
            dp_moe = 1
        self.dtype = compute_dtype
        self.hidden_act = None if self.use_fused_swiglu else hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.ffn_concat = ffn_concat
        self.use_3d_tensor_parallel = use_3d_tensor_parallel
        self.tp_x = tp_x
        self.tp_y = tp_y
        self.tp_z = tp_z

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        if self.ffn_concat:
            self.w_gate_hidden = Linear(in_channels=dim,
                                        out_channels=hidden_dim * 2,
                                        init_method_std=init_method_std,
                                        expert_num=expert_num,
                                        outer_batch=dp_moe,
                                        has_bias=False,
                                        compute_dtype=compute_dtype,
                                        param_init_type=param_init_type)
            self.activate = self.hidden_act()
            self.split = ms.ops.auto_generate.SplitWithSize()
            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             init_method_std=init_method_std,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        else:
            self.w1 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             init_method_std=init_method_std,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             activation=self.hidden_act,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             init_method_std=init_method_std,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

            self.w3 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             init_method_std=init_method_std,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.w13_concat = P.Concat(-1)
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        if self.use_fused_swiglu:
            self.swiglu = ms.ops.auto_generate.gen_ops_prim.Swiglu()
            self.expand_dims = P.ExpandDims()
            self.squeeze = P.Squeeze(axis=-1)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)
        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp
            bs, seq_len, _ = gate_hidden_out.shape
            if not self.use_fused_swiglu:
                reshape_out = self.reshape(gate_hidden_out, (bs, seq_len, self.hidden_dim, 2))
                gate, hidden = self.split(reshape_out, (1, 1), 3)
                gate = self.reshape(gate, (bs, seq_len, self.hidden_dim))
                hidden = self.reshape(hidden, (bs, seq_len, self.hidden_dim))
                gate = self.activate(gate)
        else:
            # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
            gate = self.w1(x)  # dp,1 -> dp, mp
            hidden = self.w3(x)  # dp,1 -> dp, mp
            if self.use_fused_swiglu:
                gate = self.expand_dims(gate, -1)
                hidden = self.expand_dims(hidden, -1)
            gate_hidden_out = self.w13_concat((gate, hidden))
        if self.use_fused_swiglu:
            hidden = self.swiglu(gate_hidden_out, -1)
            hidden = self.squeeze(hidden)
        else:
            hidden = self.mul(hidden, gate)  # dp,mp -> dp, mp

        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        if self.use_3d_tensor_parallel:
            self._shard_ndtp(parallel_config)
            return
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        ep = parallel_config.expert_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            if self.expert_num == 1:
                if self.ffn_concat:
                    if not self.rmsnorm_compute_2d:
                        self.w_gate_hidden.shard(((dp, 1), (mp, 1)))
                        self.activate.shard(((dp, 1, mp),))
                        self.w2.shard(((dp, mp), (1, mp)))
                        self.split.add_prim_attr("skip_redistribution", True)
                        self.split.shard(((dp, 1, mp, 1),))
                    else:
                        self.w_gate_hidden.shard(((dp, 1), (mp, 1)))
                        self.activate.shard(((dp, mp),))
                        self.w2.shard(((dp, mp), (1, mp)))
                        self.split.shard(((dp, 1),))
                        self.mul.shard(((dp, mp), (dp, mp)))
                else:
                    self.w1.shard(((dp * cp, 1), (mp, 1)), strategy_activation=((dp * cp, mp),))
                    if not self.use_fused_swiglu:
                        self.w1.activation.shard(((dp * cp, mp),))
                    self.w2.shard(((dp * cp, mp), (1, mp)))
                    self.w3.shard(((dp * cp, 1), (mp, 1)))
            else:
                logger.info("shard ffn with MoE")
                ep = parallel_config.expert_parallel
                dp = parallel_config.data_parallel * parallel_config.context_parallel // ep
                self.w1.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                              strategy_activation=((dp, ep, mp, 1),))
                self.w2.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)))
                self.w3.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)))
        else:
            if self.expert_num == 1:
                if self.ffn_concat:
                    if not self.rmsnorm_compute_2d:
                        self.w_gate_hidden.shard(((dp, 1), (mp, 1)))
                        self.activate.shard(((dp, 1, mp),))
                        self.w2.shard(((dp, mp), (1, mp)))
                        self.split.shard(((dp, 1, mp, 1),)).add_prim_attr("skip_redistribution", True)
                        self.mul.shard(((dp, mp), (dp, mp)))
                    else:
                        self.w_gate_hidden.shard(((dp, 1), (mp, 1)))
                        self.activate.shard(((dp, mp),))
                        self.w2.shard(((dp, mp), (1, mp)))
                        self.split.shard(((dp, 1),))
                        self.mul.shard(((dp, mp), (dp, mp)))
                else:
                    self.w1.shard(((dp * cp, 1), (mp, 1)), strategy_activation=((dp * cp, mp),))
                    if not self.use_fused_swiglu:
                        self.w1.activation.shard(((dp * cp, mp),))
                    self.w2.shard(((dp * cp, mp), (1, mp)))
                    self.w3.shard(((dp * cp, 1), (mp, 1)))
                    self.mul.shard(((dp, cp, mp), (dp, cp, mp)))
                    self.w13_concat.shard(((dp, cp, mp, 1), (dp, cp, mp, 1)))
                if self.use_fused_swiglu:
                    layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
                    self.swiglu.shard((layout("dp", "cp", "mp", "None"),), (layout("dp", "cp", "mp", "None"),))
                    self.swiglu.add_prim_attr("self_define_shard", True)
                    self.expand_dims.shard(((dp, cp, mp),))
                    self.squeeze.shard(((dp, cp, mp, 1),))
            else:
                logger.info("shard ffn with MoE")
                if self.mp_moe_flag:
                    dp = dp * cp * mp // (ep * self.mp_moe)
                    mp = self.mp_moe
                else:
                    dp = dp * cp // ep
                self.w1.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                              strategy_activation=((dp, ep, mp, 1),))
                self.w2.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)))
                self.w3.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)))
                if self.use_allgather_dispatcher:
                    self.mul.shard(((dp, ep, 1, mp), (dp, ep, 1, mp)))
                else:
                    mul_shard = (dp * ep, mp)
                    if parallel_config.use_seq_parallel:
                        mul_shard = (dp, ep, mp)
                    self.mul.shard((mul_shard, mul_shard))

                self.w13_concat.shard(((dp, ep * cp, mp, 1), (dp, ep * cp, mp, 1)))
                if self.use_fused_swiglu:
                    layout = Layout((dp, ep, cp, mp), ("dp", "ep", "cp", "mp"))
                    self.swiglu.shard((layout("dp", ("ep", "cp"), "mp", "None"),),
                                      (layout("dp", ("ep", "cp"), "mp", "None"),))
                    self.swiglu.add_prim_attr("self_define_shard", True)
                    self.expand_dims.shard(((dp, ep * cp, mp),))
                    self.squeeze.shard(((dp, ep * cp, mp, 1),))

    def _shard_ndtp(self, parallel_config):
        """sharding for feedforward with use_3d_tensor_parallel"""
        dp = parallel_config.data_parallel
        cp = parallel_config.context_parallel

        if not self.use_3d_tensor_parallel:
            raise ValueError("'use_3d_tensor_parallel' must be True when _shard_ndtp.")

        if self.hidden_dim % self.tp_x != 0 or self.hidden_dim % self.tp_y != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of tp_x and tp_y, but got the hidden_dim is {} and the num of tp_x "
                             "is {} and tp_y is {}.".format(self.hidden_dim, self.tp_x, self.tp_y))
        if self.dim % self.tp_x != 0 or self.dim % self.tp_y != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the"
                             "num of tp_x and tp_y, but got the dim is {} and the num of tp_x "
                             "is {} and tp_y is {}.".format(self.dim, self.tp_x, self.tp_y))
        layout_ndtp = Layout((dp, cp, self.tp_z, self.tp_x, self.tp_y), ("dp", "cp", "z", "x", "y"))
        if self.expert_num == 1:
            if self.ffn_concat:
                if not self.rmsnorm_compute_2d:
                    self.w_gate_hidden.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                             enable_nd_tp=True)
                    self.activate.shard((layout_ndtp("dp", ("cp", "z", "x"), "y"),))
                    self.w2.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("x", ("y", "z"))),
                                  enable_nd_tp=True)
                    self.split.add_prim_attr("skip_redistribution", True)
                    self.split.shard(((dp, cp * self.tp_z * self.tp_x, self.tp_y, 1),))
                    self.mul.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp(("dp", "cp", "z", "x"), "y")))
                else:
                    self.w_gate_hidden.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                             enable_nd_tp=True)
                    self.activate.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"),))
                    self.w2.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("x", ("y", "z"))),
                                  enable_nd_tp=True)
                    self.split.shard(((dp * cp * self.tp_z * self.tp_x, 1),))  # y will be allgathered for precision
                    self.mul.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"),
                                    layout_ndtp(("dp", "cp", "z", "x"), "y")))
            else:
                self.w1.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                              enable_nd_tp=True)
                self.w1.activation.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"),))
                self.w2.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("x", ("y", "z"))),
                              enable_nd_tp=True)
                self.w3.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                              enable_nd_tp=True)
                self.mul.shard((layout_ndtp("dp", ("cp", "z", "x"), "y"), layout_ndtp("dp", ("cp", "z", "x"), "y")))


class LlamaMoeInferFeedForward(Cell):
    r"""
    LLaMA FeedForward for MoE Infer implemented with grouped matmul.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 intermediate_size=None,
                 hidden_dim=None,
                 expert_num=1,
                 multiple_of=256,
                 hidden_act=LlamaSiLU,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 use_gmm=True,
                 init_method_std=0.01):
        super().__init__()

        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")

        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            if ffn_dim_multiplier is not None:
                hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * \
                         ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.use_gmm = use_gmm

        self.mul = P.Mul()
        self.cast = P.Cast()

        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         init_method_std=init_method_std,
                         expert_num=expert_num,
                         activation=hidden_act,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         init_method_std=init_method_std,
                         expert_num=expert_num,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         init_method_std=init_method_std,
                         expert_num=expert_num,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

    def construct(self, x, group_list=None):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x, group_list)  # dp,1 -> dp, mp
        hidden = self.w3(x, group_list)  # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate)  # dp,mp -> dp, mp
        output = self.w2(hidden, group_list)  # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for moe infer feedforward"""
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        if self.expert_num == 1:
            raise ValueError("For 'LlamaMoEFFNInfer', the class variable 'expert_num' must be greater than 1.")
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.w1.shard(strategy_matmul=(((1, 1),), ((1, 1, mp),), ((),), ((),), ((),), ((),), ((),), (1,)))
            self.w3.shard(strategy_matmul=(((1, 1),), ((1, 1, mp),), ((),), ((),), ((),), ((),), ((),), (1,)))
            self.w2.shard(strategy_matmul=(((1, mp),), ((1, mp, 1),), ((),), ((),), ((),), ((),), ((),), (1,)))
        else:
            self.mul.shard(((1, mp), (1, mp)))
            self.w1.shard(strategy_matmul=(((1, 1),), ((1, 1, mp),), ((),), ((),), ((),), ((),), ((),), (1,)),
                          strategy_activation=((1, 1, mp, 1),))
            self.w3.shard(strategy_matmul=(((1, 1),), ((1, 1, mp),), ((),), ((),), ((),), ((),), ((),), (1,)))
            self.w2.shard(strategy_matmul=(((1, mp),), ((1, mp, 1),), ((),), ((),), ((),), ((),), ((),), (1,)))



class LlamaFeedForwardWithMoE(Cell):
    r"""
    LLaMA FeedForward with MoE
    """
    def __init__(self, hidden_size,
                 intermediate_size=None,
                 hidden_act=LlamaSiLU,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 moe_config=None,
                 parallel_config=default_dpmp_config,
                 use_moe_infer=True,
                 return_extra_loss=False,
                 use_fused_swiglu=False,
                 init_method_std=0.01
                 ):
        super().__init__()
        self.expert_num = moe_config.expert_num
        self.shared_expert_num = moe_config.shared_expert_num
        self.use_shared_expert_gating = moe_config.use_shared_expert_gating
        self.router_dense_type = moe_config.router_dense_type
        self.compute_dtype = compute_dtype
        self.use_moe_infer = use_moe_infer
        self.return_extra_loss = return_extra_loss
        self.cast = P.Cast()

        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.use_seq_parallel = parallel_config.use_seq_parallel
        self.use_gmm = moe_config.use_gmm
        if self.use_seq_parallel:
            self.dp = parallel_config.data_parallel * parallel_config.model_parallel
        if moe_config.moe_shared_expert_overlap:
            self.add.add_prim_attr("parallel_branch", 1)
        if self.use_moe_infer:
            self.routed_experts = MoEInfer(
                ffn=LlamaMoeInferFeedForward(dim=hidden_size,
                                             intermediate_size=intermediate_size,
                                             hidden_act=hidden_act,
                                             expert_num=self.expert_num,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type,
                                             use_gmm=self.use_moe_infer,
                                             init_method_std=init_method_std),
                dim=hidden_size,
                moe_config=moe_config,
                parallel_config=parallel_config
            )
        elif self.use_gmm:
            self.routed_experts = MoEV3(
                dim=hidden_size,
                intermediate_size=intermediate_size,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                return_extra_loss=return_extra_loss,
                moe_config=moe_config,
                parallel_config=parallel_config,
                init_method_std=init_method_std
            )

        else:
            self.routed_experts = MoEV2(
                ffn=LlamaFeedForward(dim=hidden_size,
                                     intermediate_size=intermediate_size,
                                     hidden_act=hidden_act,
                                     expert_num=self.expert_num,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type,
                                     moe_config=moe_config,
                                     parallel_config=parallel_config,
                                     use_fused_swiglu=use_fused_swiglu,
                                     init_method_std=init_method_std),
                dim=hidden_size,
                moe_config=moe_config,
                parallel_config=parallel_config,
                return_extra_loss=return_extra_loss,
                init_method_std=init_method_std
            )

        self.shared_experts = LlamaFeedForward(dim=hidden_size,
                                               intermediate_size=int(intermediate_size * moe_config.shared_expert_num),
                                               hidden_act=hidden_act,
                                               expert_num=1,
                                               compute_dtype=compute_dtype,
                                               param_init_type=param_init_type,
                                               use_fused_swiglu=use_fused_swiglu,
                                               parallel_config=parallel_config)

        if self.use_shared_expert_gating:
            self.shared_experts_gate = Dense(in_channels=hidden_size,
                                             out_channels=1,
                                             has_bias=False,
                                             dtype=self.router_dense_type)

    def construct(self, x, extra_loss=0., seq_chunk=None):
        r"""Forward process of the LlamaFeedForwardWithMoE"""
        if self.use_seq_parallel:
            shared_x = self.reshape(x, (self.dp, -1, x.shape[-1]))
            shared_experts_output = self.shared_experts(shared_x)
            shared_experts_output = self.reshape(shared_experts_output,
                                                 (x.shape[0], -1, x.shape[-1]))
        else:
            shared_experts_output = self.shared_experts(x)
        if self.use_shared_expert_gating:
            gate = self.sigmoid(self.shared_experts_gate(self.cast(x, self.router_dense_type)))
            shared_experts_output = self.mul(shared_experts_output, self.cast(gate, self.compute_dtype))
        if self.return_extra_loss:
            routed_experts_output, extra_loss = self.routed_experts(x, extra_loss, seq_chunk=seq_chunk)
            output = self.add(routed_experts_output, shared_experts_output)
            return output, extra_loss

        routed_experts_output = self.routed_experts(x)
        output = self.add(routed_experts_output, shared_experts_output)
        return output


    def shard(self, parallel_config):
        r"""set parallel strategy"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.use_seq_parallel:
            parallel_config_sp = copy.deepcopy(parallel_config)
            parallel_config_sp.data_parallel = parallel_config.data_parallel * parallel_config.model_parallel
            parallel_config_sp.model_parallel = 1
            self.routed_experts.ffn.shard(parallel_config)
            self.shared_experts.shard(parallel_config_sp)
            self.add.shard(((dp, mp, 1), (dp, mp, 1)))
        else:
            self.routed_experts.ffn.shard(parallel_config)
            self.shared_experts.shard(parallel_config)
            self.shared_experts.mul.shard(((dp, 1, mp), (dp, 1, mp)))
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))

        self.mul.shard(((dp, 1, 1), (dp, 1, 1)))
        self.sigmoid.shard(((dp, 1, 1),))
        if self.use_shared_expert_gating:
            self.shared_experts_gate.matmul.shard(((dp, 1), (1, 1)))
