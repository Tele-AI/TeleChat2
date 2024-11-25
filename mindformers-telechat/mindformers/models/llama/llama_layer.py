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

import mindspore as ms
from mindspore.common.parameter import Parameter
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Sigmoid
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode
from mindformers.version_control import check_valid_big_kernel
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, \
    _valid_value_checks
from mindformers.tools.logger import _LogActionOnce
from mindformers.version_control import check_rmsnorm_big_kernel_valid
from mindformers.modules.transformer.moe import MoEV2, MoEInfer
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
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.gather = P.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output

    def shard(self, parallel_config):
        """sharding for embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if parallel_config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp, cp)))
            logger.info(f"Using {dp*cp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % (mp * cp) != 0:
                logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s"
                               "model_parallel: %s * context_parallel: %s.",
                               self.vocab_table_size, mp, cp)
                logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
                self.gather.shard(((1, 1), (dp, cp)))
            else:
                self.gather.shard(((mp * cp, 1), (dp, 1)))
                logger.info(f"Using {dp} data parallel, {cp} context parallel and {mp} "
                            f"model parallel for the embedding lookup.")


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
        if self.self_define:
            self.square.shard((strategy_in,))
            self.mean.shard((strategy_in,))
            self.rsqrt.shard((strategy_in,))
            self.add.shard((strategy_in, ()))
            self.mul.shard((strategy_in, strategy_in))
            self.mul2.shard((strategy_in, (1,)))
        else:
            self.norm.shard((strategy_in, (1,)))


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
                 is_dynamic=False,
                 parallel_config=default_dpmp_config):
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

        if expert_num > 1:
            ep = parallel_config.expert_parallel
            dp_moe = parallel_config.data_parallel // ep
        else:
            dp_moe = 1
        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.ffn_concat = ffn_concat

        self.mul = P.Mul()
        self.cast = P.Cast()

        if self.ffn_concat:
            self.w_gate_hidden = Linear(in_channels=dim,
                                        out_channels=hidden_dim * 2,
                                        expert_num=expert_num,
                                        outer_batch=dp_moe,
                                        has_bias=False,
                                        compute_dtype=compute_dtype,
                                        param_init_type=param_init_type,
                                        skip_redistribution=is_dynamic)
            self.activate = self.hidden_act()
            self.split = ms.ops.auto_generate.SplitWithSize()
            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
        else:
            self.w1 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             activation=hidden_act,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)

            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)

            self.w3 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)

        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp
            gate, hidden = self.split(gate_hidden_out, (self.hidden_dim, self.hidden_dim), 2)
            gate = self.activate(gate)
        else:
            # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
            gate = self.w1(x)  # dp,1 -> dp, mp
            hidden = self.w3(x)  # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate)  # dp,mp -> dp, mp
        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        if self.expert_num == 1:
            if self.ffn_concat:
                self.w_gate_hidden.shard(((dp, 1), (mp, 1)))
                self.activate.shard(((dp, 1, mp),))
                self.w2.shard(((dp, mp), (1, mp)))
                self.split.add_prim_attr("skip_redistribution", True)
                self.split.shard(((dp, 1, mp),))
                self.mul.shard(((dp, mp), (dp, mp)))
            else:
                self.w1.shard(((dp * cp, 1), (mp, 1)), strategy_activation=((dp * cp, mp),))
                self.w1.activation.shard(((dp * cp, mp),))
                self.w2.shard(((dp * cp, mp), (1, mp)))
                self.w3.shard(((dp * cp, 1), (mp, 1)))
                self.mul.shard(((dp, cp, mp), (dp, cp, mp)))
        else:
            logger.info("shard ffn with MoE")
            ep = parallel_config.expert_parallel
            dp = parallel_config.data_parallel // ep
            self.w1.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                          strategy_activation=((dp, ep, mp, 1),))
            self.w2.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)))
            self.w3.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)))
            self.mul.shard(((dp * ep, mp), (dp * ep, mp)))


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
                 is_dynamic=False,
                 use_gmm=True):
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
                         expert_num=expert_num,
                         activation=hidden_act,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         expert_num=expert_num,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         expert_num=expert_num,
                         has_bias=False,
                         use_gmm=use_gmm,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

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

        self.mul.shard((1, mp), (1, mp))
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
                 is_dynamic=False,
                 moe_config=None,
                 parallel_config=default_dpmp_config,
                 use_moe_infer=True,
                 return_extra_loss=False,
                 ):
        super().__init__()
        self.expert_num = moe_config.expert_num
        self.shared_expert_num = moe_config.shared_expert_num
        self.use_shared_expert_gating = moe_config.use_shared_expert_gating
        self.router_dense_type = moe_config.router_dense_type
        self.compute_dtype = compute_dtype
        self.use_moe_infer = use_moe_infer
        self.return_extra_loss = return_extra_loss

        self.sigmoid = Sigmoid()
        self.mul = P.Mul()
        self.add = P.Add()

        if self.use_moe_infer:
            self.routed_experts = MoEInfer(
                ffn=LlamaMoeInferFeedForward(dim=hidden_size,
                                             intermediate_size=intermediate_size,
                                             hidden_act=hidden_act,
                                             expert_num=self.expert_num,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type,
                                             is_dynamic=is_dynamic,
                                             use_gmm=self.use_moe_infer),
                dim=hidden_size,
                moe_config=moe_config,
                parallel_config=parallel_config
            )
        else:
            self.routed_experts = MoEV2(
                ffn=LlamaFeedForward(dim=hidden_size,
                                     intermediate_size=intermediate_size,
                                     hidden_act=hidden_act,
                                     expert_num=self.expert_num,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type,
                                     is_dynamic=is_dynamic,
                                     parallel_config=parallel_config),
                dim=hidden_size,
                moe_config=moe_config,
                parallel_config=parallel_config,
                return_extra_loss=return_extra_loss
            )

        self.shared_experts = LlamaFeedForward(dim=hidden_size,
                                               intermediate_size=int(intermediate_size * moe_config.shared_expert_num),
                                               hidden_act=hidden_act,
                                               expert_num=1,
                                               compute_dtype=compute_dtype,
                                               param_init_type=param_init_type,
                                               is_dynamic=is_dynamic,
                                               parallel_config=parallel_config)

        if self.use_shared_expert_gating:
            self.shared_experts_gate = Dense(in_channels=hidden_size,
                                             out_channels=1,
                                             has_bias=False,
                                             dtype=self.router_dense_type)

    def construct(self, x, extra_loss=0.):
        r"""Forward process of the LlamaFeedForwardWithMoE"""
        shared_experts_output = self.shared_experts(x)
        if self.use_shared_expert_gating:
            gate = self.sigmoid(self.shared_experts_gate(self.cast(x, self.router_dense_type)))
            shared_experts_output = self.mul(shared_experts_output, self.cast(gate, self.compute_dtype))
        if self.return_extra_loss:
            routed_experts_output, extra_loss = self.routed_experts(x, extra_loss)
            output = self.add(routed_experts_output, shared_experts_output)
            return output, extra_loss

        routed_experts_output = self.routed_experts(x)
        output = self.add(routed_experts_output, shared_experts_output)
        return output


    def shard(self, parallel_config):
        r"""set parallel strategy"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.mul.shard(((dp, 1, 1), (dp, 1, 1)))
        self.add.shard(((dp, 1, 1), (dp, 1, 1)))
        self.sigmoid.shard(((dp, 1, 1),))

        self.routed_experts.ffn.shard(parallel_config)
        self.shared_experts.shard(parallel_config)
        self.shared_experts.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if self.use_shared_expert_gating:
            self.shared_experts_gate.matmul.shard(((dp, 1), (1, 1)))
