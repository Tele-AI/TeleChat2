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
"""Mixtral Model Layers' APIs."""

import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, \
    _valid_value_checks
from mindformers.tools.logger import _LogActionOnce
from mindformers.models.llama.llama_layer import LlamaSiLU


class MixtralFeedForward(Cell):
    r"""
    Mixtral FeedForward.

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
            dp = parallel_config.data_parallel
            ep = parallel_config.expert_parallel
            cp = parallel_config.context_parallel
            mp = parallel_config.model_parallel
            dp_moe = dp * cp // ep
            if parallel_config.use_seq_parallel:
                dp_moe *= ep * mp if dp * cp > ep else ep
        else:
            dp_moe = 1
        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.ffn_concat = ffn_concat
        self.use_seq_parallel = parallel_config.use_seq_parallel

        self.mul = P.Mul()
        self.cast = P.Cast()

        if self.ffn_concat:
            self.w_gate_hidden = Linear(in_channels=dim,
                                        out_channels=hidden_dim * 2,
                                        expert_num=expert_num,
                                        outer_batch=dp_moe,
                                        has_bias=False,
                                        compute_dtype=compute_dtype,
                                        param_init_type=param_init_type)
            self.activate = self.hidden_act()
            self.split = ms.ops.auto_generate.SplitWithSize()
            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        else:
            self.w1 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             activation=hidden_act,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

            self.w2 = Linear(in_channels=hidden_dim,
                             out_channels=dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

            self.w3 = Linear(in_channels=dim,
                             out_channels=hidden_dim,
                             expert_num=expert_num,
                             outer_batch=dp_moe,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)

        if self.ffn_concat:
            gate_hidden_out = self.w_gate_hidden(x)  # dp,1 -> dp, mp
            if self.use_seq_parallel:
                gate, hidden = self.split(gate_hidden_out, (self.hidden_dim, self.hidden_dim), 2)
            else:
                gate, hidden = self.split(gate_hidden_out, (self.hidden_dim, self.hidden_dim), 1)
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
            dp = parallel_config.data_parallel * parallel_config.context_parallel // ep
            if self.ffn_concat:
                self.w_gate_hidden.shard(((dp, ep, 1, 1), (ep, mp, 1)))
                self.w2.shard(((dp, ep, 1, mp), (ep, 1, mp)))
                self.split.add_prim_attr("skip_redistribution", True)
                self.split.shard(((dp * ep, mp),))
                self.activate.shard(((dp * ep, mp),))
                mul_shard = (dp * ep, mp)
                if parallel_config.use_seq_parallel:
                    self.split.shard(((dp, ep, mp),))
                    self.activate.shard(((dp, ep, mp),))
                    mul_shard = (dp, ep, mp)
                self.mul.shard((mul_shard, mul_shard))
            else:
                self.w1.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                              strategy_activation=((dp, ep, mp, 1),))
                self.w2.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)))
                self.w3.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)))
                mul_shard = (dp * ep, mp)
                if parallel_config.use_seq_parallel:
                    mul_shard = (dp, ep, mp)
                self.mul.shard((mul_shard, mul_shard))
