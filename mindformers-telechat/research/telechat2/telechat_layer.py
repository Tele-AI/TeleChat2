# Copyright 2024  Huawei Technologies Co., Ltd
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
"""Telechat Model Layers' APIs."""

from mindspore.common.parameter import Parameter
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
from mindspore.common.initializer import initializer, Normal
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode

from mindformers.models.llama.llama_layer import LlamaSiLU
from mindformers.modules.layers import Linear, Dropout, _check_input_dtype, _args_type_validator_check, \
    _valid_value_checks
from mindformers.tools.logger import _LogActionOnce


class TelechatEmbedding(Cell):
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
    def __init__(self, vocab_table_size, embedding_size, sigma=0.0048, mean=0.0, param_init_type=mstype.float32,
                 parallel_optimizer=True):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(Normal(sigma=sigma, mean=mean), [self.vocab_table_size, self.embedding_size],
                        dtype=param_init_type), name='embedding_weight', parallel_optimizer=parallel_optimizer)
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
        if parallel_config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % mp != 0:
                logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                               self.vocab_table_size, mp)
                logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
                self.gather.shard(((1, 1), (dp, 1)))
            else:
                self.gather.shard(((mp, 1), (dp, 1)))
                logger.info(f"Using {dp} data parallel and {mp} "
                            f"model parallel for the embedding lookup.")


class TelechatLinear(Linear):
    # pylint: disable=W0212
    """
    Linear function for Telechat.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 sigma=0.0048,
                 mean=0.0,
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 skip_redistribution=False,
                 keep_prob=1.0):
        super(TelechatLinear, self).__init__(
            in_channels,
            out_channels,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=activation,
            transpose_b=transpose_b,
            param_init_type=param_init_type,
            skip_redistribution=skip_redistribution,
            compute_dtype=compute_dtype)
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.weight = Parameter(initializer(Normal(sigma=sigma, mean=mean), weight_shape, param_init_type),
                                name="weight")
        self.dropout = Dropout(keep_prob=keep_prob)

    def construct(self, x):
        """construct of linear."""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = self.dropout(x)
        if self.activation_flag:
            x = self.activation(x)
        x = F.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output


class TelechatFeedForward(Cell):
    r"""
    Telechat FeedForward.

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

    @_LogActionOnce(m_logger=logger, key='TelechatFeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "TelechatFeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "TelechatFeedForward"))
    def __init__(self, dim,
                 intermediate_size=None,
                 hidden_dim=None,
                 sigma=0.0048,
                 mean=0.0,
                 multiple_of=256,
                 hidden_dropout_prob=1.0,
                 hidden_act=LlamaSiLU,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 is_dynamic=False):
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

        self.hidden_dropout_prob = hidden_dropout_prob
        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.w1 = TelechatLinear(in_channels=dim,
                                 out_channels=hidden_dim,
                                 activation=hidden_act,
                                 has_bias=False,
                                 sigma=sigma,
                                 mean=mean,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic)

        self.w2 = TelechatLinear(in_channels=hidden_dim,
                                 out_channels=dim,
                                 has_bias=True,
                                 sigma=sigma,
                                 mean=mean,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic,
                                 keep_prob=1 - self.hidden_dropout_prob)

        self.w3 = TelechatLinear(in_channels=dim,
                                 out_channels=hidden_dim,
                                 has_bias=False,
                                 sigma=sigma,
                                 mean=mean,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)
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
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)))
        self.w1.activation.shard(((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))
