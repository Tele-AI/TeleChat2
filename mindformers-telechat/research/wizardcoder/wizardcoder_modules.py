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
"""Wizardcoder modules."""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode
from mindspore import log as logger

from mindformers.modules.flash_attention import FlashAttention

from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.transformer import TransformerEncoderLayer, MultiHeadAttention, \
    VocabEmbedding, TransformerOpParallelConfig, EmbeddingOpParallelConfig
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.layers import Linear, LayerNorm

default_transformer_config = TransformerOpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()


class WizardCoderVocabEmbedding(VocabEmbedding):
    def __init__(self, vocab_size, embedding_size, parallel_config=default_embedding_parallel_config,
                 param_init='normal'):
        super(WizardCoderVocabEmbedding, self).__init__(vocab_size, embedding_size, parallel_config, param_init)
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if parallel_config.vocab_emb_dp:
            self.gather = P.Gather().shard(((mp, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel for the embedding lookup.")


class MultiQueryAttention(MultiHeadAttention):
    r"""
        This is an implementation of multi query attention.
        Supported Platforms:
            ``Ascend``
    """

    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 compute_dtype,
                 softmax_compute_type,
                 param_init_type,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 use_past=False,
                 use_seq_parallel=False,
                 use_flash_attention=True,
                 parallel_config=default_dpmp_config):
        super(MultiQueryAttention, self).__init__(batch_size,
                                                  src_seq_length,
                                                  tgt_seq_length,
                                                  hidden_size,
                                                  num_heads,
                                                  hidden_dropout_rate,
                                                  attention_dropout_rate,
                                                  compute_dtype,
                                                  softmax_compute_type,
                                                  param_init_type,
                                                  use_past,
                                                  parallel_config)
        if not self._is_ascend:
            raise ValueError("For 'MultiQueryAttention', now only support Ascend")
        self.compute_dtype = compute_dtype
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            if use_seq_parallel:
                self.projection.shard(strategy_bias=((dp, 1), (1,)),
                                      strategy_matmul=((dp, mp), (mp, 1)),
                                      out_strategy_matmul=((dp * mp, 1),))
                logger.info("Enabling matmul recompuation when seq parallel enabled")
                self.projection.matmul.add_prim_attr("recompute", True)
                self.projection.matmul.add_prim_attr("recompute_comm_op", True)
        else:
            if use_seq_parallel:
                self.dropout.dropout.shard(((dp * mp, 1),))
                self.projection.shard(
                    strategy_bias=((dp * mp, 1), (1,)),
                    strategy_matmul=((dp, mp), (mp, 1)),
                    out_strategy_matmul=((dp * mp, 1),))
                logger.info("Enabling matmul recompuation when seq parallel enabled")
                self.projection.matmul.add_prim_attr("recompute", True)
                self.projection.matmul.add_prim_attr("recompute_comm_op", True)

            self.batch_matmul = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

            self.kv_heads = 1
            self.kv_dim = self.kv_heads * self.size_per_head

            self.transpose_one_head = P.Transpose().shard(((dp, 1, 1, 1),))
            self.tile_for_batch_matmul = P.Tile().shard(((dp, mp, 1, 1),))
            self.real_div_one_head = P.RealDiv().shard(((dp, 1, 1, 1), ()))
            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense1.shard(strategy_matmul=((dp, 1), (mp, 1)),
                              strategy_bias=((dp, mp), (mp,)))
            old_mp = parallel_config.model_parallel
            parallel_config.model_parallel = 1
            # Key
            self.dense2 = Linear(hidden_size,
                                 self.kv_dim,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            self.dense2.weight.parallel_optimizer = False

            # Value
            self.dense3 = Linear(hidden_size,
                                 self.kv_dim,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))

            self.dense3.weight.parallel_optimizer = False
            parallel_config.model_parallel = old_mp
            self.cast_rec = P.Cast()
            self.reshape_rec = P.Reshape()
        self.flash_attention_flag = use_flash_attention
        if self.flash_attention_flag:
            self.flash_attention = FlashAttention(self.size_per_head, attention_dropout_rate, prev_block_num=65536,
                                                  next_block_num=0, tiling_stgy_name="sparse",
                                                  dp=parallel_config.data_parallel, mp=parallel_config.model_parallel)
            self.flash_attention.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                        (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                        (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                        (parallel_config.data_parallel, 1, 1),
                                        (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.flash_attention.drop_gen_mask.recompute(False)
            self.flash_attention.fill_v2.recompute(False)
            self.flash_attention.flash_attention.recompute(False)
        self.squeeze = P.Squeeze(1)
        logger.info("dp_num = {}, mp_num = {}".format(parallel_config.data_parallel, parallel_config.model_parallel))
        logger.info("Using FlashAttention in this round of operation = ", self.flash_attention_flag)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.get_dtype = P.DType()

    def set_select_recompute(self):
        """operator select recompute"""
        self.batch_matmul.recompute()
        self.real_div.recompute()
        self.real_div_one_head.recompute()
        self.sub.recompute()
        self.add.recompute()
        self.prob_dropout.dropout.recompute()
        self.softmax_3d.softmax.recompute()
        self.softmax.softmax.recompute()
        self.cast_rec.recompute()
        self.mul.recompute()
        self.reshape_rec.recompute()

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        """Forward process of the MultiQueryAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = self.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = self.get_dtype(query_tensor)
        query_tensor = self.cast(query_tensor, self.dtype)
        key_tensor = self.cast(key_tensor, self.dtype)
        value_tensor = self.cast(value_tensor, self.dtype)
        # multi query attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            self.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        if self.flash_attention_flag:
            key = self.transpose_one_head(
                self.reshape(
                    key,
                    (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                     self.kv_heads, self.size_per_head)),
                (0, 2, 1, 3))
        else:
            key = self.transpose_one_head(
                self.reshape(
                    key,
                    (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                     self.kv_heads, self.size_per_head)),
                (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose_one_head(
            self.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.kv_heads, self.size_per_head)),
            (0, 2, 1, 3))

        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and self.flash_attention_flag is False and len(self.shape(attention_mask)) == 3:
            attention_mask = self.expand_dims(attention_mask, 1)
        if attention_mask is not None and self.flash_attention_flag is True and len(self.shape(attention_mask)) == 4:
            attention_mask = self.squeeze(attention_mask)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = self.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = self.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = self.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        if self.flash_attention_flag:
            key = self.tile_for_batch_matmul(key, (1, self.n_head, 1, 1))
            value = self.tile_for_batch_matmul(value, (1, self.n_head, 1, 1))
            attention = self.flash_attention(query, key, value, attention_mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, attention_mask)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = self.reshape(output, ori_shape)
        output = self.cast(output, ori_dtype)
        return output, layer_present

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = self.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                self.reshape_rec(attention_scores,
                                 (shape[0], -1, shape[-1])))
            attention_probs = self.reshape_rec(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]

        factor = self.cast(self.scale_factor, self.get_dtype(query))
        query = self.real_div(query, factor)
        key = self.real_div_one_head(key, factor)
        query = self.cast(query, self.compute_dtype)
        key = self.cast(key, self.compute_dtype)
        score = self.batch_matmul(query, key)

        ori_dtype = self.get_dtype(score)
        attention_scores = self.cast_rec(score, self.softmax_dtype)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                bs, *_ = self.shape(query)
                tmp = self.not_equal(self.slice(key, (0, 0, 0, 0), (bs, 1, 1, self.seq_length), (1, 1, 1, 1)), 0)
                current_index = self.reducesum(self.cast(tmp, mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(self.cast(current_index, mstype.int32), 1)
                index = self.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = self.cast(self.tensor_le(self.range, index), mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(
                self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                self.cast_rec(attention_mask, self.get_dtype(attention_scores)))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = self.cast_rec(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)

        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        attention_probs = self.cast(attention_probs, self.compute_dtype)
        value = self.cast(value, self.compute_dtype)
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class WizardCoderTransformerDecoderLayer(TransformerEncoderLayer):
    r"""WizardCoder Transformer Decoder Layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            num_heads(int): The number of the heads.
            seq_length(int): The input sequence length.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, seq_length),
              (batch_size, num_heads, seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 compute_dtype,
                 layernorm_compute_type,
                 softmax_compute_type,
                 param_init_type,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 hidden_act='gelu',
                 use_past=False,
                 use_seq_parallel=False,
                 use_flash_attention=True,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(WizardCoderTransformerDecoderLayer, self).__init__(
            batch_size=batch_size,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_heads=num_heads,
            seq_length=seq_length,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            post_layernorm_residual=post_layernorm_residual,
            layernorm_compute_type=layernorm_compute_type,
            softmax_compute_type=softmax_compute_type,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype,
            hidden_act=hidden_act,
            use_past=use_past,
            moe_config=moe_config,
            parallel_config=parallel_config
        )
        self.is_first_iteration = True
        self.layernorm1 = LayerNorm((hidden_size,), param_init_type=layernorm_compute_type)
        self.layernorm2 = LayerNorm((hidden_size,), param_init_type=layernorm_compute_type)
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            if use_seq_parallel:
                self.add.shard(((dp * mp, 1), (dp * mp, 1)))
                self.layernorm1.shard(((dp * mp, 1),))
                self.layernorm2.shard(((dp * mp, 1),))
                if not self.use_moe:
                    self.output.projection.shard(
                        strategy_bias=((dp * mp, 1), (1,)),
                        strategy_matmul=((dp, mp), (mp, 1)),
                        out_strategy_matmul=((dp * mp, 1),))
                    self.output.dropout.dropout.shard(((dp * mp, 1),))
            self.output.projection.matmul.add_prim_attr("recompute_comm_op", True)
            self.layernorm1.layer_norm.add_prim_attr("recompute_comm_op", True)
            self.layernorm2.layer_norm.add_prim_attr("recompute_comm_op", True)
        attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
        self.attention = MultiQueryAttention(batch_size=batch_size,
                                             src_seq_length=seq_length,
                                             tgt_seq_length=seq_length,
                                             hidden_size=hidden_size,
                                             num_heads=num_heads,
                                             hidden_dropout_rate=hidden_dropout_rate,
                                             attention_dropout_rate=attention_dropout_rate,
                                             compute_dtype=compute_dtype,
                                             softmax_compute_type=softmax_compute_type,
                                             param_init_type=param_init_type,
                                             use_past=use_past,
                                             use_seq_parallel=use_seq_parallel,
                                             use_flash_attention=use_flash_attention,
                                             parallel_config=attention_parallel_config)

        self.dtype = compute_dtype
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.depend = P.Depend()
        if self.use_past:
            size_per_head = hidden_size // num_heads
            self.key_shape = (batch_size, 1, size_per_head, seq_length)
            self.value_shape = (batch_size, 1, seq_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")

    def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = self.shape(x)
        x = self.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = self.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past and self.is_first_iteration:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, self.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, self.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = self.depend(input_x, key_reset)
            input_x = self.depend(input_x, value_reset)
        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)

        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.cast(x, self.dtype)
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = self.cast(output_x, self.dtype)
        aux_loss = None
        # feedforwad construct dtype should be set as bf16 or fp32
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)


        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = self.depend(key_update, key_reset)
            value_update = self.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = self.depend(mlp_logit, value_update)
        mlp_logit = self.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = self.reshape(output_x, x_shape)
            mlp_logit = self.reshape(mlp_logit, x_shape)
            x = self.reshape(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = self.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = self.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = self.reshape(output, x_shape)

        if self.use_moe:
            return output, aux_loss
        return output
