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
"""Bloom custom layers"""
import math
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore import nn, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.modules import AttentionMask
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.transformer import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.transformer.transformer import default_transformer_config, _get_lambda_func
from mindformers.version_control import check_valid_flash_attention, choose_flash_attention_dtype

from mindformers.tools.logger import logger


class BloomAttention(MultiHeadAttention):
    r"""The implementation of Bloom attention."""

    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_seq_parallel=False,
                 use_select_recompute=False,
                 use_flash_attention=False,
                 parallel_config=default_dpmp_config):

        super(BloomAttention, self).__init__(batch_size,
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

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.inv_norm_factor = Tensor([1.0 / math.sqrt(self.size_per_head)])
            self.beta = Tensor([1.0])
        else:
            self.add_alibi = P.Add().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.mul_alibi = P.Mul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), (1,)))
            self.mul_score = P.Mul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), (1,)))
            # [B, 1, S, S] + [B, H, S, S]
            self.add_atten_score = P.Add().shard(
                ((parallel_config.data_parallel, 1, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # [B, H, S, D] * [B, 1, S, 1]
            self.mul_key = P.Mul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (1, 1, 1, 1)))
            # [B, H, S, D] * [B, 1, S, 1]
            self.mul_value = P.Mul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (1, 1, 1, 1)))
            # [B, H, S, D] * [B, H, S, D]
            self.add_key = P.Add().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # [B, H, S, D] * [B, H, S, D]
            self.add_value = P.Add().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.inv_norm_factor = Tensor([1.0 / math.sqrt(self.size_per_head)])
            self.beta = Tensor([1.0])
            self.cast = P.Cast().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True).shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.use_flash_attention = use_flash_attention
            if self.use_flash_attention:
                self.attention_mask_dtype = choose_flash_attention_dtype()
                self.flash_attention = FlashAttention(head_num=self.n_head,
                                                      scale_value=1. / math.sqrt(self.size_per_head),
                                                      input_layout='BNSD',
                                                      keep_prob=1. - attention_dropout_rate,
                                                      pre_tokens=65536,
                                                      next_tokens=0,
                                                      use_alibi_mask=True,
                                                      use_attention_mask=True)
                self.flash_attention.shard(parallel_config)
            if use_select_recompute:
                logger.info("Using select recompute mode!")
                self.cast.recompute()
                self.batch_matmul.recompute()
                self.batch_matmul_trans_b.recompute()
                self.sub.recompute()
                self.add.recompute()
                self.add_alibi.recompute()
                self.prob_dropout.recompute()
                self.softmax.softmax.recompute()
                self.softmax_3d.recompute()
            if use_seq_parallel:
                logger.info("Using seq parallel mode!")
                self.dropout.dropout.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                self.projection.shard(
                    strategy_bias=((parallel_config.data_parallel * parallel_config.model_parallel, 1), (1,)),
                    strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                     (parallel_config.model_parallel, 1)),
                    out_strategy_matmul=((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
        self.sub_attention = P.Sub().shard(((1,), (1, 1, 1, 1)))
        self.tile = P.Tile()
        self.reshape = P.Reshape()

    # pylint: disable=arguments-differ
    def construct(self, query_tensor, key_tensor, value_tensor, alibi_tensor, attention_mask,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the BloomAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = F.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = F.dtype(query_tensor)
        query_tensor = self.cast(query_tensor, self.dtype)
        key_tensor = self.cast(key_tensor, self.dtype)
        value_tensor = self.cast(value_tensor, self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)

        query = self.reshape(
            query,
            (batch_size, self._get_seq_length_under_incremental(self.src_seq_length), self.n_head, self.size_per_head))
        query = self.transpose(query, (0, 2, 1, 3))  # the returned shape is [bs, num_heads, seq_length, size_per_head]
        key = self.reshape(
            key,
            (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length), self.n_head, self.size_per_head)
        )
        key = self.transpose(key, (0, 2, 1, 3))  # the returned shape is [bs, num_heads, size_per_head, seq_length]
        value = self.reshape(
            value,
            (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length), self.n_head, self.size_per_head)
        )
        value = self.transpose(value, (0, 2, 1, 3))  # the returned shape is [bs, num_heads, seq_length, size_per_head]

        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and attention_mask.ndim == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = self.cast((self.less(self.range, batch_valid_length.view(-1, 1, 1))), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul_key(key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul_value(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the input size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                current_index = batch_valid_length - 1
                current_index = self.reshape(current_index, (-1, 1, 1))
                current_mask = self.cast((self.equal(self.range, current_index)), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul_key(key, self.expand_dims(current_mask, 3))
                current_value = self.mul_value(value, self.expand_dims(current_mask, 3))
                # Concat the previous saved state and current state
                key = self.add_key(key_past, current_key)
                value = self.add_value(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        if self.use_flash_attention:
            attention_mask = self.sub_attention(Tensor((1.0,), mstype.float16), attention_mask)
            attention_mask = self.cast(attention_mask, mstype.uint8)
            # repeat alibi tensor shape as attention_mask
            attention_mask_dim = F.shape(attention_mask)[-1]
            alibi_tensor = self.tile(alibi_tensor, (1, 1, attention_mask_dim, 1))
            attention = self.flash_attention(query, key, value, attention_mask, alibi_tensor)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, alibi_tensor, attention_mask, batch_valid_length)

        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = self.reshape(output, ori_shape)
        output = self.cast(output, ori_dtype)
        return output, layer_present

    def _softmax(self, attention_scores):
        """
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """
        attention_probs = self.softmax(attention_scores)
        return attention_probs

    def _attn(self, query, key, value, alibi_tensor, attention_mask, batch_valid_length=None):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            alibi_tensor: the alibi matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        ori_dtype = F.dtype(query)
        score = self.batch_matmul_trans_b(self.cast(query, self.dtype), self.cast(key, self.dtype))
        score = self.add_alibi(
            self.mul_score(score, self.cast(self.inv_norm_factor, ori_dtype)),
            self.mul_alibi(alibi_tensor, self.cast(self.beta, ori_dtype))
        )
        attention_scores = self.cast(score, self.softmax_dtype)
        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                # bs, 1 -> bs, 1, 1
                batch_valid_length = batch_valid_length - 1
                index = self.reshape(batch_valid_length, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = self.cast((self.tensor_le(self.range, index)), mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiple_out = self.sub(
                self.cast(Tensor((1.0,)), F.dtype(attention_scores)),
                self.cast(attention_mask, F.dtype(attention_scores)))

            adder = self.mul(multiple_out, self.multiply_data)
            attention_scores = self.add_atten_score(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = self.cast(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]

        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype),
                                            self.cast(value, self.dtype))
        weighted_values = self.cast(weighted_values, self.softmax_dtype)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class BloomBlock(TransformerEncoderLayer):
    r"""A block of Bloom model."""

    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 use_seq_parallel=False,
                 use_select_recompute=False,
                 use_flash_attention=False,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):

        super(BloomBlock, self).__init__(batch_size,
                                         hidden_size,
                                         ffn_hidden_size,
                                         num_heads,
                                         seq_length,
                                         attention_dropout_rate,
                                         hidden_dropout_rate,
                                         post_layernorm_residual,
                                         layernorm_compute_type,
                                         softmax_compute_type,
                                         param_init_type,
                                         hidden_act,
                                         use_past,
                                         moe_config,
                                         parallel_config)

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            self.use_past = use_past
            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = BloomAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            hidden_size=hidden_size,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            softmax_compute_type=softmax_compute_type,
                                            param_init_type=param_init_type,
                                            use_past=use_past,
                                            use_seq_parallel=use_seq_parallel,
                                            use_select_recompute=use_select_recompute,
                                            use_flash_attention=use_flash_attention,
                                            parallel_config=attention_parallel_config)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.use_past = use_past
            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = BloomAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            hidden_size=hidden_size,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            softmax_compute_type=softmax_compute_type,
                                            param_init_type=param_init_type,
                                            use_past=use_past,
                                            use_seq_parallel=use_seq_parallel,
                                            use_select_recompute=use_select_recompute,
                                            use_flash_attention=use_flash_attention,
                                            parallel_config=attention_parallel_config)
        if self.use_past:
            size_per_head = hidden_size // num_heads
            self.key_shape = (batch_size, num_heads, seq_length, size_per_head)
            self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")

            # [B, H, S, D]
            self.assign_past = P.Assign().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))

        if use_seq_parallel:
            self.add.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),
                            (parallel_config.data_parallel * parallel_config.model_parallel, 1)))
            self.layernorm1.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
            self.layernorm2.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
            if not self.use_moe:
                self.output.projection.shard(
                    strategy_bias=((parallel_config.data_parallel * parallel_config.model_parallel, 1), (1,)),
                    strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                     (parallel_config.model_parallel, 1)),
                    out_strategy_matmul=((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                self.output.dropout.dropout.shard(
                    ((parallel_config.data_parallel * parallel_config.model_parallel, 1),))

        self.depend = P.Depend()
        self.cast = P.Cast()

    # pylint: disable=arguments-differ
    def construct(self, x, alibi_tensor, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(x, input_mask, init_reset, batch_valid_length)
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
            self.assign_past(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign_past(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = self.depend(input_x, key_reset)
            input_x = self.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, alibi_tensor, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign_past(self.key_past, key_present)
            key_update = self.key_past
            self.assign_past(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = self.depend(key_update, key_reset)
            value_update = self.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = self.depend(mlp_logit, value_update)
        mlp_logit = self.depend(mlp_logit, key_update)

        output = self.add(x, mlp_logit)
        if self.use_moe:
            return output, layer_present, aux_loss
        return output, layer_present


class BloomBlocks(TransformerEncoder):
    r"""All blocks of Bloom model."""

    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 use_past=False,
                 use_seq_parallel=False,
                 use_select_recompute=False,
                 use_flash_attention=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):

        super(BloomBlocks, self).__init__(batch_size,
                                          num_layers,
                                          hidden_size,
                                          ffn_hidden_size,
                                          seq_length,
                                          num_heads,
                                          attention_dropout_rate,
                                          hidden_dropout_rate,
                                          hidden_act,
                                          post_layernorm_residual,
                                          layernorm_compute_type,
                                          softmax_compute_type,
                                          param_init_type,
                                          lambda_func,
                                          offset,
                                          use_past,
                                          moe_config,
                                          parallel_config)

        config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if use_flash_attention:
            use_flash_attention = check_valid_flash_attention(fa_type="FlashAttention")
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = BloomBlock(hidden_size=hidden_size,
                                   batch_size=batch_size,
                                   ffn_hidden_size=ffn_hidden_size,
                                   seq_length=seq_length,
                                   attention_dropout_rate=attention_dropout_rate,
                                   hidden_dropout_rate=hidden_dropout_rate,
                                   layernorm_compute_type=layernorm_compute_type,
                                   softmax_compute_type=softmax_compute_type,
                                   num_heads=num_heads,
                                   hidden_act=hidden_act,
                                   post_layernorm_residual=post_layernorm_residual,
                                   param_init_type=param_init_type,
                                   use_past=use_past,
                                   use_seq_parallel=use_seq_parallel,
                                   use_select_recompute=use_select_recompute,
                                   use_flash_attention=use_flash_attention,
                                   moe_config=moe_config,
                                   parallel_config=config_to_layer)

                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config,
                            use_select_recompute=use_select_recompute)
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = BloomBlock(hidden_size=hidden_size,
                                   batch_size=batch_size,
                                   ffn_hidden_size=ffn_hidden_size,
                                   seq_length=seq_length,
                                   attention_dropout_rate=attention_dropout_rate,
                                   hidden_dropout_rate=hidden_dropout_rate,
                                   layernorm_compute_type=layernorm_compute_type,
                                   softmax_compute_type=softmax_compute_type,
                                   num_heads=num_heads,
                                   hidden_act=hidden_act,
                                   post_layernorm_residual=post_layernorm_residual,
                                   param_init_type=param_init_type,
                                   use_past=use_past,
                                   use_seq_parallel=use_seq_parallel,
                                   use_select_recompute=use_select_recompute,
                                   use_flash_attention=use_flash_attention,
                                   moe_config=moe_config,
                                   parallel_config=config_to_layer)

                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config,
                            use_select_recompute=use_select_recompute)
                self.blocks.append(block)

    # pylint: disable=arguments-differ
    def construct(self, hidden_states, alibi_tensor, attention_mask, init_reset=True, batch_valid_length=None):
        """forward process"""
        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    alibi_tensor,
                                                    attention_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer


class CausalMask(AttentionMask):
    r"""
        Get the Lower triangular matrix from the input mask. The input mask is a 2D tensor (batch_size, seq_length)
        with 1 and 0, where 1 indicates the current position is a valid token, otherwise not.

        Args:
            seq_length(int): The sequence length of the input tensor.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                               an instance of `OpParallelConfig` with default args.

        Inputs:
            - **input_mask** (Tensor) - The mask indicating whether each position is a valid input with
              (batch_size, seq_length).

        Outputs:
            Tensor. The attention mask matrix with shape (batch_size, seq_length, seq_length).
    """

    def __init__(self, seq_length, parallel_config=default_dpmp_config):
        super(CausalMask, self).__init__(seq_length, parallel_config)
        self.seq_length = seq_length
        self.parallel_config = parallel_config
        self.cast = P.Cast()

    def construct(self, input_mask):
        """Forward process of the CausalMask"""
        input_mask = self.cast(self.not_equal(input_mask, 0), mstype.float32)
        input_shape = input_mask.shape
        shape_right = (input_shape[0], 1, input_shape[1])
        # Mask the padded inputs
        attention_mask = self.reshape(input_mask, shape_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)
        return attention_mask
