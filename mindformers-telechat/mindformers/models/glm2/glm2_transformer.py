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
"""ChatGLM2 Transformer."""
import math

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules import LayerNorm
from mindformers.modules.layers import Linear
from mindformers.modules.flash_attention import FlashAttention
from mindformers.models.utils import LayerSetting
from mindformers.pet.tuners.ptuning2_adapter import Ptuning2Adapter
from mindformers.version_control import get_dropout

from .glm2_config import ChatGLM2Config
from .glm2_modules import ChatGLM2MLP, ChatGLM2RMSNorm




class CoreAttention(nn.Cell):
    """ChatGLM2 core attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(CoreAttention, self).__init__()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.head_dim = config.kv_channels
        projection_size = config.kv_channels * config.num_attention_heads

        self.n_head = config.num_attention_heads
        self.norm_factor = math.sqrt(self.head_dim)
        self.mul_mask = P.Mul()
        self.add = P.Add()

        # Strided linear layer.
        self.attention_dropout = get_dropout(config.attention_dropout)

        parallel_config = config.parallel_config

        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul_q_k.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = nn.Softmax(axis=-1)

        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.reshape = P.Reshape()

        self.compute_dtype = config.compute_dtype
        self.multi_query_attention = config.multi_query_attention
        if self.multi_query_attention:
            self.n_kv_head = config.multi_query_group_num
            self.qkv_hidden_size = (
                projection_size + 2 * self.head_dim * config.multi_query_group_num)
        self.transpose = P.Transpose()
        self.cast = P.Cast()

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        calculate attention function
        """
        # query_layer [b, heads, seq, hidden_size_per_head]
        # key_layer [b, heads, seq, hidden_size_per_head]
        # value_layer # [bs, heads, seq_len, hidden_size_per_head]

        # seqlen, batch, head, hidden_size

        if self.apply_query_key_layer_scaling:
            query_layer = query_layer / self.norm_factor

        # ===================================
        # Raw attention scores. [b, heads, s, s]
        # ===================================
        # [b, heads, seq_q, hidden_size_per_head] × [b, heads, seq_k, hidden_size_per_head]^T -> [b, heads, seq_q, seq_k]
        matmul_result = self.batch_matmul_q_k(query_layer, key_layer)

        # record original score dtype
        attention_scores_dtype = matmul_result.dtype
        # [b, heads, seq, seq]
        attention_scores = matmul_result

        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = ops.ones((attention_scores.shape[0],
                                       1,
                                       attention_scores.shape[2],
                                       attention_scores.shape[3]), dtype=mstype.bool_)
            attention_mask.tril()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_mask = self.mul_mask(attention_mask, -10000)
            attention_scores = self.add(attention_scores, attention_mask)

        if self.attention_softmax_in_fp32:
            attention_scores = F.cast(attention_scores, mstype.float32)

        attention_probs = self.softmax(attention_scores)
        attention_probs = F.cast(attention_probs, attention_scores_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        # [bs, heads, seq_q, seq_k] x [bs, heads, seq_v, hidden_size_per_head] -> [b, heads, seq_q, hidden_size_per_head]
        context_layer = self.batch_matmul(attention_probs, value_layer)
        context_layer = F.cast(context_layer, self.compute_dtype)
        context_layer = self._merge_heads(context_layer)

        return context_layer

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge


class ChatGLM2SelfAttention(nn.Cell):
    """ChatGLM2 self-attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(ChatGLM2SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.head_dim = config.kv_channels
        self.projection_size = config.kv_channels * config.num_attention_heads
        # Per attention head and per partition values.
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.head_dim)
        self.n_head = config.num_attention_heads
        self.params_dtype = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.batch_size
        self.pre_seq_len = config.pre_seq_len
        self.n_rep = self.n_head // config.multi_query_group_num

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size

        if self.multi_query_attention:
            self.n_kv_head = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.head_dim * config.multi_query_group_num)

        parallel_config = config.parallel_config

        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        self.query_key_value = Linear(config.hidden_size,
                                      self.qkv_hidden_size,
                                      has_bias=config.add_bias_linear or config.add_qkv_bias,
                                      param_init_type=self.params_dtype,
                                      compute_dtype=self.compute_dtype)
        self.query_key_value.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.shape = P.Shape()
        self.dense = Linear(self.projection_size,
                            config.hidden_size,
                            has_bias=config.add_bias_linear,
                            param_init_type=self.params_dtype,
                            compute_dtype=self.compute_dtype)
        self.dense.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, 1), (1,)))
        self.use_flash_attention = config.use_flash_attention
        self.use_past = config.use_past
        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.head_dim,
                                                  self.n_kv_head,
                                                  pa_n_head_split=self.n_head // mp,
                                                  pa_n_kv_head_split=self.n_kv_head // mp,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  block_size=config.block_size,
                                                  num_blocks=config.num_blocks,
                                                  use_flash_attention=self.use_flash_attention,
                                                  rotary_cos_format=3,
                                                  compute_dtype=self.compute_dtype)
            self.infer_attention.shard(parallel_config)
        else:
            self.core_attention = CoreAttention(config, self.layer_number)
            self.reshape = P.Reshape()
            self.stack = P.Stack(axis=-1)
            self.mul = P.Mul()
            self.sub = P.Sub()
            self.add = P.Add()
            self.concat = P.Concat(axis=-1)
            self.transpose = P.Transpose()
            self.cast = P.Cast()
            self.tile_kv = P.Tile()

            if self.use_flash_attention:
                self.flash_attention = FlashAttention(head_num=config.num_attention_heads,
                                                      scale_value=1. / math.sqrt(self.head_dim),
                                                      input_layout='BNSD',
                                                      keep_prob=1. - config.attention_dropout,
                                                      pre_tokens=65536,
                                                      next_tokens=0)
                self.flash_attention.shard(parallel_config)
            self.merger_head_transpose = P.Transpose().shard(((dp, mp, 1, 1),))

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def apply_rotary_pos_emb(self, x: Tensor, rotary_pos_emb: Tensor) -> Tensor:
        """apply rotary position embedding to q,k."""
        # x: [b, heads, seq, hidden_size_per_head]
        bs, num_heads, seq_len, _ = x.shape  # 1, 32，4, 128
        # rope_cache: first (seq_len, kv_channels//4, 2), other (1, kv_channels//4, 2)
        _, _, rope_cache = rotary_pos_emb
        rot_dim = rope_cache.shape[-2] * 2  # kv_channels // 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # ms not support variable sizes
        # truncate to support variable sizes
        # [bs, nh, sq, kv_channels//4, 2]
        xshaped = self.reshape(x, (bs, num_heads, seq_len, rot_dim // 2, 2))
        # [bs, 1, sq, kv_channels//4, 2]
        if rope_cache.dtype == mstype.bfloat16:
            rope_cache = self.cast(rope_cache, mstype.float32)
        rope_cache = self.reshape(rope_cache, (-1, 1, seq_len, xshaped.shape[3], 2))

        xshaped_0, xshaped_1 = ops.split(xshaped, 1, -1)
        rope_cache_0, rope_cache_1 = ops.split(rope_cache, 1, -1)
        x_out1 = self.sub(self.mul(xshaped_0, rope_cache_0), self.mul(xshaped_1, rope_cache_1))
        x_out2 = self.add(self.mul(xshaped_1, rope_cache_0), self.mul(xshaped_0, rope_cache_1))
        x_out = self.stack((x_out1, x_out2))
        x_out = self.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2], -1))
        x_out = self.cast(x_out, x_pass.dtype)
        # [bs, sq, nh, hidden_size_per_head]
        return self.concat((x_out, x_pass))

    def add_prefix_if_need(self, prefix_key_value, key_layer, value_layer, attention_mask):
        """
        add p-tuning v2 prefix if need
        """
        if not isinstance(self.pre_seq_len, int) or self.pre_seq_len <= 0:
            return key_layer, value_layer, attention_mask

        seq_len = key_layer.shape[2]

        key_layer, value_layer = Ptuning2Adapter.add_prefix(
            prefix_key_value,
            key_layer,
            value_layer
        )

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            prefix_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.pre_seq_len))
            m_cat = P.Concat(3)
            # [bs, 1, seq_len, pre_seq_len + seq_len]
            attention_mask = m_cat((prefix_mask, attention_mask))

        return key_layer, value_layer, attention_mask

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, batch_valid_length=None, prefix_key_value=None,
                  block_tables=None, slot_mapping=None):
        """Forward process of self-attention."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask: (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first -> (sen length, kv_channels//4, 2), after -> (1, kv_channels//4, 2]
        bs, seq_len, _ = self.shape(hidden_states)
        # [bs, seq_len, qkv_hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # not compatible with ms below 2.0
        (query, key, value) = mixed_raw_layer.split(
            [self.n_head * self.head_dim,
             self.n_kv_head * self.head_dim,
             self.n_kv_head * self.head_dim,
             ],
            axis=-1,
        )

        # key and value for current token(s)
        if self.use_past:
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 rotary_pos_emb, attention_mask)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))

            query = self.apply_rotary_pos_emb(query, rotary_pos_emb)
            key = self.apply_rotary_pos_emb(key, rotary_pos_emb)

            key, value, attention_mask = self.add_prefix_if_need(
                prefix_key_value,
                key,
                value,
                attention_mask
            )

            if self.use_flash_attention:
                context_layer = self.flash_attention(query, key, value, attention_mask)
                context_layer = self._merge_heads(context_layer)
            else:
                key = self._repeat_kv(key, self.n_rep)
                value = self._repeat_kv(value, self.n_rep)
                context_layer = self.core_attention(query, key, value, attention_mask)

        # # =================
        # # Output. [bs, seq_len, hidden_size]
        # # =================

        output = self.dense(context_layer)

        return output


class ChatGLM2Block(nn.Cell):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLM2Config, layer_number: int):
        super(ChatGLM2Block, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layernorm_dtype = config.layernorm_compute_type
        self.compute_dtype = config.compute_dtype

        layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                               param_init_type=self.layernorm_dtype)
        self.input_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        # Self attention.
        self.self_attention = ChatGLM2SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                        param_init_type=self.layernorm_dtype)

        # MLP
        self.mlp = ChatGLM2MLP(config)

        self.dropout = get_dropout(self.hidden_dropout)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))

        self.cast = P.Cast()


    def set_select_recompute(self):
        self.input_layernorm.recompute(False)
        self.post_attention_layernorm.recompute(False)
        self.self_attention.recompute()
        self.mlp.recompute()
        self.dropout.dropout.recompute(False)
        self.cast.recompute(False)

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, batch_valid_length=None, prefix_key_value=None,
                  block_tables=None, slot_mapping=None):
        """Forward process of the transformer layer."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask first: (bs, 1, seq_len, seq_len), after: (bs, 1, 1, seq_len)
        # rotary_pos_emb: first: (seq_len, kv_channels//4, 2)， after: (1, kv_channels//4, 2)
        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.cast(hidden_states, self.layernorm_dtype)
        layernorm_output = self.input_layernorm(hidden_states)
        # fp32 -> fp16
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        # Self attention.
        attention_output = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            batch_valid_length,
            prefix_key_value,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

        # Residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout(attention_output)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout(mlp_output)
        output = residual + output

        return output


class ChatGLM2Transformer(nn.Cell):
    """Transformer class."""

    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2Transformer, self).__init__()

        self.post_layer_norm = config.post_layer_norm
        self.compute_dtype = config.compute_dtype

        # Number of layers.
        self.num_layers = config.num_layers

        self.pre_seq_len = config.pre_seq_len

        # Transformer layers.
        def build_layer(layer_number):
            return ChatGLM2Block(config, layer_number)

        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for i in range(self.num_layers):
            layer = build_layer(i + 1)
            self.layer_setting(layer, i)
            self.layers.append(layer)

        if self.post_layer_norm:
            layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                   param_init_type=config.layernorm_compute_type)
            self.final_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

    def construct(self,
                  hidden_states,
                  attention_mask,
                  rotary_pos_emb,
                  batch_valid_length=None,
                  prefix_key_values=None,
                  block_tables=None,
                  slot_mapping=None):
        """Forward process of the transformer."""
        # hidden_states -> (bs, seq_len, hs)
        # attention_mask -> (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//2, 2)， after:[1, kv_channels // 2, 2]

        if batch_valid_length is not None and isinstance(self.pre_seq_len, int):
            batch_valid_length = batch_valid_length + self.pre_seq_len

        for i in range(self.num_layers):
            prefix_key_value = None
            if prefix_key_values is not None:
                prefix_key_value = prefix_key_values[i]
            layer = self.layers[i]

            hidden_states = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                batch_valid_length=batch_valid_length,
                prefix_key_value=prefix_key_value,
                block_tables=block_tables,
                slot_mapping=slot_mapping
            )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.cast(hidden_states, self.compute_dtype)

        return hidden_states
