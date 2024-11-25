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
"""attention modules."""
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindformers.modules.transformer import OpParallelConfig
from mindformers.modules.layers import Linear
from mindformers.version_control import get_dropout

default_dpmp_config = OpParallelConfig()


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the division value."""
    if numerator % denominator != 0:
        raise ValueError(f'{numerator} is not divisible by {denominator}')
    return numerator // denominator


class RotaryEmbedding(nn.Cell):
    """
    Rotary embedding layer

    Args:
        dim (int): Hidden layer dimension number.
        base (int, optional): Rotary embedding base. Default: 10000.
        params_dtype (ms.dtype, optional): Default: mstype.float32.
        compute_dtype (ms.dtype, optional): Default: mstype.float16.
        max_seq_len (int): Max sequence length.
        parallel_config (optional): Operator parallel strategy. Default: `OpParallelConfig()`.
    """

    def __init__(self, dim, base=10000, params_dtype=mstype.float32, compute_dtype=mstype.float16, max_seq_len=512,
                 parallel_config=None):
        super(RotaryEmbedding, self).__init__()
        if not parallel_config:
            parallel_config = default_dpmp_config
        self.max_seq_len = max_seq_len
        self.params_dtype = params_dtype
        self.compute_dtype = compute_dtype
        inv_freq = 1. / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        inv_freq = inv_freq.astype(np.float16)
        t_range = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t_range, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.emb = emb

        # cos_cached: [max_seq_len, 1, dim]
        self.cos_cached = np.expand_dims(np.cos(emb), 1)
        self.sin_cached = np.expand_dims(np.sin(emb), 1)
        self.cos_cached = Tensor(self.cos_cached, params_dtype)
        self.sin_cached = Tensor(self.sin_cached, params_dtype)

        self.stride_slice = ops.StridedSlice()
        self.mul = ops.Mul()
        self.add = ops.Add()

        self.squeeze = ops.Squeeze(axis=1).shard(((1, 1, 1),))
        self.concat = P.Concat(axis=-1).shard(((parallel_config.data_parallel, 1, 1, 1),
                                               (parallel_config.data_parallel, 1, 1, 1)))

    def rotate_half(self, x):
        """rotate half"""
        x1 = x[:, :, :, : x.shape[-1] // 2]
        x2 = x[:, :, :, x.shape[-1] // 2:]
        return self.concat((-x2, x1))

    def apply_rotary_pos_emb_index(self, q, k, position_id):
        """
        apply rotary pos emb index

        Inputs:
            q, k (Tensor): Querry and key.
            position_id (Tensor): Used to identify each token's position in the list of tokens.

        returns:
            q, k (Tensor): Embedded querry and key layer.
        """
        # position_id: [bs, seq_len]
        # cos, sin: [max_seq_len, 1, hidden_size] -> [max_seq_len, hidden_size]
        #           -> [seq_len, bs, hidden_size] -> [seq_len, bs, 1, hidden_size]
        q, k = ops.cast(q, self.compute_dtype), ops.cast(k, self.compute_dtype)
        cos = ops.expand_dims(self.squeeze(self.cos_cached)[position_id], 2)
        sin = ops.expand_dims(self.squeeze(self.sin_cached)[position_id], 2)
        cos, sin = ops.cast(cos, q.dtype), ops.cast(sin, q.dtype)
        # q, k: [bs, seq_len, size_per_head, hidden_size]
        q = self.add(self.mul(q, cos), self.mul(self.rotate_half(q), sin))
        k = self.add(self.mul(k, cos), self.mul(self.rotate_half(k), sin))
        return q, k

    def construct(self, q, k, position_id):
        return self.apply_rotary_pos_emb_index(q, k, position_id)


def split_tensor_along_last_dim(tensor, num_partitions):
    """
    Split a tensor along its last dimension.
    Used in construct function.

    Arguments:
        tensor (Tensor): Input tensor.
        num_partitions (int): Number of partitions to split the tensor.
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    # Split.
    tensor_list = ops.Split(axis=last_dim, output_num=num_partitions)(tensor)

    return tensor_list


class RotaryEmbeddingFP32SoftmaxSelfAttention(nn.Cell):
    """
    Attention layer with rotary embedding and softmax

    Args:
        hidden_size (int): Hidden layer size.
        batch_size (int): Batch size.
        num_attention_heads (int): Number of attention heads.
        parallel_config: Operator parallel strategy.
        attention_dropout_prob (float, [0, 1.0]): Attention layer dropout probability.
        output_dropout_prob (float, [0, 1.0]): Output dropout probability.
        layer_id (int): Layer id.
        max_seq_len (int): Max sequence length.
        hidden_size_per_attention_head (ms.dtype, optional): Default: None.
        position_encoding_2d (ms.dtype, optional): Use 2d position encoding or not. Default: True.
        bias (bool, optional): Use bias or not. Default: True.
        params_dtype (ms.dtype, optional): Parameter data type. Default: mstype.float32.
        softmax_dtype (ms.dtype, optional): Calculate softmax data type. Default: mstype.float32.
        compute_dtype (ms.dtype, optional): Other compute data type. Default: mstype.float16.
        use_past (bool, optional): Use infer cache or not. Default: False.
    """

    def __init__(
            self,
            hidden_size,
            batch_size,
            num_attention_heads,
            parallel_config,
            attention_dropout_prob,
            output_dropout_prob,
            layer_id,
            max_seq_len=512,
            hidden_size_per_attention_head=None,
            position_encoding_2d=True,
            bias=True,
            params_dtype=mstype.float32,
            softmax_dtype=mstype.float32,
            compute_dtype=mstype.float16,
            use_past=False,
    ):
        super(RotaryEmbeddingFP32SoftmaxSelfAttention, self).__init__()
        self.params_dtype = params_dtype
        self.softmax_dtype = softmax_dtype
        self.compute_dtype = compute_dtype

        self.layer_id = layer_id
        # Per attention head values.
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        else:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.inner_hidden_size = (num_attention_heads * self.hidden_size_per_attention_head)
        self.use_past = use_past
        self.is_first_iteration = True
        self.position_encoding_2d = position_encoding_2d

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if self.position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            params_dtype=params_dtype,  # need to use float32 for proper ops.cos/ops.sin
            compute_dtype=compute_dtype,
            max_seq_len=max_seq_len,
            parallel_config=parallel_config,
        )

        # Strided linear layer.
        self.query_key_value = Linear(
            hidden_size,
            3 * self.inner_hidden_size,
            has_bias=bias,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype
        )
        self.query_key_value.shard(
            strategy_matmul=((parallel_config.data_parallel, 1),
                             (parallel_config.model_parallel, 1)),
            strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                           (parallel_config.model_parallel,))
        )

        self.attention_dropout = get_dropout(attention_dropout_prob)
        self.dense = Linear(
            self.inner_hidden_size,
            hidden_size,
            has_bias=bias,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype
        )
        self.dense.shard(
            strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
            strategy_bias=((parallel_config.data_parallel, 1), (1,)))
        self.output_dropout = get_dropout(output_dropout_prob)
        self.matmul = P.BatchMatMul()
        self.matmul.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                           (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = nn.Softmax(axis=-1)
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.split = P.Split(axis=-1, output_num=2)

        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))

        self.concat_query = P.Concat(axis=3).shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1))
        )
        self.concat_key = P.Concat(axis=3).shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1))
        )

        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(max_seq_len).reshape(1, 1, -1)  # [1, 1, config.seq_length]
            self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)  # [bs, 1, config.seq_len]
            self.seq_length = max_seq_len
            self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
            self.add = P.TensorAdd().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.sub1 = P.Sub().shard(((1,), ()))
            self.tile = P.Tile().shard(((1, 1, 1, 1),))
            self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
            self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.concat_query = P.Concat(axis=3).shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.concat_key = P.Concat(axis=3).shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def attention_fn(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            layer_id,
            scaling_attention_score=True,
    ):
        """
        calculate attention function
        """
        # seqlen, batch, head, hidden_size
        query_key_layer_scaling_coeff = F.cast(layer_id + 1, self.compute_dtype)
        if scaling_attention_score:
            query_layer_dtype = query_layer.dtype
            query_layer = F.cast(query_layer, self.compute_dtype)
            sqrt_value = F.sqrt(F.cast(self.hidden_size_per_attention_head, self.compute_dtype))
            query_layer = query_layer / (sqrt_value * query_key_layer_scaling_coeff)
            query_layer = F.cast(query_layer, query_layer_dtype)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, hn] * [b * np, hn, sk]
        matmul_result = self.matmul(
            query_layer.swapaxes(1, 2), F.transpose(key_layer, (0, 2, 3, 1))
        )

        # record original score dtype
        attention_scores_dtype = matmul_result.dtype
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result
        attention_mask = F.cast(attention_mask, mstype.bool_)
        masked_fill_constant = Tensor(-10000.0, attention_scores_dtype)
        attention_scores = attention_scores.masked_fill(attention_mask, masked_fill_constant)
        attention_scores = F.cast(attention_scores, self.softmax_dtype)
        attention_scores = attention_scores * query_key_layer_scaling_coeff
        # softmax under `softmax_dtype` mode
        attention_probs = self.softmax(attention_scores)
        # cast back to original score dtype
        attention_probs = F.cast(attention_probs, attention_scores_dtype)

        if self.training:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # matmul: [b * np, sq, sk] x [b * np, sk, hn] --> [b * np, sq, hn]
        context_layer = self.matmul(attention_probs, value_layer)
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
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (x_shape[0], x_shape[1], x_shape[-2] * x_shape[-1])
        x_merge = F.reshape(x, new_shape)
        return x_merge

    def attention_forward(self, hidden_states, mask, position_ids, layer_id, key_past=None, value_past=None,
                          batch_valid_length=None):
        """
        attention forward

        Input:
            hidden_states (Tensor): Hidden layer states.
            mask (Tensor): Same as `attention_mask`, used when batching sequences together.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            layer_id (int): Layer id.
            key_past (Tensor, optional): Default: None.
            value_past (Tensor, optional): Default: None.
            batch_valid_length (bool, optional): Default: None.

        return:
            output (Tensor): Attention output.
            layer_present (Tensor): Layer present, used for infer cache.
        """
        mixed_raw_layer = self.query_key_value(hidden_states)
        mixed_raw_layer = F.cast(mixed_raw_layer, self.compute_dtype)

        # [b, seq, (heads * 3 * hidden_size_per_head)] --> [b, seq, heads, 3 * hidden_size_per_head]
        new_tensor_shape = mixed_raw_layer.shape[:-1] + (
            self.num_attention_heads, 3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
        # [b, seq, heads, hidden_size_per_head]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = self.split(query_layer)
            k1, k2 = self.split(key_layer)
            position_ids, block_position_ids = position_ids[:, 0, :], \
                                               position_ids[:, 1, :]
            q1, k1 = self.rotary_emb(q1, k1, position_ids)
            q2, k2 = self.rotary_emb(q2, k2, block_position_ids)
            query_layer = self.concat_query((q1, q2))
            key_layer = self.concat_query((k1, k2))
        else:
            # apply rotary embed on q, k: [bs, seq,  num_heads, hidden_size]
            # position_ids: bs, 2, seq_length
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer, position_ids)

        # key and value for current token(s)
        # [bs, num_heads, hidden_size, seq_len]
        value_layer = F.transpose(value_layer, (0, 2, 1, 3))
        key_present = key_layer
        value_present = value_layer
        if self.use_past:
            # reshape
            key_present = F.transpose(key_present, (0, 2, 3, 1))
            value_present = F.transpose(value_present, (0, 1, 3, 2))
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)),
                                             self.params_dtype)  # [bs, 1, seq_len]
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key_present, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value_present, self.expand_dims(valid_length_vector, 2))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                # key_past: [batch_size, num_heads, size_per_head, seq_length]
                valid_length = batch_valid_length - 1
                valid_length = F.reshape(valid_length, (-1, 1, 1))  # [bs, 1, 1]
                # self.range: [bs, 1, config.seq_len]
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.params_dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key_present, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value_present, (1, 1, 1, self.seq_length)),
                                          self.expand_dims(valid_length_vector, 2))
                # Concat the previous saved state and current state
                key_present = self.add(key_past, current_key)  # [batch_size, num_heads, size_per_head, seq_length]
                value_present = self.add(value_past, current_value)
            # update k v for attention
            # [batch_size, num_heads, size_per_head, seq_length] -> [bs, num_heads, hidden_size, seq_len]
            key_layer = F.transpose(key_present, (0, 3, 1, 2))
            # [batch_size, num_heads, size_per_head, seq_length] -> [bs, num_heads, seq_len, hidden_size]
            value_layer = F.transpose(value_present, (0, 1, 3, 2))

        layer_present = (key_present, value_present)

        # [batch_size, num_heads, size_per_head, seq_length] -> [seq_len, bs, num_heads, hidden_size]
        query_layer = F.cast(query_layer, self.compute_dtype)
        key_layer = F.cast(key_layer, self.compute_dtype)
        value_layer = F.cast(value_layer, self.compute_dtype)

        context_layer = self.attention_fn(query_layer, key_layer, value_layer, mask, layer_id, True)

        output = self.dense(context_layer)
        output = F.cast(output, self.params_dtype)

        if self.training:
            output = self.output_dropout(output)

        return output, layer_present

    def construct(self, hidden_states, mask, position_ids, layer_id, key_past=None, value_past=None,
                  batch_valid_length=None):
        return self.attention_forward(hidden_states, mask, position_ids, layer_id, key_past, value_past,
                                      batch_valid_length)
