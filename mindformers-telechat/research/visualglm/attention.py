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
"""glm attention adaptor for visualglm."""


from mindspore import ops
from mindspore.ops import functional as F

from mindformers.models.glm.attention import RotaryEmbeddingFP32SoftmaxSelfAttention


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


def transpose_for_scores(raw_tensor, last_size):
    """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
    size [b, np, s, hn].
    """
    new_tensor_shape = raw_tensor.shape[:-1] + (-1, last_size)
    raw_tensor = raw_tensor.view(*new_tensor_shape)
    return raw_tensor


class SelfAttentionAdapter(RotaryEmbeddingFP32SoftmaxSelfAttention):
    """
    RotaryEmbeddingFP32SoftmaxSelfAttention adaptor for visualglm.
    """

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

        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = \
            split_tensor_along_last_dim(mixed_raw_layer, 3)
        # [1, 64, 32, 128]
        query_layer = transpose_for_scores(mixed_query_layer, self.hidden_size_per_attention_head)
        key_layer = transpose_for_scores(mixed_key_layer, self.hidden_size_per_attention_head)
        value_layer = transpose_for_scores(mixed_value_layer, self.hidden_size_per_attention_head)

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
