# Copyright 2025 Huawei Technologies Co., Ltd
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
"""SelfAttention."""
import math
from typing import Union
import mindspore.common.dtype as mstype
from mindspore import Tensor, mint, nn, ops

from mindformers.experimental.infer.core import get_attn_mask_func
from mindformers.experimental.infer.core.rotary_embedding import RotaryEmbedding
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.graph.transformer.spec_utils import (
    ModuleSpec, build_module
)
from mindformers.experimental.parallel_core.pynative.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax
)
from mindformers.experimental.parallel_core.pynative.utils import divide


class SelfAttentionSubmodules:
    """Configuration class for specifying the submodules of a self-attention."""
    def __init__(self,
                 linear_qkv: Union[ModuleSpec, type] = None,
                 core_attention: Union[ModuleSpec, type] = None,
                 linear_proj: Union[ModuleSpec, type] = None,
                 linear_q: Union[ModuleSpec, type] = None,
                 linear_k: Union[ModuleSpec, type] = None,
                 linear_v: Union[ModuleSpec, type] = None):
        self.linear_qkv = linear_qkv
        self.core_attention = core_attention
        self.linear_proj = linear_proj
        self.linear_q = linear_q
        self.linear_k = linear_k
        self.linear_v = linear_v


class CoreAttention(nn.Cell):
    """
    Get the weighted score along the seq_length.

    Args:
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.

    Inputs:
        - **query** (Tensor) - Tensor of query matrix.
        - **key** (Tensor) - Tensor of key matrix.
        - **value** (Tensor) - Tensor of value matrix.
        - **attention_mask** (Tensor) - Tensor of attention mask matrix.

    Outputs:
        - **attn_output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, layer_number, config, attn_mask_type=None):
        super().__init__()
        if attn_mask_type:
            raise NotImplementedError(
                "For CoreAttention, `attn_mask_type` is not supported for now."
            )
        self.config = config
        self.layer_index = max(1, layer_number)
        self.compute_dtype = self.config.compute_dtype
        self.softmax_compute_dtype = self.config.softmax_compute_dtype
        self.sequence_parallel = self.config.sequence_parallel
        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)

        coeff = None
        norm_factor = math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_index
            norm_factor *= coeff
        self.inv_norm_factor = Tensor(1.0 / norm_factor,
                                      dtype=self.compute_dtype)

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = ScaleMaskSoftmax(
            self.mask_func, softmax_compute_type=self.softmax_compute_dtype)

        self.attention_dropout = mint.nn.Dropout(
            p=self.config.attention_dropout_rate)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward process of the CoreAttention."""
        # score: [B, N, S_q, S_k]
        score = ops.bmm(query_layer, key_layer.transpose(0, 1, 3, 2))
        score = mint.mul(score, self.inv_norm_factor)

        # attention scores and attention mask [B, N, S_q, S_k]
        attention_probs = self.scale_mask_softmax(score, attention_mask)

        attention_probs = self.attention_dropout(attention_probs)

        # [B, N, S_q, S_k] * [B, N, S_v, D] -> [B, N, S_q, D]
        weighted_values = ops.bmm(attention_probs, value_layer)

        return weighted_values


class SelfAttention(nn.Cell):
    """
    SelfAttention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        submodules (SelfAttentionSubmodules): submodules of SelfAttention
        config (TransformerConfig): Configuration.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config,
                 submodules: SelfAttentionSubmodules,
                 layer_number,
                 attn_mask_type=None):
        super().__init__(config)
        if attn_mask_type:
            raise NotImplementedError(
                "For SelfAttention, `attn_mask_type` is not supported for now."
            )
        self.config = config
        self.submodules = submodules
        self.layer_index = max(1, layer_number)
        self.param_init_type = self.config.param_init_type
        self.compute_dtype = self.config.compute_dtype
        self.is_prefill = True
        self.qkv_concat = self.config.qkv_concat

        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = (self.num_attention_heads
                             if config.num_query_groups is None else
                             config.num_query_groups)
        self.hidden_size = self.config.hidden_size
        self.head_dim = divide(self.hidden_size, self.num_heads)
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.n_rep = divide(self.num_heads, self.kv_num_heads)

        self.sequence_parallel = self.config.sequence_parallel
        self.use_flash_attention = self.config.use_flash_attention
        self.norm_factor = math.sqrt(self.head_dim)

        self.tp_group_size = get_tp_world_size()
        self.num_heads_per_partition = divide(self.num_heads,
                                              self.tp_group_size)

        self.use_gqa = (self.num_heads != self.kv_num_heads)

        if self.use_gqa:
            self._check_gqa_valid()
            self.kv_num_heads_per_partition = divide(self.kv_num_heads,
                                                     self.tp_group_size)
            self.repeat_num = divide(self.num_heads, self.kv_num_heads)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        self._init_self_attn()

        self.reshape = ops.Reshape()

        # submodules.linear_proj: RowParallelLinear
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config,
            bias=self.config.out_proj_has_bias,
            transpose_b=True,
            param_init_type=self.param_init_type,
            compute_dtype=self.compute_dtype,
        )

        if self.use_flash_attention:
            kv_cache_shape = (self.config.num_blocks, self.config.block_size,
                              self.kv_num_heads_per_partition, self.head_dim)
            self.flash_attention = build_module(
                submodules.core_attention,
                head_num=self.num_heads_per_partition,
                kv_cache_shape=kv_cache_shape,
                head_dim=self.head_dim,
                kv_head_num=self.kv_num_heads_per_partition,
                scale_value=1.0 / self.norm_factor,
                next_tokens=0,
                compute_dtype=self.compute_dtype,
            )
        else:
            self.core_attention = build_module(submodules.core_attention,
                                               config=self.config,
                                               layer_number=self.layer_index)

        self.rotary_embedding = RotaryEmbedding(kv_channels=self.head_dim,
                                                rotary_cos_format=2)

    def construct(self,
                  x,
                  batch_valid_length=None,
                  kv_cache=None,
                  block_tables=None,
                  slot_mapping=None,
                  rotary_pos_cos=None,
                  rotary_pos_sin=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  context_lens_tensor=None,
                  attn_mask=None,
                  alibi_mask=None,
                  prefix_keys_values=None):
        """Forward process of the SelfAttention."""
        # hidden_states: [B, S, H]
        ori_dtype = x.dtype
        bs, seq_len, _ = x.shape

        # apply query, key, value projection
        if self.sequence_parallel:
            seq_len = seq_len * self.tp_group_size
        if self.qkv_concat:
            qkv = self.cast(self.linear_qkv(x), self.compute_dtype)
            # [B, S, H] --> [B, S, N, D]
            reshape_qkv = self.reshape(
                qkv, (bs, seq_len, self.kv_num_heads_per_partition,
                      (self.n_rep + 2) * self.head_dim))
            query, key, value = mint.split(
                reshape_qkv,
                (self.head_dim * self.n_rep, self.head_dim, self.head_dim), -1)
            # [B, S, N, D] --> [B, S, H] ReshapeAndCache only supports 'BSH'
            query = self.reshape(query,
                                 (bs, seq_len, self.hidden_size_per_partition))
            key = self.reshape(
                key, (bs, seq_len, self.kv_hidden_size_per_partition))
            value = self.reshape(
                value, (bs, seq_len, self.kv_hidden_size_per_partition))
        else:
            query = self.cast(self.linear_q(x), self.compute_dtype)
            key = self.cast(self.linear_k(x), self.compute_dtype)
            value = self.cast(self.linear_v(x), self.compute_dtype)

        # [B, S, H]
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            query, key = self.rotary_embedding(query, key, rotary_pos_cos,
                                               rotary_pos_sin,
                                               batch_valid_length)

        if prefix_keys_values is not None:
            prefix_len = prefix_keys_values.shape[2]
            slot_mapping = slot_mapping + self.cast(mint.ne(slot_mapping, -1),
                                                    mstype.int32) * prefix_len
            if self.is_first_iteration:
                key, value = self._cat_prefix(key, value, prefix_keys_values)

        if self.use_flash_attention:
            bs, seq_len, _ = query.shape
            context_layer = self.flash_attention(
                query, key, value, kv_cache, slot_mapping, block_tables,
                batch_valid_length, context_lens_tensor, actual_seq_qlen,
                actual_seq_kvlen, attn_mask, alibi_mask)
            context_layer = self.reshape(
                context_layer,
                (bs, seq_len, self.num_heads_per_partition * self.head_dim))
        else:
            # [B, S, H] --> [B, S, N, D]
            query = query.reshape(bs, seq_len, -1, self.head_dim)
            key = key.reshape(bs, seq_len, -1, self.head_dim)
            value = value.reshape(bs, seq_len, -1, self.head_dim)
            # [B, S, N_kv, D] --> [B, S, N, D]
            if self.use_gqa:
                key = mint.repeat_interleave(key,
                                             repeats=self.repeat_num,
                                             dim=2)
                value = mint.repeat_interleave(value,
                                               repeats=self.repeat_num,
                                               dim=2)
            # [B, S, N, D] --> [B, N, S, D]
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            context_layer = self.core_attention(query, key, value, attn_mask)
            # [B, N, S, D] --> [B, S, H]
            context_layer = context_layer.transpose(0, 2, 1, 3).reshape(
                bs, seq_len, self.hidden_size_per_partition)

        # apply output projection
        output = self.linear_proj(context_layer)
        output = self.cast(output, ori_dtype)
        return output

    def _cat_prefix(self, key, value, prefix_keys_values):
        """
        concat prefix_keys_values to key and value
        prefix_keys_values: shape(2, bs, pre_len, num_heads * kv_channels)
        """
        if prefix_keys_values is not None:
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.cast(past_key, key.dtype)
            past_value = self.cast(past_value, value.dtype)
            key = ops.concat((past_key, key), 1)
            value = ops.concat((past_value, value), 1)
        return key, value

    def _check_gqa_valid(self):
        """check whether the config is valid for grouped-query-attention"""
        if self.num_heads % self.kv_num_heads != 0:
            raise ValueError(f"num_heads must be divisible by kv_num_heads, "
                             f"but got num_heads {self.num_heads} "
                             f"and kv_num_heads {self.kv_num_heads}")
        if self.kv_num_heads % self.tp_group_size != 0:
            raise ValueError(
                f"kv_num_heads must be divisible by tp_group_size, "
                f"but got kv_num_heads {self.kv_num_heads} "
                f"and kv_num_heads {self.tp_group_size}")

    def _init_self_attn(self):
        """init qkv linears of self-attention"""
        self.hidden_size_per_partition = divide(self.hidden_size,
                                                self.tp_group_size)
        self.kv_hidden_size_per_partition = divide(self.kv_hidden_size,
                                                   self.tp_group_size)
        if self.qkv_concat:
            if self.submodules.linear_qkv is not None:
                # ColumnParallelLinear
                self.linear_qkv = build_module(
                    self.submodules.linear_qkv,
                    self.hidden_size,
                    self.hidden_size + 2 * self.kv_hidden_size,
                    config=self.config,
                    bias=self.config.qkv_has_bias,
                    gather_output=False,
                    transpose_b=True,
                    param_init_type=self.param_init_type,
                    compute_dtype=self.compute_dtype,
                )
        else:
            if self.submodules.linear_q is not None:
                # ColumnParallelLinear
                self.linear_q = build_module(
                    self.submodules.linear_q,
                    self.hidden_size,
                    self.hidden_size,
                    config=self.config,
                    bias=self.config.qkv_has_bias,
                    gather_output=False,
                    transpose_b=True,
                    param_init_type=self.param_init_type,
                    compute_dtype=self.compute_dtype,
                )
            if self.submodules.linear_k is not None:
                # ColumnParallelLinear
                self.linear_k = build_module(
                    self.submodules.linear_k,
                    self.hidden_size,
                    self.kv_hidden_size,
                    config=self.config,
                    bias=self.config.qkv_has_bias,
                    gather_output=False,
                    transpose_b=True,
                    param_init_type=self.param_init_type,
                    compute_dtype=self.compute_dtype,
                )
            if self.submodules.linear_v is not None:
                # ColumnParallelLinear
                self.linear_v = build_module(
                    self.submodules.linear_v,
                    self.hidden_size,
                    self.kv_hidden_size,
                    config=self.config,
                    bias=self.config.qkv_has_bias,
                    gather_output=False,
                    transpose_b=True,
                    param_init_type=self.param_init_type,
                    compute_dtype=self.compute_dtype,
                )
