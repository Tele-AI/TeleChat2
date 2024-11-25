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

"""Paged Attention Manager for inference."""
import math

import mindspore.common.dtype as mstype
from mindspore import nn, Parameter
from mindspore.common.tensor import Tensor
from mindspore import ops as P
from mindspore.common.initializer import Zero


class PagedAttentionMgr(nn.Cell):
    """Paged Attention Manager."""

    def __init__(self,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 kv_shape,
                 compute_dtype=mstype.float16,
                 parallel_decoding=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.key_cache = Parameter(Tensor(shape=kv_shape, dtype=compute_dtype, init=Zero()), name="key_cache",
                                   requires_grad=False)
        self.value_cache = Parameter(Tensor(shape=kv_shape, dtype=compute_dtype, init=Zero()), name="value_cache",
                                     requires_grad=False)

        self.reshape_and_cache = P.auto_generate.ReshapeAndCache()
        self.paged_attention = P.auto_generate.PagedAttention(self.n_heads,
                                                              self.scale_value,
                                                              self.n_kv_heads)
        self.paged_attention_with_alibi = P.auto_generate.PagedAttentionMask(self.n_heads,
                                                                             self.scale_value,
                                                                             self.n_kv_heads)
        self.parallel_decoding = parallel_decoding

    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)

    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        if self.parallel_decoding:
            return self.paged_attention(query, self.key_cache, self.value_cache, block_tables, batch_valid_length,
                                        None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, self.key_cache, self.value_cache, block_tables, batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.paged_attention_with_alibi(query, self.key_cache, self.value_cache,
                                               block_tables, batch_valid_length, None, None, alibi_tensor)

    def shard(self, parallel_config):
        """The shard strategy."""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        mp = 1 if parallel_config is None else parallel_config.model_parallel
        self.reshape_and_cache.shard(((dp, 1, mp), (dp, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (1,)))
        if self.parallel_decoding:
            self.paged_attention.shard(((dp, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (dp, 1), (dp,), (dp, 1), (1,)))
        else:
            self.paged_attention.shard(((dp, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (dp, 1), (dp,)))
        self.paged_attention_with_alibi.shard(((dp, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (dp, 1), (dp,),
                                               (dp, mp, 1, 1)))
