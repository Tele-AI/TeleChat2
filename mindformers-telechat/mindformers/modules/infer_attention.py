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
"""Infer Attention Layer"""
import math

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P

from mindformers.modules import PagedAttentionMgr
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.layers import RotaryEmbedding
from mindformers.tools.utils import get_disable_custom_fa, get_use_rope_self_define


class InferRotaryEmbedding(Cell):
    r"""
    Infer Rotary Position Embedding.

    Args:
            rotary_cos_format (int): - Choose the rotary embedding cos format. Default 0.

    Inputs:
            query: the query matrix.
            key: the key matrix.
            freqs_cis: The precompute freqs and mask for rotary position embedding used in attention.
            batch_valid_length: Int32 tensor with shape [batch_size] the past calculated the index.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, rotary_cos_format=0):
        super().__init__()
        self.rotary_cos_format = rotary_cos_format
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(self.rotary_cos_format)

    def construct(self, query: Tensor, key: Tensor, freqs_cis, batch_valid_length):
        """Forward of rotary position embedding."""
        freqs_cos, freqs_sin, _ = freqs_cis
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)

    def shard(self, parallel_config):
        """sharding for rotary embedding"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        mp = 1 if parallel_config is None else parallel_config.model_parallel
        self.rotary_embedding_op.shard(((dp, 1, mp), (dp, 1, mp), (1, 1), (1, 1), (dp,)))


class InferAttention(Cell):
    """Infer Attention Layer.

    This function contains the InferAttention primitives used with FlashAttention and PagedAttention for kbk infer.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D

    Args:
        n_head (int): The head num of query.
        head_dim (int): The dim of head.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        Default: "BSH".
        sparse_mode (int): Indicates sparse mode. Default 0.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        block_size (int): Block size for paged attention.
        num_blocks (int): Block num for paged attention.
        use_flash_attention (bool): The value is True if chosen to use flash attention. Default: True.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        use_rope_rotary_emb (bool): If use rotary embedding. Default True.
        rotary_cos_format (int): Choose the rotary embedding cos format. Default 0.
        rotary_dtype (mstype): Compute dtype for rope op. Default mstype.float16.
        compute_dtype (mstype): Compute dtype for infer attention. Default mstype.float16.


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)` or :math:`(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)` or :math:`(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)` or :math:`(B, N2, S2, D)`.
        - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
          Used for incremental prediction when the use_past is True. Default None.
        - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
        - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
        - **freqs_cos** (Tensor[float16, bfloat16]) - The precompute freqs cos for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **freqs_sin** (Tensor[float16, bfloat16]) - The precompute freqs sin for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, :math:`(B, 1, S1, S2)`,
          :math:`(S1, S2)` or (2048, 2048).
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization.
          Input tensor of shape :math:`(B, N1, S1, S2)`, :math:`(1, N1, S1, S2)`, :math:`(B, N1, 1024, S2)`,
          :math:`(1, N1, 1024, S2)` or (1024, 1024).

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    Examples:
        >>> import numpy as np
        >>> import math
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindformers.modules.infer_attention import InferAttention
        >>> bsz, head_num, seq_len, head_dim = 1, 16, 4096, 128
        >>> n_kv_head = 16
        >>> block_size = 1024
        >>> num_blocks = 16
        >>> hidden_size = head_num * head_dim
        >>> query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> key = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> value = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
        >>> batch_valid_length = Tensor(np.ones((bsz, 1)), mstype.int32)
        >>> block_tables = Tensor(np.ones((bsz, num_blocks)), mstype.int64)
        >>> slot_mapping = Tensor(np.ones((bsz,)), mstype.int32)
        >>> freqs_cos = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
        >>> freqs_sin = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
        >>> attn_mask = Tensor(np.ones((bsz, 1, seq_len, seq_len)), mstype.uint8)
        >>> infer_attention = InferAttention(head_num,
                                             head_dim,
                                             n_kv_head,
                                             scale_value=1. / math.sqrt(head_dim),
                                             pre_tokens=65536,
                                             next_tokens=0,
                                             block_size=block_size,
                                             num_blocks=num_blocks)
        >>> output = infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping, attn_mask)
        >>> print(output.shape)
        (1, 4096, 2048)
    """

    def __init__(self,
                 n_head,
                 head_dim,
                 n_kv_head,
                 pa_n_head_split=None,
                 pa_n_kv_head_split=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 block_size=16,
                 num_blocks=1024,
                 use_flash_attention=True,
                 use_alibi_mask=False,
                 use_rope_rotary_emb=True,
                 rotary_cos_format=0,
                 rotary_dtype=mstype.float32,
                 compute_dtype=mstype.float16,
                 parallel_decoding=False,
                 ):
        super(InferAttention, self).__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_kv_head = n_kv_head
        self.pa_n_head_split = pa_n_head_split if pa_n_head_split is not None else n_head
        self.pa_n_kv_head_split = pa_n_kv_head_split if pa_n_kv_head_split is not None else n_kv_head
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.sparse_mode = sparse_mode
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_flash_attention = use_flash_attention
        self.use_alibi_mask = use_alibi_mask
        self.use_rope_rotary_emb = use_rope_rotary_emb
        self.use_rope_self_define = get_use_rope_self_define()
        self.rotary_cos_format = rotary_cos_format
        self.rotary_dtype = rotary_dtype
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.shape = P.Shape()
        self.tile_kv = P.Tile()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)
        self.not_equal = P.NotEqual()
        self.n_rep = self.n_head // self.n_kv_head
        if self.use_alibi_mask:
            self.add_alibi = P.Add()

        self.disable_custom_fa = get_disable_custom_fa()
        self.use_attention_mask = False
        self.input_layout = "BSH"

        if self.disable_custom_fa:
            self.use_attention_mask = True
            self.input_layout = "TH"

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.n_head,
                                                  pre_tokens=self.pre_tokens,
                                                  next_tokens=self.next_tokens,
                                                  keep_prob=self.keep_prob,
                                                  scale_value=self.scale_value,
                                                  sparse_mode=self.sparse_mode,
                                                  use_attention_mask=self.use_attention_mask,
                                                  use_alibi_mask=self.use_alibi_mask,
                                                  input_layout=self.input_layout)

        kv_shape = (self.num_blocks, self.block_size, self.n_kv_head, self.head_dim)
        self.paged_attention_mgr = PagedAttentionMgr(self.pa_n_head_split,
                                                     self.head_dim,
                                                     self.pa_n_kv_head_split,
                                                     kv_shape,
                                                     compute_dtype=self.compute_dtype,
                                                     parallel_decoding=parallel_decoding,
                                                     )
        if use_rope_rotary_emb:
            if self.use_rope_self_define:
                self.rotary_embedding = RotaryEmbedding(self.head_dim, self.rotary_dtype)
            else:
                self.rotary_embedding = InferRotaryEmbedding(self.rotary_cos_format)
        self.parallel_decoding = parallel_decoding

    def _core_attention(self, query, key, value, attn_mask, alibi_mask=None):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attn_mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
            alibi_mask: the alibi matrix
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        if alibi_mask is not None:
            score = self.add_alibi(score, alibi_mask)
        score = self.add(attn_mask, score)

        attention_probs = self.softmax(self.cast(score, mstype.float32))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.compute_dtype), value)
        # [bs, n_head, seq/1, head_dim]
        weighted_values = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return weighted_values

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
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _apply_rotary_pos_emb(self, query, key, freqs_cis, batch_valid_length):
        """
        apply rotary pos embedding

        Inputs:
            query: the query matrix
            key: the key matrix
            freqs_cis: The precompute freqs and mask for rotary position embedding used in attention.
            batch_valid_length: Int32 tensor with shape [batch_size] the past calculated the index.
        Outputs:
            query: the query matrix
            key: the key matrix
        """
        if self.use_rope_self_define:
            bs, seq_len, _ = query.shape
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            if not self.is_first_iteration:
                freqs_cos, freqs_sin, swap_mask = freqs_cis
                freqs_cos = self.reshape(freqs_cos, (bs, 1, seq_len, self.head_dim))
                freqs_sin = self.reshape(freqs_sin, (bs, 1, seq_len, self.head_dim))
                freqs_cis = (freqs_cos, freqs_sin, swap_mask)
            query, key = self.rotary_embedding(query, key, freqs_cis)  # dp, mp, 1, 1
            query = self._merge_heads(query)
            key = self._merge_heads(key)
        else:
            query, key = self.rotary_embedding(query, key, freqs_cis, batch_valid_length)  # dp, mp, 1, 1
        return query, key

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

    def _prefill_attention(self, query, key, value, attn_mask, alibi_mask, actual_seq_qlen=None, actual_seq_kvlen=None):
        """
        prefill attention
        """
        if self.use_flash_attention:
            if self.disable_custom_fa:
                bs, seq_len, _ = query.shape
                query = self.reshape(query, (-1, self.n_head * self.head_dim))
                key = self.reshape(key, (-1, self.n_kv_head * self.head_dim))
                value = self.reshape(value, (-1, self.n_kv_head * self.head_dim))
                output = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None, actual_seq_qlen,
                                              actual_seq_kvlen)  # B*S, N, D
                output = self.reshape(output, (bs, seq_len, self.n_head * self.head_dim))  # B, S, H
                return output
            return self.flash_attention(query, key, value, attn_mask, alibi_mask)
        bs, seq_len, _ = query.shape
        key_seq_len = key.shape[1]
        value_seq_len = value.shape[1]
        query = self.transpose(self.reshape(query, (bs, -1, self.n_head, self.head_dim)), (0, 2, 1, 3))
        key = self.transpose(self.reshape(key, (bs, key_seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
        value = self.transpose(self.reshape(value, (bs, value_seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        return self._core_attention(query, key, value, attn_mask, alibi_mask)

    def _incre_attention(self, query, batch_valid_length, block_tables, alibi_mask=None, attn_mask=None,
                         q_seq_lens=None):
        if self.use_alibi_mask:
            return self.paged_attention_mgr.paged_attn_with_alibi(query, batch_valid_length, block_tables, alibi_mask)
        return self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables, attn_mask=attn_mask,
                                                   q_seq_lens=q_seq_lens)

    def construct(self, query, key, value, batch_valid_length, block_tables, slot_mapping, freqs_cis=None,
                  attn_mask=None, alibi_mask=None, prefix_keys_values=None, q_seq_lens=None):
        """Forward process of the Infer Attention Cell"""
        if self.use_rope_rotary_emb:
            query, key = self._apply_rotary_pos_emb(query, key, freqs_cis, batch_valid_length)

        if prefix_keys_values is not None:
            prefix_len = prefix_keys_values.shape[2]
            slot_mapping = slot_mapping + self.cast(self.not_equal(slot_mapping, -1), mstype.int32) * prefix_len
            if self.is_first_iteration:
                key, value = self._cat_prefix(key, value, prefix_keys_values)

        key_out = self.paged_attention_mgr(key, value, slot_mapping)
        query = ops.depend(query, key_out)

        if self.is_first_iteration:
            return self._prefill_attention(query, key, value, attn_mask, alibi_mask, batch_valid_length,
                                           batch_valid_length)
        return self._incre_attention(query, batch_valid_length, block_tables, alibi_mask, attn_mask, q_seq_lens)

    def shard(self, parallel_config):
        """Parallel strategy configuratiuon interface."""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        mp = 1 if parallel_config is None else parallel_config.model_parallel

        if self.use_rope_rotary_emb:
            self.rotary_embedding.shard(parallel_config)
        if self.use_flash_attention:
            self.flash_attention.shard(parallel_config)
        self.paged_attention_mgr.shard(parallel_config)

        self.transpose.shard(((dp, 1, mp, 1),))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        if self.use_alibi_mask:
            self.add_alibi.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.softmax.shard(((dp, mp, 1, 1),))
        self.tile_kv.shard(((dp, mp, 1, 1),))
        return self
