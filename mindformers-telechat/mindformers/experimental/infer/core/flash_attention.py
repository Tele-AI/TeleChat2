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
"""Flash Attention Layer"""
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper) and PagedAttention.
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
        - **kv_cache** (Tensor) - reserved field
        - **slot_mapping** (Tensor) - Store token cache physical slot index.
        - **block_tables** (Tensor) - The block mapping table with data type of int32.
        - **batch_valid_length** (Tensor) -  In incremental inference, a tensor used for calculating the index
          of the previous step. It is of int32 type and has a shape of [batch_size].
        - **context_lens_tensor** (Tensor) - The context length of each sequence with data type of int32.
        - **actual_seq_qlen** (Union[List[int64], Tuple[int64], None]) - Size of query corresponding to each batch,
          array with increasing values and the last value equal to T1.
        - **actual_seq_kvlen** (Union[List[int64], Tuple[int64], None]) - Size of key and value corresponding to each
          batch, array with increasing values and the last value equal to T2.
        - **attn_mask** (Union[Tensor, None]) - The attention mask tensor.
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization.
          Input tensor of shape :math:`(B, N1, S1, S2)`, :math:`(1, N1, S1, S2)`, :math:`(B, N1, 1024, S2)`,
          :math:`(1, N1, 1024, S2)` or (1024, 1024).
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            head_num,
            kv_cache_shape,
            head_dim=None,
            kv_head_num=None,
            keep_prob=1.0,
            scale_value=1.0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            sparse_mode=0,
            use_alibi_mask=False,
            compute_dtype=mstype.float16,
            input_layout="TH"
    ):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.kv_head_num = kv_head_num
        self.enable_dropout = keep_prob < 1.0
        self.sparse_mode = sparse_mode
        self.use_alibi_mask = use_alibi_mask
        self.is_prefill = True
        self.input_layout = input_layout

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, self.kv_head_num)

        if self.use_alibi_mask:
            self.alibi_rescale_factor = Tensor([1.0 / scale_value],
                                               dtype=compute_dtype)
            self.alibi_rescale_mul = ops.Mul()

        self.key_cache = Parameter(initializer('zeros', kv_cache_shape,
                                               compute_dtype),
                                   name="key_cache",
                                   requires_grad=False)
        self.value_cache = Parameter(initializer('zeros', kv_cache_shape,
                                                 compute_dtype),
                                     name="value_cache",
                                     requires_grad=False)

    # pylint: disable=W0613
    def construct(self,
                  query,
                  key,
                  value,
                  kv_cache=None,
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  attn_mask=None,
                  alibi_mask=None,
                  padding_mask=None,
                  prefix=None):
        """Forward process of the FlashAttention."""

        key_cache = self.key_cache
        value_cache = self.value_cache
        self.reshape_and_cache(key, value, self.key_cache, self.value_cache,
                               slot_mapping)

        if self.is_prefill:

            if self.use_alibi_mask and alibi_mask is not None:
                alibi_mask = self.alibi_rescale_mul(
                    alibi_mask,
                    F.cast(self.alibi_rescale_factor, alibi_mask.dtype))
            else:
                alibi_mask = None

            if self.input_layout == "TH":
                query = query.reshape((-1, self.head_num * self.head_dim))
                key = key.reshape((-1, self.kv_head_num * self.head_dim))
                value = value.reshape((-1, self.kv_head_num * self.head_dim))

            if self.input_layout == "TNH":
                query = query.reshape((-1, self.head_num, self.head_dim))
                key = key.reshape((-1, self.kv_head_num, self.head_dim))
                value = value.reshape((-1, self.kv_head_num, self.head_dim))

            _, _, _, output = self.flash_attention(query, key, value,
                                                   alibi_mask, None,
                                                   padding_mask, attn_mask,
                                                   prefix, actual_seq_qlen,
                                                   actual_seq_kvlen)
            return output

        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None,
                                    None, attn_mask,
                                    batch_valid_length - context_lens_tensor)
