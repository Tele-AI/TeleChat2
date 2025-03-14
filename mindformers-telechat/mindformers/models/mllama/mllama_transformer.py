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
"""LLaMA transformer Layer's APIs."""
from typing import Tuple
import math

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.models.llama.llama_transformer import LLamaAttention, LLamaDecodeLayer
from mindformers.models.utils import predict_lazy_inline
from mindformers.modules.flash_attention import FlashAttention


class MllamaCrossAttention(LLamaAttention):
    """MllamaCrossAttention"""

    def __init__(self, seq_length, layernorm_compute_dtype, norm_eps, fused_kernel, **kwargs):
        super().__init__(seq_length=seq_length, **kwargs)
        parallel_config = kwargs.get("parallel_config")

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        self.wq_norm = LlamaRMSNorm(self.head_dim, norm_eps, compute_type=layernorm_compute_dtype,
                                    fused_kernel=fused_kernel)
        self.wk_norm = LlamaRMSNorm(self.head_dim, norm_eps, compute_type=layernorm_compute_dtype,
                                    fused_kernel=fused_kernel)
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.wq_norm.shard((dp, mp, cp, 1))
            self.wk_norm.shard((dp, mp, cp, 1))

        if self.use_flash_attention:
            self.input_layout = "BSH" if cp > 1 else "BNSD"
            self.sparse_mode = 1
            self.flash_attention = FlashAttention(head_num=self.n_head,
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  input_layout=self.input_layout,
                                                  keep_prob=1.,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  sparse_mode=self.sparse_mode,
                                                  use_attention_mask=True,
                                                  use_ring_attention=self.use_ring_attention)
            self.flash_attention.shard(parallel_config)

    # pylint: disable=W0221
    def construct(self, x: Tensor, cross_attention_states: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None,
                  batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, q_seq_lens=None,
                  kv_mask=None, seq_chunk=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        if not self.rmsnorm_compute_2d:
            bs, seq_len, _ = self.shape(x)
        else:
            seq_len = self.seq_length
            bs = self.shape(x)[0] // seq_len

        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key = self.cast(self.wk(cross_attention_states), self.dtype)  # dp, 1 -> dp,mp
        value = self.cast(self.wv(cross_attention_states), self.dtype)  # dp, 1 -> dp, mp

        # key and value for current token(s)
        query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
        query = self.wq_norm(query)
        key = self.transpose(self.reshape(key, (bs, -1, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
        key = self.wk_norm(key)

        value = self.transpose(self.reshape(value, (bs, -1, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
        key, value = self._cat_prefix(key, value, prefix_keys_values)

        if self.use_flash_attention:
            # with ulysses context parallel, insert all to all after FA
            if self.context_parallel > 1 and self.cp_ds > 1:
                context_layer = self.flash_attention(query, key, value, mask)
                context_layer = self._ulysses_context_layer_a2a(context_layer)
            elif self.context_parallel > 1:
                context_layer = self.flash_attention(query, key, value, mask)
            else:
                context_layer = self.flash_attention(query, key, value, mask)
                context_layer = self._merge_heads(context_layer)
        else:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            context_layer = self._attn(query, key, value, mask)

        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(context_layer)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)
        return output


# pylint: disable=C0326
class MllamaCrossAttentionDecoderLayer(LLamaDecodeLayer):
    """MllamaCrossAttentionDecoderLayer"""

    @predict_lazy_inline
    def __init__(self, seq_length, layer_id, **kwargs):
        super().__init__(seq_length=seq_length, layer_id=layer_id, **kwargs)
        dim = kwargs.get("dim")
        n_heads = kwargs.get("n_heads")
        n_kv_heads = kwargs.get("n_kv_heads")
        qkv_concat = kwargs.get("qkv_concat")
        compute_dtype = kwargs.get("compute_dtype")
        param_init_type = kwargs.get("param_init_type")
        softmax_compute_dtype = kwargs.get("softmax_compute_dtype")
        layernorm_compute_dtype = kwargs.get("layernorm_compute_dtype")
        rotary_dtype = kwargs.get("rotary_dtype")
        qkv_has_bias = kwargs.get("qkv_has_bias")
        attn_proj_has_bias = kwargs.get("attn_proj_has_bias")
        is_dynamic = kwargs.get("is_dynamic")
        use_rope_slice = kwargs.get("use_rope_slice")
        use_flash_attention = kwargs.get("use_flash_attention")
        use_ring_attention = kwargs.get("use_ring_attention")
        use_attn_mask_compression = kwargs.get("use_attn_mask_compression")
        rmsnorm_compute_2d = kwargs.get("rmsnorm_compute_2d")
        block_size = kwargs.get("block_size")
        num_blocks = kwargs.get("num_blocks")
        batch_size = kwargs.get("batch_size")
        parallel_config = kwargs.get("parallel_config")
        init_method_std = kwargs.get("init_method_std")
        chunk_prefill = kwargs.get("chunk_prefill")
        parallel_decoding = kwargs.get("parallel_decoding")
        norm_eps = kwargs.get("norm_eps")
        fused_kernel = kwargs.get("fused_kernel")

        self.cross_attn_attn_gate = Parameter(initializer("zeros", [1], mstype.float32))
        self.cross_attn_ff_gate = Parameter(initializer("zeros", [1], mstype.float32))
        self.tanh = P.Tanh()
        self.mul = P.Mul()
        self.mul1 = P.Mul()
        self.slice = P.StridedSlice()

        self.attention = MllamaCrossAttention(seq_length,
                                              layernorm_compute_dtype,
                                              norm_eps,
                                              fused_kernel,
                                              dim=dim,
                                              n_heads=n_heads,
                                              n_kv_heads=n_kv_heads,
                                              qkv_concat=qkv_concat,
                                              compute_dtype=compute_dtype,
                                              softmax_compute_dtype=softmax_compute_dtype,
                                              rotary_dtype=rotary_dtype,
                                              param_init_type=param_init_type,
                                              qkv_has_bias=qkv_has_bias,
                                              attn_proj_has_bias=attn_proj_has_bias,
                                              use_past=False,
                                              is_dynamic=is_dynamic,
                                              use_rope_slice=use_rope_slice,
                                              use_flash_attention=use_flash_attention,
                                              use_ring_attention=use_ring_attention,
                                              use_attn_mask_compression=use_attn_mask_compression,
                                              rmsnorm_compute_2d=rmsnorm_compute_2d,
                                              block_size=block_size,
                                              num_blocks=num_blocks,
                                              batch_size=batch_size,
                                              parallel_config=parallel_config,
                                              parallel_decoding=parallel_decoding,
                                              init_method_std=init_method_std,
                                              chunk_prefill=chunk_prefill)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tanh.shard(((1,),))
            self.mul.shard(((dp, cp, 1), (1,)))
            self.mul1.shard(((dp, cp, 1), (dp, cp, 1)))
            self.slice.shard(((dp, mp, 1),))

    # pylint: disable=W0221
    def construct(self, x, freqs_cis, mask=None, cross_attention_mask=None, cross_attention_states=None,
                  full_text_row_masked_out_mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, q_seq_lens=None, kv_mask=None, seq_chunk=None):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, cross_attention_states, freqs_cis, cross_attention_mask, batch_valid_length,
                           block_tables,
                           slot_mapping, prefix_keys_values, q_seq_lens, kv_mask, seq_chunk)

        if self.residual_cast_flag:
            x = self.cast(x, self.residual_dtype)
            h = self.cast(h, self.residual_dtype)
        attn_gate = self.tanh(self.cross_attn_attn_gate)
        h = self.add(x, self.mul(h, self.cast(attn_gate, self.dtype)))
        if self.residual_cast_flag:
            h = self.cast(h, self.dtype)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        if self.residual_cast_flag:
            h = self.cast(h, self.residual_dtype)
            ffn_out = self.cast(ffn_out, self.residual_dtype)

        if full_text_row_masked_out_mask is not None:
            bz, _, seq_len, _ = full_text_row_masked_out_mask.shape
            full_text_row_masked_out_mask = self.reshape(full_text_row_masked_out_mask, (bz, seq_len, -1))
            ffn_out = self.mul1(full_text_row_masked_out_mask, ffn_out)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ff_gate = self.tanh(self.cross_attn_ff_gate)
        out = self.add(h, self.mul(ffn_out, self.cast(ff_gate, self.dtype)))
        if self.residual_cast_flag:
            out = self.cast(out, self.dtype)
        return out
