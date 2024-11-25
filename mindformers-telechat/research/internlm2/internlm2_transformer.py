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
"""InternLM2 transformer Layer's APIs."""
import math
from typing import Tuple, Optional

from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.modules.layers import Linear, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention


class InternLM2Attention(nn.Cell):
    """
    This is an implementation of multihead attention in InternLM2.

    Args:
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """

    def __init__(self,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.qkv_concat = qkv_concat

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        if self.qkv_concat:
            self.w = Linear(in_channels=self.hidden_size,
                            out_channels=self.hidden_size + self.kv_dim * 2,
                            has_bias=False,
                            compute_dtype=compute_dtype,
                            param_init_type=param_init_type,
                            skip_redistribution=is_dynamic)
            self.w.shard(((dp, 1), (mp, 1)))
            self.slice = P.StridedSlice()
            self.slice.add_prim_attr("skip_redistribution", True)
            self.slice.shard(((dp, mp, 1, 1),))
        else:
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wk = Linear(self.hidden_size,
                             self.n_kv_head * self.head_dim,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wv = Linear(self.hidden_size,
                             self.n_kv_head * self.head_dim,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
            self.wq.shard(((dp, 1), (mp, 1)))
            self.wk.shard(((dp, 1), (mp, 1)))
            self.wv.shard(((dp, 1), (mp, 1)))

        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)
        self.wo.shard(((dp, mp), (1, mp)))

        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.head_dim,
                                                  self.n_kv_head,
                                                  pa_n_head_split=self.n_head // mp,
                                                  pa_n_kv_head_split=self.n_kv_head // mp,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  block_size=self.block_size,
                                                  num_blocks=self.num_blocks,
                                                  use_flash_attention=self.use_flash_attention,
                                                  rotary_cos_format=2,
                                                  rotary_dtype=rotary_dtype,
                                                  compute_dtype=compute_dtype)
            self.infer_attention.shard(parallel_config)
        else:
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            self.batch_matmul = P.BatchMatMul()
            self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
            self.mul = P.Mul()
            self.add = P.Add()
            self.softmax = P.Softmax()
            self.cast_attn = P.Cast()
            self.tile_kv = P.Tile()

            self.apply_rotary_emb = RotaryEmbedding(self.head_dim, rotary_dtype, use_rope_slice=use_rope_slice)

            if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
                self.transpose.shard(((dp, 1, mp, 1),))
                self.merger_head_transpose.shard(((dp, mp, 1, 1),))
                self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul.shard(((dp, mp, 1, 1), ()))
                self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
                self.softmax.shard(((dp, mp, 1, 1),))
                self.tile_kv.shard(((dp, mp, 1, 1),))

                self.apply_rotary_emb.shard(parallel_config)

                if parallel_config.use_seq_parallel and self.is_first_iteration:
                    self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
                if parallel_config.recompute.select_recompute and not self.use_flash_attention:
                    self.apply_rotary_emb.recompute()
                    self.tile_kv.recompute()
                    self.batch_matmul_q_k.recompute()
                    self.mul.recompute()
                    self.add.recompute()
                    self.cast_attn.recompute()
                    self.softmax.recompute()
                    self.batch_matmul.recompute()

            if self.use_flash_attention:
                self.flash_attention = FlashAttention(head_num=self.n_head,
                                                      pre_tokens=65536,
                                                      next_tokens=0,
                                                      input_layout="BNSD",
                                                      keep_prob=1.,
                                                      scale_value=1. / math.sqrt(self.head_dim),
                                                      sparse_mode=0,
                                                      use_attention_mask=True)
                self.flash_attention.shard(parallel_config)

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, q_seq_lens=None):
        """Forward process of the MultiHeadAttention"""
        _ = prefix_keys_values
        _ = q_seq_lens
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)
        if self.qkv_concat:
            x = self.reshape(x, (-1, x.shape[-1]))
            bs_seq = x.shape[0]
            qkv = self.cast(self.w(x), self.dtype)
            qkv = self.reshape(qkv, (bs_seq, -1, (2 + self.n_rep), self.head_dim))  # b*q (h gs d) -> b*q h gs d
            h = qkv.shape[1]
            query = self.slice(qkv, (0, 0, 0, 0), (bs_seq, h, self.n_rep, self.head_dim), (1, 1, 1, 1))
            query = self.reshape(query, (bs, seq_len, -1))  # b*q h gs d -> b*q (h gs d)
            key = self.slice(qkv, (0, 0, self.n_rep, 0), (bs_seq, h, self.n_rep + 1, self.head_dim), (1, 1, 1, 1))
            key = self.reshape(key, (bs, seq_len, -1))  # b*q h gs d -> b*q (h gs d)
            value = self.slice(qkv, (0, 0, self.n_rep + 1, 0), (bs_seq, h, self.n_rep + 2, self.head_dim), (1, 1, 1, 1))
            value = self.reshape(value, (bs, seq_len, -1))  # b*q h gs d -> b*q (h gs d)
        else:
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp

        # key and value for current token(s)
        if self.use_past:
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 freqs_cis, mask)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, 1, 1
            if self.use_flash_attention:
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
        convert a 4d input to a 3d output

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

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class InternLM2DecodeLayer(LLamaDecodeLayer):
    """InternLM2 Transformer Layer inherits from LLamaDecodeLayer.

    Args:
        layer_id (int): The layer id of current transformer block layer.
        **kwargs: keyword arguments of [`LLamaDecodeLayer`].

    """

    def __init__(self,
                 layer_id,
                 qkv_concat,
                 **kwargs):
        super().__init__(layer_id=layer_id,
                         **kwargs)
        kwargs.pop("multiple_of")
        kwargs.pop("intermediate_size")
        kwargs.pop("ffn_dim_multiplier")
        kwargs.pop("norm_eps")
        kwargs.pop("layernorm_compute_dtype")
        self.attention = InternLM2Attention(qkv_concat=qkv_concat, **kwargs)
