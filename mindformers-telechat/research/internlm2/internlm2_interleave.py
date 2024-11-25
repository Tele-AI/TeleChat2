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
"""InternLM2 fine grain interleave transformer Layer's APIs."""

from typing import Optional
import math

from mindspore import nn, __version__
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.models.llama.llama_interleave import LLamaDecodeLayerInterleave
from mindformers.modules.layers import Linear, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention


class InternLM2AttentionInterleave(nn.Cell):
    r"""
    This is an implementation of multihead attention in InternLM2.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **qkv_concat** (bool):
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **has_bias** (bool): Whether Q/K/V/O in attention has bias or not.
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
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

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
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.qkv_concat = qkv_concat
        self.use_flash_attention = use_flash_attention

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = nn.Softmax().to_float(softmax_compute_dtype)
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()
        self.slice = P.StridedSlice()
        self.slice.add_prim_attr("skip_redistribution", True)

        self.apply_rotary_emb = RotaryEmbedding(self.head_dim, rotary_dtype)
        if self.qkv_concat:
            self.w = Linear(in_channels=self.hidden_size,
                            out_channels=self.hidden_size + self.kv_dim * 2,
                            has_bias=False,
                            compute_dtype=compute_dtype,
                            param_init_type=param_init_type)
            self.w.matmul.add_prim_attr("skip_redistribution", True)
        else:
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wk = Linear(self.hidden_size,
                             self.kv_dim,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wv = Linear(self.hidden_size,
                             self.kv_dim,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.n_head,
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  keep_prob=1.,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  input_layout="BNSD",
                                                  sparse_mode=0,
                                                  use_attention_mask=True)
            self.flash_attention.shard(parallel_config)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.softmax.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))
            self.slice.shard(((dp, mp, 1, 1),))

            self.apply_rotary_emb.shard(parallel_config)
            if self.qkv_concat:
                self.w.shard(((dp, 1), (mp, 1)))
            else:
                self.wq.shard(((dp, 1), (mp, 1)))
                self.wk.shard(((dp, 1), (mp, 1)))
                self.wv.shard(((dp, 1), (mp, 1)))
            self.wo.shard(((dp, mp), (1, mp)))
            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
            if parallel_config.recompute.select_recompute and not self.use_flash_attention:
                self.apply_rotary_emb.recompute()
                self.tile_kv.recompute()
                self.batch_matmul_q_k.recompute()
                self.mul.recompute()
                self.add.recompute()
                self.cast_attn.recompute()
                self.softmax.softmax.recompute()
                self.batch_matmul.recompute()

    def compute_qkv(self, x):
        """compute the qkv with interleave number"""
        x = self.reshape(x, (-1, x.shape[-1]))
        if self.qkv_concat:
            bs_seq = x.shape[0]
            qkv = self.cast(self.w(x), self.dtype)
            qkv = self.reshape(qkv, (bs_seq, -1, (2 + self.n_rep), self.head_dim))  # b*q (h gs d) -> b*q h gs d
            h = qkv.shape[1]
            query = self.slice(qkv, (0, 0, 0, 0), (bs_seq, h, self.n_rep, self.head_dim), (1, 1, 1, 1))
            query = self.reshape(query, (bs_seq, -1))  # b*q h gs d -> b*q (h gs d)
            key = self.slice(qkv, (0, 0, self.n_rep, 0), (bs_seq, h, self.n_rep + 1, self.head_dim), (1, 1, 1, 1))
            key = self.reshape(key, (bs_seq, -1))  # b*q h gs d -> b*q (h gs d)
            value = self.slice(qkv, (0, 0, self.n_rep + 1, 0), (bs_seq, h, self.n_rep + 2, self.head_dim), (1, 1, 1, 1))
            value = self.reshape(value, (bs_seq, -1))  # b*q h gs d -> b*q (h gs d)
        else:
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)    # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        return query, key, value

    def cal_attn(self, query, key, value, mask, freqs_cis):
        """cal_attn"""
        query = self.reshape(query, (-1, self.seq_length, self.n_head, self.head_dim))
        key = self.reshape(key, (-1, self.seq_length, self.n_kv_head, self.head_dim))
        value = self.reshape(value, (-1, self.seq_length, self.n_kv_head, self.head_dim))

        # [bs, seq/1, n_head/n_kv_head, head_dim]
        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))

        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis) # dp, mp, 1, 1
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        bs, n_head, seq, head_dim = query.shape
        n_kv_head = key.shape[1]
        query = self.reshape(query, (bs, n_head, seq, head_dim))
        key = self.reshape(key, (bs, n_kv_head, seq, head_dim))
        value = self.reshape(value, (bs, n_kv_head, seq, head_dim))

        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            attention = self._attn(query, key, value, mask)
        return attention

    def cal_output_proj(self, attention):
        """cal_output_proj"""
        output = self.wo(attention) # dp, mp -> dp, 1 / dp * mp, 1
        return output

    def _repeat_kv(self, x, rep):
        """repeat_kv"""
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
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
        x = self.merger_head_transpose(x, (0, 2, 1, 3)) # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        x_shape = x.shape
        # [bs * seq/1, hidden_dim]
        new_shape = (-1, x_shape[-2] * x_shape[-1])
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


class InternLM2DecodeLayerInterleave(LLamaDecodeLayerInterleave):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            qkv_concat(bool):
    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 qkv_concat=False,
                 **kwargs):
        super().__init__(batch_size=batch_size, seq_length=seq_length, layer_id=layer_id, **kwargs)
        kwargs.pop("num_layers")
        kwargs.pop("multiple_of")
        kwargs.pop("intermediate_size")
        kwargs.pop("ffn_dim_multiplier")
        kwargs.pop("norm_eps")
        kwargs.pop("layernorm_compute_dtype")
        kwargs.pop("fine_grain_interleave")
        self.attention = InternLM2AttentionInterleave(batch_size=batch_size,
                                                      seq_length=seq_length,
                                                      qkv_concat=qkv_concat,
                                                      **kwargs)
