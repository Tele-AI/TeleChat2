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
"""Telechat transformer Layer's APIs."""
import math
from typing import Tuple, Optional

import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.models.utils import predict_lazy_inline
from mindformers.modules.layers import _check_input_dtype, Dropout, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode

from research.telechat2.telechat_layer import TelechatLinear, TelechatFeedForward


class TelechatAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in Telechat.

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
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
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
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 sigma: float = 0.0048,
                 mean: float = 0.0,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 wo_has_bias=True,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 use_attn_mask_compression=False,
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
        self.sigma = sigma
        self.mean = mean
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression

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
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)

        self.wq = TelechatLinear(self.hidden_size,
                                 self.hidden_size,
                                 sigma=self.sigma,
                                 mean=self.mean,
                                 has_bias=qkv_has_bias,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic)
        self.wk_v = TelechatLinear(self.hidden_size,
                                   self.n_kv_head * self.head_dim * 2,
                                   has_bias=qkv_has_bias,
                                   sigma=self.sigma,
                                   mean=self.mean,
                                   compute_dtype=compute_dtype,
                                   param_init_type=param_init_type,
                                   skip_redistribution=is_dynamic)

        if qkv_has_bias:
            self.wq.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
            self.wk_v.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
        else:
            self.wq.shard(((dp, 1), (mp, 1)))
            self.wk_v.shard(((dp, 1), (mp, 1)))
        self.split_kv = ms.ops.auto_generate.SplitWithSize()
        self.split_kv.add_prim_attr("skip_redistribution", True)
        self.split_kv.shard(((dp, mp, 1),))

        self.wo = TelechatLinear(in_channels=self.hidden_size,
                                 out_channels=self.hidden_size,
                                 sigma=self.sigma,
                                 mean=self.mean,
                                 has_bias=wo_has_bias,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic,
                                 keep_prob=1 - self.hidden_dropout_prob)
        if wo_has_bias:
            self.wo.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)), out_strategy_matmul=((dp, 1),))
        else:
            self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp, 1),))

        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.head_dim,
                                                  self.n_kv_head,
                                                  pa_n_head_split=self.n_head // mp,
                                                  pa_n_kv_head_split=self.n_kv_head // mp,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=2147483647,
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
            self.attention_dropout = Dropout(1 - self.attention_dropout_prob)

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
                self.sparse_mode = 2 if self.use_attn_mask_compression else 0
                self.flash_attention = FlashAttention(head_num=self.n_head,
                                                      pre_tokens=65536,
                                                      next_tokens=0,
                                                      input_layout="BNSD",
                                                      keep_prob=1. - attention_dropout_prob,
                                                      scale_value=1. / math.sqrt(self.head_dim),
                                                      sparse_mode=self.sparse_mode,
                                                      use_attention_mask=True)
                self.flash_attention.shard(parallel_config)


    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key_value = self.cast(self.wk_v(x), self.dtype)  # dp, 1 -> dp, mp
        key_value = self.reshape(key_value, (-1, self.n_kv_head, self.head_dim * 2))
        key, value = self.split_kv(key_value, (self.head_dim, self.head_dim), 2)

        # key and value for current token(s)
        if self.use_past:
            key = self.reshape(key, (bs, seq_len, self.n_kv_head * self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.n_kv_head * self.head_dim))
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 freqs_cis, mask, prefix_keys_values=prefix_keys_values)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, 1, 1

            value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            key, value = self._cat_prefix(key, value, prefix_keys_values)

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

    def _cat_prefix(self, key, value, prefix_keys_values):
        r'''
        concat prefix_keys_values to key and value
        prefix_keys_values: shape(2, bs, pre_len, num_heads * kv_channels)
        '''
        if prefix_keys_values is not None:
            bs, n_kv_head, _, head_dim = key.shape
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.transpose(self.reshape(past_key, (bs, -1, n_kv_head, head_dim)), (0, 2, 1, 3))
            past_value = self.transpose(self.reshape(past_value, (bs, -1, n_kv_head, head_dim)), (0, 2, 1, 3))
            past_key = self.cast(past_key, self.dtype)
            past_value = self.cast(past_value, self.dtype)
            cat = P.Concat(2)
            key = cat((past_key, key))
            value = cat((past_value, value))
        return key, value

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
        attention_probs = self.attention_dropout(attention_probs)
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


# pylint: disable=C0326
class TelechatDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            multiple_of(int): The SwiGLU hidden layer size multiple of large power of 2.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            qkv_has_bias(bool): Whether Q/K/V in attention has bias or not.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, head_dim, seq_length),
              (batch_size, num_heads, seq_length, head_dim)).

    """

    @predict_lazy_inline
    def __init__(self,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 sigma: float = 0.0048,
                 mean: float = 0.0,
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 res_dtype=mstype.float32,
                 qkv_has_bias=False,
                 wo_has_bias=True,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 use_attn_mask_compression=False,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.res_dtype = res_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.sigma = sigma
        self.mean = mean
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.add = P.Add()
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = TelechatAttention(dim=dim,
                                           n_heads=n_heads,
                                           n_kv_heads=n_kv_heads,
                                           sigma=self.sigma,
                                           mean=self.mean,
                                           hidden_dropout_prob=hidden_dropout_prob,
                                           attention_dropout_prob=attention_dropout_prob,
                                           compute_dtype=compute_dtype,
                                           softmax_compute_dtype=softmax_compute_dtype,
                                           rotary_dtype=rotary_dtype,
                                           param_init_type=param_init_type,
                                           qkv_has_bias=qkv_has_bias,
                                           wo_has_bias=wo_has_bias,
                                           use_past=use_past,
                                           is_dynamic=is_dynamic,
                                           use_rope_slice=use_rope_slice,
                                           use_flash_attention=use_flash_attention,
                                           use_attn_mask_compression=use_attn_mask_compression,
                                           block_size=block_size,
                                           num_blocks=num_blocks,
                                           parallel_config=parallel_config)

        self.feed_forward = TelechatFeedForward(dim=self.hidden_size,
                                                intermediate_size=intermediate_size,
                                                hidden_dim=4 * self.hidden_size,
                                                sigma=self.sigma,
                                                mean=self.mean,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                multiple_of=multiple_of,
                                                ffn_dim_multiplier=ffn_dim_multiplier,
                                                compute_dtype=compute_dtype,
                                                param_init_type=param_init_type,
                                                is_dynamic=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.add.shard(((dp, mp, 1), (dp, mp, 1)))
            self.attention_norm.shard((dp, mp, 1))
            self.ffn_norm.shard((dp, mp, 1))
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        self.predict_run_mode = get_predict_run_mode()
        logger.info("Predict run mode:{}".format(self.predict_run_mode))

        if self.predict_run_mode:
            self.no_inline = False

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables,
                           slot_mapping, prefix_keys_values)
        h = self.add(self.cast(x, self.res_dtype), self.cast(h, self.res_dtype))
        h = self.cast(h, ori_dtype)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        h = self.add(self.cast(h, self.res_dtype), self.cast(ffn_out, self.res_dtype))
        out = self.cast(h, ori_dtype)
        return out

    def _check_input(self, x, freqs_cis, mask):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if swap_mask is not None:
            _check_input_dtype(swap_mask.dtype, "swap_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16, mstype.uint8, mstype.bool_],
                               self.cls_name)
        return True
