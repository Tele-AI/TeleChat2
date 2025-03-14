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
"""Mixtral transformer Layer's APIs."""
import math
from typing import Tuple, Optional

import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.parallel.shard import Layout

from mindformers.models.llama.llama_layer import LlamaRMSNorm, LlamaMoeInferFeedForward, LlamaFeedForwardWithMoE
from mindformers.models.utils import predict_lazy_inline
from mindformers.modules.layers import _check_input_dtype, Linear, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.transformer.moe import MoEInfer
from mindformers.tools.utils import get_predict_run_mode

from .mixtral_layer import MixtralFeedForward
from .mixtral_moe import MixtralMoEV2


class MixtralAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in LLaMA.

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
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 use_ring_attention=False,
                 use_attn_mask_compression=False,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 parallel_decoding=False,
                 ):
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
        self.use_ring_attention = use_ring_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.qkv_concat = qkv_concat

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        self.data_parallel = dp
        self.model_parallel = mp
        self.context_parallel = cp
        # define ulysses context parallel
        self.cp_ds = parallel_config.get_ulysses_cp_num()
        # define colossal ai context parallel
        self.cp_co = cp // self.cp_ds

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_head % (mp * self.cp_ds) != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                             "'parallel_config.model_parallel * ulysses_cp_num', but got the n_head is {}, "
                             "the parallel_config.model_parallel is {}, and ulysses_cp_num is {}"
                             .format(self.n_head, mp, self.cp_ds))
        if self.n_kv_head % (mp * self.cp_ds) != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel * ulysses_cp_num', but got the n_kv_head is {}, "
                             "the parallel_config.model_parallel is {}, and ulysses_cp_num is {}"
                             .format(self.n_kv_head, mp, self.cp_ds))

        self.shape = P.Shape()
        self.cast = P.Cast()

        if self.qkv_concat:
            self.w_qkv = Linear(in_channels=self.hidden_size,
                                out_channels=self.hidden_size + self.kv_dim * 2,
                                has_bias=qkv_has_bias,
                                compute_dtype=compute_dtype,
                                param_init_type=param_init_type)
            if qkv_has_bias:
                self.w_qkv.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
            else:
                self.w_qkv.shard(((dp * cp, 1), (mp, 1)))
            self.split_qkv = ms.ops.auto_generate.SplitWithSize()
            self.split_qkv.add_prim_attr("skip_redistribution", True)
            self.split_qkv.shard(((dp, cp, mp),))
        else:
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wk = Linear(self.hidden_size,
                             self.kv_dim,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wv = Linear(self.hidden_size,
                             self.kv_dim,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            if qkv_has_bias:
                self.wq.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
                self.wk.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
                self.wv.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
            else:
                self.wq.shard(((dp * cp, 1), (mp, 1)))
                self.wk.shard(((dp * cp, 1), (mp, 1)))
                self.wv.shard(((dp * cp, 1), (mp, 1)))
        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wo.shard(((dp * cp, mp), (1, mp)), out_strategy_matmul=((dp * cp, 1),))

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
                                                  is_dynamic=is_dynamic,
                                                  use_flash_attention=self.use_flash_attention,
                                                  rotary_cos_format=2,
                                                  compute_dtype=compute_dtype,
                                                  parallel_decoding=parallel_decoding,
                                                  )
            self.infer_attention.shard(parallel_config)
        else:
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

            self.reshape = P.Reshape()
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

            # ulysses context parallel, initial related ops
            if self.cp_ds > 1:
                self._ulysses_initial()

            if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
                self.transpose.shard(((dp, cp, mp, 1),))
                if cp > 1:
                    layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
                    layout_merger_head_transpose = (layout("dp", "mp", "cp", "None"),)
                    self.merger_head_transpose.shard(in_strategy=layout_merger_head_transpose)
                else:
                    self.merger_head_transpose.shard(((dp, mp, 1, 1),))
                self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul.shard(((dp, mp, 1, 1), ()))
                self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
                self.softmax.shard(((dp, mp, 1, 1),))
                self.tile_kv.shard(((dp, mp, 1, 1),))

                self.apply_rotary_emb.shard(parallel_config)
                # if parallel_config.use_seq_parallel and cp > 1:
                #     logger.warning(
                #         "The context parallel way conflicts with sequence parallel way."
                #         "The Sequence parallel way has no effect here and is ignored"
                #     )
                if parallel_config.use_seq_parallel and self.is_first_iteration and cp == 1:
                    self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
                elif parallel_config.use_seq_parallel and self.is_first_iteration:
                    self.wo.shard(((dp * cp, mp), (1, mp)), out_strategy_matmul=((dp * cp * mp, 1),))
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
                self.input_layout = "BSH" if cp > 1 else "BNSD"
                self.sparse_mode = 2 if self.use_attn_mask_compression and not self.use_ring_attention else 0
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

    def _ulysses_initial(self):
        """initial ulysses related ops."""
        self.transpose_back = P.Transpose()
        self.transpose_ulysses = P.Transpose()
        self.transpose_a2a = P.Transpose()
        self.transpose_ulysses_merger_a2a = P.Transpose()
        self.transpose_ulysses_merger = P.Transpose()
        dp = self.data_parallel
        mp = self.model_parallel
        cp = self.context_parallel
        # ulysses shard strategy
        if self.is_first_iteration:
            self.wo.shard(((dp * cp, mp), (1, mp)), out_strategy_matmul=((dp * cp * mp, 1),))
            layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
            layout_transpose_back = (layout("dp", "mp", "cp", "None"),)
            self.transpose_back.shard(in_strategy=layout_transpose_back)
            self.transpose_ulysses.shard(((dp, cp, mp, 1, 1),))
            self.transpose_a2a.shard(((dp, self.cp_co, self.cp_ds, mp, 1),))
            self.transpose_ulysses_merger_a2a.shard(((dp, self.cp_co, self.cp_ds, mp, 1),))
            self.transpose_ulysses_merger.shard(((dp, cp, 1, mp, 1),))

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, q_seq_lens=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)
        if self.qkv_concat:
            qkv = self.cast(self.w_qkv(x), self.dtype)
            query, key, value = self.split_qkv(qkv, (self.hidden_size, self.kv_dim, self.kv_dim), 2)
        else:
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp

        # key and value for current token(s)
        if self.use_past:
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 freqs_cis, mask, prefix_keys_values=prefix_keys_values,
                                                 q_seq_lens=q_seq_lens)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, cp, 1
            # with ulysses context parallel, insert all to all before FA
            if self.context_parallel > 1 and self.cp_ds > 1:
                # for query & key, transpose from BNSD back to BSND
                query = self.transpose_back(query, (0, 2, 1, 3))
                query = self._ulysses_qkv_a2a(query)
                key = self.transpose_back(key, (0, 2, 1, 3))
                key = self._ulysses_qkv_a2a(key)
                # value is BSND, no need for transpose back
                value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))
                value = self._ulysses_qkv_a2a(value)
            elif self.context_parallel > 1:
                query = self._merge_heads(query)
                key = self._merge_heads(key)
            else:
                value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
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

    def _ulysses_qkv_a2a(self, qkv):
        """Given a qkv tensor with shape of (bs, seq_len, n_head, head_dim),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape of (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all to all commu.
        """
        bs, seq_len, head_num, hidden_size = F.shape(qkv)
        new_shape = (bs, seq_len, head_num // self.cp_ds, self.cp_ds, hidden_size)
        # [bs, seq_len, n_head, head_dim] -> [bs, seq_len, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [bs, seq_len, n_head/cp_ds, cp_ds, head_dim] -> [bs, seq_len, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4))
        # insert all-to-all (dp, cp, 1, mp, 1) -> (dp, cp_co, cp_ds, mp, 1)
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4))
        # reshape to BSH, here set -1 to H, for kv head could be different from q head
        qkv = F.reshape(qkv, (bs, seq_len, -1))
        return qkv

    def _ulysses_context_layer_a2a(self, context_layer):
        """Given the context_layer tensor after fa, with shape of (bs, seq_len, hidden_size),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            context_layer (Tensor): context layer after attention, with shape of (B, S, H)

        Returns:
            Tensor: context layer tensor after all to all commu.
        """
        bs, seq_len, _ = F.shape(context_layer)
        new_shape = (bs, seq_len, self.cp_ds, self.n_head // self.cp_ds, -1)
        context_layer = F.reshape(context_layer, new_shape)
        # insert all-to-all back (dp, cp_co, cp_ds, mp, 1) -> (dp, cp, 1, mp, 1)
        context_layer = self.transpose_ulysses_merger_a2a(context_layer, (0, 1, 2, 3, 4))
        context_layer = self.transpose_ulysses_merger(context_layer, (0, 1, 3, 2, 4))
        # reshape back to BSH
        context_layer = F.reshape(context_layer, (bs, seq_len, self.hidden_size))
        return context_layer

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


# pylint: disable=C0326
class MixtralDecodeLayer(nn.Cell):
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
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 moe_config=None,
                 use_flash_attention=False,
                 use_ring_attention=False,
                 use_attn_mask_compression=False,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 parallel_decoding=False,
                 ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.add = P.Add()
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = MixtralAttention(dim=dim,
                                          n_heads=n_heads,
                                          n_kv_heads=n_kv_heads,
                                          qkv_concat=qkv_concat,
                                          compute_dtype=compute_dtype,
                                          softmax_compute_dtype=softmax_compute_dtype,
                                          rotary_dtype=rotary_dtype,
                                          param_init_type=param_init_type,
                                          qkv_has_bias=qkv_has_bias,
                                          use_past=use_past,
                                          is_dynamic=is_dynamic,
                                          use_rope_slice=use_rope_slice,
                                          use_flash_attention=use_flash_attention,
                                          use_ring_attention=use_ring_attention,
                                          use_attn_mask_compression=use_attn_mask_compression,
                                          block_size=block_size,
                                          num_blocks=num_blocks,
                                          parallel_config=parallel_config,
                                          parallel_decoding=parallel_decoding)

        self.expert_num = 1 if moe_config is None else moe_config.expert_num
        self.shared_expert_num = 0 if moe_config is None else moe_config.shared_expert_num
        # set kbk infer for moe structural models.
        self.use_moe_infer = use_past and (self.expert_num > 1)
        if self.use_moe_infer:
            ffn = LlamaMoeInferFeedForward(dim=self.hidden_size,
                                           intermediate_size=intermediate_size,
                                           hidden_dim=4 * self.hidden_size,
                                           multiple_of=multiple_of,
                                           expert_num=self.expert_num,
                                           ffn_dim_multiplier=ffn_dim_multiplier,
                                           compute_dtype=compute_dtype,
                                           param_init_type=param_init_type,
                                           use_gmm=self.use_moe_infer)
        else:
            ffn = MixtralFeedForward(dim=self.hidden_size,
                                     intermediate_size=intermediate_size,
                                     hidden_dim=4 * self.hidden_size,
                                     multiple_of=multiple_of,
                                     expert_num=self.expert_num,
                                     ffn_dim_multiplier=ffn_dim_multiplier,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type,
                                     ffn_concat=qkv_concat,
                                     parallel_config=parallel_config) if self.shared_expert_num == 0 else None
        if self.expert_num == 1:
            self.feed_forward = ffn
        else:
            if self.shared_expert_num == 0:
                if self.use_moe_infer:
                    self.feed_forward = MoEInfer(
                        ffn=ffn,
                        dim=self.hidden_size,
                        moe_config=moe_config,
                        parallel_config=parallel_config)
                else:
                    self.feed_forward = MixtralMoEV2(
                        ffn=ffn,
                        dim=self.hidden_size,
                        moe_config=moe_config,
                        parallel_config=parallel_config)
            else:
                self.feed_forward = LlamaFeedForwardWithMoE(self.hidden_size,
                                                            intermediate_size=intermediate_size,
                                                            compute_dtype=compute_dtype,
                                                            param_init_type=param_init_type,
                                                            moe_config=moe_config,
                                                            parallel_config=parallel_config,
                                                            use_moe_infer=self.use_moe_infer)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            if self.expert_num == 1:
                self.feed_forward.shard(parallel_config)
            elif self.shared_expert_num == 0:
                self.feed_forward.ffn.shard(parallel_config)
            else:
                self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, cp, 1), (dp, cp, 1)))
            if cp > 1:
                self.attention_norm.shard((dp, cp * mp, 1))
                self.ffn_norm.shard((dp, cp * mp, 1))
            else:
                self.attention_norm.shard((dp, 1, 1))
                self.ffn_norm.shard((dp, 1, 1))
            if moe_config is None or not moe_config.expert_num > 1:
                self.feed_forward.mul.shard(((dp, cp, mp), (dp, cp, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            if cp > 1:
                self.add.shard(((dp, cp * mp, 1), (dp, cp * mp, 1)))
                self.attention_norm.shard((dp, cp * mp, 1))
                self.ffn_norm.shard((dp, cp * mp, 1))
            else:
                self.add.shard(((dp, mp, 1), (dp, mp, 1)))
                self.attention_norm.shard((dp, mp, 1))
                self.ffn_norm.shard((dp, mp, 1))
            if moe_config is None or not moe_config.expert_num > 1:
                self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        self.predict_run_mode = get_predict_run_mode()
        if self.predict_run_mode:
            self.no_inline = False

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, q_seq_lens=None):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables,
                           slot_mapping, prefix_keys_values, q_seq_lens)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
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
