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
"""DeepSeekV2 transformer Layer's APIs."""
import math
from typing import Tuple, Optional, Dict
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.models.llama.llama_layer import (LlamaFeedForward, LlamaFeedForwardWithMoE, LlamaRMSNorm)
from mindformers.models.utils import predict_lazy_inline
from mindformers.modules.layers import _check_input_dtype, _yarn_get_mscale, Linear
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.transformer.moe import MoEV2
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode
from research.deepseek2.deepseek2_layer import DeepSeekV2RotaryEmbedding, DeepSeekV2MoEInfer


class DeepSeekV2Attention(nn.Cell):
    r"""
    This is an implementation of multihead attention in DeepSeekV2.

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
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 kv_lora_rank=512,
                 q_lora_rank=1536,
                 qk_rope_head_dim=64,
                 v_head_dim=128,
                 qk_nope_head_dim=128,
                 max_position_embeddings=2048,
                 scaling_factor: Optional[Dict] = None,
                 norm_eps=1e-5
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

        # additional params for qkv computation
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling_factor = scaling_factor
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
        cp = parallel_config.context_parallel
        self.context_parallel = cp
        self.shape = P.Shape()
        self.cast = P.Cast()

        self.q2l_proj = Linear(
            self.hidden_size,
            self.q_lora_rank,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic
        )
        # for reason of inference precision, we do not use fused_kernel. This will be corrected once the fused kernel
        # meets accuracy requirement for bfloat16 data.
        self.lq_norm = LlamaRMSNorm(self.q_lora_rank, norm_eps, compute_type=mstype.float32,
                                    fused_kernel=not get_predict_run_mode())

        self.l2q_proj = Linear(
            self.q_lora_rank,
            self.n_head * self.q_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic
        )

        # 1. kv2l: kv latent vector; 2. lkv_norm: latent vector of kv normalization
        self.kv2l = Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic
        )
        # for reason of inference precision, we do not use fused_kernel. This will be corrected once the fused kernel
        # meets accuracy requirement for bfloat16 data.
        self.lkv_norm = LlamaRMSNorm(self.kv_lora_rank, norm_eps, compute_type=mstype.float32,
                                     fused_kernel=not get_predict_run_mode())
        self.lkv2kv = Linear(
            self.kv_lora_rank,
            self.n_head * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            skip_redistribution=is_dynamic
        )

        self.q2l_proj.shard(((dp, 1), (1, 1)))
        self.l2q_proj.shard(((dp, 1), (mp, 1)))
        self.kv2l.shard(((dp, 1), (1, 1)))
        self.lkv2kv.shard(((dp, 1), (mp, 1)))

        if parallel_config.use_seq_parallel:
            self.q2l_proj.shard(((dp, 1), (mp, 1)))

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.lq_norm.shard((dp, 1, 1))
            self.lkv_norm.shard((dp, 1, 1))

        if parallel_config.use_seq_parallel:
            self.lq_norm.shard((dp, mp, 1))
            self.lkv_norm.shard((dp, mp, 1))

        self.wo = Linear(in_channels=self.n_head * self.v_head_dim,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)
        self.wo.shard(((dp, mp), (1, mp)))

        self.inv_norm_factor = self.q_head_dim ** (-0.5)
        # this is intended for Flash attention scale_factor args which is a python math scalar.
        self.scale_fa = 1. / math.sqrt(self.q_head_dim)
        if self.scaling_factor is not None:
            mscale_all_dim = self.scaling_factor.get("mscale_all_dim", 0)
            factor = self.scaling_factor["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(factor, mscale_all_dim)
                self.inv_norm_factor = self.inv_norm_factor * mscale * mscale
                self.scale_fa = mscale * mscale / (math.sqrt(self.q_head_dim))

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.dp_only_transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()

        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()
        self.slice_qkv = P.StridedSlice()
        self.dim_slice_4d = P.Slice()
        self.dim_slice_3d = P.Slice()
        self.pe_concat = P.Concat(3)
        self.sum_test = P.ReduceSum()
        self.padv = P.PadV3()
        self.pad_zero = Tensor(0, compute_dtype)

        self.apply_rotary_emb = DeepSeekV2RotaryEmbedding(self.qk_rope_head_dim,
                                                          rotary_dtype, use_rope_slice=use_rope_slice)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))
            self.slice_qkv.shard(((dp, mp, 1, 1),))
            self.dim_slice_3d.shard(((dp, 1, 1),))
            self.dim_slice_4d.shard(((dp, mp, 1, 1),))
            self.pe_concat.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.padv.shard(((dp, mp, 1, 1), (1,), ()))

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
                                                  scale_value=1. / math.sqrt(self.q_head_dim),
                                                  sparse_mode=0,
                                                  use_attention_mask=True)
            self.flash_attention.shard(parallel_config)
        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.qk_nope_head_dim + self.qk_rope_head_dim,
                                                  self.n_kv_head,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  block_size=self.block_size,
                                                  num_blocks=self.num_blocks,
                                                  use_rope_rotary_emb=False,
                                                  use_flash_attention=False,
                                                  rotary_cos_format=2)
            self.infer_attention.shard(parallel_config=parallel_config)

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)

        q = self.q2l_proj(x)  # [bs, seq/1, hidden_size] -> [bs, seq/1, q_lora_rank]
        # redistribution allgather needed   !!!! allgather
        norm_q = self.lq_norm(q)   # [bs, seq/1, q_lora_rank]
        q = self.l2q_proj(norm_q) # [bs, seq/1, q_lora_rank] ->  [bs, seq/1,  n_head * q_head_dim]
        q = self.reshape(q, (bs, seq_len, self.n_head, self.q_head_dim))  # [bs, seq/1,  n_head, q_head_dim]
        # redistribution allgather:
        q = self.transpose(q, (0, 2, 1, 3))  # [bs, n_head, seq/1, q_head_dim]

        # redistribution allgather:  !!!!  2 allgather??
        q_nope = self.dim_slice_4d(q, (0, 0, 0, 0), (bs, self.n_head, seq_len, self.qk_nope_head_dim))  # [bs, n_head, seq/1, qk_nope_head_dim]
        q_pe = self.dim_slice_4d(q, (0, 0, 0, self.qk_nope_head_dim), (bs, self.n_head, seq_len, self.qk_rope_head_dim))  # [bs, n_head, seq/1, qk_rope_head_dim]

        latent_kv_all = self.kv2l(x)  # [bs, seq/1, hidden_size] -> [bs, seq/1, kv_lora_rank + self.qk_rope_head_dim]
        # redistribution allgather:
        latent_kv = self.dim_slice_3d(latent_kv_all, (0, 0, 0), (bs, seq_len, self.kv_lora_rank)) # [bs, seq/1, kv_lora_rank]
        k_pe = self.dim_slice_3d(latent_kv_all, (0, 0, self.kv_lora_rank), (bs, seq_len, self.qk_rope_head_dim))  # [bs, seq/1, qk_rope_head_dim]
        k_pe = self.reshape(k_pe, (bs, 1, seq_len, self.qk_rope_head_dim)) # [bs, 1, seq, qk_rope_head_dim]

        # redistribution allgather:
        i_kv = self.lkv_norm(latent_kv)  # [bs, seq/1, kv_lora_rank]
        o_kv = self.lkv2kv(i_kv) # [bs, seq/1, kv_lora_rank] ->  bs, seq/1, n_head * (qk_nope_head_dim + v_head_dim)
        kv = self.reshape(o_kv, (bs, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim))
        kv = self.transpose(kv, (0, 2, 1, 3))  # [bs, n_head, seq/1, qk_nope_head_dim + v_head_dim]

        # redistribution allgather:
        k_nope = self.dim_slice_4d(kv, (0, 0, 0, 0), (bs, self.n_head, seq_len, self.qk_nope_head_dim))   # bs, n_head, seq/1, qk_nope_head_dim
        k_pe = self.tile_kv(k_pe, (1, self.n_head, 1, 1))  # [bs, n_head, seq, qk_rope_head_dim]
        q_pe, k_pe = self.apply_rotary_emb(q_pe, k_pe, freqs_cis)

        # redistribution allgather:
        query_states = self.pe_concat((q_nope, q_pe))
        # redistribution allgather:
        key_states = self.pe_concat((k_nope, k_pe))
        value_states = self.dim_slice_4d(kv, (0, 0, 0, self.qk_nope_head_dim),
                                         (bs, self.n_head, seq_len, self.v_head_dim))  # bs, n_head, seq/1, v_head_dim

        if self.use_past:
            value_states = self.pe_concat((value_states, k_pe))
            key_states = self.reshape(self.transpose(key_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            value_states = self.reshape(self.transpose(value_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            query_states = self.reshape(self.transpose(query_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            context_layer = self.infer_attention(query_states, key_states, value_states, batch_valid_length,
                                                 block_tables, slot_mapping, freqs_cis, mask,
                                                 prefix_keys_values=prefix_keys_values)
            attn_out = self.dim_slice_3d(context_layer, (0, 0, 0), (bs, seq_len, self.n_head * self.v_head_dim))
        else:
            if self.use_flash_attention:
                # PadV3 operator does not support bflat16, cast value_states to float32 first
                value_states_ = self.padv(self.cast(value_states, mstype.float32),
                                          (0, self.q_head_dim - self.v_head_dim, 0, 0, 0, 0, 0, 0,), self.pad_zero)
                context_layer = self.flash_attention(self.cast(query_states, self.dtype),
                                                     self.cast(key_states, self.dtype),
                                                     self.cast(value_states_, self.dtype), mask)
                context_layer = self.dim_slice_4d(context_layer, (0, 0, 0, 0),
                                                  (bs, self.n_head, seq_len, self.v_head_dim))
                attn_out = self._merge_heads(context_layer)
            else:
                attn_out = self._attn(query_states, key_states, value_states, mask)

        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        # output reshape allreduce
        output = self.wo(attn_out)  # dp, mp -> dp, 1 / dp * mp, 1
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
        # redistribution allgather
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
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), self.cast(value, self.dtype))
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class DeepSeekV2DecodeLayer(nn.Cell):
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
                 compute_dtype=mstype.float32,
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
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 kv_lora_rank=512,
                 q_lora_rank=1536,
                 qk_rope_head_dim=64,
                 v_head_dim=128,
                 qk_nope_head_dim=128,
                 max_position_embeddings=2048,
                 scaling_factor: Optional[Dict] = None
                 ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.return_extra_loss = True
        self.use_past = use_past
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = P.Add()
        # for reason of inference precision, we do not use fused_kernel. This will be corrected once the fused kernel
        # meets accuracy requirement for bfloat16 data.
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype,
                                     fused_kernel=not get_predict_run_mode())
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype,
                                           fused_kernel=not get_predict_run_mode())
        self.attention = DeepSeekV2Attention(dim=dim,
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
                                             block_size=block_size,
                                             num_blocks=num_blocks,
                                             parallel_config=parallel_config,
                                             kv_lora_rank=kv_lora_rank,
                                             q_lora_rank=q_lora_rank,
                                             qk_rope_head_dim=qk_rope_head_dim,
                                             v_head_dim=v_head_dim,
                                             qk_nope_head_dim=qk_nope_head_dim,
                                             max_position_embeddings=max_position_embeddings,
                                             scaling_factor=scaling_factor,
                                             norm_eps=norm_eps)

        self.expert_num = 1 if moe_config is None else moe_config.expert_num
        self.shared_expert_num = 0 if moe_config is None else moe_config.shared_expert_num
        # set kbk infer for moe structural models.
        self.use_moe_infer = False # temporarily disable using moe infer

        ffn = LlamaFeedForward(dim=self.hidden_size,
                               intermediate_size=intermediate_size,
                               hidden_dim=4 * self.hidden_size,
                               multiple_of=multiple_of,
                               expert_num=self.expert_num,
                               ffn_dim_multiplier=ffn_dim_multiplier,
                               compute_dtype=compute_dtype,
                               param_init_type=param_init_type,
                               ffn_concat=qkv_concat,
                               is_dynamic=is_dynamic,
                               parallel_config=parallel_config) if self.shared_expert_num == 0 else None

        # Feed Forward Network
        self.first_k_dense = (moe_config.first_k_dense_replace and layer_id < moe_config.first_k_dense_replace)
        if self.first_k_dense:
            logger.warning("first_k_dense_replace is provided in MoEConfig, "
                           "a normal dense FFN will be used in this block.")
            self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                                 intermediate_size=intermediate_size,
                                                 hidden_dim=4 * self.hidden_size,
                                                 multiple_of=multiple_of,
                                                 expert_num=1,
                                                 ffn_dim_multiplier=ffn_dim_multiplier,
                                                 compute_dtype=compute_dtype,
                                                 param_init_type=param_init_type,
                                                 is_dynamic=is_dynamic,
                                                 parallel_config=parallel_config)
        else:
            if self.expert_num == 1:
                self.feed_forward = ffn
            else:
                if self.use_moe_infer:
                    self.feed_forward = DeepSeekV2MoEInfer(hidden_size=self.hidden_size,
                                                           intermediate_size=moe_config.moe_intermediate_size,
                                                           compute_dtype=compute_dtype,
                                                           param_init_type=param_init_type,
                                                           is_dynamic=is_dynamic,
                                                           moe_config=moe_config,
                                                           parallel_config=parallel_config)
                elif self.shared_expert_num == 0:
                    self.feed_forward = MoEV2(ffn=ffn,
                                              dim=self.hidden_size,
                                              moe_config=moe_config,
                                              parallel_config=parallel_config,
                                              return_extra_loss=self.return_extra_loss)
                else: # There are shared experts to initialize them
                    logger.info("MoE config is provided, use MoE FFN with shared ffn")
                    self.feed_forward = LlamaFeedForwardWithMoE(hidden_size=self.hidden_size,
                                                                intermediate_size=moe_config.moe_intermediate_size,
                                                                compute_dtype=compute_dtype,
                                                                param_init_type=param_init_type,
                                                                is_dynamic=is_dynamic,
                                                                moe_config=moe_config,
                                                                parallel_config=parallel_config,
                                                                use_moe_infer=self.use_moe_infer,
                                                                return_extra_loss=self.return_extra_loss)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            if self.expert_num == 1 or self.first_k_dense:
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
            if moe_config is None or not moe_config.expert_num > 1 or self.first_k_dense:
                self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.add.shard(((dp, mp, 1), (dp, mp, 1)))
            self.attention_norm.shard((dp, mp, 1))
            self.ffn_norm.shard((dp, mp, 1))
            if moe_config is None or not moe_config.expert_num > 1 or self.first_k_dense:
                self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        self.predict_run_mode = get_predict_run_mode()
        if self.predict_run_mode:
            self.no_inline = False

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, extra_loss=Tensor([0], mstype.float32)):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables,
                           slot_mapping, prefix_keys_values)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        if hasattr(self.feed_forward, "return_extra_loss") and self.return_extra_loss:
            ffn_out, extra_loss = self.feed_forward(ffn_norm, extra_loss)
        else:
            # [bs, seq/1, hidden_dim]
            ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out, extra_loss

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
