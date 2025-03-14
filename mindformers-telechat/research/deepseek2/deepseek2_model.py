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
"""DeepseekV2 models' APIs."""
import copy
import math
from typing import Tuple, Optional, Dict
import numpy as np

from mindspore import Tensor, nn, mint
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.parallel.shard import Layout
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear, FreqsMgr, _check_input_dtype, _yarn_get_mscale
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline, LayerSetting, check_fine_grain_interleave_valid, predict_lazy_inline
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode
from mindformers.models.llama.llama_layer import LlamaFeedForward, LlamaFeedForwardWithMoE, LlamaRMSNorm, \
                                                 LlamaEmbedding, LlamaMoeInferFeedForward
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.transformer.moe import MoEV2
from mindformers.modules.transformer.moe import MoEInfer

from research.deepseek2.deepseek2_config import DeepseekV2Config

__all__ = ['DeepseekV2ForCausalLM', 'DeepseekV2Model']


class MTPHiddenFuser(Cell):
    """State fuser for Multi-Token Prediction module."""
    def __init__(self, config):
        super(MTPHiddenFuser, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.norm = LlamaRMSNorm(self.hidden_size, config.rms_norm_eps,
                                 compute_type=config.layernorm_compute_type,
                                 fused_kernel=not get_predict_run_mode())
        self.norm_emb = LlamaRMSNorm(self.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type,
                                     fused_kernel=not get_predict_run_mode())
        self.concat = P.Concat(axis=-1)
        self.dense = Linear(self.hidden_size * 2,
                            self.hidden_size,
                            has_bias=False,
                            compute_dtype=config.compute_dtype,
                            param_init_type=config.param_init_type)
        self.cast = P.Cast()
        self.dtype = config.compute_dtype
        if config.parallel_config.use_seq_parallel:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
            self.norm.shard((dp, mp, 1))
            self.norm_emb.shard((dp, mp, 1))
            self.concat.shard(((dp, mp, 1), (dp, mp, 1)))
            self.dense.shard(((dp * mp, 1), (1, 1)))

    def construct(self, h, h_emb):
        norm_h = self.norm(h)
        norm_emb = self.norm_emb(h_emb)
        norm_emb = self.cast(norm_emb, self.dtype)
        h_concat = self.concat([norm_h, norm_emb.astype(norm_h.dtype)])
        output = self.dense(h_concat)
        return output


class MtpEmbeddingLayer(nn.Cell):
    """Embedding layer used in Multi-Token Prediction module, same to standard embedding."""
    def __init__(self, vocab_table_size, rmsnorm_compute_2d=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        self.gather = P.Gather()

    def construct(self, embedding_weight, tokens):
        return self.gather(embedding_weight, tokens, 0)

    def shard(self, parallel_config):
        """sharding for embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if parallel_config.vocab_emb_dp:
            if not self.rmsnorm_compute_2d:
                self.gather.shard(((1, 1), (dp, cp)))
                logger.info(f"Using {dp*cp} data parallel for the embedding lookup.")
            else:
                self.gather.shard(((1, 1), (dp * cp,)))
                logger.info(f"Using {dp * cp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % (mp * cp) != 0:
                logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s"
                               "model_parallel: %s * context_parallel: %s.",
                               self.vocab_table_size, mp, cp)
                logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
                if not self.rmsnorm_compute_2d:
                    self.gather.shard(((1, 1), (dp, cp)))
                else:
                    self.gather.shard(((1, 1), (dp * cp,)))
            else:
                if not self.rmsnorm_compute_2d:
                    self.gather.shard(((mp * cp, 1), (dp, 1)))
                    logger.info(f"Using {dp} data parallel, {cp} context parallel and {mp} "
                                f"model parallel for the embedding lookup.")
                else:
                    self.gather.shard(((1, 1), (dp,)))
                    logger.info(f"Using {dp} data parallel for the embedding lookup.")


class DeepSeekV2RotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding for DeepSeekV2. This matches official implementation in Hugginface.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **parallel_config** (dict): - Parallel Config.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32, use_rope_slice=True):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.dtype = compute_dtype
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True

        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose()
        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()
        self.mul_inc = P.Mul()
        self.neg = P.Neg()
        self.slice = P.StridedSlice()
        self.concat = P.Concat(axis=-1)
        self.shape = P.Shape()
        self.cast = P.Cast()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        x = self.bmm_swap(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = self.shape(x)
        x1 = self.slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = self.slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = self.concat((self.neg(x2), x1))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        freqs_cos = self.cast(freqs_cos, self.dtype)
        freqs_sin = self.cast(freqs_sin, self.dtype)
        swap_mask = self.cast(swap_mask, self.dtype)
        mul = self.mul if self.is_first_iteration else self.mul_inc
        if self.use_rope_slice:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.slice_half(xq), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.rotate_half(xq, swap_mask), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.rotate_half(xk, swap_mask), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out

    def shard(self, parallel_config):
        """sharding for rotary embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        strategy_in = (dp, mp, 1, 1)
        if cp > 1:
            layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
            layout_add = (layout("dp", "mp", "cp", "None"), layout("dp", "mp", "cp", "None"))
            layout_bmm_swap = (layout("dp", "mp", "cp", "None"), layout("None", "None"))
            layout_mul = (layout("dp", "mp", "cp", "None"), layout("None", "None", "cp", "None"))
            self.add.shard(in_strategy=layout_add)
            self.bmm_swap.shard(in_strategy=layout_bmm_swap)
            self.mul.shard(in_strategy=layout_mul)
        else:
            self.add.shard((strategy_in, strategy_in))
            self.bmm_swap.shard((strategy_in, (1, 1)))
            self.mul.shard((strategy_in, (1, 1, 1, 1)))
        self.mul_inc.shard((strategy_in, (strategy_in[0], 1, 1, 1)))
        self.neg.shard((strategy_in,))
        self.slice.shard((strategy_in,))
        self.concat.shard((strategy_in, strategy_in))
        transpose_strategy_in = (dp, mp, 1, 1, 1)
        self.transpose.shard((transpose_strategy_in,))


class DeepSeekV2MoEInfer(Cell):
    r"""
    MoE inferernce inherited from MoEInfer, where shared experts are added.
    """
    def __init__(self, hidden_size, intermediate_size, compute_dtype,
                 param_init_type, moe_config, parallel_config):
        super(DeepSeekV2MoEInfer, self).__init__()
        ffn = LlamaMoeInferFeedForward(dim=hidden_size,
                                       intermediate_size=intermediate_size,
                                       expert_num=moe_config.expert_num,
                                       compute_dtype=compute_dtype,
                                       param_init_type=param_init_type,
                                       use_gmm=True)
        self.routed_experts = MoEInfer(ffn, hidden_size, moe_config, parallel_config)
        intermediate_size_all = int(moe_config.moe_intermediate_size * moe_config.shared_expert_num)
        self.shared_experts = LlamaFeedForward(dim=hidden_size,
                                               intermediate_size=intermediate_size_all,
                                               expert_num=1,
                                               compute_dtype=compute_dtype,
                                               param_init_type=param_init_type,
                                               parallel_config=parallel_config)
        self.add = P.Add()

    def construct(self, x):
        routed_experts_output = self.routed_experts(x)
        shared_experts_output = self.shared_experts(x)
        output = self.add(routed_experts_output, shared_experts_output)
        return output

    def shard(self, parallel_config):
        r"""set parallel strategy"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.add.shard(((dp, 1, 1), (dp, 1, 1)))

        self.routed_experts.ffn.shard(parallel_config)
        self.shared_experts.shard(parallel_config)
        self.shared_experts.mul.shard(((dp, 1, mp), (dp, 1, mp)))


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
        self.use_seq_parallel = parallel_config.use_seq_parallel
        self.mp = mp
        self.context_parallel = cp
        self.shape = P.Shape()
        self.cast = P.Cast()

        self.q2l_proj = Linear(
            self.hidden_size,
            self.q_lora_rank,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )
        # for reason of inference precision, we do not use fused_kernel. This will be corrected once the fused kernel
        # meets accuracy requirement for bfloat16 data.
        self.lq_norm = LlamaRMSNorm(self.q_lora_rank, norm_eps, compute_type=mstype.float32,
                                    fused_kernel=not get_predict_run_mode())

        self.l2q_nope_proj = Linear(
            self.q_lora_rank,
            self.n_head * self.qk_nope_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )
        self.l2q_pe_proj = Linear(
            self.q_lora_rank,
            self.n_head * self.qk_rope_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )

        # 1. kv2l: kv latent vector; 2. lkv_norm: latent vector of kv normalization
        self.kv2l_k_pe = Linear(
            self.hidden_size,
            self.qk_rope_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )
        self.kv2l_latent_kv = Linear(
            self.hidden_size,
            self.kv_lora_rank,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )
        # for reason of inference precision, we do not use fused_kernel. This will be corrected once the fused kernel
        # meets accuracy requirement for bfloat16 data.
        self.lkv_norm = LlamaRMSNorm(self.kv_lora_rank, norm_eps, compute_type=mstype.float32,
                                     fused_kernel=not get_predict_run_mode())
        self.lkv2kv_k_nope = Linear(
            self.kv_lora_rank,
            self.n_head * self.qk_nope_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )
        self.lkv2kv_v = Linear(
            self.kv_lora_rank,
            self.n_head * self.v_head_dim,
            has_bias=qkv_has_bias,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )

        self.q2l_proj.shard(((dp, 1), (1, 1)))
        self.l2q_nope_proj.shard(((dp, 1), (mp, 1)))
        self.l2q_pe_proj.shard(((dp, 1), (mp, 1)))
        self.kv2l_k_pe.shard(((dp, 1), (1, 1)))
        self.kv2l_latent_kv.shard(((dp, 1), (1, 1)))
        self.lkv2kv_k_nope.shard(((dp, 1), (mp, 1)))
        self.lkv2kv_v.shard(((dp, 1), (mp, 1)))

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.lq_norm.shard((dp, 1, 1))
            self.lkv_norm.shard((dp, 1, 1))

        if parallel_config.use_seq_parallel:
            self.q2l_proj.shard(((dp * mp, 1), (1, 1)))
            self.kv2l_k_pe.shard(((dp * mp, 1), (1, 1)))
            self.kv2l_latent_kv.shard(((dp * mp, 1), (1, 1)))
            self.lq_norm.shard((dp * mp, 1))
            self.lkv_norm.shard((dp * mp, 1))

        self.wo = Linear(in_channels=self.n_head * self.v_head_dim,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
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
        self.dim_slice_4d = P.StridedSlice()
        self.dim_slice_3d = P.StridedSlice()
        self.pe_concat = P.Concat(3)
        self.sum_test = P.ReduceSum()
        self.mul_zeros = P.Mul()
        self.v_zeros = Tensor(np.array([0] * (self.q_head_dim - self.v_head_dim)))

        self.apply_rotary_emb = DeepSeekV2RotaryEmbedding(self.qk_rope_head_dim,
                                                          rotary_dtype, use_rope_slice=use_rope_slice)

        if parallel_config.recompute.select_recompute and not self.use_flash_attention:
            self.apply_rotary_emb.recompute()
            self.tile_kv.recompute()
            self.batch_matmul_q_k.recompute()
            self.mul.recompute()
            self.add.recompute()
            self.cast_attn.recompute()
            self.softmax.recompute()
            self.batch_matmul.recompute()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.tile_kv.shard(((dp, mp, 1, 1),))
            self.slice_qkv.shard(((dp, mp, 1, 1),))
            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
        else:
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
            self.mul_zeros.shard(((dp, mp, 1, 1), ()))

            self.apply_rotary_emb.shard(parallel_config)

            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.n_head,
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  input_layout="BNSD",
                                                  keep_prob=1.,
                                                  scale_value=self.scale_fa,
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
        # [bs, seq/1, hidden_size] -> [bs, seq/1, q_lora_rank]
        q = self.q2l_proj(x)
        # [bs, seq/1, q_lora_rank]
        if self.use_seq_parallel:
            input_q_shape = self.shape(q)
            q = self.reshape(q, (-1, q.shape[-1]))
            norm_q = self.lq_norm(q)
            norm_q = self.reshape(norm_q, input_q_shape)
        else:
            norm_q = self.lq_norm(q)

        q_nope = self.l2q_nope_proj(norm_q)
        q_nope = self.reshape(q_nope, (bs, seq_len, self.n_head, self.qk_nope_head_dim))
        q_nope = self.transpose(q_nope, (0, 2, 1, 3))

        q_pe = self.l2q_pe_proj(norm_q)
        q_pe = self.reshape(q_pe, (bs, seq_len, self.n_head, self.qk_rope_head_dim))
        q_pe = self.transpose(q_pe, (0, 2, 1, 3))

        k_pe = self.kv2l_k_pe(x)
        k_pe = self.reshape(k_pe, (bs, 1, seq_len, self.qk_rope_head_dim))
        k_pe = self.tile_kv(k_pe, (1, self.n_head, 1, 1))

        latent_kv = self.kv2l_latent_kv(x)
        latent_kv = self.reshape(latent_kv, (bs, seq_len, self.kv_lora_rank))
        if self.use_seq_parallel:
            latent_kv_shape = self.shape(latent_kv)
            latent_kv = self.reshape(latent_kv, (-1, latent_kv.shape[-1]))
            i_kv = self.lkv_norm(latent_kv)
            i_kv = self.reshape(i_kv, latent_kv_shape)
        else:
            i_kv = self.lkv_norm(latent_kv)
        k_nope = self.lkv2kv_k_nope(i_kv)
        k_nope = self.reshape(k_nope, (bs, seq_len, self.n_head, self.qk_nope_head_dim))
        k_nope = self.transpose(k_nope, (0, 2, 1, 3))

        value_states = self.lkv2kv_v(i_kv)
        value_states = self.reshape(value_states, (bs, seq_len, self.n_head, self.v_head_dim))
        value_states = self.transpose(value_states, (0, 2, 1, 3))

        q_pe, k_pe = self.apply_rotary_emb(q_pe, k_pe, freqs_cis)
        query_states = self.pe_concat((q_nope, q_pe))
        key_states = self.pe_concat((k_nope, k_pe))

        if self.use_past:
            value_states = self.pe_concat((value_states, k_pe))
            key_states = self.reshape(self.transpose(key_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            value_states = self.reshape(self.transpose(value_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            query_states = self.reshape(self.transpose(query_states, (0, 2, 1, 3)), (bs, seq_len, -1))
            context_layer = self.infer_attention(query_states, key_states, value_states, batch_valid_length,
                                                 block_tables, slot_mapping, freqs_cis, mask,
                                                 prefix_keys_values=prefix_keys_values)
            attn_out = self.dim_slice_3d(context_layer, (0, 0, 0),
                                         (bs, seq_len, self.n_head * self.v_head_dim),
                                         (1, 1, 1))
        else:
            if self.use_flash_attention:
                pad_zeros = self.mul_zeros(q_pe, 0)
                value_states = self.pe_concat((value_states, pad_zeros))
                context_layer = self.flash_attention(self.cast(query_states, self.dtype),
                                                     self.cast(key_states, self.dtype),
                                                     self.cast(value_states, self.dtype), mask)
                context_layer = self.dim_slice_4d(context_layer, (0, 0, 0, 0),
                                                  (bs, self.n_head, seq_len, self.v_head_dim),
                                                  (1, 1, 1, 1))
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
                 scaling_factor: Optional[Dict] = None,
                 return_extra_loss=True
                 ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.return_extra_loss = return_extra_loss
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


class DeepseekV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV2Config
    base_model_prefix = "deepseekv2"


class DeepseekV2Model(DeepseekV2PreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepSeekV2DecoderLayer`]
    Args:
        config(DeepseekV2Config): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of deepseek decoderlayer
    """

    def __init__(self,
                 config: DeepseekV2Config = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.max_position_embeddings = config.max_position_embeddings  # used for yarn rotary embedding.
        self.mtp_depth = 0
        if hasattr(config, "mtp_depth"):
            self.mtp_depth = config.mtp_depth

        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_flash_attention = config.use_flash_attention
        # only support flash attention in train and prefill predict process.
        if self.use_past:
            self.use_flash_attention = False
        if self.use_flash_attention:
            logger.info("Enable flash attention.")

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()
        self.concat = P.Concat(axis=1)
        self.concat_2d = P.Concat(axis=-1)
        self.zeros_op = P.Zeros()

        self.freqs_mgr = FreqsMgr(head_dim=self.qk_rope_head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embeddings,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)
        self.freqs_mgr.shard(config.parallel_config)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_past=config.use_past)

        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type,
                                             parallel_optimizer=True)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers + self.mtp_depth,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers + self.mtp_depth):
            layer = DeepSeekV2DecodeLayer(layer_id,
                                          dim=config.hidden_size,
                                          n_heads=config.num_heads,
                                          n_kv_heads=config.n_kv_heads,
                                          intermediate_size=config.intermediate_size,
                                          multiple_of=config.multiple_of,
                                          ffn_dim_multiplier=config.ffn_dim_multiplier,
                                          norm_eps=config.rms_norm_eps,
                                          qkv_has_bias=config.qkv_has_bias,
                                          qkv_concat=config.qkv_concat,
                                          compute_dtype=config.compute_dtype,
                                          layernorm_compute_dtype=config.layernorm_compute_type,
                                          softmax_compute_dtype=config.softmax_compute_type,
                                          rotary_dtype=config.rotary_dtype,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          use_flash_attention=config.use_flash_attention,
                                          block_size=config.block_size,
                                          num_blocks=config.num_blocks,
                                          use_rope_slice=config.use_rope_slice,
                                          parallel_config=config.parallel_config,
                                          moe_config=config.moe_config,
                                          kv_lora_rank=config.kv_lora_rank,
                                          q_lora_rank=config.q_lora_rank,
                                          qk_rope_head_dim=config.qk_rope_head_dim,
                                          v_head_dim=config.v_head_dim,
                                          qk_nope_head_dim=config.qk_nope_head_dim,
                                          max_position_embeddings=config.max_position_embeddings,
                                          scaling_factor=config.scaling_factor,
                                          return_extra_loss=config.return_extra_loss)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)

        self.mtp_hidden_fusers = nn.CellList()
        for i in range(self.mtp_depth):
            layer = MTPHiddenFuser(config)
            self.layer_setting(layer, config.num_layers + i)
            self.mtp_hidden_fusers.append(layer)
        self.mtp_embeddings = None
        if self.mtp_depth > 0:
            self.mtp_embeddings = MtpEmbeddingLayer(vocab_table_size=config.vocab_size)

        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type,
                                     fused_kernel=not get_predict_run_mode())

        self.tok_embeddings.pipeline_stage = 0
        if self.mtp_embeddings is not None:
            self.mtp_embeddings.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if config.parallel_config.pipeline_stage > 1:
            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            self.tok_embeddings.set_comm_fusion(2)
            self.norm_out.set_comm_fusion(2)
        else:
            self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.tok_embeddings.shard(config.parallel_config)
        if self.mtp_embeddings is not None:
            self.mtp_embeddings.shard(config.parallel_config)
        self.casual_mask.shard(config.parallel_config)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.norm_out.shard((dp, 1, 1))
            self.concat.shard(((dp, 1, 1), (dp, 1, 1)))
            self.slice.shard(((dp, 1),))
            self.concat_2d.shard(((dp, 1), (dp, 1)))
            self.zeros_op.shard(((dp, 1),))

        if config.parallel_config.use_seq_parallel:
            mp = config.parallel_config.model_parallel
            self.norm_out.shard((dp, mp, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  extra_loss=None):
        """
        Forward of deepseekv2 model.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.

        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask.prefill()
                if prefix_keys_values is not None:
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        h = self.cast(self.tok_embeddings(tokens), self.dtype)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))

        #h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h, extra_loss = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                           block_tables=block_tables, slot_mapping=slot_mapping,
                                           prefix_keys_values=prefix_kv, extra_loss=extra_loss)
        output = self.norm_out(h)
        for i in range(self.mtp_depth):
            layer_id = i + self.num_layers
            # shift tokens to match up next token
            tokens = self._shift_and_pad(tokens)
            h = self.mtp_hidden_fusers[i](h, self.mtp_embeddings(self.tok_embeddings.embedding_weight, tokens))
            prefix_kv = prefix_keys_values[layer_id] if prefix_keys_values is not None else None
            h, extra_loss = self.layers[layer_id](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                                  block_tables=block_tables, slot_mapping=slot_mapping,
                                                  prefix_keys_values=prefix_kv, extra_loss=extra_loss)
            output = self.concat((output, self.norm_out(h)))
        return output, extra_loss

    def _shift_and_pad(self, x):
        """implement roll with shift and pad."""
        bs, seq_len = self.shape(x)
        pad_zeros = self.zeros_op((bs, 1))
        x = self.slice(x, (0, 1), (bs, seq_len), (1, 1))
        x = self.concat_2d((x, self.cast(pad_zeros, x.dtype)))

        return x


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    r"""
    Provide DeepseekV2 training loss or logits through network.
    Args:
        config (DeepseekV2Config): The config of DeepseekV2 model.

    Inputs:
        input_ids(Tensor): The tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
        labels(Tensor): The tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
        input_position(Tensor): Current position, used by model.predict.
        position_ids(Tensor): Reserved param, not used.
        attention_mask(Tensor): Reserved param, not used.
        input_embeds(Tensor): Reserved param, not used.
        init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
            past value parameter used in the incremental prediction. Default True.
        batch_valid_length(Tensor): The past calculated the index with datatype int32, used for incremental
            prediction. Tensor of shape :math:`(batch_size,)`. Default None.
        batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
            Tensor of shape :math:`(batch_size,)`. Default None.
        zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
            Tensor of shape :math:`(seq_length,)`. Default None.

    Returns:
        Tensor, the loss or logits of the network.
    """

    @lazy_inline
    def __init__(self, config: DeepseekV2Config = None):
        super(DeepseekV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.init_extra_loss = Tensor([0], mstype.float32)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = DeepseekV2Model(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)

        if config.parallel_config.pipeline_stage > 1:
            self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.not_equal.shard(((dp, 1), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))
        else:
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()
        logger.info("Predict run mode:{}".format(self.predict_run_mode))

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get deepseekv2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        """Mindspore's feature, Set dynamic input for DeepSeekV2."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_input_position = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_init_reset = Tensor([False], mstype.bool_)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)
        logger.info("Set dynamic input for DeepSeekV2.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model when the use_past is enabled."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """DeepseekV2ForCausalLM forward.
        """
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output, extra_loss = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables,
                                        slot_mapping, prefix_keys_values, self.init_extra_loss)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                return logits
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask) + extra_loss
        return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
