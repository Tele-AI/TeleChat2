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
"""DeepseekV3 models' APIs."""
import math
from enum import Enum
from typing import Tuple, Optional, Dict
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, mint, ops, Parameter
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.common.initializer import Zero
from mindspore.communication._comm_helper import _is_initialized
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline, LayerSetting, check_fine_grain_interleave_valid, predict_lazy_inline
from mindformers.modules.layers import Linear, FreqsMgr, _check_input_dtype, _yarn_get_mscale
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.infer_attention import InferRotaryEmbedding, FlashAttention
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode, is_pynative
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_info, initialize_model_parallel
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.infer.core.norm import RMSNorm
from mindformers.experimental.infer.core.moe import RoutedParallelMLP, SharedParallelMLP, ParallelMoEV2
from mindformers.experimental.infer.core.transformer import ParallelMLP, VocabEmbedding

from deepseek3_config import DeepseekV3Config
from utils import convert_model_config

__all__ = ['InferenceDeepseekV3ForCausalLM', 'DeepseekV3Model']


class CacheConfig(Enum):
    KEY_VALUE_CACHE = 0
    KEY_CACHE = 1
    KEY_VALUE_CACHE_KVSCALE_CACHE = 2


class MLAPagedAttentionMgr(nn.Cell):
    r""" Paged Attention Manager for MLA, which only stores the cache of key_cache.

    Args:
            - **n_head** (int): The head num of query.
            - **head_dim** (int): The dim of head.
            - **n_kv_head** (int): The head num of key and value.
            - **kv_shape** (tuple): Shape of key and value: math:`(num_blocks, block_size, self.n_kv_head, head_dim)`.
            - **compute_dtype** (mstype): Compute dtype for infer attention. Default mstype.float16.
            - **parallel_decoding** (mstype): If open parallel decoding. Default False.
            - **scale_value** (mstype): The scale factor of score. Default None.
            - **mla_v_dim** (int): The dim of value in Multi-Latent Attention. Default 512.

    Inputs:
            - **key** (Tensor[float16, bfloat16]) - The key tensor.
            Input tensor of shape: math:`(B, S2, H2)` or math:`(B, N2, S2, D)`.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.

    Outputs:
            - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
            are the same as the key.
    """

    def __init__(self,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 kv_shape,
                 compute_dtype=mstype.float16,
                 parallel_decoding=False,
                 scale_value=None,
                 mla_v_dim=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.scale_value = 1 / math.sqrt(self.head_dim) if scale_value is None else scale_value

        self.key_cache = Parameter(Tensor(shape=kv_shape, dtype=compute_dtype, init=Zero()), name="key_cache",
                                   requires_grad=False)

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.paged_attention = ops.auto_generate.PagedAttention(self.n_heads,
                                                                self.scale_value,
                                                                self.n_kv_heads,
                                                                mla_v_dim=mla_v_dim)
        self.parallel_decoding = parallel_decoding

    def construct(self, key, slot_mapping):
        """The forward compute of single cache for Paged Attention."""
        return self.reshape_and_cache(key, None, self.key_cache, None, slot_mapping)

    def paged_attn(self, query, batch_valid_length, block_tables):
        """The forward compute of Paged Attention."""
        return self.paged_attention(query, self.key_cache, self.key_cache, block_tables, batch_valid_length)


class MLAInferAttention(nn.Cell):
    r"""Multi-Latent-Attention Layer for infer.

    This function contains the InferAttention primitives used with FlashAttention and PagedAttention for MLA infer.

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
        n_kv_head (int): The head num of key and value.
        pa_n_head_split (int): The query head num of paged attention op after split.
        pa_n_kv_head_split (int): The key and value head num of paged attention op after split.
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
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        use_rope_rotary_emb (bool): If use rotary embedding. Default True.
        rotary_cos_format (int): Choose the rotary embedding cos format. Default 0.
        rotary_dtype (mstype): Compute dtype for rope op. Default mstype.float16.
        compute_dtype (mstype): Compute dtype for infer attention. Default mstype.float16.
        parallel_decoding (mstype): If open parallel decoding. Default False.
        prefill_head_dim (int): The dim of head for prefill attention. Default None.

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
        - **prefix_keys_values** (Union[Tensor[float16, bfloat16], None]) - The prefix keys values.
        - **q_seq_lens** (Union[Tensor[int32], None]) - The query actual seq len.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.
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
                 use_alibi_mask=False,
                 compute_dtype=mstype.float16,
                 parallel_decoding=False,
                 prefill_head_dim=None,
                 ):
        super(MLAInferAttention, self).__init__()
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
        self.use_alibi_mask = use_alibi_mask
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.reshape = P.Reshape()

        self.is_pynative = is_pynative()
        if self.is_pynative:
            self.input_layout = "BSH"
        else:
            self.input_layout = "TH"
        self.use_attention_mask = not self.use_alibi_mask

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
        self.paged_attention_mgr = MLAPagedAttentionMgr(self.pa_n_head_split,
                                                        self.head_dim,
                                                        self.pa_n_kv_head_split,
                                                        kv_shape,
                                                        compute_dtype=self.compute_dtype,
                                                        parallel_decoding=parallel_decoding,
                                                        scale_value=self.scale_value)
        self.prefill_head_dim = prefill_head_dim

    def _prefill_attention(self, query, key, value, attn_mask, alibi_mask, actual_seq_qlen=None,
                           actual_seq_kvlen=None):
        """
        prefill attention
        """
        bs, seq_len, _ = query.shape
        prefill_head_dim = self.prefill_head_dim if self.prefill_head_dim else self.head_dim
        if not self.is_pynative:
            query = self.reshape(query, (-1, self.n_head * prefill_head_dim))
            key = self.reshape(key, (-1, self.n_head * prefill_head_dim))
            value = self.reshape(value, (-1, self.n_head * prefill_head_dim))
        output = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None,
                                      actual_seq_qlen, actual_seq_kvlen)
        output = self.reshape(output, (bs, seq_len, self.n_head * prefill_head_dim))
        return output

    def _incre_attention(self, query, batch_valid_length, block_tables):
        return self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables)

    def construct(self, query, key, value, batch_valid_length, block_tables,
                  attn_mask=None, alibi_mask=None):
        """ Forward process of the MLA Infer Attention Cell """
        if self.is_first_iteration:
            return self._prefill_attention(query, key, value, attn_mask, alibi_mask, batch_valid_length,
                                           batch_valid_length)
        return self._incre_attention(query, batch_valid_length, block_tables)


class DeepseekV3Attention(nn.Cell):
    r"""
    This is an implementation of self-attention mechanism in DeepSeek-V3.

    Args:
            - **dim** (int): The hidden size of the input.
            - **n_heads** (int): The number of the heads.
            - **n_kv_heads** (int): The number of key_value heads that should be used to implement
                                    Grouped Query Attention.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
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
                pass the single step's input tensor, and loop it. Default True.
            - **use_flash_attention** (bool): Whether to enable flash attention ops. Default True.
            - **block_size** (int): The maximum number of tokens in one block can have when using paged attention.
                Default 16.
            - **num_blocks** (int): The maximum number of blocks when using paged attention. Default 512.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.
            - **kv_lora_rank** (int): kv_lora_rank for Multi-Latent-Attention. Default 512.
            - **q_lora_rank** (int): q_lora_rank for Multi-Latent-Attention. Default 1536.
            - **qk_rope_head_dim** (int): qk_rope_head_dim for Multi-Latent-Attention. Default 64.
            - **v_head_dim** (int): v_head_dim for Multi-Latent-Attention. Default 128.
            - **qk_nope_head_dim** (int): qk_nope_head_dim for Multi-Latent-Attention. Default 128.
            - **max_position_embeddings** (int): The maximum sequence length that this model might ever be used with.
                Default 2048.
            - **scaling_factor** (float): Scaling factor of Multi-Latent Attention. Default None.
            - **norm_eps** (float): The epsilon value of the denominator. Default 1e-5.
            - **layernorm_compute_dtype** (dtype.Number): The computation type of layernorm. Default mstype.float32.

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
                 dim=512,
                 n_heads=8,
                 n_kv_heads=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=True,
                 use_flash_attention=True,
                 block_size=None,
                 num_blocks=None,
                 parallel_config=TransformerOpParallelConfig(),
                 kv_lora_rank=512,
                 q_lora_rank=1536,
                 qk_rope_head_dim=64,
                 v_head_dim=128,
                 qk_nope_head_dim=128,
                 max_position_embeddings=2048,
                 scaling_factor=None,
                 norm_eps=1e-5,
                 layernorm_compute_dtype=mstype.float32,
                 config: DeepseekV3Config = None
                 ):
        super().__init__()
        self.hidden_size = dim
        self.tensor_parallel_group_size = get_tp_world_size()
        self.n_head = n_heads
        self.n_local_heads = n_heads // self.tensor_parallel_group_size
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.kv_dim = self.n_kv_head * self.head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling_factor = scaling_factor
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.qkv_concat = config.qkv_concat

        if not self.use_past:
            raise ValueError("For 'DeepseekV3Attention', the use_past must be enabled.")

        if not self.use_flash_attention:
            raise ValueError("For 'DeepseekV3Attention', the use_flash_attention must be enabled.")

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))
        self.shape = P.Shape()
        self.cast = P.Cast()

        if self.q_lora_rank == 0:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.n_head * self.q_head_dim,
                config=parallel_config,
                bias=qkv_has_bias,
                param_init_type=param_init_type,
                compute_dtype=compute_dtype
            )
            # 1. kv2l: kv latent vector; 2. lkv_norm: latent vector of kv normalization
            self.kv2l = Linear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=qkv_has_bias,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type
            )
        else:
            if self.qkv_concat:
                self.qkv2l = Linear(
                    self.hidden_size,
                    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                    has_bias=qkv_has_bias,
                    compute_dtype=compute_dtype,
                    param_init_type=param_init_type
                )
            else:
                self.q2l_proj = Linear(
                    self.hidden_size,
                    self.q_lora_rank,
                    has_bias=qkv_has_bias,
                    compute_dtype=compute_dtype,
                    param_init_type=param_init_type
                )
                # 1. kv2l: kv latent vector; 2. lkv_norm: latent vector of kv normalization
                self.kv2l = Linear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    has_bias=qkv_has_bias,
                    compute_dtype=compute_dtype,
                    param_init_type=param_init_type
                )
            self.lq_norm = RMSNorm(self.q_lora_rank, norm_eps, compute_type=layernorm_compute_dtype)
            self.l2q_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.n_head * self.q_head_dim,
                config=parallel_config,
                bias=qkv_has_bias,
                param_init_type=param_init_type,
                compute_dtype=compute_dtype
            )

        self.lkv_norm = RMSNorm(self.kv_lora_rank, norm_eps, compute_type=layernorm_compute_dtype)
        self.lkv2kv_k_nope = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_head * self.qk_nope_head_dim,
            config=parallel_config,
            bias=qkv_has_bias,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype
        )

        self.lkv2kv_v = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_head * self.v_head_dim,
            config=parallel_config,
            bias=qkv_has_bias,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype
        )

        self.wo = RowParallelLinear(
            self.n_head * self.v_head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            config=parallel_config,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
        )

        self.scale_fa = 1. / math.sqrt(self.q_head_dim)
        if self.scaling_factor is not None:
            mscale_all_dim = self.scaling_factor.get("mscale_all_dim", 0)
            factor = self.scaling_factor["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(factor, mscale_all_dim)
                self.scale_fa = mscale * mscale / (math.sqrt(self.q_head_dim))

        self.reshape = P.Reshape()
        self.tile_kv = P.Tile()
        self.dim_slice_4d = P.Slice()
        self.kpe_concat = P.Concat(2)
        self.pe_concat = P.Concat(3)
        self.qabsorb_matmul = P.BatchMatMul()
        self.outabsorb_matmul = P.BatchMatMul(transpose_b=True)

        self.infer_attention = MLAInferAttention(self.n_local_heads,
                                                 self.kv_lora_rank + self.qk_rope_head_dim,
                                                 1,
                                                 scale_value=self.scale_fa,
                                                 next_tokens=0,
                                                 block_size=self.block_size,
                                                 num_blocks=self.num_blocks,
                                                 compute_dtype=compute_dtype,
                                                 prefill_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim)

        self.apply_rotary_emb = InferRotaryEmbedding(rotary_cos_format=2)


    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None):
        """ Forward process of the DeepseekV3Attention. """
        ori_dtype = x.dtype
        bs, seq_len, _ = self.shape(x)

        if self.q_lora_rank == 0:
            q = self.q_proj(x)
            latent_kv_all = self.kv2l(x)
            latent_kv, k_pe = mint.split(latent_kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        else:
            if self.qkv_concat:
                qkv2l = self.qkv2l(x)
                q, latent_kv, k_pe = mint.split(qkv2l, [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
                                                dim=-1)
                norm_q = self.lq_norm(q)
                q = self.l2q_proj(norm_q)
            else:
                q = self.q2l_proj(x)
                norm_q = self.lq_norm(q)
                q = self.l2q_proj(norm_q)
                latent_kv_all = self.kv2l(x)
                latent_kv, k_pe = mint.split(latent_kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q = self.reshape(q, (bs, seq_len, self.n_local_heads, self.q_head_dim))
        q_nope, q_pe = mint.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        i_kv = self.lkv_norm(latent_kv)

        k_pe = self.reshape(k_pe, (bs, seq_len, 1, self.qk_rope_head_dim))
        q_pe, k_pe = self.apply_rotary_emb(q_pe, k_pe, freqs_cis, batch_valid_length)
        q_pe = self.reshape(q_pe, (bs, seq_len, self.n_local_heads, self.qk_rope_head_dim))
        k_pe = self.reshape(k_pe, (bs, seq_len, 1, self.qk_rope_head_dim))

        key_states_cache = self.kpe_concat((i_kv, k_pe.view(bs, seq_len, self.qk_rope_head_dim)))
        key_out = self.infer_attention.paged_attention_mgr(key_states_cache, slot_mapping)
        q_nope = ops.depend(q_nope, key_out)

        if self.is_first_iteration:
            o_k_nope = self.lkv2kv_k_nope(i_kv)
            o_v = self.lkv2kv_v(i_kv)
            k_nope = self.reshape(o_k_nope, (bs, seq_len, self.n_local_heads, self.qk_nope_head_dim))
            value_states = self.reshape(o_v, (bs, seq_len, self.n_local_heads, self.v_head_dim))
            query_states = self.pe_concat((q_nope, q_pe))
            k_pe = self.tile_kv(k_pe, (1, 1, self.n_local_heads, 1))
            key_states = self.pe_concat((k_nope, k_pe))
            value_states = self.pe_concat((value_states, k_pe))

            key_states = key_states.view(bs, seq_len, -1)
            value_states = value_states.view(bs, seq_len, -1)
            query_states = query_states.view(bs, seq_len, -1)

            context_layer = self.infer_attention(query_states, key_states, value_states, batch_valid_length,
                                                 block_tables, mask)

            context_layer = context_layer.view(bs, seq_len, self.n_local_heads, self.q_head_dim)
            context_layer = self.dim_slice_4d(context_layer, (0, 0, 0, 0), (bs, seq_len, self.n_local_heads,
                                                                            self.v_head_dim))
            attn_out = context_layer.view(bs, seq_len, self.n_local_heads * self.v_head_dim)
            output = self.wo(attn_out)
            output = self.cast(output, ori_dtype)
            return output

        q_absorb = self.lkv2kv_k_nope.weight.view(self.n_local_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.lkv2kv_v.weight.view(self.n_local_heads, self.v_head_dim, self.kv_lora_rank)
        q_nope = self.qabsorb_matmul(q_nope.transpose(0, 2, 1, 3), q_absorb).transpose(0, 2, 1, 3)
        query_states = self.pe_concat((q_nope, q_pe))
        query_states = query_states.view(bs, seq_len, -1)
        key_states = key_states_cache
        context_layer = self.infer_attention(query_states, key_states, key_states, batch_valid_length,
                                             block_tables, attn_mask=mask)
        context_layer = context_layer.view(bs, seq_len, self.n_local_heads, -1).transpose(0, 2, 1, 3)
        attn_out = self.outabsorb_matmul(context_layer, out_absorb).transpose(0, 2, 1, 3)
        attn_out = attn_out.view(bs, seq_len, self.n_local_heads * self.v_head_dim)
        output = self.wo(attn_out)
        output = self.cast(output, ori_dtype)
        return output


class DeepseekV3MoE(Cell):
    r"""
    This is an implementation of self-attention mechanism in DeepSeek-V3.

    Args:
        - **config** (Config): Model config of DeepSeek-V3.

    Inputs:
        - **x** (Tensor): Should be `[batch, seq_length, hidden_size]`. Float tensor.

    Outputs:
        - **output** (Tensor): The output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.
    """

    def __init__(self, config):
        super(DeepseekV3MoE, self).__init__()
        self.config = config
        self.parallel_config = config.parallel_config
        self.moe_config = config.moe_config
        self.moe_config.router_dense_type = config.router_dense_type
        intermediate_size = self.moe_config.moe_intermediate_size

        if self.parallel_config.expert_parallel == 1:
            ffn = RoutedParallelMLP(config)
            self.routed_experts = ParallelMoEV2(ffn, self.config.hidden_size, self.moe_config)
        else:
            raise NotImplementedError("For ParallelMoEV2, `expert_parallel` is not supported for now.")

        if self.moe_config.shared_expert_num is not None:
            intermediate_size = intermediate_size * self.moe_config.shared_expert_num
            self.shared_experts = SharedParallelMLP(config, intermediate_size)
        self.add = P.Add()

    def construct(self, x):
        output = self.routed_experts(x)
        if self.moe_config.shared_expert_num is not None:
            output = self.add(output, self.shared_experts(x))
        return output


class DeepseekV3DecodeLayer(nn.Cell):
    r"""
    Transformer Layer. This is an implementation of the single layer of the transformer
    encoder layer, including multihead attention and feedward layer.

    Args:
            - **layer_id** (int): The layer id of current transformer block layer.
            - **dim** (int): The hidden size of the input.
            - **num_heads** (int): The number of the heads.
            - **n_kv_heads** (int): The number of key_value heads that should be used to implement
                                    Grouped Query Attention.
            - **norm_eps** (float): The epsilon value of the denominator. Default 1e-5.
            - **compute_dtype** (dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            - **layernorm_compute_type** (dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction. In the first step,
                set the is_first_iteration to be True by `model.add_flags_recursive(is_first_iteration=True)`,
                and pass the full inputs. Then, set the is_first_iteration to be False by
                `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default True.
            - **moe_config** (MoEConfig): The MoE configuration. Default: ``default_moe_config`` ,
                an instance of `MoEConfig` with default args.
            - **use_flash_attention** (bool): Whether to enable flash attention ops. Default True.
            - **block_size** (int): The maximum number of tokens in one block can have when using paged attention.
                Default 16.
            - **num_blocks** (int): The maximum number of blocks when using paged attention. Default 512.
            - **parallel_config** (OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.
            - **kv_lora_rank** (int): kv_lora_rank for Multi-Latent-Attention. Default 512.
            - **q_lora_rank** (int): q_lora_rank for Multi-Latent-Attention. Default 1536.
            - **qk_rope_head_dim** (int): qk_rope_head_dim for Multi-Latent-Attention. Default 64.
            - **v_head_dim** (int): v_head_dim for Multi-Latent-Attention. Default 128.
            - **qk_nope_head_dim** (int): qk_nope_head_dim for Multi-Latent-Attention. Default 128.
            - **max_position_embeddings** (int): The maximum sequence length that this model might ever be used with.
                Default 2048.
            - **scaling_factor** (float): Scaling factor of Multi-Latent Attention. Default None.
            - **config** (Config): Model config of DeepSeek-V3. Default None.

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
                shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past
                is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

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
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float32,
                 layernorm_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 use_past=True,
                 moe_config=None,
                 use_flash_attention=True,
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
                 config: DeepseekV3Config = None
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
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.add = P.Add()
        self.ffn_norm = RMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention_norm = RMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)

        self.attention = DeepseekV3Attention(dim=dim,
                                             n_heads=n_heads,
                                             n_kv_heads=n_kv_heads,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type,
                                             qkv_has_bias=qkv_has_bias,
                                             use_past=use_past,
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
                                             norm_eps=norm_eps,
                                             layernorm_compute_dtype=layernorm_compute_dtype,
                                             config=config)

        self.expert_num = 1 if moe_config is None else moe_config.expert_num
        self.shared_expert_num = 0 if moe_config is None else moe_config.shared_expert_num

        # Feed Forward Network
        self.first_k_dense = (moe_config.first_k_dense_replace and layer_id < moe_config.first_k_dense_replace)
        if self.first_k_dense:
            logger.warning("first_k_dense_replace is provided in MoEConfig, "
                           "a normal dense FFN will be used in this block.")
            self.feed_forward = ParallelMLP(config)
        else:
            self.feed_forward = DeepseekV3MoE(config=config)

        self.predict_run_mode = get_predict_run_mode()
        if self.predict_run_mode:
            self.no_inline = False

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)

        input_x = self.attention_norm(x)
        h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables, slot_mapping)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        ffn_out = self.feed_forward(ffn_norm)
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


class DeepseekV3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV3Config
    base_model_prefix = "deepseekv3"


class DeepseekV3Model(DeepseekV3PreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]
    Args:
        config(DeepseekV3Config): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of deepseek decoderlayer
    """

    def __init__(self,
                 config: DeepseekV3Config = None):
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
        self.max_position_embeddings = config.max_position_embeddings

        self.is_first_iteration = True
        self.is_pynative = is_pynative()
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()

        self.freqs_mgr = FreqsMgr(head_dim=self.qk_rope_head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embeddings,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)

        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_past=config.use_past)

        if config.parallel_config.vocab_emb_dp:
            self.tok_embeddings = VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                         embedding_dim=config.hidden_size,
                                                         parallel_config=config.parallel_config,
                                                         init_method="normal",
                                                         init_type=config.param_init_type)
        else:
            self.tok_embeddings = VocabEmbedding(num_embeddings=config.vocab_size,
                                                 embedding_dim=config.hidden_size,
                                                 param_init_type=config.param_init_type,
                                                 param_init="normal")

        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)

        for layer_id in range(config.num_layers):
            layer = DeepseekV3DecodeLayer(layer_id,
                                          dim=config.hidden_size,
                                          n_heads=config.num_heads,
                                          n_kv_heads=config.n_kv_heads,
                                          norm_eps=config.rms_norm_eps,
                                          qkv_has_bias=config.qkv_has_bias,
                                          compute_dtype=config.compute_dtype,
                                          layernorm_compute_dtype=config.layernorm_compute_type,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          use_flash_attention=config.use_flash_attention,
                                          block_size=config.block_size,
                                          num_blocks=config.num_blocks,
                                          parallel_config=config.parallel_config,
                                          moe_config=config.moe_config,
                                          kv_lora_rank=config.kv_lora_rank,
                                          q_lora_rank=config.q_lora_rank,
                                          qk_rope_head_dim=config.qk_rope_head_dim,
                                          v_head_dim=config.v_head_dim,
                                          qk_nope_head_dim=config.qk_nope_head_dim,
                                          max_position_embeddings=config.max_position_embeddings,
                                          scaling_factor=config.scaling_factor,
                                          config=config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                compute_type=config.layernorm_compute_type)

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """
        Forward of deepseekv3 model.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.
            block_tables(Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping(Tensor[int32]): Store token cache physical slot index.

        Returns:
            output: Tensor, the output of deepseekv3 decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                if not self.is_pynative:
                    mask = self.casual_mask.prefill()
                else:
                    mask = self.casual_mask(tokens)
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)

        else:
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)

        h = self.cast(self.tok_embeddings(tokens), self.dtype)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))

        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                               block_tables=block_tables, slot_mapping=slot_mapping)
        output = self.norm_out(h)
        return output


class InferenceDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    r"""
    Provide DeepseekV3 logits through network.
    Args:
        config (DeepseekV3Config): The config of DeepseekV3 model.

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
        block_tables(Tensor, optional): Int64 type Tensor, store mapping tables for each sequence. Default None.
        slot_mapping(Tensor, optional): Int32 type Tensor, token cache physical slot index. Default None.

    Returns:
        Tensor. If it is in prediction mode, the output Tensor contains logits;
        If it is in evaluation mode, the output Tensor contains logits, tokens, and input masks.
    """

    @lazy_inline
    def __init__(self, config: DeepseekV3Config = None):
        super(InferenceDeepseekV3ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)

        self.config = convert_model_config(config)
        self.parallel_config = self.config.parallel_config

        tp_group = get_group_info('tp').group is None
        ep_group = get_group_info('ep').group is None
        pp_group = get_group_info('pp').group is None
        all_groups_initialized = tp_group and ep_group and pp_group

        if all_groups_initialized and _is_initialized():
            initialize_model_parallel(pipeline_model_parallel_size=self.parallel_config.pipeline_model_parallel_size,
                                      expert_model_parallel_size=self.parallel_config.expert_parallel,
                                      tensor_model_parallel_size=self.parallel_config.tensor_model_parallel_size,
                                      order='tp-ep-dp-pp')

        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = DeepseekV3Model(config=config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config.parallel_config,
            bias=False,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            weight_init="normal",
            gather_output=True
        )
        self.prefill_gather_flatten = P.Gather()

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()
        logger.info("Predict run mode:{}".format(self.predict_run_mode))

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """ Get deepseekv3 model input tuple for transform ckpt. """
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, labels, None, None, None, None, None, None, None, None, None, \
               slot_mapping

    def set_dynamic_inputs(self, **kwargs):
        """ Mindspore's feature, Set dynamic input for DeepseekV3. """
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_init_reset = True
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, dynamic_init_reset,
                        dynamic_batch_valid_length, None, None, dynamic_block_tables,
                        dynamic_slot_mapping)
        logger.info("Set dynamic input for DeepseekV3.")

    def pre_gather_func(self, pre_gather, output, batch_valid_length):
        """Pre gather operation in infer mode."""
        if not pre_gather:
            return output
        if pre_gather:
            if self.config.is_dynamic:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = self.prefill_gather_flatten(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
            else:
                output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        return output

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """ DeepseekV3ForCausalLM forward. """
        bsz, _ = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        output = self.pre_gather_func(pre_gather, output, batch_valid_length)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is not None and labels.ndim > 1:
            label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
            input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if self.predict_run_mode:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
            return logits
        return logits, tokens, input_mask
