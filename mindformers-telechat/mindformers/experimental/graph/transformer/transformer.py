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
""" For transformer """
import math
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindformers.experimental.graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.graph.transformer.dropout import Dropout
from mindformers.experimental.graph.transformer.fused_softmax import FusedScaleMaskSoftmax
from mindformers.experimental.graph.transformer.rotary_pos_embedding import ApplyRotaryPosEmb
from mindformers.experimental.graph.transformer.utils import get_attn_mask_func, LayerSetting
from mindformers.experimental.graph.activation import get_activation
from mindformers.experimental.graph.transformer.norm import get_norm
from mindformers.experimental.graph.transformer.flash_attention import FlashAttention
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

__all__ = [
    "ParallelMLP",
    "CoreAttention",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformer",
    "CausalMaskGenerate"
]


class ParallelMLP(nn.Cell):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (dict): Configuration.
        is_expert (book): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

        - **output_bias** (Tensor) - output_bias tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config: TransformerConfig, is_expert: bool = False):
        super(ParallelMLP, self).__init__()
        if is_expert:
            raise NotImplementedError("For ParallelMPL, `is_expert` is not supported for now")
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        if config.intermediate_size is not None:
            self.ffn_hidden_size = config.intermediate_size
        elif config.multiple_of is not None:
            if config.ffn_dim_multiplier is not None:
                self.ffn_hidden_size = int((config.ffn_dim_multiplier + 0.01) * self.ffn_hidden_size)
            self.ffn_hidden_size = int(2 * self.ffn_hidden_size / 3)
            self.ffn_hidden_size = config.multiple_of * (
                (self.ffn_hidden_size + config.multiple_of - 1) // config.multiple_of
            )
        self.mlp_has_bias = config.add_bias_linear
        self.mlp_has_gate = getattr(config, 'mlp_has_gate', False)
        self.gated_linear_unit = getattr(config, 'gated_linear_unit', False)
        if self.mlp_has_gate and self.gated_linear_unit:
            raise ValueError(
                "For 'ParallelMLP', 'mlp_has_gate' and 'gated_linear_unit' cannot be True at the same time.")
        self.init_method = config.init_method
        self.activation_type = config.hidden_act
        self.compute_dtype = config.compute_dtype
        self.parallel_config = config

        if self.gated_linear_unit:
            self.mapping_ffn_hidden_size = self.ffn_hidden_size * 2
        else:
            self.mapping_ffn_hidden_size = self.ffn_hidden_size

        if self.mlp_has_gate:
            self.gating = ColumnParallelLinear(self.hidden_size,
                                               self.ffn_hidden_size,
                                               self.parallel_config,
                                               bias=self.mlp_has_bias,
                                               compute_dtype=self.compute_dtype,
                                               is_expert=is_expert,
                                               skip_bias_add=True,
                                               init_method=self.init_method
                                               )
            self.mul = P.Mul()

        self.mapping = ColumnParallelLinear(self.hidden_size,
                                            self.mapping_ffn_hidden_size,
                                            self.parallel_config,
                                            bias=self.mlp_has_bias,
                                            compute_dtype=self.compute_dtype,
                                            is_expert=is_expert,
                                            skip_bias_add=True,
                                            init_method=self.init_method
                                            )

        if self.activation_type is not None:
            self.activation_func = get_activation(self.activation_type, config=self.parallel_config)
        else:
            self.activation_func = None

        self.projection = RowParallelLinear(self.ffn_hidden_size,
                                            self.hidden_size,
                                            self.parallel_config,
                                            bias=self.mlp_has_bias,
                                            compute_dtype=self.compute_dtype,
                                            is_expert=is_expert,
                                            skip_bias_add=True,
                                            init_method=self.init_method
                                            )
        self.add = P.Add()
        self.shard(self.parallel_config)

    def construct(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """ Construct function of mlp block. """
        # [bs, seq_len, hidden_size] -> [bs, seq_len, ffn_hidden_size]
        intermediate_parallel, bias_parallel = self.mapping(hidden_states)
        if self.mlp_has_gate:
            # [bs, seq_len, hidden_size] -> [bs, seq_len, ffn_hidden_size]
            gate, gate_bias = self.gating(hidden_states)
            if self.activation_func is not None:
                gate = self.activation_func(gate, gate_bias)
            if bias_parallel is not None:
                intermediate_parallel = self.add(intermediate_parallel, bias_parallel)
            # [bs, seq_len, ffn_hidden_size]
            intermediate_parallel = self.mul(intermediate_parallel, gate)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel, bias_parallel)
        # [bs, seq_len, ffn_hidden_size] -> [bs, seq_len, hidden_size]
        output, output_bias = self.projection(intermediate_parallel)
        return output, output_bias

    def shard(self, config: TransformerConfig):
        if self.mlp_has_gate:
            dp = config.data_parallel if config.data_parallel is not None else 1
            cp = config.context_parallel if config.context_parallel is not None else 1
            tp = config.tensor_parallel if config.tensor_parallel is not None else 1
            mul_in_strategy = ((dp, cp, tp), (dp, cp, tp))
            self.mul.shard(in_strategy=mul_in_strategy)


class CoreAttention(nn.Cell):
    r"""
    multi-head self attention.

    Args:
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attn_mask_type (str): Attention type. Support ['self_attn', 'cross_attn'].

    Inputs:
        - **query** (Tensor) - Tensor of query matrix.
        - **key** (Tensor) - Tensor of key matrix.
        - **value** (Tensor) - Tensor of value matrix.
        - **attention_mask** (Tensor) - Tensor of attention mask matrix.

    Outputs:
        - **attn_output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, layer_number, config: TransformerConfig, attn_mask_type=None):
        super(CoreAttention, self).__init__()
        if attn_mask_type:
            raise NotImplementedError("For CoreAttention, 'attn_mask_type' is not supported for now.")

        self.config = config
        self.layer_index = max(1, layer_number)
        self.compute_dtype = self.config.compute_dtype
        if self.config.softmax_compute_dtype == mstype.float32:
            self.attention_softmax_in_fp32 = True
        else:
            self.attention_softmax_in_fp32 = False
        self.dropout_rate = self.config.attention_dropout
        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(self.hidden_size, self.num_heads))

        self.head_dim = self.hidden_size // self.num_heads

        coeff = None
        norm_factor = math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_index
            norm_factor *= coeff
        self.inv_norm_factor = Tensor(1.0 / norm_factor, dtype=self.compute_dtype)

        self.scale_mask_softmax = FusedScaleMaskSoftmax(config=config,
                                                        mask_func=get_attn_mask_func(self.config.mask_func_type)(
                                                            config),
                                                        scale=coeff,
                                                        softmax_in_fp32=self.attention_softmax_in_fp32,
                                                        input_in_fp16=self.fp16,
                                                        input_in_bf16=self.bf16,
                                                        softmax_compute_dtype=config.softmax_compute_dtype)
        self.dropout = Dropout(self.dropout_rate)
        self.bmm_qk = P.BatchMatMul(transpose_b=True)
        self.bmm_qkv = P.BatchMatMul(transpose_b=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.merge_head_transpose = P.Transpose()
        self.mul = P.Mul()
        self.cast = P.Cast()

        self.shard(config)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """construct function."""
        # score: [B, N, S, D]
        score = self.bmm_qk(query_layer, key_layer)
        score = self.mul(score, self.inv_norm_factor)

        attention_probs = self.scale_mask_softmax(score, attention_mask)
        attention_probs = self.dropout(attention_probs)

        if self.fp16:
            value_layer = self.cast(value_layer, mstype.float16)
        if self.bf16:
            value_layer = self.cast(value_layer, mstype.bfloat16)

        # [B, N, S, S] * [B, N, S, D] -> [B, N, S, D]
        weighted_values = self.bmm_qkv(self.cast(attention_probs, self.compute_dtype), value_layer)
        # [B, N, S, D] -> [B, S, N*D]
        attn_output = self._merge_heads(weighted_values)
        attn_output = self.cast(attn_output, self.compute_dtype)

        return attn_output

    def _merge_heads(self, x):
        # [bs, n_head, seq, head_dim]
        x = self.merge_head_transpose(x, (0, 2, 1, 3))
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def shard(self, config: TransformerConfig):
        """sharding parameters"""
        dp = 1 if config is None else config.data_parallel
        tp = 1 if config is None else config.tensor_parallel
        cp = 1 if config is None else config.context_parallel

        dropout_strategy = (dp, tp, cp, 1)
        self.dropout.shard(strategy=dropout_strategy)
        self.bmm_qkv.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.mul.shard(((dp, tp, cp, 1), ()))
        self.bmm_qk.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.merge_head_transpose.shard(((dp, tp, cp, 1),))


class ParallelAttention(nn.Cell):
    r"""
    Parallel attention block.

    Args:
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): Configuration.
        attention_type (str): Attention type. Support ['self_attn', 'cross_attn']. Default: 'self_attn'.
        attn_mask_type (str): attention mask type. Default None.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **bias** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config: TransformerConfig, layer_number, attention_type='self_attn', attn_mask_type=None):
        super(ParallelAttention, self).__init__()
        if attn_mask_type:
            raise NotImplementedError("For ParallelAttention, 'attn_mask_type' is not supported for now.")
        self.config = config
        self.compute_dtype = self.config.compute_dtype
        self.use_gqa = self.config.group_query_attention
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = self.config.num_query_groups if self.use_gqa else self.num_heads
        self.hidden_size = self.config.hidden_size
        self.use_flash_attention = self.config.use_flash_attn
        self.parallel_config = self.config
        self.qkv_concat = self.config.qkv_concat
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        tp = 1 if self.config is None else self.config.tensor_parallel
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(self.hidden_size, self.num_heads))
        if self.kv_num_heads % tp != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'kv_num_heads' must be a multiple of "
                             "'parallel_config.tensor_parallel', but got the kv_num_heads is {} "
                             "and the parallel_config.tensor_parallel  is {}."
                             .format(self.kv_num_heads, tp))

        self.head_dim = self.hidden_size // self.num_heads
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.n_rep = self.num_heads // self.kv_num_heads
        self.layer_index = max(1, layer_number)
        self.attn_type = attention_type
        self.norm_factor = math.sqrt(self.head_dim)

        if self.attn_type == 'self_attn':
            if self.qkv_concat:
                self.qkv_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size + 2 * self.kv_hidden_size,
                                                     config=self.config,
                                                     bias=self.config.add_qkv_bias or self.config.add_bias_linear,
                                                     compute_dtype=self.config.compute_dtype,
                                                     init_method=self.config.init_method
                                                     )
                self.reshape_concat = P.Reshape()
                self.split_qkv = ms.ops.auto_generate.SplitWithSize().add_prim_attr("skip_redistribution", True)
            else:
                self.q_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size, config=self.config,
                                                   bias=self.config.add_qkv_bias or self.config.add_bias_linear,
                                                   compute_dtype=self.config.compute_dtype,
                                                   init_method=self.config.init_method
                                                   )
                self.k_proj = ColumnParallelLinear(self.hidden_size, self.kv_hidden_size, config=self.config,
                                                   bias=self.config.add_qkv_bias or self.config.add_bias_linear,
                                                   compute_dtype=self.config.compute_dtype,
                                                   init_method=self.config.init_method
                                                   )
                self.v_proj = ColumnParallelLinear(self.hidden_size, self.kv_hidden_size, config=self.config,
                                                   bias=self.config.add_qkv_bias or self.config.add_bias_linear,
                                                   compute_dtype=self.config.compute_dtype,
                                                   init_method=self.config.init_method
                                                   )
        elif self.attn_type == 'cross_attn':
            self.q_proj, self.kv_proj, self.split_kv = self._cross_attn_init()
        else:
            raise NotImplementedError(f"attention_type shuold be self_attn or cross_attn, but got {self.attn_type}")

        self.core_attention = CoreAttention(self.layer_index, self.config)
        self.out_proj = RowParallelLinear(self.hidden_size, self.hidden_size, input_is_parallel=False,
                                          config=self.config,
                                          bias=self.config.add_bias_linear,
                                          compute_dtype=self.config.compute_dtype,
                                          init_method=self.config.init_method
                                          )
        if self.use_flash_attention:
            self.input_layout = "BNSD"
            self.sparse_mode = 2 if self.use_attn_mask_compression else 0
            self.flash_attention = FlashAttention(head_num=self.num_heads,
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  input_layout=self.input_layout,
                                                  keep_prob=1.0,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  sparse_mode=self.sparse_mode,
                                                  use_attention_mask=True)
        self.apply_rotary_pos_emb = ApplyRotaryPosEmb(self.parallel_config)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merge_head_transpose = P.Transpose()
        self.tile_kv = P.Tile()
        self.cat = P.Concat(2)
        self.shard(self.config)

    def _cross_attn_init(self):
        """ cross attention init. """
        if self.use_gqa:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")

        if self.hidden_size != self.kv_hidden_size:
            raise ValueError("self.hidden_size and self.kv_hidden_size must be equal in cross_attn!!")

        q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            compute_dtype=self.config.compute_dtype,
            init_method=self.config.init_method
        )
        kv_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.kv_hidden_size,
            config=self.config,
            bias=self.config.add_bias_linear,
            compute_dtype=self.config.compute_dtype,
            init_method=self.config.init_method
        )
        split_kv = ms.ops.auto_generate.SplitWithSize().add_prim_attr("skip_redistribution", True)

        return q_proj, kv_proj, split_kv

    def construct(
            self,
            hidden_states,
            attention_mask,
            encoder_output=None,
            inference_params=None,
            rotary_pos_emb=None,
            prefix_keys_values=None
    ):
        """ Construct function of attention block. """
        if inference_params is not None:
            raise NotImplementedError("For ParallelAttention, `inference_params` is not supported for now")
        # hidden_states: [B, S, H]
        ori_dtype = hidden_states.dtype
        bs, seq_len, _ = hidden_states.shape
        # apply query, key, value projection
        if self.attn_type == 'self_attn':
            if self.qkv_concat:
                qkv, _ = self.qkv_proj(hidden_states)
                qkv = self.cast(qkv, self.compute_dtype)
                new_tensor_shape = (bs, seq_len, -1, (self.n_rep + 2) * self.head_dim)
                mixed_x_layer = self.reshape_concat(qkv, new_tensor_shape)
                query, key, value = self.split_qkv(mixed_x_layer,
                                                   (self.head_dim * self.n_rep, self.head_dim, self.head_dim), 3)
            else:
                query, _ = self.q_proj(hidden_states)
                query = self.cast(query, self.compute_dtype)
                key, _ = self.k_proj(hidden_states)
                key = self.cast(key, self.compute_dtype)
                value, _ = self.v_proj(hidden_states)
                value = self.cast(value, self.compute_dtype)
        else:
            query, _ = self.q_proj(hidden_states)
            query = self.cast(query, self.compute_dtype)
            kv, _ = self.kv_proj(encoder_output)
            kv = self.cast(kv, self.compute_dtype)
            key, value = self.split_kv(kv, (self.kv_hidden_size, self.kv_hidden_size), 2)

        # transpose and reshape
        query = self.transpose(self.reshape(query, (bs, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key = self.transpose(self.reshape(key, (bs, seq_len, self.kv_num_heads, self.head_dim)), (0, 2, 1, 3))
        value = self.transpose(self.reshape(value, (bs, seq_len, self.kv_num_heads, self.head_dim)), (0, 2, 1, 3))

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = self.apply_rotary_pos_emb(query, q_pos_emb)
            key = self.apply_rotary_pos_emb(key, k_pos_emb)

        key, value = self._cat_prefix(key, value, prefix_keys_values)

        if not self.use_flash_attention:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            context_layer = self.core_attention(query, key, value, attention_mask)
        else:
            if attention_mask is not None:
                attention_mask = self.cast(attention_mask, mstype.uint8)

            if query.dtype not in (mstype.float16, mstype.bfloat16):
                query = self.cast(query, mstype.float16)
            if key.dtype not in (mstype.float16, mstype.bfloat16):
                key = self.cast(key, mstype.float16)
            if value.dtype not in (mstype.float16, mstype.bfloat16):
                value = self.cast(value, mstype.float16)
            output = self.flash_attention(query, key, value, attention_mask)
            context_layer = self._merge_heads(output)
            context_layer = self.cast(context_layer, self.compute_dtype)

        # apply output projection
        output, bias = self.out_proj(context_layer)
        output = self.cast(output, ori_dtype)

        return output, bias

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
            past_key = self.cast(past_key, self.compute_dtype)
            past_value = self.cast(past_value, self.compute_dtype)
            key = self.cat((past_key, key))
            value = self.cat((past_value, value))
        return key, value

    def _merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merge_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def shard(self, config: TransformerConfig):
        """sharding parameters"""
        dp = 1 if config is None else config.data_parallel
        tp = 1 if config is None else config.tensor_parallel
        cp = 1 if config is None else config.context_parallel

        if self.attn_type == 'self_attn':
            if self.qkv_concat:
                self.split_qkv.shard(((dp, cp, tp, 1),))
        else:
            self.split_kv.shard(((dp, cp, tp),))

        if self.use_flash_attention:
            self.flash_attention.shard(config)

        self.transpose.shard(((dp, cp, tp, 1),))
        self.merge_head_transpose.shard(((dp, tp, cp, 1),))
        self.tile_kv.shard(((dp, tp, 1, cp),))
        self.cat.shard(((dp, tp, 1, 1), (dp, tp, 1, 1)))


class ParallelTransformerLayer(nn.Cell):
    r"""
    Single parallel transformer layer.

    Args:
        config (dict): Configuration.
        layer_index (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        layer_type: Default None.
        self_attn_mask_type: Default None.
        drop_path_rate (flat): default 0.0.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: TransformerConfig,
                 layer_number: int,
                 self_attn_mask_type=None,
                 drop_path_rate: float = 0.0,
                 layer_type=None):
        super(ParallelTransformerLayer, self).__init__()
        if layer_type:
            raise NotImplementedError("For ParallelTransformerLayer, `layer_type` is not supported for now")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformerLayer, `self_attn_mask_type` is not supported for now")
        if drop_path_rate > 0.0:
            raise NotImplementedError("For ParallelTransformerLayer, `drop_path_rate > 0.0` is not supported for now")
        self.layer_number = layer_number
        self.apply_residual_connection_post_norm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout_rate = config.hidden_dropout
        self.add = P.Add()
        self.input_norm = get_norm(config)
        self.self_attention = ParallelAttention(config, layer_number)
        self.post_attention_norm = get_norm(config)
        self.mlp = ParallelMLP(config)
        self.hidden_states_droupout = Dropout(drop_prob=self.hidden_dropout_rate)
        self.add_bias = P.Add()
        self.shard(config)

    def construct(self, hidden_states: Tensor, attention_mask: Tensor, encoder_output=None, enc_dec_attn_mask=None,
                  retriever_input=None, retriever_output=None, retriever_attn_mask=None, inference_params=None,
                  rotary_pos_emb: Tensor = None, prefix_keys_values: Tensor = None):
        """ Construct function of transformer layer. """
        if (encoder_output is not None or enc_dec_attn_mask is not None or
                retriever_input is not None or retriever_output is not None or
                retriever_attn_mask is not None or inference_params is not None):
            raise ValueError(
                "encoder_output or enc_dec_attn_mask or retriever_input or retriever_output"
                " or retriever_attn_mask or inference_params is not None. Not support yet!!")
        # [bs, seq_len, hidden_size]
        # Layer norm at the beginning
        hidden_states_norm_output = self.input_norm(hidden_states)

        # Self-Attention
        attention_output, attention_bias = self.self_attention(hidden_states_norm_output, attention_mask,
                                                               rotary_pos_emb=rotary_pos_emb,
                                                               prefix_keys_values=prefix_keys_values)
        if attention_bias is not None:
            attention_output = self.add_bias(attention_output, attention_bias)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = hidden_states_norm_output
        else:
            residual = hidden_states

        # Dropout
        dropout_output = self.hidden_states_droupout(attention_output)

        norm_input = self.add(residual, dropout_output)

        # Layer norm post the self attention
        norm_output = self.post_attention_norm(norm_input)

        # MLP
        mlp_output, mlp_output_bias = self.mlp(norm_output)
        if mlp_output_bias is not None:
            mlp_output = self.add_bias(mlp_output, mlp_output_bias)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        # Dropout
        dropout_output = self.hidden_states_droupout(mlp_output)

        output = self.add(residual, dropout_output)
        return output

    def shard(self, config: TransformerConfig):
        dp = config.data_parallel if config.data_parallel is not None else 1
        cp = config.context_parallel if config.context_parallel is not None else 1
        tp = config.tensor_parallel if config.tensor_parallel is not None else 1
        self.add.shard(((dp, cp, tp), (dp, cp, tp)))
        self.hidden_states_droupout.shard((dp, cp, tp))
        self.add_bias.shard(((dp, cp, tp), (tp,)))


class ParallelTransformer(nn.Cell):
    r"""
    Parallel transformer class.

    Args:
        config (dict): Configuration.
        model_type : Default None.
        layer_type : Default None.
        self_attn_mask_type : Default None.
        post_norm (bool): Insert normalization layer at the end of transformer block. Default: True.
        pre_process : Default None.
        post_process : Default None.
        drop_path_rate : Default 0.0.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            model_type=None,
            layer_type=None,
            self_attn_mask_type=None,
            post_norm: bool = True,
            pre_process=False,
            post_process=False,
            drop_path_rate: float = 0.0
    ):
        super(ParallelTransformer, self).__init__()
        if model_type:
            raise NotImplementedError("For ParallelTransformer, `model_type` is not supported for now.")
        if layer_type:
            raise NotImplementedError("For ParallelTransformer, `layer_type` is not supported for now.")
        if self_attn_mask_type:
            raise NotImplementedError("For ParallelTransformer, `self_attn_mask_type` is not supported for now.")
        if pre_process:
            raise NotImplementedError("For ParallelTransformer, `pre_process=True` is not supported.")
        if post_process:
            raise NotImplementedError("For ParallelTransformer, `post_process=True` is not supported.")
        if drop_path_rate > 0.0:
            raise NotImplementedError("For ParallelTransformer, `drop_path_rate > 0` is not supported for now.")

        self.post_norm = post_norm
        self.num_layers = config.num_layers

        offset = 0
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = ParallelTransformerLayer(config, layer_id + 1 + offset)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)

        if self.post_norm:
            self.final_norm = get_norm(config)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(self, hidden_states: Tensor, attention_mask: Tensor, rotary_pos_emb: Tensor = None,
                  prefix_keys_values=None):
        """ Construct function of transformer. """
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            prefix_kv = prefix_keys_values[index] if prefix_keys_values is not None else None
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb,
                                  prefix_keys_values=prefix_kv)

        # final layernorm.
        if self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


class CausalMaskGenerate(nn.Cell):
    """Get the upper triangular matrix from the input_ids.

    Args:
        seq_length (int): The length of the input sequence.
        compute_type (mstype): The compute type of the input tensor. Default: mstype.float16.
        is_dynamic (bool): Whether the input_ids is dynamic. Default: False.
        pad_token_id (int): The pad token id. Default: 0.
        use_flash_attention (bool): Whether to use the flash attention. Default: False.
        use_prompt_flash_attention (bool): Whether to use the prompt flash attention. Default: False.
        use_incre_flash_attention (bool): Whether to use the incremental flash attention. Default: False.
        use_attn_mask_compression (bool): Whether to use the attention mask compression. Default: False.
    """

    def __init__(self,
                 seq_length: int,
                 compute_type: mstype = mstype.float16,
                 is_dynamic: bool = False,
                 pad_token_id: int = 0,
                 use_flash_attention: bool = False,
                 use_prompt_flash_attention: bool = False,
                 use_incre_flash_attention: bool = False,
                 use_attn_mask_compression: bool = False,
                 config: TransformerConfig = None
                 ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.use_incre_flash_attention = use_incre_flash_attention
        self.is_first_iteration = True
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        if use_attn_mask_compression:
            if seq_length < 2048:
                raise ValueError("seq_length should be larger than 2048 when use mask_compression")
            self.lower_triangle_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
        else:
            self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                              dtype=compute_type)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.expand_dim_post = P.ExpandDims()
        self.shard(config)

    def construct(self, tokens=None, masks=None):
        """Forward process of the CausalMask

        Args:
            tokens (Tensor): The input tokens. Default: None.
            masks (Tensor): The input masks. Default: None.

        Returns:
            Tensor, the upper triangle attention mask carrying 0 and 1 values
        """
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention or self.use_prompt_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
        return attention_mask

    def shard(self, config: TransformerConfig):
        """sharding operators
        """
        dp = config.data_parallel if config.data_parallel is not None else 1
        self.not_equal.shard(((dp, 1), ()))
        self.expand_dim.shard(((1, 1),))
        self.mul.shard(((dp, 1, 1), (1, 1, 1)))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.expand_dim_post.shard(((dp, 1, 1),))
