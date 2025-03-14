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
"""Llava-NeXT models' APIs."""
from typing import Optional
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.modules import Linear
from mindformers.modules.flash_attention import FlashAttention


class LayerNorm(nn.LayerNorm):
    r"""Implementation That Supports Fp16 Inputs But Fp32 Gains Biases.

    Args:
        x (ms.Tensor): Input tensor.
            The detailed function could refer to mindspore.nn.LayerNorm.

    Return:
        y (ms.Tensor): Normalized tensor.
    """

    # pylint: disable=C0111
    def construct(self, x: ms.Tensor):
        y = super().construct(P.Cast()(x, ms.float32))
        return P.Cast()(y, x.dtype)


class MultiheadAttention(nn.Cell):
    r"""MultiheadAttention, With Layers As Input For Initialization

    Args:
        d_model (int): The feature dimension
        n_head (int): The number of attention heads
        layers (int): The number of transformers, used for weight initialization
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, d_model: int, n_head: int, layers: int, compute_dtype: mstype, param_init_type: mstype,
                 use_flash_attention: bool):
        super(MultiheadAttention, self).__init__()
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        attn_std = d_model ** -0.5

        self.scaling = self.head_dim ** -0.5
        self.reshape = P.Reshape()
        self.slice = P.StridedSlice()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.softmax = P.Softmax()
        self.shape = P.Shape()
        self.batch_matmul = P.BatchMatMul()
        self.merger_head_transpose = P.Transpose()
        self.add = P.Add()
        self.transpose = P.Transpose()
        self.out_proj = Linear(d_model, d_model, weight_init=Normal(mean=0.0, sigma=proj_std),
                               compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.in_proj = Linear(d_model, 3 * d_model, weight_init=Normal(mean=0.0, sigma=attn_std),
                              compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.num_heads,
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  input_layout="BNSD",
                                                  keep_prob=1.,
                                                  scale_value=self.scaling,
                                                  sparse_mode=0,
                                                  use_attention_mask=False)

        self.softmax_dtype = mstype.float32

        self.cast_attn = P.Cast()
        self.cast = P.Cast()
        self.split_qkv = ms.ops.auto_generate.SplitWithSize()
        self.split_qkv.add_prim_attr("skip_redistribution", True)
        self.dtype = compute_dtype

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        batch_size, len_tgt, width = query.shape
        qkv = self.in_proj(query)
        query, key, value = self.split_qkv(qkv, (width, width, width), 2)

        query = self.cast(self.transpose(self.reshape(query, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                         (0, 2, 1, 3)), self.dtype)
        key = self.cast(self.transpose(self.reshape(key, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                       (0, 2, 1, 3)), self.dtype)
        value = self.cast(self.transpose(self.reshape(value, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                         (0, 2, 1, 3)), self.dtype)
        if self.use_flash_attention:
            context_layer = self.flash_attention(query, key, value, attn_mask)
            attn = self._merge_heads(context_layer)
        else:
            attn = self._attn(query, key, value, attn_mask)
        return self.out_proj(attn)

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
        score = self.batch_matmul_q_k(query, key)

        score = self.mul(score, self.scaling)
        if mask is not None:
            score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        return self._merge_heads(weighted_values)

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
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        return self.reshape(x, new_shape)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.in_proj.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.split_qkv.shard(((dp, 1, 1),))
        self.transpose.shard(((dp, 1, mp, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.softmax.shard(((dp, mp, 1, 1),))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.out_proj.shard(strategy_matmul=((dp, mp), (1, mp)), out_strategy_matmul=((dp, 1),),
                            strategy_bias=((dp, 1), (1,)))
        if self.use_flash_attention:
            self.flash_attention.shard(parallel_config)


class QuickGELU(nn.Cell):
    r"""QuickGELU of CLIP"""

    def __init__(self, ratio: Optional[int] = 1.702):
        super(QuickGELU, self).__init__()
        self.ratio = ratio
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()
        self.mul2 = P.Mul()

    # pylint: disable=C0111
    def construct(self, input_x: ms.Tensor):
        return self.mul(input_x, self.sigmoid(self.mul2(input_x, self.ratio)))

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.mul2.shard(((dp, 1, mp), ()))
        self.sigmoid.shard(((dp, 1, mp),))
        self.mul.shard(((dp, 1, mp), (dp, 1, mp)))


class MLP(nn.Cell):
    """
    A multilayer perceptron for ViT
    """

    def __init__(self, layers: int, input_channel_dim: int, output_channel_dim: int, compute_dtype, param_init_type):
        super().__init__()

        proj_std = (input_channel_dim ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * input_channel_dim) ** -0.5
        self.c_fc = Linear(input_channel_dim, output_channel_dim, weight_init=Normal(mean=0.0, sigma=fc_std),
                           compute_dtype=compute_dtype, param_init_type=param_init_type)

        self.c_proj = Linear(output_channel_dim, input_channel_dim, weight_init=Normal(mean=0.0, sigma=proj_std),
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

        self.gelu = QuickGELU()
        self.cast = P.Cast()
        self.dtype = compute_dtype

    # pylint: disable=C0111
    def construct(self, x):
        ori_dtype = x.dtype
        x = self.cast(x, self.dtype)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.cast(x, ori_dtype)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.c_fc.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.gelu.shard(parallel_config)
        self.c_proj.shard(strategy_matmul=((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))


class ResidualAttentionBlock(nn.Cell):
    r"""
    ResidualAttentionBlock of CLIP

    Args:
        d_model (int): The dimension of features.
        n_head (int): The number of attention heads.
        layers (int): The number of transformer layers for weight initialization.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
        attn_mask (Optional[ms.Tensor]): attention mask.
    """

    def __init__(self, d_model: int, n_head: int, layers: int,
                 dtype: mstype, attn_mask: Optional[ms.Tensor] = None, **kwargs):
        super(ResidualAttentionBlock, self).__init__()
        param_init_type = kwargs.get("param_init_type", mstype.float16)
        use_flash_attention = kwargs.get("use_flash_attention", False)
        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, layers, self.dtype, param_init_type, use_flash_attention)
        self.ln_1 = LayerNorm([d_model], epsilon=1e-5)

        self.mlp = MLP(layers, d_model, d_model * 4, self.dtype, param_init_type)
        self.ln_2 = LayerNorm([d_model], epsilon=1e-5)

        self.attn_mask = attn_mask
        self.add = P.Add()

    # pylint: disable=C0111
    def construct(self, input_x: ms.Tensor):
        ln_1 = self.ln_1(input_x)
        attn_tensor = self.attention(ln_1)
        input_x = self.add(input_x, attn_tensor)
        ln_2 = self.ln_2(input_x)
        mlp_2 = self.mlp(ln_2)
        return self.add(input_x, mlp_2)

    # pylint: disable=C0111
    def attention(self, input_x: ms.Tensor):
        return self.attn(input_x, self.attn_mask)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.ln_1.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.ln_2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.add.shard(((dp, 1, 1), (dp, 1, 1)))
        self.mlp.shard(parallel_config)
        self.attn.shard(parallel_config)


class Transformer(nn.Cell):
    r"""
    Text Transformer of CLIP

    Args:
        width (int): The dimension of input features.
        layers (int): The number of transformer layers.
        heads (int): The number of attention heads.
        attn_mask (ms.Tensor):  Attention mask.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, width, layers, heads, dtype, attn_mask=None, **kwargs):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, layers, dtype, attn_mask, **kwargs) for _ in range(layers)]
        )

    # pylint: disable=C0111
    def construct(self, input_x):
        return self.resblocks(input_x)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        for layer in self.resblocks:
            layer.shard(parallel_config)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlavaVisionEncoder(nn.Cell):
    r"""VisionTransformer Of CLIPModel

    Args: config: Llavaconfig for Llava model
    """

    def __init__(self, config: PretrainedConfig, **kwargs):
        super(LlavaVisionEncoder, self).__init__(config, **kwargs)
        self.config = config
        input_resolution = config.image_size
        patch_size = config.patch_size
        width = config.hidden_size
        layers = config.num_hidden_layers + config.vision_feature_layer + 1
        if layers <= 0:
            raise ValueError("num of layers is invalid, please set number of layers larger than 0, at least 1.")
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

        heads = config.num_attention_heads
        self.dtype = config.compute_dtype
        parallel_config = config.parallel_config
        param_init_type = config.param_init_type
        self.conv1 = \
            nn.Conv2d(
                in_channels=3, out_channels=width, kernel_size=patch_size,
                stride=patch_size, has_bias=False, pad_mode="pad").to_float(param_init_type)

        scale = width ** -0.5
        self.class_embedding = \
            Parameter(scale * Tensor(np.random.normal(0, 1, size=(width))).astype(param_init_type))
        self.positional_embedding = \
            Parameter(scale * Tensor(
                np.random.normal(0, 1, size=(
                    (input_resolution // patch_size) ** 2 + 1, width))).astype(param_init_type),
                      parallel_optimizer=False)
        self.ln_pre = LayerNorm([width], epsilon=1e-5)
        self.transformer = Transformer(width, layers, heads, self.dtype, param_init_type=param_init_type,
                                       use_flash_attention=config.use_flash_attention)
        self.ln_post = LayerNorm([width], epsilon=1e-5)
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)

        self.position_shape = self.positional_embedding.shape
        self.expand_dims = P.ExpandDims()
        self.shape = P.Shape()
        self.cat = P.Concat(1)
        self.tile = P.Tile()
        self.add = P.Add()
        self.transpose = P.Transpose()
        self.slice = P.StridedSlice()
        self.cast = P.Cast()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            dp = parallel_config.data_parallel
            self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
            self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
            self.tile.shard(((1, 1, 1),))
            self.transpose.shard(((dp, 1, 1),))
            self.cat.shard(((dp, 1, 1), (dp, 1, 1)))
            self.add.shard(((dp, 1, 1), (1, 1, 1)))
            self.ln_pre.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
            self.slice.shard(((dp, 1, 1),))
            self.transformer.shard(parallel_config)
            self.expand_dims.shard(((1, 1),))

    def construct(self, input_x: ms.Tensor):
        r"""Construct

        Args:
            input_x (ms.Tensor): Input tensor.

        Returns:
            input_x (ms.Tensor): Output tensor.
        """
        input_x = self.cast(input_x, self.dtype)
        input_x = self.conv1(input_x)
        bs, dim, seq1, seq2 = self.shape(input_x)
        input_x = self.reshape(input_x, (bs, dim, seq1 * seq2))

        input_x = self.transpose(input_x, (0, 2, 1))  #

        class_embedding = self.cast(self.tile(self.class_embedding, (input_x.shape[0], 1, 1)), self.dtype)
        input_x = self.cat([class_embedding, self.cast(input_x, self.dtype)])
        positional_embedding = self.expand_dims(self.positional_embedding, 0)
        input_x = self.add(input_x, self.cast(positional_embedding, self.dtype))
        input_x = self.ln_pre(input_x)
        input_x = self.transformer(input_x)
        if self.vision_feature_select_strategy == 'default':
            bs, seq_length, dim = input_x.shape
            output = self.slice(input_x, (0, 1, 0), (bs, seq_length, dim), (1, 1, 1))
        elif self.vision_feature_select_strategy == "full":
            output = input_x
        else:
            raise ValueError("Please select valuable vision feature select strategy in ['full' and 'default']!")
        return output
