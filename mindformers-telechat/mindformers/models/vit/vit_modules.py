# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Modules of ViTForImageClassification, including Linear, Block, MLP, Attention, PatchEmbed, etc.
"""
import math
import numpy as np
from mindspore import nn, Parameter, Tensor, context, ParallelMode
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P, constexpr
from mindspore.ops import functional as F
import mindspore.common.initializer as init
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.modules import layers, FeedForward
from mindformers.modules.transformer.moe import default_moe_config, _check_moe_config
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, _check_config



@constexpr
def gen_shape(x_shape, ndim):
    return (x_shape,) + (1,) * (ndim + 1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim should be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim should be divisible by 2")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(layers.LayerNorm):
    # pylint: disable=W0212
    r"""
    A self-defined layer norm operation using reduce sum and reduce mean.
    """

    def __init__(self, normalized_shape, eps=1e-6, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__(
            normalized_shape,
            eps=eps,
            param_init_type=param_init_type)


class Identity(nn.Cell):

    def construct(self, x):
        return x


class RelativePositionBias(nn.Cell):
    """relative position bias"""

    def __init__(self, window_size, num_heads):
        super(RelativePositionBias, self).__init__()

        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        self.relative_position_bias_table = Parameter(
            init.initializer(
                init.TruncatedNormal(sigma=.02),
                (self.num_relative_distance, num_heads)),
            name='relative_position_bias_table')

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(np.arange(window_size[0]), mstype.int32)
        coords_w = Tensor(np.arange(window_size[1]), mstype.int32)
        coords = P.Stack(axis=0)(P.Meshgrid(indexing='ij')((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = P.Flatten()(coords)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).asnumpy()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = \
            np.zeros(((window_size[0] * window_size[1] + 1),) * 2, dtype=int)

        relative_position_index[1:, 1:] = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        relative_position_index = Tensor(relative_position_index, mstype.int32)
        self.relative_position_index = Parameter(
            relative_position_index,
            requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

        self.gather = P.Gather()

    def construct(self):
        relative_position_index = self.relative_position_index.view(-1)
        relative_position_bias = self.gather(self.relative_position_bias_table, relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1] + 1,
             self.window_size[0] * self.window_size[1] + 1, -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        return relative_position_bias


class Linear(layers.Linear):
    # pylint: disable=W0212
    r"""
    Linear function for RingMo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 activation_compute_type=mstype.float16,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            weight_init=weight_init,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=activation,
            transpose_b=transpose_b,
            expert_num=expert_num,
            outer_batch=outer_batch,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype)

        self.activation_compute_type = activation_compute_type

    # pylint: disable=W0221
    def construct(self, x):
        """construct of layer"""
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        if self.expert_flag:
            x = P.Reshape()(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(self.cast(x, self.activation_compute_type))
            x = self.cast(x, self.dtype)
        output = P.Reshape()(x, out_shape)
        return output


class Dropout(layers.Dropout):
    # pylint: disable=W0212
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for context training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__(keep_prob=keep_prob, dtype=dtype)


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop = Dropout(keep_prob=1 - drop_prob)
        self.mask = Tensor(np.ones(1,), dtype=mstype.float32)
        self.tile = P.Tile()
        self.mul = P.Mul()

    def construct(self, x):
        if not self.training:
            return x
        mask = self.tile(self.mask, (x.shape[0],) + (1,) * (x.ndim - 1))
        out = self.drop(mask)
        out = self.mul(out, x)
        return out

    def shard(self, strategy):
        self.mul.shard(strategy)


class Block(nn.Cell):
    """Block of ringmo"""

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 drop_rate=0.,
                 attention_dropout_rate=0.,
                 hidden_dropout_rate=0.,
                 layer_norm_eps=1e-6,
                 qkv_bias=True,
                 window_size=None,
                 post_layernorm_residual=False,
                 init_values=None,
                 weight_init='XavierUniform',
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(Block, self).__init__()
        _check_config(parallel_config)
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError(
                "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                "'parallel_config.model_parallel', but got the num_heads is {} and "
                "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
        if hidden_size % parallel_config.model_parallel != 0:
            raise ValueError(
                "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
        if ffn_hidden_size % parallel_config.model_parallel != 0:
            raise ValueError(
                "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                "and parallel_config. model_parallel is {}."
                .format(ffn_hidden_size, parallel_config.model_parallel))
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.post_layernorm_residual = post_layernorm_residual
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dtype = mstype.float16
        self.key_past = None
        self.value_past = None

        self.layernorm1 = LayerNorm((hidden_size,), eps=layer_norm_eps).to_float(layernorm_compute_type)
        self.layernorm1.shard(((parallel_config.data_parallel, 1),))
        self.layernorm2 = LayerNorm((hidden_size,), eps=layer_norm_eps).to_float(layernorm_compute_type)
        self.layernorm2.shard(((parallel_config.data_parallel, 1),))
        parallel_config_args = parallel_config.dpmp if self.use_moe else parallel_config
        self.attention = Attention(src_seq_length=seq_length,
                                   tgt_seq_length=seq_length,
                                   hidden_size=hidden_size,
                                   window_size=window_size,
                                   num_heads=num_heads,
                                   weight_init=weight_init,
                                   hidden_dropout_rate=hidden_dropout_rate,
                                   attention_dropout_rate=attention_dropout_rate,
                                   qkv_bias=qkv_bias,
                                   softmax_compute_type=softmax_compute_type,
                                   param_init_type=param_init_type,
                                   parallel_config=parallel_config_args)
        self.output = MLP(hidden_size=hidden_size,
                          dropout_rate=drop_rate,
                          ffn_hidden_size=ffn_hidden_size,
                          param_init_type=param_init_type,
                          weight_init=weight_init,
                          hidden_act=hidden_act,
                          use_dropout=False,
                          parallel_config=parallel_config)
        if init_values is not None:
            self.gamma_1 = Parameter(
                Tensor(init_values * np.ones((hidden_size,)), mstype.float32),
                name="gamma1", requires_grad=True)
            self.gamma_2 = Parameter(
                Tensor(init_values * np.ones((hidden_size,)), mstype.float32),
                name="gamma2", requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.mul_gamma = P.Mul().shard(((parallel_config.data_parallel, 1), (1,)))
        self.drop_path = DropPath(hidden_dropout_rate)
        self.drop_path.shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.drop_path3d = DropPath(hidden_dropout_rate)
        self.drop_path3d.shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.mul = P.Mul().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.reshape = P.Reshape()

    def construct(self, x, input_mask, rel_pos_bias=None):
        """construct of Block"""
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        attention = self.attention(
            input_x, input_x, input_x, input_mask, rel_pos_bias)

        if self.gamma_1 is not None:
            attention = self.mul_gamma(attention, self.gamma_1)

        if len(x_shape) == 3:
            attention = P.Reshape()(attention, x_shape)
            attention = self.drop_path3d(attention)
            attention = P.Reshape()(attention, (-1, x_shape[-1]))
        else:
            attention = self.drop_path(attention)

        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)

        mlp_logit = self.output(output_x)

        if self.gamma_2 is not None:
            mlp_logit = self.mul_gamma(mlp_logit, self.gamma_2)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            mlp_logit = self.drop_path3d(mlp_logit)
            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            mlp_logit = self.drop_path(mlp_logit)
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)
        return output


class MLP(FeedForward):
    r"""MLP for ring-mo."""

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size=None,
                 out_features=None,
                 dropout_rate=0.,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 use_dropout=True,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        ffn_hidden_size = ffn_hidden_size or hidden_size
        super(MLP, self).__init__(
            hidden_size,
            ffn_hidden_size,
            dropout_rate=dropout_rate,
            hidden_act=hidden_act,
            param_init_type=mstype.float32,
            parallel_config=parallel_config)
        mp = parallel_config.model_parallel
        dp = parallel_config.data_parallel

        out_features = out_features or hidden_size

        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=hidden_size,
                              out_channels=ffn_hidden_size,
                              activation=hidden_act,
                              transpose_b=False,
                              outer_batch=dp,
                              weight_init=weight_init,
                              param_init_type=param_init_type)

        self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                           strategy_bias=((dp, mp), (mp,)),
                           strategy_activation=((dp, mp),))
        # Project back to hidden_size
        self.projection = Linear(in_channels=ffn_hidden_size,
                                 out_channels=out_features,
                                 transpose_b=False,
                                 outer_batch=dp,
                                 weight_init=weight_init,
                                 param_init_type=param_init_type)
        self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                              strategy_bias=((dp, 1), (1,)))
        self.projection.bias.parallel_optimizer = False
        self.dropout = Dropout(1 - dropout_rate)
        self.dropout.shard(((dp, 1),))
        self.dropout_3d = Dropout(1 - dropout_rate)
        self.dropout_3d.shard(((dp, 1, 1),))
        self.use_dropout = use_dropout

    def construct(self, x):
        """construct of mlp"""
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)

        if self.use_dropout:
            if len(F.shape(hidden)) == 3:
                hidden = self.dropout_3d(hidden)
            elif len(F.shape(hidden)) == 2:
                hidden = self.dropout(hidden)

        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        return output


class Attention(nn.Cell):
    r"""
        This is an implementation of multihead attention in the paper `Attention is all you need
        <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
        key and value vector with target length, the attention will be performed as the following
    """

    def __init__(self,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 window_size=None,
                 hidden_dropout_rate=0.,
                 attention_dropout_rate=0.,
                 qkv_bias=True,
                 weight_init='XavierUniform',
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(Attention, self).__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        _check_config(parallel_config)
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.hidden_size = hidden_size
        if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
        if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
        if hidden_size % num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(hidden_size, num_heads))
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                             "'parallel_config.model_parallel', but got the num_heads is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(num_heads, parallel_config.model_parallel))
        self.is_first_iteration = True
        self.transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = num_heads
        # embedding size per head
        self.size_per_head = hidden_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=softmax_compute_type)
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.real_div = P.RealDiv().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (parallel_config.data_parallel, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((parallel_config.data_parallel, 1, 1, 1), (1,)))
        self.add = P.Add().shard(
            ((parallel_config.data_parallel, 1, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        # Normalize factor for attention, sqrt(dk) as widely used
        self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
        self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_type
        # Output layer
        self.projection = Linear(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 transpose_b=False,
                                 weight_init=weight_init,
                                 param_init_type=param_init_type).to_float(compute_dtype)
        self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                              strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                               (parallel_config.model_parallel, 1)))
        self.projection.bias.parallel_optimizer = False

        self.dropout = Dropout(1 - hidden_dropout_rate)
        self.dropout.shard(((parallel_config.data_parallel, 1),))
        self.prob_dropout = Dropout(1 - attention_dropout_rate)
        self.prob_dropout.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))

        # Query
        self.dense1 = Linear(hidden_size,
                             hidden_size,
                             has_bias=qkv_bias,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        # Key
        self.dense2 = Linear(hidden_size,
                             hidden_size,
                             has_bias=qkv_bias,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        # Value
        self.dense3 = Linear(hidden_size,
                             hidden_size,
                             has_bias=qkv_bias,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        if window_size:
            self.relative_position_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
            self.relative_position_bias = None

        self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1, 1), (1, 1, 1)))
        self.expand_dims_rpb = P.ExpandDims().shard(((1, 1, 1),))
        self.add_rpb = P.Add().shard(((parallel_config.data_parallel, 1, 1, 1), (1, 1, 1, 1)))

        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, rel_pos_bias=None):
        """construct of attention"""
        ori_shape = F.shape(query_tensor)
        batch_size = F.shape(attention_mask)[0]
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)

        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            F.reshape(
                key, (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if len(F.shape(attention_mask)) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, attention_mask, rel_pos_bias)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        return output

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask, rel_pos_bias):
        r"""Get the weighted score along the seq_length."""
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, self.softmax_dtype)

        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        # add window cross module
        if self.relative_position_bias is not None:
            relative_position_bias = self.expand_dims_rpb(self.relative_position_bias(), 0)
            attention_scores = self.add_rpb(attention_scores, relative_position_bias)

        if rel_pos_bias is not None:
            attention_scores = self.add_3d(attention_scores, rel_pos_bias)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = P.Cast()(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor


class PatchEmbed(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_features=3,
                 out_features=768,
                 norm_layer=False,
                 use_mask=False,
                 parallel_config=None):
        super(PatchEmbed, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=patch_size, stride=patch_size,
            weight_init=init.TruncatedNormal(sigma=0.02),
            has_bias=True,
            pad_mode='pad')
        self.proj.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.proj.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        # usually not use norm
        self.norm = LayerNorm((out_features,), eps=1e-6).shard(((dp, 1, 1),)) if norm_layer else Identity()
        self.mask_token = Parameter(init.initializer(init.Zero(), (1, 1, out_features)),
                                    requires_grad=True) if use_mask else None
        self.tile = P.Tile()
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, x, mask=None):
        """construct"""
        # True x: bs  False x: bs * dp
        x = self.proj(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        if mask is not None:
            mask_tokens = self.tile(self.mask_token, (b, self.num_patches, 1))
            x = self.mul(x, (1.0 - mask))
            x = self.add(x, self.mul(mask_tokens, mask))
        return x


def get_kernel_size(patch_size):
    """
        input: 2^i & i <= 5 | 14
        output: a list of kernel size
    """
    x = None
    y = None
    z = None
    ans = False
    for i in range(1, patch_size + 1):
        if patch_size % i == 0:
            x = i
            mul_y_z = patch_size // i
            for j in range(1, mul_y_z + 1):
                if mul_y_z % j == 0:
                    y = j
                    z = mul_y_z // j
                    if x >= y >= z:
                        ans = True
                        break
            if ans:
                break
    if not ans:
        raise ValueError(patch_size)
    return [x, y, z]


class Patchify(nn.Cell):
    """Patchify"""

    def __init__(self, patch_size, parallel_config=None):
        super(Patchify, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        patches = self.reshape(x, (bs, (h // p) * (w // p), channels * p * p))
        return patches


class UnPatchify(nn.Cell):
    """UnPatchify"""

    def __init__(self, patch_size, seq_length, parallel_config=None):
        super(UnPatchify, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.p = patch_size
        self.h = self.w = int(seq_length ** .5)
        if self.h * self.w != seq_length:
            raise ValueError("self.h * self.w should equal to seq_length")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, x):
        bs = x.shape[0]
        x = self.reshape(x, (bs, self.h, self.w, self.p, self.p, 3))
        x = self.transpose(x, (0, 5, 1, 3, 2, 4))
        images = self.reshape(x, (bs, 3, self.h * self.p, self.w * self.p))
        return images


class PixelShuffle(nn.Cell):
    """PixelShuffle"""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        b, _, h, w = x.shape
        x = self.reshape(x, (b, -1, self.upscale_factor, self.upscale_factor, h, w))
        x = self.transpose(x, (0, 1, 4, 2, 5, 3))
        x = self.reshape(x, (b, -1, h * self.upscale_factor, w * self.upscale_factor))
        return x
