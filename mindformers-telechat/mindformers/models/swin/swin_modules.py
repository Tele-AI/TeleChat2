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
# This file was refer to project:
# https://github.com/microsoft/Swin-Transformer
# ============================================================================
"""Swin Transformer API."""
import collections.abc
from itertools import repeat
import numpy as np

from mindspore import nn
from mindspore import numpy
from mindspore import context
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
import mindspore.common.dtype as mstype
import mindspore.common.initializer as weight_init_

from mindformers.modules import layers, FeedForward
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config


# utils
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


@constexpr
def gen_shape(x_shape, ndim):
    """generate shape for x."""
    return (x_shape,) + (1,) * (ndim + 1)


class LayerNorm(layers.LayerNorm):
    # pylint: disable=W0212
    """
    A self-defined layer norm operation using reduce sum and reduce mean.
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__(
            normalized_shape,
            eps=eps,
            param_init_type=param_init_type)


class Linear(layers.Linear):
    # pylint: disable=W0212
    """
    Linear function for Swin.
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
        """construct of linear."""
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


class Identity(nn.Cell):
    """No construct for x."""
    def __init__(self, auto_prefix=True):
        super(Identity, self).__init__(auto_prefix=auto_prefix)

    def construct(self, x):
        """No construct."""
        return x


class Dropout(layers.Dropout):
    # pylint: disable=W0212
    """
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for context training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__(keep_prob=keep_prob, dtype=dtype)


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob, ndim=1, parallel_config=None):
        # pylint: disable=W0613
        super(DropPath, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.drop = Dropout(keep_prob=1 - drop_prob)
        self.drop.shard(((1, 1, 1),))
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)
        self.tile = P.Tile().shard(((1, 1, 1),))
        self.mul = P.Mul().shard(((dp, 1, 1),))

    def construct(self, x):
        """Dropout."""
        if not self.training:
            return x
        shape = gen_shape(x.shape[0], self.ndim)
        mask = self.tile(self.mask, shape)
        out = self.drop(mask)
        out = self.mul(out, x)
        return out


class SwinPatchEmbeddings(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config):
        super(SwinPatchEmbeddings, self).__init__(config)
        if config.parallel_config:
            dp = config.parallel_config.data_parallel
        else:
            dp = 1
        img_size = (config.image_size, config.image_size)
        patch_size = (config.patch_size, config.patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Conv2d(in_channels=config.num_channels,
                                    out_channels=config.embed_dim,
                                    kernel_size=patch_size[0],
                                    stride=patch_size[0],
                                    weight_init=weight_init_.TruncatedNormal(sigma=0.02),
                                    has_bias=True,
                                    pad_mode='pad')

        self.projection.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.projection.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        # usually not use norm
        if config.patch_norm:
            self.norm = LayerNorm((config.embed_dim,), eps=config.layer_norm_eps).shard(((dp, 1, 1),))
        else:
            self.norm = Identity()

    def construct(self, x):
        """construct"""
        # True x: bs  False x: bs * dp
        x = self.projection(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x


class SwinStage(nn.Cell):
    """ Swin Basic Layer
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path,
                 norm_layer=LayerNorm, downsample=None, parallel_config=default_dpmp_config):
        super(SwinStage, self).__init__(config)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(config=config,
                                 dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 parallel_config=parallel_config)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim, input_resolution, config.weight_init,
                                         norm_layer=norm_layer, parallel_config=parallel_config)
        else:
            self.downsample = None

    def construct(self, x):
        """construct"""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinTransformerBlock(nn.Cell):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        shift_size (int): Shift size for SW-MSA.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm/LayerNorm
    """
    def __init__(self, config, dim, input_resolution, num_heads, shift_size, drop_path,
                 norm_layer=LayerNorm, parallel_config=default_dpmp_config):
        super(SwinTransformerBlock, self).__init__(config)
        self.use_moe = (config.moe_config.expert_num > 1)
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = config.window_size
        self.shift_size = shift_size
        self.mlp_ratio = config.mlp_ratio
        self.reshape = P.Reshape()
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer((dim,), eps=config.layer_norm_eps)
        self.norm1.shard(((dp, 1, 1),))
        self.attn = SwinAttention(dim,
                                  window_size=to_2tuple(self.window_size),
                                  num_heads=num_heads,
                                  qkv_bias=config.qkv_bias,
                                  attn_drop=config.attention_probs_dropout_prob,
                                  proj_drop=config.hidden_dropout_prob)

        self.drop_path = DropPath(drop_path, ndim=1, parallel_config=parallel_config) \
            if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,), eps=config.layer_norm_eps)
        self.norm2.shard(((dp, 1, 1),))

        self.mlp = SwinIntermediate(hidden_size=dim,
                                    ffn_hidden_size=int(dim * config.mlp_ratio),
                                    dropout_rate=config.hidden_dropout_prob,
                                    weight_init=config.weight_init,
                                    hidden_act=config.hidden_act,
                                    parallel_config=parallel_config)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            in_h, in_w = self.input_resolution
            img_mask = np.zeros((1, in_h, in_w, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            # mask: [64, 49, 49]
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False, name="attention_mask")
            self.roll_pos = Roll(self.shift_size, parallel_config=parallel_config)
            self.roll_neg = Roll(-self.shift_size, parallel_config=parallel_config)
        else:
            self.attn_mask = None

        self.dtype = P.DType()
        self.add_3d = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()

    def construct(self, x):
        """construct function"""
        h, w = self.input_resolution
        b, _, c = x.shape
        ori_type = self.dtype(x)
        shortcut = x
        x = self.norm1(x)
        x = self.reshape(x, (b, h, w, c))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = self.reshape(x_windows,
                                 (-1, self.window_size * self.window_size, c))  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = self.reshape(attn_windows, (-1, self.window_size, self.window_size, c))
        shifted_x = self.window_reverse(attn_windows, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x = self.reshape(x, (b, h * w, c))

        # FFN
        x = self.drop_path(x)
        x = self.cast(x, ori_type)
        x = self.add_3d(shortcut, x)
        x_tmp = self.norm2(x)
        x_tmp = self.mlp(x_tmp)
        x_tmp = self.drop_path(x_tmp)
        output = self.add_3d(x, x_tmp)
        output = self.cast(output, ori_type)
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


# pylint: disable=C0326
class SwinPatchMerging(nn.Cell):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm/LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 weight_init='normal',
                 norm_layer=LayerNorm,
                 parallel_config=default_dpmp_config):
        super(SwinPatchMerging, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = Linear(in_channels=4 * dim,
                                out_channels=2 * dim,
                                has_bias=False,
                                weight_init=weight_init).to_float(mstype.float16)
        self.reduction.shard(strategy_matmul=((dp, mp), (mp, 1)), strategy_bias=((dp, 1), (1,)))
        self.norm = norm_layer([dim * 4, ], eps=1e-5)
        self.norm.shard(((dp, 1, 1),))
        self.h, self.w = self.input_resolution
        self.h_2, self.w_2 = self.h // 2, self.w // 2
        self.h2w2 = int(self.h * self.w // 4)
        self.dim_mul_4 = int(dim * 4)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, x):
        """
        x: B, H*W, C
        """
        b = x.shape[0]
        x = self.reshape(x, (b, self.h_2, 2, self.w_2, 2, self.dim))
        x = self.transpose(x, (0, 1, 3, 4, 2, 5))
        x = self.reshape(x, (b, self.h2w2, self.dim_mul_4))
        x = self.norm(x)
        x = self.cast(x, mstype.float16)
        x = self.reduction(x)
        x = self.cast(x, mstype.float32)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class SwinAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) Cell with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(SwinAttention, self).__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        if parallel_config:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
        else:
            dp = mp = 1
        self._is_ascend = context.get_context('device_target') in ["Ascend"]

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale_factor = Tensor(qk_scale or head_dim ** -0.5, mstype.float32)
        self.relative_position_bias = SwinRelativePositionBias(self.window_size, num_heads)

        # get pair-wise relative position index for each token inside the window
        self.q = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.q.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.k = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.k.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.v = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.v.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.attn_drop = Dropout(keep_prob=1.0 - attn_drop)
        self.proj = Linear(
            in_channels=dim, out_channels=dim, has_bias=True,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.proj.shard(strategy_bias=((dp, 1), (1,)), strategy_matmul=((dp, mp), (mp, 1)))

        self.proj_drop = Dropout(keep_prob=1.0 - proj_drop)
        self.proj_drop.shard(((dp, mp, 1, 1),))

        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((dp, mp, 1, 1),))
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))
        self.matmul = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.real_div = P.RealDiv().shard(((dp, mp, 1, 1), ()))
        self.sub = P.Sub().shard(((1,), (dp, 1, 1, 1)))
        self.mul = P.Mul().shard(((dp, 1, 1, 1), (1,)))
        self.add_4d = P.Add().shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.add_5d = P.Add().shard(((dp, 1, 1, 1, 1), (dp, 1, 1, 1, 1)))
        self.dtype = P.DType()
        self.shape = P.Shape()
        self.compute_type = compute_dtype
        self.softmax_dtype = softmax_compute_type

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """
        attention_probs = self.softmax(attention_scores)
        return attention_probs

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, seq, c = x.shape
        ori_type = self.dtype(x)
        x = self.cast(x, self.compute_type)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = self.transpose(self.reshape(q, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        k = self.transpose(self.reshape(k, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 3, 1))
        v = self.transpose(self.reshape(v, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        factor = self.cast(self.scale_factor, self.compute_type)
        q = self.mul(q, factor)
        attn = self.matmul(q, k)

        attn = self.cast(attn, ori_type)
        attn = self.add_4d(attn, self.relative_position_bias())

        if mask is not None:
            nw, ws2, _ = mask.shape
            # mask: [64, 49, 49] ==> [1, 64, 1, 49, 49]
            mask = self.reshape(mask, (1, nw, 1, ws2, ws2))
            attn = self.reshape(attn, (b // nw, nw, self.num_heads, seq, seq))
            attn = self.add_5d(attn, mask)
            attn = self.reshape(attn, (-1, self.num_heads, seq, seq))
            attn = self._softmax(attn)
        else:
            attn = self._softmax(attn)
        attn = self.cast(attn, self.compute_type)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (b, seq, c))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinIntermediate(FeedForward):
    """MLP for Swin Transformer Blocks."""

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size=None,
                 out_features=None,
                 dropout_rate=0.,
                 hidden_act='gelu',
                 weight_init='normal',
                 use_dropout=True,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        ffn_hidden_size = ffn_hidden_size or hidden_size
        super(SwinIntermediate, self).__init__(
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


class SwinRelativePositionBias(nn.Cell):
    """relative position bias for swin"""
    def __init__(self, window_size, num_heads):
        super(SwinRelativePositionBias, self).__init__()
        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)

        self.relative_position_bias_table = Parameter(
            weight_init_.initializer(
                weight_init_.TruncatedNormal(sigma=.02),
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

        relative_position_index = Tensor(np.sum(relative_coords, axis=-1), mstype.int32)  # Wh*Ww, Wh*Ww
        self.relative_position_index = Parameter(
            relative_position_index, requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.expand_dim = P.ExpandDims()
        self.gather = P.Gather()

    def construct(self):
        relative_position_index = self.relative_position_index.view(-1)
        relative_position_bias = self.gather(self.relative_position_bias_table, relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = self.expand_dim(relative_position_bias, 0)
        return relative_position_bias


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = x.shape
    x = np.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2), parallel_config=default_dpmp_config):
        # pylint: disable=W0613
        super(Roll, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel  # pylint: disable=W0612
        else:
            dp = 1  # pylint: disable=W0612
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x):
        x = numpy.roll(x, self.shift_size, self.shift_axis)
        return x


class WindowPartition(nn.Cell):
    """WindowPartitionConstruct Cell"""

    def __init__(self, window_size, parallel_config=None):
        super(WindowPartition, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))

        self.window_size = window_size

    def construct(self, x):
        """
        Args:
            x: (B, H, W, C)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        b, h, w, c = x.shape
        x = self.reshape(x, (b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (b * h * w // (self.window_size ** 2), self.window_size, self.window_size, c))
        return x


class WindowReverse(nn.Cell):
    """WindowReverseConstruct Cell"""

    def __init__(self, parallel_config=None):
        super(WindowReverse, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))

    def construct(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = self.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (b, h, w, -1))
        return x
