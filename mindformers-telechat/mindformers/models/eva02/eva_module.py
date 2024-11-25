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
# This file was refer to project:
# https://github.com/facebookresearch/mae
# ============================================================================
"""EVA-02 Modules' APIs."""
import math
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.common.initializer as init
from mindspore import nn, ops, Parameter, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.modules.layers import Linear

from .eva_mlp import SwiGLU, GluMlp, Mlp


class PatchEmbed(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_features: int = 3,
                 out_features: int = 768,
                 has_bias: bool = True,
                 pad_mode: str = 'pad',
                 compute_dtype=mstype.float32):
        super(PatchEmbed, self).__init__()
        self.dtype = compute_dtype

        # build modules
        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=patch_size,
            weight_init=init.TruncatedNormal(sigma=0.02),
            has_bias=has_bias,
            pad_mode=pad_mode,
            dtype=compute_dtype)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """Image Patch Embedding Forward."""
        x = self.cast(x, self.dtype)
        x = self.proj(x)
        b, d, h, w = F.shape(x)
        x = self.reshape(x, (b, d, h * w))  # NDHW -> NLC
        x = self.transpose(x, (0, 2, 1))  # NCL -> NLC
        return x


class RotaryEmbeddingCat(nn.Cell):
    """ Rotary position embedding w/ concatenate sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self,
                 head_dim,
                 feat_shape,
                 ref_feat_shape,
                 temperature=10000.,
                 max_res=224.,
                 linear_bands=False,
                 in_pixels=True,
                 rotary_emb_type=mstype.float32):
        super().__init__()
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands

        # cache full sin/cos embeddings if shape provided up front
        embeds = self.build_rotary_pos_embed(
            head_dim=head_dim,
            feat_shape=feat_shape,
            ref_feat_shape=ref_feat_shape,
            in_pixels=in_pixels,
            dtype=rotary_emb_type
        )
        self.pos_embed = F.cat(embeds, -1)

    def get_pos_embed(self):
        """Get rotary position embedding."""
        return self.pos_embed

    def build_rotary_pos_embed(self, head_dim, feat_shape, ref_feat_shape, in_pixels=True, dtype=mstype.float32):
        """Build rotary position embedding."""
        sin_emb, cos_emb = self.build_fourier_pos_embed(
            head_dim // 4,
            feat_shape,
            ref_feat_shape,
            in_pixels=in_pixels,
            dtype=dtype,
        )
        num_spatial_dim = 1
        for x in feat_shape:
            num_spatial_dim *= x

        sin_emb = F.reshape(sin_emb, (num_spatial_dim, -1))
        sin_emb = F.repeat_interleave(sin_emb, repeats=2, axis=-1)
        cos_emb = F.reshape(cos_emb, (num_spatial_dim, -1))
        cos_emb = F.repeat_interleave(cos_emb, repeats=2, axis=-1)
        return sin_emb, cos_emb

    @staticmethod
    def freq_bands(num_bands, temperature=10000., step=2, dtype=mstype.float32):
        """Get bands frequency."""
        exp = F.arange(0, num_bands, step, dtype=dtype)
        exp = F.pow(temperature, F.div(exp, num_bands))
        bands = F.div(1., exp)
        return bands

    @staticmethod
    def pixel_freq_bands(num_bands, max_freq=224., linear_bands=True, dtype=mstype.float32):
        """Get pixel bands frequency."""
        if linear_bands:
            bands = F.linspace(1.0, max_freq / 2, num_bands)
        else:
            bands = F.pow(2., F.linspace(0, math.log(max_freq, 2) - 1, num_bands))
        bands = F.cast(bands, dtype=dtype)
        bands = F.mul(bands, math.pi)
        return bands

    def build_fourier_pos_embed(self, num_bands, feat_shape, ref_feat_shape, in_pixels=True, dtype=mstype.float32):
        """Get input position sin and cos value."""
        if in_pixels:
            bands = self.pixel_freq_bands(num_bands,
                                          float(self.max_res),
                                          linear_bands=self.linear_bands,
                                          dtype=dtype)
        else:
            bands = self.freq_bands(num_bands,
                                    temperature=self.temperature,
                                    step=1,
                                    dtype=dtype)

        t = [F.linspace(-1., 1., steps=s) if in_pixels else F.arange(s, dtype=dtype) for s in feat_shape]

        if ref_feat_shape is not None:
            # eva's scheme for resizing rope embeddings (ref shape = pretrain)
            t = [F.mul(F.div(x, f), r) for x, f, r in zip(t, feat_shape, ref_feat_shape)]

        ndgrid = F.meshgrid(t[0], t[1], indexing='ij')
        grid = F.stack(ndgrid, axis=-1)
        grid = F.expand_dims(grid, -1)

        pos = F.mul(grid, bands)
        pos_sin = F.sin(pos)
        pos_cos = F.cos(pos)

        return pos_sin, pos_cos


class EvaAttention(nn.Cell):
    """
    The implementation of attention module in EVA model.
    """

    def __init__(self,
                 hidden_size: int,
                 num_attn_heads: int = 8,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 layer_norm_eps: float = 1e-6,
                 qkv_bias: bool = True,
                 use_qkv_fused: bool = True,
                 use_qkv_simple: bool = False,
                 use_attn_norm: bool = True,
                 attn_head_dim=None,
                 compute_dtype=mstype.float32,
                 layer_norm_type=mstype.float32,
                 param_init_type=mstype.float32):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        head_dim = hidden_size // num_attn_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        real_head_dim = head_dim * self.num_attn_heads
        self.scale = Tensor(head_dim ** -0.5, dtype=param_init_type)

        self.use_qkv_fused = use_qkv_fused
        self.use_qkv_simple = use_qkv_simple
        if not use_qkv_fused:  # for eva-clip use_qkv_fused is equal to use_attn_norm
            self.q_proj = Linear(hidden_size, real_head_dim, has_bias=qkv_bias,
                                 param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.k_proj = Linear(hidden_size, real_head_dim, has_bias=False,
                                 param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.v_proj = Linear(hidden_size, real_head_dim, has_bias=qkv_bias,
                                 param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None
        elif use_qkv_simple:
            self.qkv = Linear(hidden_size, real_head_dim * 3, has_bias=True,
                              param_init_type=param_init_type, compute_dtype=compute_dtype)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv = Linear(hidden_size, real_head_dim * 3, has_bias=False,
                              param_init_type=param_init_type, compute_dtype=compute_dtype)
            if qkv_bias:
                self.q_bias = Parameter(ops.zeros(real_head_dim, dtype=param_init_type))
                # k_bias not used in network
                self.k_bias = Parameter(ops.zeros(real_head_dim, dtype=param_init_type), requires_grad=False)
                self.v_bias = Parameter(ops.zeros(real_head_dim, dtype=param_init_type))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
            self.q_proj = self.k_proj = self.v_proj = None

        self.attn_drop = nn.Dropout(p=attn_drop, dtype=param_init_type)
        if use_attn_norm:
            self.layer_norm = nn.LayerNorm((real_head_dim,), layer_norm_eps, layer_norm_type)
        else:
            self.layer_norm = nn.Identity()
        self.proj = Linear(real_head_dim, hidden_size, param_init_type=param_init_type, compute_dtype=compute_dtype)
        self.proj_drop = nn.Dropout(p=proj_drop, dtype=param_init_type)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.stride_slice = P.StridedSlice()
        self.mul = P.Mul()
        self.add = P.Add()
        self.div = P.Div()
        self.real_div = P.RealDiv()
        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        self.bmm = P.BatchMatMul(transpose_b=True)
        self.bmm_static = P.BatchMatMul(transpose_b=False)
        self.cast = P.Cast()
        self.stack = P.Stack(-1)
        self.masked_fill = P.MaskedFill()
        self.expand_dim = P.ExpandDims()
        self.assign_qkv = P.Assign()
        self.softmax = P.Softmax()

    def construct(self, x, rope=None, attn_mask=None):
        """EVA Attention Forward."""
        bs, seq_len, dim = F.shape(x)

        if not self.use_qkv_fused:  # B, num_heads, N, C
            q = self.q_proj(x)
            q = self.reshape(q, (bs, seq_len, self.num_attn_heads, -1))
            q = self.transpose(q, (0, 2, 1, 3))
            k = self.k_proj(x)
            k = self.reshape(k, (bs, seq_len, self.num_attn_heads, -1))
            k = self.transpose(k, (0, 2, 1, 3))
            v = self.v_proj(x)
            v = self.reshape(v, (bs, seq_len, self.num_attn_heads, -1))
            v = self.transpose(v, (0, 2, 1, 3))
        elif self.use_qkv_simple:
            qkv = self.qkv(x)
            q, k, v = self.extract_qkv(qkv, (bs, seq_len, 3, self.num_attn_heads, -1))
        else:
            qkv = self._fused_qkv(x, (bs, seq_len, dim))
            q, k, v = self.extract_qkv(qkv, (bs, seq_len, 3, self.num_attn_heads, -1))

        if rope is not None:
            q = self._apply_rope(q, v, rope)
            k = self._apply_rope(k, v, rope)

        q = self.mul(q, self.scale)
        attn = self.bmm(q, k)
        if attn_mask is not None:
            attn_bool = self.cast(attn, mstype.bool_)
            attn_shape = F.shape(attn)
            if len(attn_shape) != 2:
                raise ValueError(f"attn_shape length should be 2, but got {len(attn_shape)}")
            attn = F.broadcast_to(~attn, (attn_shape[0], 1, 1, attn_shape[1]))
            attn = self.masked_fill(attn, attn_bool, float("-inf"))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.bmm_static(attn, v)
        x = self.reshape(self.transpose(x, (0, 2, 1, 3)), (bs, seq_len, dim))
        x = self.layer_norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extract_qkv(self, qkv, shape):
        """Split qkv after fused linear."""
        qkv = self.reshape(qkv, shape)  # b, l, 3, n, d
        qkv = F.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = F.unbind(qkv, 0)  # B, num_heads, N, head_dim
        return q, k, v

    def _fused_qkv(self, x, x_shape):
        """Get fused qkv from linear."""
        qkv_bias = F.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
        qkv_weight = self.qkv.weight
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        qkv = self.matmul(x, qkv_weight)
        qkv = self.bias_add(qkv, qkv_bias)
        return qkv

    def _apply_rope(self, qk, v, rope):
        """Apply rope on query or key."""
        b, n, t, d = F.shape(qk)
        v_dtype = F.dtype(v)
        qk_token = self.stride_slice(qk, (0, 0, 0, 0), (b, n, 1, d), (1, 1, 1, 1))
        slice_qk = self.stride_slice(qk, (0, 0, 1, 0), (b, n, t, d), (1, 1, 1, 1))
        qk_emb = self.apply_rot_embed_cat(slice_qk, rope)
        qk = F.cat((qk_token, qk_emb), 2)
        qk = self.cast(qk, v_dtype)
        return qk

    def rot(self, x):
        """Rotary input data."""
        x_shape = F.shape(x)
        sub_x = self.stride_slice(-x, (0, 0, 0, 1), x_shape, (1, 1, 1, 2))
        x = self.stride_slice(x, (0, 0, 0, 0), x_shape, (1, 1, 1, 2))
        x = self.stack((sub_x, x))
        x = self.reshape(x, x_shape)
        return x

    def apply_rot_embed_cat(self, x, emb):
        """Apply rope on input data."""
        sin_emb, cos_emb = F.tensor_split(emb, 2, axis=-1)
        rot_x = self.rot(x)
        if sin_emb.ndim == 3:
            cos_emb = cos_emb.expand_as(x)
            cos_emb = self.expand_dim(cos_emb, 1)
            sin_emb = sin_emb.expand_as(x)
            sin_emb = self.expand_dim(sin_emb, 1)

        cos_x = self.mul(x, cos_emb)
        sin_x = self.mul(rot_x, sin_emb)
        x = self.add(cos_x, sin_x)
        return x


class EvaBlock(nn.Cell):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_hidden_size: int,
                 qkv_bias: bool = True,
                 swiglu_mlp: bool = True,
                 scale_mlp: bool = True,
                 use_qkv_fused: bool = True,
                 use_qkv_simple: bool = False,
                 use_attn_norm: bool = True,
                 proj_drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 layer_norm_eps: float = 1e-6,
                 use_post_norm: bool = False,
                 layer_norm=nn.LayerNorm,
                 compute_dtype=mstype.float32,
                 layer_norm_type=mstype.float32,
                 param_init_type=mstype.float32):
        super().__init__()
        self.use_post_norm = use_post_norm

        self.attn = EvaAttention(hidden_size=hidden_size,
                                 num_attn_heads=num_heads,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,
                                 layer_norm_eps=layer_norm_eps,
                                 qkv_bias=qkv_bias,
                                 use_qkv_fused=use_qkv_fused,
                                 use_qkv_simple=use_qkv_simple,
                                 use_attn_norm=use_attn_norm,
                                 compute_dtype=compute_dtype,
                                 layer_norm_type=layer_norm_type,
                                 param_init_type=param_init_type)

        self.norm1 = layer_norm((hidden_size,), layer_norm_eps, layer_norm_type)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_size=hidden_size,
                    hidden_size=mlp_hidden_size,
                    norm_layer=scale_mlp,
                    drop_prob=proj_drop,
                    compute_dtype=compute_dtype,
                    param_init_type=param_init_type,
                    layer_norm_type=layer_norm_type
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_size=hidden_size,
                    hidden_size=mlp_hidden_size * 2,
                    norm_layer=scale_mlp,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop_prob=proj_drop,
                    compute_dtype=compute_dtype,
                    param_init_type=param_init_type,
                    layer_norm_type=layer_norm_type
                )
        else:
            self.mlp = Mlp(
                in_size=hidden_size,
                hidden_size=mlp_hidden_size,
                norm_layer=scale_mlp,
                drop_prob=proj_drop,
                compute_dtype=compute_dtype,
                param_init_type=param_init_type,
                layer_norm_type=layer_norm_type
            )
        self.norm2 = layer_norm((hidden_size,), layer_norm_eps, layer_norm_type)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, x, rope=None, attn_mask=None):
        """Eva Block Forward."""
        if self.use_post_norm:
            # forward attention
            residual = x
            x = self.attn(x, rope=rope, attn_mask=attn_mask)
            x = self.add(residual, self.drop_path1(self.norm1(x)))

            # forward mlp
            residual = x
            x = self.mlp(x)
            x = self.add(residual, self.drop_path2(self.norm2(x)))
        else:
            # forward attention
            residual = x
            x = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
            x = self.add(residual, self.drop_path1(x))

            # forward mlp
            residual = x
            x = self.mlp(self.norm2(x))
            x = self.add(residual, self.drop_path2(x))
        return x


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.init_tensor = Tensor(np.ones(1,), dtype=mstype.float32)
        self.tile = P.Tile()
        self.div = P.Div()
        self.mul = P.Mul()

    def construct(self, x):
        """DropPath Forward."""
        epsilon = 1e-15
        if abs(self.drop_prob - 0.) < epsilon or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.tile(self.init_tensor, shape)
        random_tensor = F.bernoulli(random_tensor, p=self.keep_prob)
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = self.div(random_tensor, self.keep_prob)
        x = self.mul(x, random_tensor)
        return x
