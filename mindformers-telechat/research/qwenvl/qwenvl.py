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
# https://huggingface.co/Qwen/Qwen-VL
# ============================================================================
"""QwenVL models' APIs."""

import math
from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import nn, Parameter, Tensor
from mindspore import ops
from mindspore.common.initializer import initializer, TruncatedNormal, Normal
from mindspore.ops import operations as P
from mindformers import MultiHeadAttention, MindFormerRegister, MindFormerModuleType, PreTrainedModel, \
    TransformerOpParallelConfig
from mindformers.models import build_network
from mindformers.models.utils import lazy_inline
from mindformers.models.vit.vit_modules import get_2d_sincos_pos_embed
from mindformers.modules.activation import GELU
from mindformers.modules.layers import LayerNorm, Linear
from mindformers.tools.logger import logger

from qwenvl_config import QwenVLConfig, QwenVLVisionConfig


class AbsPos(nn.Cell):
    r"""
    Module to resize position embedding if src size do not equal to target size
    Args:
        src_size(int): the src size
        tgt_size(int): the target size

    Returns:
            x: Tensor, the resized input if the input needs to be resized, otherwise return the original input
    """

    def __init__(self, src_size: int, tgt_size: int):
        super().__init__()
        self.src_size = int(math.sqrt(src_size))
        self.tgt_size = int(math.sqrt(tgt_size))

        self.cast = P.Cast()
        self.reshape = P.Reshape().shard(((1, 1),))
        self.flatten = nn.Flatten(start_dim=0, end_dim=2)
        self.resize_shape = ms.Tensor([self.tgt_size, self.tgt_size], ms.int32)
        # Resize does not support shard
        self.resize = P.ResizeBicubic(align_corners=False, half_pixel_centers=False)
        self.transpose = P.Transpose().shard(((1, 1, 1, 1),))

    def construct(self, x: Tensor):
        """forward of AbsPos"""
        if self.src_size != self.tgt_size:
            ori_dtype = x.dtype
            x = self.reshape(x, (1, self.src_size, self.src_size, -1))
            x = self.transpose(x, (0, 3, 1, 2))
            x = self.cast(x, ms.float32)
            x = self.resize(x, self.resize_shape)
            x = self.cast(x, ori_dtype)
            x = self.transpose(x, (0, 2, 3, 1))
            x = self.flatten(x)
        return x


class Resampler(nn.Cell):
    """
    A 2D perceiver-resampler network with one cross attention layers by num_queries learnable queries and 2d
    sin_cos pos_emb

    Args:
        image_size(int): Size of image.
        patch_size(int): Patch size of image.
        hidden_size(int): The dimension of embed.
        num_queries(int): Nums of query tokens.
        output_dim(int): The target embed dim of output.
        parallel_config(TransformerOpParallelConfig): The parallel configure.
        compute_dtype: The type of Linear computation module.
        param_init_type:  The parameter initialization type of the module.
        softmax_compute_type: The type of softmax computation module.

    Returns:
        out: A tensor with the shape of (bs, num_queries, output_dim)
    """

    def __init__(self, image_size: int,
                 patch_size: int,
                 hidden_size: int,
                 num_queries: int,
                 output_dim: int,
                 parallel_config: TransformerOpParallelConfig,
                 compute_dtype,
                 param_init_type,
                 softmax_compute_type):
        super().__init__()

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.num_queries = num_queries
        self.query_grid_size = int(math.sqrt(num_queries))

        self.embed_dim = output_dim
        self.num_heads = self.embed_dim // 128

        self.kv_dim = hidden_size

        self.pos_embed = Parameter(get_2d_sincos_pos_embed(self.embed_dim, self.query_grid_size), requires_grad=False,
                                   parallel_optimizer=False)

        self.query = Parameter(initializer(TruncatedNormal(mean=0.0, sigma=0.02), [self.num_queries, self.embed_dim]),
                               requires_grad=False, parallel_optimizer=False)

        if self.kv_dim is not None and self.kv_dim != self.embed_dim:
            self.kv_proj = Linear(in_channels=self.kv_dim, out_channels=self.embed_dim,
                                  has_bias=False,
                                  compute_dtype=compute_dtype,
                                  param_init_type=param_init_type)
            self.kv_proj.shard(strategy_matmul=((dp, 1),
                                                (mp, 1)),
                               strategy_bias=((dp, mp), (mp,))
                               )
        else:
            self.kv_proj = nn.Identity()

        self.img_grid_size = image_size // patch_size
        self.attn = MultiHeadAttention(hidden_size=self.embed_dim,
                                       num_heads=self.num_heads,
                                       batch_size=None,
                                       src_seq_length=self.num_queries,
                                       tgt_seq_length=self.img_grid_size ** 2,
                                       hidden_dropout_rate=0.0,
                                       attention_dropout_rate=0.0,
                                       softmax_compute_type=softmax_compute_type,
                                       use_past=False,
                                       param_init_type=param_init_type,
                                       parallel_config=parallel_config.dp_mp_config)

        self.ln_q = LayerNorm((self.embed_dim,), eps=1e-6)
        self.ln_q.shard(((1, 1,),))
        self.ln_kv = LayerNorm((self.embed_dim,), eps=1e-6)
        self.ln_kv.shard(((dp, 1, 1),))

        self.abs_pos = AbsPos(self.pos_embed.shape[0], self.img_grid_size ** 2)

        self.shape = P.Shape()
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.expand_dims = P.ExpandDims().shard(((dp, 1),))
        self.expand_dims_no_shard = P.ExpandDims().shard(((1, 1),))

    def construct(self, x, attn_mask=None):
        """forward of Resampler"""
        # x: [bs, img_grid_size**2, width], [bs, 1024, 1664]
        bs, _, _ = self.shape(x)
        pos_embed = self.abs_pos(self.pos_embed)  # (img_grid_size**2, embed_dim) (1024, 4096)

        x = self.kv_proj(x)  # [bs, img_grid_size**2, embed_dim] [bs, 1024, 4096]
        x = self.ln_kv(x)  # [bs, img_grid_size**2, embed_dim] [bs, 1024, 4096]

        q = self.ln_q(self.query)  # [num_queries, embed_dim] [256, 4096]]

        query_tensor = self.add(self.tile(q, (bs, 1, 1)), self.expand_dims_no_shard(self.pos_embed, 0))
        key_tensor = self.add(x, self.expand_dims_no_shard(pos_embed, 0))

        out = self.attn(
            query_tensor,  # [bs, 256, 4096]
            key_tensor,  # [bs, 1024, 4096]
            x,  # [bs, 1024, 4096]
            attention_mask=attn_mask
        )[0]

        return out  # (bs, num_queries, embed_dim)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenVLVisionModel(PreTrainedModel):
    r"""VisionModel Of Qwen-VL

    Args:
        config (QwenVLVisionConfig): The config of VisionConfig for QwenVL
        num_queries(int): num of query tokens

    Returns:
        input_x: A tensor with the shape of (bs, num_queries, output_dim)
    """

    def __init__(self, config: QwenVLVisionConfig, num_queries: int = 256, **kwargs):
        super().__init__(config, **kwargs)
        self.num_queries = num_queries

        parallel_config = config.parallel_config
        hidden_size = config.hidden_size
        dtype = config.compute_dtype
        patch_size = config.patch_size
        self.conv1 = \
            nn.Conv2d(
                in_channels=3, out_channels=hidden_size, kernel_size=patch_size,
                stride=patch_size, has_bias=False, pad_mode='pad').to_float(dtype)
        self.conv1.conv2d.shard(((parallel_config.data_parallel, 1, 1, 1), (1, 1, 1, 1)))
        self.conv1.bias_add.shard(((parallel_config.data_parallel, 1, 1, 1), (1,)))

        scale = hidden_size ** -0.5
        self.positional_embedding = \
            Parameter(scale * Tensor(
                np.random.normal(0, 1, size=(256, hidden_size))).astype(dtype),
                      parallel_optimizer=False)
        self.ln_pre = LayerNorm((hidden_size,), eps=1e-6)
        self.ln_pre.shard(((parallel_config.data_parallel, 1, 1),))
        self.transformer = QwenVLTransformer(image_size=config.image_size,
                                             patch_size=patch_size,
                                             hidden_size=hidden_size,
                                             intermediate_size=config.intermediate_size,
                                             n_head=config.num_attention_heads,
                                             layers=config.num_hidden_layers,
                                             dtype=config.dtype,
                                             softmax_compute_type=config.softmax_compute_type,
                                             compute_dtype=config.compute_dtype,
                                             param_init_type=config.param_init_type,
                                             gelu_dtype=config.gelu_dtype,
                                             parallel_config=config.parallel_config,
                                             use_flash_attention=config.use_flash_attention,
                                             enable_fa_opt=config.enable_fa_opt)

        self.attn_pool = Resampler(image_size=config.image_size,
                                   patch_size=patch_size,
                                   hidden_size=hidden_size,
                                   num_queries=self.num_queries,
                                   output_dim=config.output_dim,
                                   parallel_config=config.parallel_config,
                                   compute_dtype=config.compute_dtype,
                                   param_init_type=config.param_init_type,
                                   softmax_compute_type=config.softmax_compute_type)
        self.transpose = P.Transpose().shard(((parallel_config.data_parallel, 1, 1),))
        self.ln_post = LayerNorm((config.output_dim,), eps=1e-6)
        self.ln_post.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.proj = \
            Parameter(scale * Tensor(np.random.normal(0, 1,
                                                      size=(config.output_dim, config.output_dim))).astype(dtype))
        self.dtype = dtype
        self.cast = P.Cast()
        self.add = P.Add().shard(((parallel_config.data_parallel, 1, 1), (1, 1)))
        img_grid_size = config.image_size // patch_size
        self.abs_pos = AbsPos(self.positional_embedding.shape[0], img_grid_size ** 2)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.matmul = P.BatchMatMul().shard(((dp, 1, mp), (mp, 1)))

    def construct(self, input_x: ms.Tensor):
        """forward of QwenVLVisionModel"""
        input_x = self.conv1(input_x)
        input_x = input_x.reshape(input_x.shape[0], input_x.shape[1], -1)
        input_x = self.transpose(input_x, (0, 2, 1))

        abs_pos = self.abs_pos(self.positional_embedding)
        input_x = self.add(input_x, abs_pos)
        input_x = self.ln_pre(input_x)
        input_x = self.transformer(input_x)
        input_x = self.attn_pool(input_x)
        input_x = self.ln_post(input_x)
        input_x = self.cast(input_x, self.dtype)
        input_x = self.matmul(input_x, self.proj)
        return input_x


class MLP(nn.Cell):
    """
    A multilayer perceptron for ViT
    """

    def __init__(self, layers: int, hidden_size: int, intermediate_size: int,
                 compute_dtype, param_init_type, gelu_dtype, parallel_config):
        super().__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        proj_std = (hidden_size ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * hidden_size) ** -0.5
        self.c_fc = Linear(hidden_size, intermediate_size, weight_init=Normal(mean=0.0, sigma=fc_std),
                           compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.c_fc.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.c_proj = Linear(intermediate_size, hidden_size, weight_init=Normal(mean=0.0, sigma=proj_std),
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.c_proj.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.gelu = GELU(approximate=False)
        self.cast = P.Cast()
        self.gelu_dtype = gelu_dtype
        self.dtype = P.DType()

    def construct(self, x):
        x = self.c_fc(x)
        ori_dtype = self.dtype(x)
        x = self.cast(x, self.gelu_dtype)
        x = self.gelu(x)
        x = self.cast(x, ori_dtype)
        x = self.c_proj(x)
        return x


class VisualFlashAttention(nn.Cell):
    """
    Flash Attention for visual module
    """

    def __init__(self, fa, parallel_config, size_per_head, enable_fa_opt=False):
        super().__init__()
        self.fa = fa
        self.enable_fa_opt = enable_fa_opt
        if self.enable_fa_opt:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
            self.stridden_slice = P.StridedSlice().shard(((dp, 1, 1, 1),))
            self.concat = P.Concat(axis=-1).shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.zeros = P.Zeros().shard(((dp, 1, 1, 1),))
            mul_value, remainder = divmod(size_per_head, 128)
            if remainder > 0:
                new_size_per_head = (mul_value + 1) * 128
            else:
                new_size_per_head = size_per_head
            self.pad_length = new_size_per_head - size_per_head
            if self.pad_length == 0:
                raise ValueError("size_per_head is divisible by 128, please disable enable_fa_opt")

    def construct(self, query, key, value, attention_mask):
        """forward for VisualFlashAttention"""
        bsz, num_head, seq, size_per_head = query.shape
        if self.enable_fa_opt:
            pad = self.zeros((bsz, num_head, seq, self.pad_length), P.DType()(query))
            query = self.concat([query, pad])
            key = self.concat([key, pad])
            value = self.concat([value, pad])
        weighted_values = self.fa(query, key, value, attention_mask)
        if self.enable_fa_opt:
            weighted_values = self.stridden_slice(weighted_values, (0, 0, 0, 0), (bsz, num_head, seq, size_per_head),
                                                  (1, 1, 1, 1))
        return weighted_values


class VisualAttention(MultiHeadAttention):
    def __init__(self, *args, use_attention_mask=False, enable_fa_opt=False, **kwargs):
        super().__init__(*args, **kwargs)
        parallel_config = kwargs.get('parallel_config')
        if self.use_flash_attention and not use_attention_mask:
            self.flash_attention = VisualFlashAttention(self.flash_attention, parallel_config, self.size_per_head,
                                                        enable_fa_opt=enable_fa_opt)


class ResidualAttentionBlock(nn.Cell):
    r"""
    ResidualAttentionBlock of QwenVLVisionModel

    Args:
        image_size(int): size of image
        patch_size(int): patch size of image
        hidden_size(int): the embed dim of input
        hidden_size(int): The dimension of embed.
        intermediate_size(int): The linear width in MLP.
        n_head(int): The number of attention heads.
        layers(int): The number of transformer layers for weight initialization.
        dtype(mstype): The type of Linear computation module.
        softmax_compute_type(mstype): The type of softmax computation module.
        compute_dtype(mstype): The type of linear computation module.
        param_init_type(mstype):  The parameter initialization type of the module.
        gelu_dtype(mstype): The type of gelu activation computation module.
        parallel_config: The parallel configure.
        use_flash_attention: Whether to use flash attention
        enable_fa_opt: Whether to enable padding MatMul operation flash attention.
        attn_mask (Optional[ms.Tensor]): attention mask.
    """

    def __init__(self, image_size: int, patch_size: int, hidden_size: int, intermediate_size: int, n_head: int,
                 layers: int,
                 dtype: mstype,
                 softmax_compute_type,
                 compute_dtype,
                 param_init_type,
                 gelu_dtype,
                 parallel_config,
                 use_flash_attention: bool = False,
                 enable_fa_opt: bool = False,
                 attn_mask: Optional[ms.Tensor] = None):
        super().__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.dtype = dtype
        img_grid_size = image_size // patch_size
        self.attn = VisualAttention(hidden_size=hidden_size,
                                    num_heads=n_head,
                                    batch_size=None,
                                    src_seq_length=img_grid_size ** 2,
                                    tgt_seq_length=img_grid_size ** 2,
                                    hidden_dropout_rate=0.0,
                                    attention_dropout_rate=0.0,
                                    softmax_compute_type=softmax_compute_type,
                                    use_past=False,
                                    compute_dtype=compute_dtype,
                                    param_init_type=param_init_type,
                                    parallel_config=parallel_config.dp_mp_config,
                                    use_flash_attention=use_flash_attention,
                                    use_attention_mask=False,
                                    enable_fa_opt=enable_fa_opt)
        self.ln_1 = LayerNorm((hidden_size,), eps=1e-6, param_init_type=param_init_type)
        self.ln_1.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.mlp = MLP(layers=layers, hidden_size=hidden_size, intermediate_size=intermediate_size,
                       compute_dtype=compute_dtype, param_init_type=param_init_type, gelu_dtype=gelu_dtype,
                       parallel_config=parallel_config)
        self.ln_2 = LayerNorm((hidden_size,), eps=1e-6, param_init_type=param_init_type)
        self.ln_2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.attn_mask = attn_mask
        self.add = P.Add().shard(((dp, 1, mp), (dp, 1, mp)))

    def construct(self, input_x: ms.Tensor):
        r"""Construct"""
        # current (1, 1024, 1664)
        input_x = self.add(input_x, self.attention(self.ln_1(input_x)))
        input_x = self.add(input_x, self.mlp(self.ln_2(input_x)))
        return input_x

    def attention(self, input_x: ms.Tensor):
        r"""Attention"""
        return self.attn(input_x, input_x, input_x, self.attn_mask)[0]


class QwenVLTransformer(nn.Cell):
    r"""
    Transformer of QwenVLVisionModel

    Args:
        image_size(int): Size of image.
        patch_size(int): Patch size of image.
        hidden_size (int): The dimension of input features.
        intermediate_size(int): The linear width in MLP.
        n_head (int): The number of attention heads.
        layers (int): The number of transformer layers.
        attn_mask (ms.Tensor):  Attention mask.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, image_size: int, patch_size: int, hidden_size: int, intermediate_size: int, n_head: int,
                 layers: int,
                 dtype: mstype,
                 softmax_compute_type,
                 compute_dtype,
                 param_init_type,
                 gelu_dtype,
                 parallel_config,
                 use_flash_attention: bool = False,
                 enable_fa_opt: bool = False,
                 attn_mask: Optional[ms.Tensor] = None):
        super().__init__()
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(image_size=image_size,
                                     patch_size=patch_size,
                                     hidden_size=hidden_size,
                                     intermediate_size=intermediate_size,
                                     n_head=n_head,
                                     layers=layers,
                                     dtype=dtype,
                                     softmax_compute_type=softmax_compute_type,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type,
                                     gelu_dtype=gelu_dtype,
                                     parallel_config=parallel_config,
                                     use_flash_attention=use_flash_attention,
                                     enable_fa_opt=enable_fa_opt,
                                     attn_mask=attn_mask)
              for _ in range(layers)]
        )

    def construct(self, input_x):
        r"""Construct"""
        return self.resblocks(input_x)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenVL(PreTrainedModel):
    """
    Provide QwenVL training loss or logits through network.

    Args:
        config (QwenVLConfig): The config of QwenVL model.
    """

    @lazy_inline
    def __init__(self, config: QwenVLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.vision_encoder = build_network(config.vision_model)
        self.llm_model = build_network(config.llm_model, default_args={"num_queries": self.config.num_queries})

        self.image_start_id = self.config.image_start_id
        self.image_pad_id = self.config.image_pad_id
        self.num_queries = self.config.num_queries
        self.image_size = self.config.vision_model.model_config.image_size
        self.is_first_iteration = True
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.ignore_token_id = ms.Tensor(config.ignore_token_id, mstype.int32)
        self.use_past = config.use_past

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        parallel_config = config.parallel_config
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.slice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.masked_fill = P.MaskedFill().shard(
            ((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1), ()))
        self.tensor_scatter_update = ops.TensorScatterUpdate().shard(((1, 1, 1),
                                                                      (1, 1, 1),
                                                                      (1, 1, 1)))
        self.gather = P.Gather().shard(((1, 1, 1), ()))
        self.equal = P.Equal().shard(((parallel_config.data_parallel, 1), ()))
        self.ones = P.Ones()
        self.img_pos_add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.base_index_adder = None

        self.freeze_component()

    def freeze_component(self):
        """freeze components according to config"""
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_encoder.trainable_params():
                if not self.config.freeze_resampler and "vision_encoder.attn_pool" in param.name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if self.config.freeze_llm:
            logger.info("freeze llm model")
            for param in self.llm_model.trainable_params():
                param.requires_grad = False

    def generate_base_index_adder(self, batch_size):
        if self.base_index_adder is None:
            self.base_index_adder = ms.Tensor(
                [[i, 0] for i in range(batch_size)], ms.int32).reshape(batch_size, 1, 1, 2)

    # pylint: disable=W0613
    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        batch_size, _ = input_ids.shape
        self.generate_base_index_adder(batch_size)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation in inference"""
        is_first_iteration = self.is_first_iteration
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
            is_first_iteration = True

        if is_first_iteration or not self.use_past:
            images = kwargs.pop("images")
            img_pos = kwargs.pop("img_pos", None)
            if img_pos is not None:
                img_pos = ms.Tensor(img_pos, mstype.int32)
        else:
            batch_size, _ = input_ids.shape
            img_shape = (batch_size, 3, 3, self.image_size, self.image_size)
            images = self.ones(img_shape, ms.float32)
            img_pos = self.ones((batch_size, 1, self.config.num_queries, 2), mstype.int32)

        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "images": images,
            "img_pos": img_pos
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """prepare inputs for predict layout"""
        input_ids = Tensor(input_ids, mstype.int32)
        bs = input_ids.shape[0]

        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((bs, 1, 3, self.image_size, self.image_size)), ms.float32)

        if "img_pos" in kwargs:
            img_pos = Tensor(kwargs.get("img_pos"))
        else:
            img_pos = Tensor(np.random.randint(0, self.num_queries, (bs, 1, self.num_queries, 2)), ms.int32)

        self.generate_base_index_adder(bs)
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return input_ids, images, img_pos, None, None, None, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self):
        """set inputs when is_dynamic=True"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None, None], dtype=mstype.float32)
        dynamic_img_pos = Tensor(shape=[None, None, None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_img_pos, None, None, None,
                        None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        self.llm_model.set_dynamic_inputs()
        logger.info("Set dynamic inputs for Qwen-VL")

    def add_flags_custom(self, is_first_iteration):
        """add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.llm_model.add_flags_custom(is_first_iteration=is_first_iteration)

    def kvcache(self, layer_idx):
        return self.llm_model.kvcache(layer_idx)

    def concat_image_text(self, text_embeds, image_embeds, img_pos):
        """update the value at a specific position of the text embedding with the image embedding"""
        if self.training:
            img_pos = img_pos.reshape((-1, self.num_queries, 2))
        else:
            img_pos = self.img_pos_add(img_pos, self.base_index_adder).reshape((-1, self.num_queries, 2))
        image_embeds = self.cast(image_embeds, text_embeds.dtype)
        text_embeds = self.tensor_scatter_update(text_embeds, img_pos, image_embeds)
        return text_embeds

    def construct(self, input_ids, images, img_pos: Tensor = None, labels=None,
                  input_position=None, position_ids=None, attention_mask=None, init_reset=None, batch_valid_length=None,
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None):
        """forward of QwenVL"""
        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
            if labels is None:
                pad_input_ids_pos = self.equal(input_ids, self.pad_token_id)
                labels = self.masked_fill(input_ids, pad_input_ids_pos, self.ignore_token_id)
                pad_label_pos = self.equal(labels, self.pad_token_id)
                labels = self.masked_fill(labels, pad_label_pos, self.ignore_token_id)
        else:
            tokens = input_ids

        input_embeds = self.llm_model.to_embeddings(tokens)

        if attention_mask is None:
            attention_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

        if self.is_first_iteration or self.training:
            if images.ndim == 5:
                images_shape = self.shape(images)
                new_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
                images = self.reshape(images, new_shape)

            image_embeds = self.vision_encoder(images)
            input_embeds = self.concat_image_text(input_embeds, image_embeds, img_pos)

        return self.llm_model(
            input_ids=None,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
