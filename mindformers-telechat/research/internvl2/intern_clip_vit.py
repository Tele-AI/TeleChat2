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
"""Internvl2 models' APIs."""
from typing import Optional

import numpy as np

import mindspore.ops.operations as P
from mindspore import nn, Parameter, Tensor
from mindspore import dtype as mstype
import mindspore.ops as ops

from mindformers import PreTrainedModel
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.modules.layers import Linear

from research.internvl2.internvl_config import InternVisionConfig

__all__ = ['InternVisionModel']


class InternRMSNorm(nn.Cell):
    """RMSNorm hidden_states for InternAttention"""
    def __init__(self, config: InternVisionConfig, hidden_size, eps=1e-6):
        super(InternRMSNorm, self).__init__()
        self.config = config
        dp = self.config.parallel_config.data_parallel
        self.weight = Parameter(Tensor(np.ones(hidden_size),
                                       dtype=mstype.float32), name="weight", parallel_optimizer=False)
        self.variance_epsilon = eps
        self.square = P.Square().shard(((dp, 1, 1),))
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.mul1 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul2 = P.Mul().shard(((1,), (dp, 1, 1)))
        self.cast = P.Cast()
        self.add = P.Add().shard(((dp, 1, 1), ()))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.real_div = P.RealDiv().shard(((), (dp, 1, 1)))

    def construct(self, hidden_states):
        """Construct

        Args:
            hidden_states (ms.Tensor): hidden_states tensor.

        Returns:
            hidden_states (ms.Tensor): Hiddenstate tensor.
        """
        input_dtype = hidden_states.dtype
        hidden_states = self.cast(hidden_states, mstype.float32)
        variance = self.square(hidden_states)
        variance = self.mean(variance, -1)
        sqrt_var = self.sqrt(self.add(variance, self.variance_epsilon))
        rsqrt_var = self.real_div(1, sqrt_var)
        hidden_states = self.mul1(hidden_states, rsqrt_var)
        hidden_states = self.mul2(self.weight, hidden_states)
        hidden_states = self.cast(hidden_states, input_dtype)
        return hidden_states


class InternVisionEmbeddings(nn.Cell):
    """InternVisionEmbeddings Of VisionModel

    Args: config: model_config for vision model
    """

    def __init__(self, config: InternVisionConfig):
        super(InternVisionEmbeddings, self).__init__()
        self.config = config
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.compute_dtype = config.compute_dtype
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.class_embedding = Parameter(ops.randn(1, 1, self.embed_dim, dtype=config.param_init_type),
                                         parallel_optimizer=False)
        self.tile = P.Tile().shard(((1, 1, 1),))
        self.slice = P.StridedSlice().shard(((1, 1, 1),))
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=True,
            pad_mode='pad',
            padding=0,
            dtype=config.param_init_type
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = Parameter(
            ops.randn(1, self.num_positions, self.embed_dim, dtype=config.param_init_type), parallel_optimizer=False)
        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.resize = P.ResizeBicubic(align_corners=False, half_pixel_centers=False)
        self.patch_embedding.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.patch_embedding.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.shape = P.Shape()
        self.transpose_pos_4dim = P.Transpose().shard(((1, 1, 1, 1),))
        self.resize.shard(((1, 1, 1, 1), (1, 1)))
        self.transpose_pos_3dim = P.Transpose().shard(((1, 1, 1),))
        self.cat_2 = P.Concat(axis=1).shard(((1, 1, 1), (1, 1, 1)))
        self.resize_shape = Tensor([32, 32], mstype.int32)

    def _get_pos_embed(self, pos_embed):
        """Get position embedding"""
        target_dtype = pos_embed.dtype
        pos_embed_shape = self.shape(pos_embed)
        pos_embed = self.cast(pos_embed, mstype.float32)

        pos_embed = self.reshape(pos_embed, (
            pos_embed_shape[0], self.image_size // self.patch_size, self.image_size // self.patch_size,
            pos_embed_shape[2]))
        pos_embed = self.transpose_pos_4dim(pos_embed, (0, 3, 1, 2))

        pos_embed = self.resize(pos_embed, self.resize_shape)
        pos_embed_shape_new = self.shape(pos_embed)
        pos_embed = self.reshape(pos_embed, (pos_embed_shape_new[0], pos_embed_shape_new[1], 32 * 32))
        pos_embed = self.transpose_pos_3dim(pos_embed, (0, 2, 1))
        pos_embed = self.cast(pos_embed, target_dtype)
        return pos_embed

    def construct(self, pixel_values):
        """Construct

        Args:
            pixel_values (ms.Tensor): pixel_values tensor.

        Returns:
            embeddings (ms.Tensor): Hiddenstate tensor.
        """
        patch_embeds = self.patch_embedding(self.cast(pixel_values, self.compute_dtype))
        batch_size, _, height, width = self.shape(patch_embeds)
        patch_embeds = self.reshape(patch_embeds, (batch_size, _, height * width))
        patch_embeds = self.transpose(patch_embeds, (0, 2, 1))
        class_embeds = self.cast(self.tile(self.class_embedding, (batch_size, 1, 1)), self.compute_dtype)
        embeddings = self.cat([class_embeds, patch_embeds])

        front_position_embedding = self.slice(self.position_embedding, (0, 0, 0), (1, 1, self.embed_dim), (1, 1, 1))
        rare_position_embedding = self.slice(self.position_embedding, (0, 1, 0),
                                             (1, self.num_positions, self.embed_dim), (1, 1, 1))
        rare_position_embeds = self._get_pos_embed(rare_position_embedding)
        position_embedding = self.cat_2((front_position_embedding, rare_position_embeds))
        embeddings = self.add(embeddings, self.cast(position_embedding, self.compute_dtype))
        return embeddings


class InternAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super(InternAttention, self).__init__()
        self.config = config
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = Linear(self.embed_dim, 3 * self.embed_dim, has_bias=config.qkv_bias,
                          param_init_type=config.param_init_type, compute_dtype=config.compute_dtype)
        self.qkv.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.matmul = P.MatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.config, self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.config, self.embed_dim, eps=config.layer_norm_eps)
        self.input_layout = "BNSD"
        self.proj = Linear(self.embed_dim, self.embed_dim,
                           param_init_type=config.param_init_type, compute_dtype=config.compute_dtype)
        self.proj.shard(strategy_matmul=((dp, mp), (1, mp)), out_strategy_matmul=((dp, 1),),
                        strategy_bias=((dp, 1), (1,)))

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.transpose1 = P.Transpose().shard(((dp, 1, 1, mp),))
        self.transpose2 = P.Transpose().shard(((dp, 1, 1, mp),))
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True).shard(((dp, 1, 1, mp), (dp, 1, 1, mp)))
        self.batch_matmul = P.BatchMatMul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.softmax = P.Softmax().shard(((dp, 1, 1, 1),))
        self.merger_head_transpose = P.Transpose().shard(((dp, 1, 1, mp),))
        self.split_qkv = ops.auto_generate.SplitWithSize()
        self.split_qkv.add_prim_attr("skip_redistribution", True)
        self.dtype = self.config.compute_dtype

    def construct(self, hidden_states):
        """Get attention score with flash attention"""
        batch, nums, channel = self.shape(hidden_states)
        qkv = self.qkv(hidden_states)
        q, k, v = self.split_qkv(qkv, (channel, channel, channel), 2)
        query = self.cast(self.transpose1(self.reshape(q, (batch, nums, self.num_heads, self.head_dim)),
                                          (0, 2, 1, 3)), self.dtype)
        key = self.cast(self.transpose1(self.reshape(k, (batch, nums, self.num_heads, self.head_dim)),
                                        (0, 2, 1, 3)), self.dtype)
        value = self.cast(self.transpose1(self.reshape(v, (batch, nums, self.num_heads, self.head_dim)),
                                          (0, 2, 1, 3)), self.dtype)
        if self.qk_normalization:
            batch_, high_, nums_, dims_ = self.shape(query)
            q = self.transpose2(query, (0, 2, 1, 3))
            q = self.reshape(q, (batch_, nums_, high_ * dims_))
            q = self.q_norm(q)
            q = self.reshape(q, (batch_, nums_, high_, dims_))
            q = self.transpose1(q, (0, 2, 1, 3))
            batch_, high_, nums_, dims_ = self.shape(key)
            k = self.transpose2(key, (0, 2, 1, 3))
            k = self.reshape(k, (batch_, nums_, high_ * dims_))
            k = self.k_norm(k)
            k = self.reshape(k, (batch_, nums_, high_, dims_))
            k = self.transpose1(k, (0, 2, 1, 3))
        attn = self.batch_matmul_q_k((q * self.scale), k)
        attn = self.softmax(self.cast(attn, mstype.float32))
        x = self.batch_matmul(self.cast(attn, self.dtype), value)
        x = self.merger_head_transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (batch, nums, channel))
        x = self.proj(x)
        return x


ACT2FN = {
    'relu': ops.ReLU(),
    'gelu': ops.GeLU(),
}


class InternMLP(nn.Cell):
    """
    Module to transformer image dimension
    Args:
        config (InternVisionConfig): The config of InternVisionConfig model.
    """
    def __init__(self, config: InternVisionConfig):
        super(InternMLP, self).__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = Linear(config.hidden_size, config.intermediate_size,
                          param_init_type=config.param_init_type, compute_dtype=config.compute_dtype)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size,
                          param_init_type=config.param_init_type, compute_dtype=config.compute_dtype)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.fc1.shard(((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.fc2.shard(((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))
        self.act.shard(((dp, 1, mp),))

    def construct(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Cell):
    """VisionTransformer Of VisionModel

    Args: config: model_config for Vision model
    """
    def __init__(self, config: InternVisionConfig):
        super(InternVisionEncoderLayer, self).__init__()
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = InternRMSNorm(config, self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = InternRMSNorm(config, self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = Parameter(config.initializer_factor * ops.ones(self.embed_dim, dtype=config.compute_dtype))
        self.ls2 = Parameter(config.initializer_factor * ops.ones(self.embed_dim, dtype=config.compute_dtype))

        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.cast = P.Cast()

    def construct(self, hidden_states):
        """
        Args:
            hidden_states (`Tuple[FloatTensor, Optional[FloatTensor]]`):
            input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        norm_hd = self.cast(self.norm1(hidden_states), hidden_states.dtype)
        hidden_states = self.add(hidden_states, self.mul(self.attn(norm_hd), self.ls1))
        hidden_states = self.add(hidden_states, self.mul(self.mlp(self.norm2(hidden_states)), self.ls2))
        return hidden_states


class InternVisionEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super(InternVisionEncoder, self).__init__()
        self.config = config
        # stochastic depth decay rule
        self.layers = nn.CellList([
            InternVisionEncoderLayer(config) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def construct(self, inputs_embeds):
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
        """
        hidden_states = inputs_embeds

        for _, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = nn.Cell.recompute(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs
        return hidden_states


class InternVisionPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = InternVisionConfig
    base_model_prefix = "InternVision"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InternVisionModel(InternVisionPreTrainedModel):
    """VisionTransformer Of CLIPModel

    Args: config: model_config for Vision model
    """
    main_input_name = 'pixel_values'
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings

    def construct(
            self,
            pixel_values: Optional[Tensor] = None,
            pixel_embeds: Optional[Tensor] = None,
    ):
        """
        InternVL forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq_length)`.
            images(Tensor): the image tensor with datatype float32, Tensor of shape :math:
            `(batch, 3, image_resolution, image_resolution)`
            image_context_pos(Tensor): the position index of the image in final input embedding. Tensor of shape :math
            `(batch, num_queries, 2)`
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): the input embedding Tensor of shape :math:`(batch, seq_length, hidden_size)`.
                Default None.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs
