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
"""LLaMA Vision models' APIs."""
import copy
import math

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, mint
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.generation import GenerationMixin
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import LayerSetting
from mindformers.modules.layers import Linear, LayerNorm
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode
from mindformers.models.build_model import build_network
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.models.llama.llama import LlamaModel
from mindformers.modules.transformer import VocabEmbedding
from mindformers.modules.activation import GELU
from mindformers.modules.flash_attention import FlashAttention

from .mllama_config import MllamaConfig, MllamaVisionConfig
from .mllama_transformer import MllamaCrossAttentionDecoderLayer
from ..utils import lazy_inline
from ...tools.logger import logger

__all__ = ['MllamaVisionModel', 'MllamaTextModel', 'MllamaForCausalLM', 'MllamaForConditionalGeneration']


class MllamaPrecomputedAspectRatioEmbedding(nn.Cell):
    """MllamaPrecomputedAspectRatioEmbedding"""
    def __init__(self, config, is_gated=True):
        super().__init__()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.add = P.Add()
        self.tanh = P.Tanh()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated
        self.param_init_type = config.param_init_type
        self.compute_dtype = config.compute_dtype

        self.embedding = VocabEmbedding(vocab_size=self.max_aspect_ratio_id + 1,
                                        embedding_size=self.max_num_tiles * self.hidden_size,
                                        param_init_type=self.param_init_type,
                                        parallel_config=config.parallel_config.embedding_dp_mp_config)
        if is_gated:
            self.gate = Parameter(initializer("zeros", [1], dtype=self.param_init_type))

    def construct(self, hidden_state, aspect_ratio_ids):
        embeddings, _ = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            gate_tanh = self.tanh(self.gate)
            embeddings = self.mul(embeddings, gate_tanh)
        hidden_state = self.add(hidden_state, embeddings)
        return hidden_state

    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.mul.shard(((dp, 1, 1, 1), (1,)))
        self.add.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))


class MllamaPrecomputedPositionEmbedding(nn.Cell):
    """MllamaPrecomputedPositionEmbedding"""
    def __init__(self, config):
        super().__init__()
        self.mul = P.Mul()
        self.mul1 = P.Mul()
        self.add = P.Add()
        self.add1 = P.Add()
        self.sub = P.Sub()
        self.tanh = P.Tanh()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size ** -0.5
        self.param_init_type = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.gate = Parameter(initializer("zeros", [1], dtype=self.param_init_type))

        # position embedding
        position_embedding = Tensor(np.random.randn(self.num_patches, self.hidden_size), dtype=self.param_init_type)
        self.embedding = Parameter(initializer(self.scale * position_embedding, [self.num_patches, self.hidden_size],
                                               dtype=self.param_init_type))

        # tile position embedding
        self.tile_embedding = VocabEmbedding(vocab_size=self.max_aspect_ratio_id + 1,
                                             embedding_size=self.max_num_tiles * self.num_patches * self.hidden_size,
                                             param_init_type=self.param_init_type,
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)

    def construct(self, hidden_state, aspect_ratio_ids):
        """compute position embeddings"""
        # position embeddings
        gate = self.tanh(self.gate)
        gate = self.sub(1, gate)
        gated_position_embedding = self.mul(gate, self.embedding)
        hidden_state = self.add(hidden_state,
                                self.reshape(gated_position_embedding, (1, 1, self.num_patches, self.hidden_size)))

        # precomputed tile position embeddings
        tile_position_embedding, _ = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = self.reshape(tile_position_embedding, (
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        ))
        gated_tile_position_embedding = self.mul1(tile_position_embedding, self.tanh(self.gate))
        gated_tile_position_embedding = self.cast(gated_tile_position_embedding, self.compute_dtype)
        hidden_state = self.add1(hidden_state, gated_tile_position_embedding)

        return hidden_state

    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.add.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.mul1.shard(((dp, 1, 1, 1), (1,)))
        self.add1.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Cell):
    """MllamaVisionMLP"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = GELU()
        self.compute_dtype = config.compute_dtype
        self.param_init_type = config.param_init_type
        self.fc1 = Linear(in_channels=config.hidden_size,
                          out_channels=config.intermediate_size,
                          compute_dtype=self.param_init_type,
                          param_init_type=self.compute_dtype)
        self.fc2 = Linear(in_channels=config.intermediate_size,
                          out_channels=config.hidden_size,
                          compute_dtype=self.param_init_type,
                          param_init_type=self.compute_dtype)

    def construct(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states

    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.fc1.shard(((dp, 1,), (mp, 1)), ((dp, mp), (mp,)))
        self.fc2.shard(strategy_matmul=((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)),
                       out_strategy_matmul=((dp, 1),))
        self.activation_fn.shard(((dp, 1, mp),))


class MllamaVisionAttention(nn.Cell):
    """MllamaVisionAttention"""
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul()
        self.mul = P.Mul()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.transpose_attn = P.Transpose()
        self.slice = P.StridedSlice()
        self.compute_dtype = config.compute_dtype
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=self.compute_dtype)
        self.use_flash_attention = config.use_flash_attention
        self.q_proj = Linear(self.embed_dim, self.num_heads * self.head_dim, has_bias=False,
                             compute_dtype=self.compute_dtype,
                             param_init_type=config.param_init_type)
        self.k_proj = Linear(self.embed_dim, self.num_heads * self.head_dim, has_bias=False,
                             compute_dtype=self.compute_dtype,
                             param_init_type=config.param_init_type)
        self.v_proj = Linear(self.embed_dim, self.num_heads * self.head_dim, has_bias=False,
                             compute_dtype=self.compute_dtype,
                             param_init_type=config.param_init_type)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.embed_dim, has_bias=False,
                             compute_dtype=self.compute_dtype,
                             param_init_type=config.param_init_type)

        if self.use_flash_attention:
            self.input_layout = "BNSD"
            self.flash_attention = FlashAttention(head_num=self.num_heads,
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  input_layout=self.input_layout,
                                                  keep_prob=1.,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  use_attention_mask=True,
                                                  sparse_mode=1)

    def construct(self, hidden_state, attention_mask=None):
        """Forward of MllamaVisionAttention"""
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = self.reshape(query, (batch_size, q_seq_len, self.num_heads, self.head_dim))
        key = self.reshape(key, (batch_size, kv_seq_len, self.num_heads, self.head_dim))
        value = self.reshape(value, (batch_size, kv_seq_len, self.num_heads, self.head_dim))

        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))
        if self.use_flash_attention:
            attn_output = self.flash_attention(query, key, value, attention_mask)
        else:
            attn_weights = self.mul(self.batch_matmul_q_k(query, key), self.inv_norm_factor)
            causal_mask = self.slice(attention_mask, (0, 0, 0, 0), (batch_size, 1, q_seq_len, key.shape[-2]),
                                     (1, 1, 1, 1))

            attn_weights = self.add(attn_weights, causal_mask)

            attn_weights = self.softmax(self.cast(attn_weights, mstype.float32))
            attn_weights = self.cast(attn_weights, self.compute_dtype)

            attn_output = self.batch_matmul(attn_weights, value)

        attn_output = self.transpose_attn(attn_output, (0, 2, 1, 3))
        attn_output = self.reshape(attn_output, (batch_size, q_seq_len, -1))
        output = self.o_proj(attn_output)

        return output

    def shard(self, parallel_config):
        """shard"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.q_proj.shard(((dp, 1), (mp, 1)))
        self.k_proj.shard(((dp, 1), (mp, 1)))
        self.v_proj.shard(((dp, 1), (mp, 1)))
        self.o_proj.shard(strategy_matmul=((dp, mp), (1, mp)), out_strategy_matmul=((dp, 1),))
        self.transpose.shard(((dp, 1, mp, 1),))
        self.transpose_attn.shard(((dp, mp, 1, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.slice.shard(((dp, 1, 1, 1),))
        self.add.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.softmax.shard(((dp, mp, 1, 1),))
        if self.use_flash_attention:
            self.flash_attention.shard(parallel_config)


class MllamaVisionEncoderLayer(nn.Cell):
    """MllamaVisionEncoderLayer"""
    def __init__(self, config, is_gated=False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size
        self.cast = P.Cast()
        self.param_init_type = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.add = P.Add()
        self.mul = P.Mul()
        self.tanh = P.Tanh()
        self.self_attn = MllamaVisionAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = LayerNorm(self.hidden_size, eps=config.norm_eps,
                                         param_init_type=self.param_init_type)
        self.post_attention_layernorm = LayerNorm(self.hidden_size, eps=config.norm_eps,
                                                  param_init_type=self.param_init_type)

        if is_gated:
            self.gate_attn = Parameter(Tensor(np.ones(1) * math.pi / 4, dtype=self.param_init_type))
            self.gate_ffn = Parameter(Tensor(np.ones(1) * math.pi / 4, dtype=self.param_init_type))

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)) and _is_sharding_propagation():
            self.input_layernorm.shard(((dp, 1, 1),))
            self.post_attention_layernorm.shard(((dp, 1, 1),))
            self.self_attn.shard(config.parallel_config)
            self.mul.shard(((dp, 1, 1), (1,)))
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.mlp.shard(config.parallel_config)

    def construct(self, hidden_state, attention_mask=None):
        """Forward of MllamaVisionEncoderLayer"""
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = self.mul(hidden_state, self.tanh(self.gate_attn))
        hidden_state = self.add(residual, hidden_state)

        # Feed forwardvi
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = self.mul(hidden_state, self.tanh(self.gate_ffn))
        outputs = self.add(residual, hidden_state)
        return outputs


class MllamaVisionEncoder(nn.Cell):
    """MllamaVisionEncoder"""
    def __init__(self, config, num_layers=32, is_gated=False):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)])
        self.config = config

    def construct(self, hidden_states, attention_mask=None, output_hidden_states=None):
        """Llama Vision Encoder"""
        encoder_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask
            )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return (hidden_states, encoder_states)


class MllamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MllamaConfig
    base_model_prefix = "mllama"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MllamaVisionModel(MllamaPreTrainedModel):
    """MllamaVisionModel"""
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_model"

    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = Tensor(config.intermediate_layers_indices)
        self.dtype = config.compute_dtype
        self.param_init_type = config.param_init_type
        self.tensor_min_type = -3.38953e+38
        if self.dtype != mstype.bfloat16:
            np_type = mstype.dtype_to_nptype(self.dtype)
            self.tensor_min_type = np.finfo(np_type).min

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size ** -0.5
        self.use_flash_attention = config.use_flash_attention

        self.gather = P.Gather()
        self.stack = P.Stack(axis=-1)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.tile1 = P.Tile()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cat = P.Concat(axis=1)
        self.cat_last = P.Concat(axis=-1)
        self.slice = P.StridedSlice()
        self.num_padding_patches = (8 - (self.num_patches % 8)) % 8
        self.pad = P.Pad(paddings=((0, 0), (0, 0), (0, self.num_padding_patches), (0, 0)))
        self.pad1 = P.Pad(paddings=((0, 0), (0, 0), (0, self.num_patches), (0, 0)))
        self.batch_matmul = P.BatchMatMul(transpose_b=True)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad_mode="valid",
            has_bias=False,
            dtype=self.param_init_type
        )

        init_data = self.scale * Tensor(np.random.randn(self.hidden_size), dtype=self.param_init_type)
        self.class_embedding = Parameter(initializer(init_data, [self.hidden_size], dtype=self.param_init_type))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = LayerNorm(self.hidden_size, eps=config.norm_eps,
                                       param_init_type=self.param_init_type)
        self.layernorm_post = LayerNorm(self.hidden_size, eps=config.norm_eps,
                                        param_init_type=self.param_init_type)

        # encoders
        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)
        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)) and _is_sharding_propagation():
            self.patch_embedding.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
            self.patch_embedding.bias_add.shard(((dp, 1, 1, 1), (1,)))
            self.transpose.shard(((dp, 1, 1),))
            self.pre_tile_positional_embedding.shard(config.parallel_config)
            self.tile.shard(((dp, 1, 1, 1),))
            self.tile1.shard(((1, 1, 1,),))
            self.cat.shard(((dp, 1, 1), (dp, 1, 1)))
            self.gated_positional_embedding.shard(config.parallel_config)
            self.layernorm_pre.shard(((dp, 1, 1, 1),))
            self.pad.shard(((dp, 1, 1, 1),))
            self.sub.shard(((), (dp, 1, 1, 1)))
            self.batch_matmul.shard(((dp, 1, 1), (dp, 1, 1)))
            self.layernorm_post.shard(((dp, 1, 1),))
            self.post_tile_positional_embedding.shard(config.parallel_config)
            self.slice.shard(((dp, 1, 1, 1),))
            strategy_stack = tuple((dp, 1, 1,) for i in range(config.num_hidden_layers + 1))
            self.stack.shard(strategy_stack)
            self.gather.shard(((dp, 1, 1, 1), (1,)))
            self.cat_last.shard(((dp, 1, 1, 1, 1), (dp, 1, 1, 1, 1)))

    def apply_class_embedding(self, hidden_state):
        batch_size, _, _ = hidden_state.shape
        class_embedding = self.cast(self.tile1(self.class_embedding, (batch_size, 1, 1)), self.dtype)
        hidden_state = self.cat([class_embedding, hidden_state])
        return hidden_state

    def prepare_aspect_ratio_attention_mask(self, aspect_ratio_mask, num_patches, target_length, dtype):
        """prepare aspect_ratio_attention_mask"""
        # Expand aspect ratio mask to target_length
        batch_size, max_num_tiles = aspect_ratio_mask.shape
        attention_mask = self.cast(self.reshape(aspect_ratio_mask, (batch_size, max_num_tiles, 1, 1)), dtype)
        attention_mask = self.tile(attention_mask, (1, 1, num_patches, 1))

        # Mask padding patches
        attention_mask = self.pad(attention_mask)

        # Invert the mask (0 -> 1, 1 -> 0)
        attention_mask = self.sub(1, attention_mask)

        # Reshape to 2D and create 4D attention mask
        # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
        attention_mask = self.reshape(attention_mask, (batch_size, max_num_tiles * target_length, 1))
        attention_mask = self.batch_matmul(attention_mask, attention_mask)
        if self.use_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
        else:
            attention_mask = self.mul(attention_mask, self.tensor_min_type)
        dim1, dim2, dim3 = attention_mask.shape
        attention_mask = self.reshape(attention_mask, (dim1, 1, dim2, dim3))

        return attention_mask

    def construct(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
        """Llama Vision Model"""
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = self.reshape(pixel_values,
                                    (batch_size * num_concurrent_media * num_tiles, num_channels, height, width))
        aspect_ratio_ids = self.reshape(aspect_ratio_ids, (batch_size * num_concurrent_media, -1))

        # Patch embedding
        pixel_values = self.cast(pixel_values, self.dtype)
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = self.reshape(patch_embeds, (patch_embeds.shape[0], patch_embeds.shape[1], -1))
        hidden_state = self.transpose(patch_embeds, (0, 2, 1))

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = self.reshape(hidden_state, (batch_size * num_concurrent_media, num_tiles, -1, dim))
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = self.reshape(hidden_state, (batch_size * num_concurrent_media * num_tiles, num_patches, dim))
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = self.reshape(hidden_state, (batch_size * num_concurrent_media, num_tiles, num_patches, dim))
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Pad the tensor
        hidden_state = self.pad(hidden_state)
        slice_index = -self.num_padding_patches if self.num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = self.reshape(aspect_ratio_mask, (batch_size * num_concurrent_media, -1))
        attention_mask = self.prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, -1, dim)

        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_state = output[0]
        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = self.reshape(hidden_state, (
            batch_size * num_concurrent_media, num_tiles, num_patches + self.num_padding_patches, dim
        ))
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = self.reshape(hidden_state, (
            batch_size * num_concurrent_media, num_tiles * (num_patches + self.num_padding_patches), dim
        ))
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        hidden_state = global_output[0]

        # Remove padding form hidden state
        hidden_state = self.reshape(hidden_state, (
            batch_size * num_concurrent_media, num_tiles, num_patches + self.num_padding_patches, dim
        ))
        hidden_state = self.slice(hidden_state, (0, 0, 0, 0),
                                  (batch_size * num_concurrent_media, num_tiles, slice_index, dim), (1, 1, 1, 1))
        hidden_state = self.reshape(hidden_state, (batch_size, num_concurrent_media, num_tiles, num_patches, dim))

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]

        intermediate_hidden_states = self.stack(all_intermediate_hidden_states)

        intermediate_hidden_states = self.gather(intermediate_hidden_states, self.intermediate_layers_indices, 3)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = self.reshape(intermediate_hidden_states, (
            batch_size * num_concurrent_media, num_tiles, num_patches + self.num_padding_patches, -1
        ))

        intermediate_hidden_states = self.slice(intermediate_hidden_states, (0, 0, 0, 0),
                                                (batch_size * num_concurrent_media, num_tiles, slice_index,
                                                 intermediate_hidden_states.shape[-1]), (1, 1, 1, 1))
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = self.cast(hidden_state, self.dtype)
        hidden_state = self.cat_last([hidden_state, intermediate_hidden_states])

        return hidden_state


class MllamaTextModel(LlamaModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    or [`MllamaCrossAttentionDecoderLayer`]
    Args:
        config(MllamaConfig): the config of network

    Returns:
            output: Tensor, the output of mllama decoderlayer

    Examples:
        >>> from mindformers import MllamaTextModel
        >>> network = MllamaTextModel.from_pretrained('llama_7b')
        >>> type(network)
        <class 'mindformers.models.mllama.mllama.MllamaTextModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['mllama']

    def __init__(self, config=None):
        super().__init__(config)
        _check_config(config.parallel_config)
        self.cross_attention_layers = config.cross_attention_layers
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size + 8,
                                             embedding_size=config.hidden_size,
                                             init_method_std=config.init_method_std,
                                             param_init_type=config.embedding_init_type,
                                             parallel_optimizer=config.parallel_optimizer,
                                             rmsnorm_compute_2d=config.rmsnorm_compute_2d)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            if layer_id in self.cross_attention_layers:
                layer = MllamaCrossAttentionDecoderLayer(config.seq_length,
                                                         layer_id,
                                                         dim=config.hidden_size,
                                                         n_heads=config.num_heads,
                                                         n_kv_heads=config.n_kv_heads,
                                                         intermediate_size=config.intermediate_size,
                                                         multiple_of=config.multiple_of,
                                                         ffn_dim_multiplier=config.ffn_dim_multiplier,
                                                         norm_eps=config.rms_norm_eps,
                                                         qkv_has_bias=config.qkv_has_bias,
                                                         attn_proj_has_bias=config.attn_proj_has_bias,
                                                         qkv_concat=config.qkv_concat,
                                                         compute_dtype=config.compute_dtype,
                                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                                         softmax_compute_dtype=config.softmax_compute_type,
                                                         rotary_dtype=config.rotary_dtype,
                                                         param_init_type=config.param_init_type,
                                                         residual_dtype=config.residual_dtype,
                                                         use_past=config.use_past,
                                                         is_dynamic=config.is_dynamic,
                                                         use_flash_attention=config.use_flash_attention,
                                                         use_ring_attention=config.use_ring_attention,
                                                         use_attn_mask_compression=config.use_attn_mask_compression,
                                                         block_size=config.block_size,
                                                         num_blocks=config.num_blocks,
                                                         use_rope_slice=config.use_rope_slice,
                                                         rmsnorm_compute_2d=config.rmsnorm_compute_2d,
                                                         batch_size=config.batch_size,
                                                         moe_config=config.moe_config,
                                                         parallel_config=config.parallel_config,
                                                         parallel_decoding=self.parallel_decoding,
                                                         fused_kernel=config.fused_rms_norm,
                                                         init_method_std=config.init_method_std,
                                                         chunk_prefill=config.chunk_prefill)
            else:
                layer = LLamaDecodeLayer(config.seq_length,
                                         layer_id,
                                         dim=config.hidden_size,
                                         n_heads=config.num_heads,
                                         n_kv_heads=config.n_kv_heads,
                                         intermediate_size=config.intermediate_size,
                                         multiple_of=config.multiple_of,
                                         ffn_dim_multiplier=config.ffn_dim_multiplier,
                                         norm_eps=config.rms_norm_eps,
                                         qkv_has_bias=config.qkv_has_bias,
                                         attn_proj_has_bias=config.attn_proj_has_bias,
                                         qkv_concat=config.qkv_concat,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         rotary_dtype=config.rotary_dtype,
                                         param_init_type=config.param_init_type,
                                         residual_dtype=config.residual_dtype,
                                         use_past=config.use_past,
                                         is_dynamic=config.is_dynamic,
                                         use_flash_attention=config.use_flash_attention,
                                         use_ring_attention=config.use_ring_attention,
                                         use_attn_mask_compression=config.use_attn_mask_compression,
                                         block_size=config.block_size,
                                         num_blocks=config.num_blocks,
                                         use_rope_slice=config.use_rope_slice,
                                         rmsnorm_compute_2d=config.rmsnorm_compute_2d,
                                         batch_size=config.batch_size,
                                         moe_config=config.moe_config,
                                         parallel_config=config.parallel_config,
                                         parallel_decoding=self.parallel_decoding,
                                         fused_kernel=config.fused_rms_norm,
                                         init_method_std=config.init_method_std,
                                         chunk_prefill=config.chunk_prefill)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.tok_embeddings.shard(config.parallel_config)
        else:
            self.tok_embeddings.shard(config.parallel_config)

    # pylint: disable=W0613,W0221
    def construct(self, tokens: Tensor, cross_attention_mask=None, cross_attention_states=None,
                  full_text_row_masked_out_mask=None, input_embeds=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  attention_mask=None, position_ids=None, q_seq_lens=None, seq_range=None):
        """
        Forward of mllama model.

        Args:
            tokens: the tokenized inputs with datatype int32
            cross_attention_mask(Tensor, optional): Cross-attention mask to control the interaction between text
                tokens and image tiles. This 4D tensor defines which image tiles each text token should attend to.
            cross_attention_states(Tensor, optional): Output of the vision model, used for cross-attention.
                This tensor contains the processed image features that the language model will attend to.
            input_embeds: the embedding Tensor of tokens, Tensor of shape:math:`(batch_size, seq/_length, hidden_size)`.
                Default None.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of mllama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        kv_mask = None
        seq_chunk = None
        rmsnorm_compute_2d = self.training and self.rmsnorm_compute_2d
        if self.chunk_prefill and self.is_first_iteration:
            # get chunk + decode masks
            if attention_mask is not None:
                mask = attention_mask
            else:
                mask = self.casual_mask.chunk_masks(seq_range)
            # get chunk + decode pos
            freqs_cis = self.freqs_mgr.chunk_with_decode(seq_range)
        elif self.parallel_decoding:
            # FA with TH layout, mask is 2D, FA with BSH layout, mask is 4D
            mask = attention_mask
            freqs_cis = self.freqs_mgr.increment_multi_ids(position_ids)
        elif attention_mask is not None:
            mask = attention_mask
            mask = self.cast(mask, mstype.uint8)
            freqs_cis = self.freqs_mgr(seq_len, position_ids)
        else:
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
                if self.seq_pipe:
                    mask = self.casual_mask(tokens, seq_chunk=self.seq_chunk)
                    seq_chunk = P.ReLU()(self.seq_chunk)
                    kv_mask = self.cast(self.equal_kv(self.kv_mask_add(self.zeros, self.kv_mask), seq_chunk),
                                        self.dtype)
                    seq_update = F.depend(self.seq_update, mask)
                    seq_update = F.depend(seq_update, kv_mask)
                    mask = F.depend(mask, self.assign_add_count(self.seq_chunk, seq_update))
                elif not self.use_ring_attention:
                    mask = self.casual_mask(tokens)
                freqs_cis = self.freqs_mgr(seq_len, seq_chunk=seq_chunk)
                if prefix_keys_values is not None:
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        if input_embeds is not None:
            h = self.cast(input_embeds, self.dtype)
        else:
            h = self.cast(self.tok_embeddings(tokens), self.dtype)
        if not rmsnorm_compute_2d:
            h = self.reshape(h, (bs, seq_len, self.hidden_size))  # h: [bs, seq/1, hidden_dim]

        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            if i in self.cross_attention_layers:
                h = self.layers[i](h, freqs_cis, mask, cross_attention_mask, cross_attention_states,
                                   full_text_row_masked_out_mask, batch_valid_length=batch_valid_length,
                                   block_tables=block_tables,
                                   slot_mapping=slot_mapping, prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                                   kv_mask=kv_mask, seq_chunk=seq_chunk)
            else:
                h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                                   slot_mapping=slot_mapping, prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                                   kv_mask=kv_mask, seq_chunk=seq_chunk)
        if rmsnorm_compute_2d:
            h = self.reshape(h, (bs * seq_len, -1))
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    r"""
    Provide mllama training logits through network.

    Args:
        config (MllamaConfig): The config of mllama model. Default: `None` .

    Inputs:
        - **input_ids** (Tensor) - the indices of input sequence tokens in the vocabulary with data type Int64/Int32,
          Tensor of shape :math:`(batch, seq\_length)`.
        - **labels** (Tensor, optional) - the labels of inputs with data type Int64/Int32, Tensor of
          shape :math:`(batch, seq\_length)` . Default: ``None`` .
        - **input_position** (Tensor, optional) - the position ids of inputs (at incremental reasoning mode) which is
          an increasing sequence with data type Int64/Int32, Tensor :math:`(batch, seq\_length)`.
          Default: ``None`` .
        - **position_ids** (Tensor, optional) - the position ids of inputs which is
          an increasing sequence with data type
          Int64/Int32, Tensor :math:`(batch, seq\_length)`. Default: ``None`` .
        - **attention_mask** (Tensor, optional) - input sentences padding mask, where 0 indicates padding position with
          data type Int64/Int32, Tensor of shape :math:`(batch, seq\_length)`. Default: ``None`` .
        - **input_embeds** (Tensor, optional) - the embedding of inputs with data type Float32/Float16, Tensor of
          shape :math:`(batch, seq\_length, hidden\_size)`. Default: ``None`` .
        - **init_reset** (Tensor, optional) - A Bool tensor with shape [1], used to clear the past key parameter and
          past value parameter used in the incremental prediction. Only valid when use_past is True.
          Tensor of shape :math:`(1)`. Default: ``Tensor([True])`` .
        - **batch_valid_length** (Tensor, optional) - Int32 tensor with shape [batch_size]
          the past calculated the index.
          Used for incremental prediction when the use_past is True. Default: ``None`` .
        - **block_tables** (Tensor, optional) - Int64 type Tensor, Store mapping tables for each sequence.
          Default: ``None`` .
        - **slot_mapping** (Tensor, optional) - Int32 type Tensor, token cache physical slot index. Default:``None`` .

    Outputs:
        Tensor. the output Tensor contains logits.
    """

    @lazy_inline
    def __init__(self, config=None):
        super(MllamaForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.gather = P.Gather(1)
        self.prefill_gather_flatten = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = MllamaTextModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.embedding_weight
        vocab_size = config.vocab_size
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)) and _is_sharding_propagation():
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.prefill_gather_flatten.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))

        if config.parallel_config.pipeline_stage > 1:
            self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.parallel_decoding = config.parallel_decoding_params is not None
        self.cross_attention_layers = config.cross_attention_layers

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for i in range(self.config.num_layers):
            layer = self.model.layers[i]
            if i not in self.cross_attention_layers:
                layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.add_flags(is_first_iteration=is_first_iteration)

    def pre_gather_func(self, pre_gather, output, batch_valid_length):
        """Pre gather operation in infer mode."""
        if not pre_gather:
            return output
        if self.parallel_decoding and self.is_first_iteration:
            output = output.reshape(-1, output.shape[-1])
            output = output[self.sub_batch_valid_len(batch_valid_length, 1)]
        elif pre_gather:
            if self.config.is_dynamic:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = self.prefill_gather_flatten(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
            else:
                output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        return output

    # pylint: disable=W0613
    def construct(self, tokens, cross_attention_mask=None, cross_attention_states=None,
                  full_text_row_masked_out_mask=None, input_embeds=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  attention_mask=None, position_ids=None, q_seq_lens=None, seq_range=None):
        r"""
        MllamaForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            cross_attention_mask(Tensor, optional): Cross-attention mask to control the interaction between text
                tokens and image tiles. This 4D tensor defines which image tiles each text token should attend to.
            cross_attention_states(Tensor, optional): Output of the vision model, used for cross-attention.
                This tensor contains the processed image features that the language model will attend to.
            labels(Tensor, optional): the tokenized labels with datatype int32,
                Tensor of shape :math:`(batch, seq\_length)`. Default: ``None`` .
            input_position(Tensor, optional): current position, used by model.predict. Default: ``None`` .
            position_ids(Tensor, optional): Reserved param, not used. Default: ``None`` .
            attention_mask(Tensor, optional): Reserved param, not used. Default: ``None`` .
            input_embeds(Tensor, optional): the input embedding Tensor of shape
                :math:`(batch, seq\_length, hidden_size)`. Default: ``None`` .
            batch_valid_length(Tensor, optional): the past calculated the index with datatype int32,
                used for incremental prediction. Tensor of shape :math:`(batch_size,)`.  Default: ``None`` .
            block_tables (Tensor[int64], optional): Store mapping tables for each sequence. Default: ``None`` .
            slot_mapping (Tensor[int32], optional): Store token cache physical slot index. Default: ``None`` .
            q_seq_lens (Tensor[int32], optional): In parallel decoding, the query may be flattened.
                The Paged Attention operator need `q_seq_lens` to obtain the length information. Default: ``None`` .

        Returns:
            Tensor, The logits of the network.
        """

        output = self.model(tokens, cross_attention_mask, cross_attention_states,
                            full_text_row_masked_out_mask, input_embeds, batch_valid_length, batch_index,
                            zactivate_len, block_tables, slot_mapping, prefix_keys_values,
                            attention_mask, position_ids, q_seq_lens, seq_range)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        output = self.pre_gather_func(pre_gather, output, batch_valid_length)
        logits = self.lm_head(output)
        return logits

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    r"""
    Provide mllama training loss or logits through network.

    Args:
        config (MllamaConfig): The config of mllama model. Default: `None` .

    Inputs:
        - **input_ids** (Tensor) - the indices of input sequence tokens in the vocabulary with data type Int64/Int32,
          Tensor of shape :math:`(batch, seq\_length)`.
        - **pixel_values** (Tensor, optional) - the tensors corresponding to the input images.
          Tensor of shape :math:`(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)`.
        - **aspect_ratio_mask** (Tensor, optional) - mask to avoid performing attention on padding tiles. Mask values
          selected in `[0, 1]`. Tensor of shape :math:`(batch_size, max_num_images, max_num_tiles)`. Default: ``None``.
        - **aspect_ratio_ids** (Tensor, optional) - Aspect ratio ids used to select the appropriate precomputed tile
          embeddings based on the aspect ratio of each input image. These ids correspond to indices in the model's list
          of supported aspect ratios, offset by 1. Tensor of shape :math:`(batch_size, max_num_images)`.
          Default: ``None``.
        - **cross_attention_mask** (Tensor, optional) - Cross-attention mask to control the interaction between text
          tokens and image tiles. This 4D tensor defines which image tiles each text token should attend to.
          Tensor of shape :math:`(batch_size, seq_length, max_num_images, max_num_tiles)`. Default: ``None``.
        - **cross_attention_states** (Tensor, optional) - Output of the vision model, used for cross-attention. This
          tensor contains the processed image features that the language model will attend to. Default: ``None``.
        - **labels** (Tensor, optional) - the labels of inputs with data type Int64/Int32, Tensor of
          shape :math:`(batch, seq\_length)` . Default: ``None``.
        - **input_position** (Tensor, optional) - the position ids of inputs (at incremental reasoning mode) which is
          an increasing sequence with data type Int64/Int32, Tensor :math:`(batch, seq\_length)`.
          Default: ``None``.
        - **position_ids** (Tensor, optional) - the position ids of inputs which is
          an increasing sequence with data type
          Int64/Int32, Tensor :math:`(batch, seq\_length)`. Default: ``None``.
        - **attention_mask** (Tensor, optional) - input sentences padding mask, where 0 indicates padding position with
          data type Int64/Int32, Tensor of shape :math:`(batch, seq\_length)`. Default: ``None``.
        - **input_embeds** (Tensor, optional) - the embedding of inputs with data type Float32/Float16, Tensor of
          shape :math:`(batch, seq\_length, hidden\_size)`. Default: ``None``.
        - **init_reset** (Tensor, optional) - A Bool tensor with shape [1], used to clear the past key parameter and
          past value parameter used in the incremental prediction. Only valid when use_past is True.
          Tensor of shape :math:`(1)`. Default: ``Tensor([True])``.
        - **batch_valid_length** (Tensor, optional) - Int32 tensor with shape [batch_size]
          the past calculated the index.
          Used for incremental prediction when the use_past is True. Default: ``None``.
        - **block_tables** (Tensor, optional) - Int64 type Tensor, store mapping tables for each sequence.
          Default: ``None``.
        - **slot_mapping** (Tensor, optional) - Int32 type Tensor, token cache physical slot index. Default: ``None``.
        - **loss_mask** (Tensor, optional) - Used to determine whether the corresponding token position participates
          in the loss calculation. If the value is :math:`(1)`, the loss of the position is calculated,
          and :math:`(0)` is not calculated. Default: ``None`` .

    Outputs:
        Tensor. If it is in training mode, the output Tensor contains loss;
        If it is in prediction mode, the output Tensor contains logits;
        If it is in evaluation mode, the output Tensor contains logits, tokens, and input masks.

    Examples:
        >>> from mindformers.models.mllama import MllamaConfig, MllamaForConditionalGeneration
        >>> import mindspore as ms
        >>> ms.set_context(mode=0)
        >>> config = MllamaConfig(batch_size=2)
        >>> network = MllamaForConditionalGeneration(config=config)
        >>> type(network)
        <class 'mindformers.models.mllama.mllama.MllamaForConditionalGeneration'>
        >>> from mindformers import MllamaForConditionalGeneration
        >>> network = MllamaForConditionalGeneration.from_pretrained('llama2_7b')
        >>> type(network)
        <class 'mindformers.models.mllama.mllama.MllamaForConditionalGeneration'>
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.dtype = config.compute_dtype

        self.tensor_min_type = -3.38953e+38
        if self.dtype != mstype.bfloat16:
            np_type = mstype.dtype_to_nptype(self.dtype)
            self.tensor_min_type = np.finfo(np_type).min
        self.use_flash_attention = config.use_flash_attention

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.slice1 = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.not_equal1 = P.NotEqual()
        self.mul = P.Mul()
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.masked_fill = P.MaskedFill()
        self.prefill_gather_flatten = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.reduce_any = P.ReduceAny(keep_dims=True)
        self.tile = P.Tile()
        self.assign = P.Assign()
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        mp = config.parallel_config.model_parallel
        self.vocab_size = config.text_model.model_config.vocab_size
        if self.vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           self.vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        loss_parallel_config.data_parallel *= loss_parallel_config.context_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", True)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)
        self.hidden_size = config.text_model.model_config.hidden_size
        self.image_size = config.vision_model.model_config.image_size
        self.num_channels = config.vision_model.model_config.num_channels
        self.max_num_tiles = config.vision_model.model_config.max_num_tiles
        self.max_num_images = config.vision_model.model_config.max_num_images
        self.vision_output_dim = config.vision_model.model_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = build_network(config.vision_model)
        self.language_model = build_network(config.text_model)
        self.multi_modal_projector = Linear(in_channels=self.vision_output_dim,
                                            out_channels=self.hidden_size,
                                            has_bias=True,
                                            compute_dtype=config.compute_dtype,
                                            param_init_type=config.param_init_type,
                                            weight_init="normal")

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.num_patches = self.vision_model.num_patches
        self.batch_size = config.batch_size // dp if config.batch_size // dp > 0 else 1
        self.cross_init_tensor = Tensor(
            np.ones((self.max_num_tiles * self.batch_size, self.num_patches, self.hidden_size)), self.dtype)
        self.cross_attention_states = Parameter(self.cross_init_tensor, name="cross_init_tensor")
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,)) and _is_sharding_propagation():
            self.multi_modal_projector.shard(((dp, 1), (1, 1)), ((dp, 1), (1,)))
            self.masked_fill.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.not_equal1.shard(((dp, 1, 1, 1), ()))
            self.not_equal.shard(((dp, 1,), ()))
            self.reduce_any.shard(((dp, 1, 1, 1),))
            self.mul.shard(((dp, 1,), (dp, 1,)))
            self.mul1.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.mul2.shard(((dp, 1, 1, 1), (1,)))
            self.slice.shard(((dp, 1),))
            self.slice1.shard(((dp, 1, 1, 1),))

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

        logger.info("Predict run mode:{}".format(self.predict_run_mode))
        self.parallel_decoding = config.text_model.model_config.parallel_decoding_params is not None
        self.input_sliced_sig = config.text_model.model_config.input_sliced_sig

    def freeze_component(self):
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_model.trainable_params():
                param.requires_grad = False

    def to_embeddings(self, tokens):
        """return embedding tokens"""
        return self.language_model.tok_embeddings(tokens)

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        pixel_values = Tensor(np.ones(shape=tuple([bs, self.max_num_images, self.max_num_tiles, self.num_channels,
                                                   self.image_size, self.image_size])), mstype.int32)
        aspect_ratio_mask = Tensor(np.ones(shape=tuple([bs, self.max_num_images, self.max_num_tiles])), mstype.int32)
        aspect_ratio_ids = Tensor(np.ones(shape=tuple([bs, self.max_num_images])), mstype.int32)
        cross_attention_mask = Tensor(np.ones(shape=tuple([bs, seq, self.max_num_images, self.max_num_tiles])),
                                      mstype.int32)
        return input_ids, labels, pixel_values, aspect_ratio_mask, aspect_ratio_ids, cross_attention_mask, \
               None, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values, None

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_pixel_values = Tensor(shape=[None, None, None, None, None, None], dtype=self.dtype)
        dynamic_aspect_ratio_mask = Tensor(shape=[None, None, None], dtype=mstype.int32)
        dynamic_aspect_ratio_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_cross_attention_mask = Tensor(shape=[None, None, None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        dynamic_position_ids = Tensor(shape=[None, None], dtype=mstype.int32) if self.parallel_decoding else None
        dynamic_mask = Tensor(shape=[None, None], dtype=mstype.float16) if self.parallel_decoding else None
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32) if self.parallel_decoding else None
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, None, dynamic_position_ids,
                            dynamic_mask,
                            None, None, dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None, dynamic_q_seq_lens, None, None,
                            None)
        elif self.use_past:
            self.set_inputs(dynamic_input_ids, None, dynamic_pixel_values, dynamic_aspect_ratio_mask,
                            dynamic_aspect_ratio_ids, dynamic_cross_attention_mask, None, None,
                            dynamic_position_ids, dynamic_mask, None, None, dynamic_batch_valid_length, None, None,
                            dynamic_block_tables, dynamic_slot_mapping, None, None, dynamic_q_seq_lens, None, None,
                            None)
        elif kwargs.get("pre_gather", False):
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, None, None, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            None, None, None, None, None, None)
        logger.info("Set dynamic input for Mllama.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.language_model.add_flags_custom(is_first_iteration=is_first_iteration)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        cross_attention_mask = kwargs.get("cross_attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        aspect_ratio_ids = kwargs.get("aspect_ratio_ids", None)
        aspect_ratio_mask = kwargs.get("aspect_ratio_mask", None)
        pixel_values = kwargs.get("pixel_values", None)

        model_inputs = {
            "input_ids": Tensor.from_numpy(input_ids.astype(np.int32)),
            "position_ids": position_ids,
            "aspect_ratio_ids": Tensor.from_numpy(aspect_ratio_ids.astype(np.int32)),
            "aspect_ratio_mask": Tensor.from_numpy(aspect_ratio_mask.astype(np.int32)),
            "pixel_values": Tensor(pixel_values, dtype=self.dtype),
            "cross_attention_mask": Tensor.from_numpy(cross_attention_mask.astype(np.int32)),
        }

        if not self.use_past:
            return model_inputs
        if kwargs.get("prefill"):
            batch_valid_length = kwargs.get("valid_length_each_example")
            slot_mapping = kwargs.get("slot_mapping")
            model_inputs = self._prepare_inputs_for_prefill_flatten(input_ids, batch_valid_length,
                                                                    slot_mapping, model_inputs)
        else:
            model_inputs["cross_attention_mask"] = model_inputs["cross_attention_mask"][:, -1:]

        return model_inputs

    def prepare_cross_attention_mask(self, cross_attention_mask, num_vision_tokens, dtype):
        """prepare cross_attention_mask"""
        # reshape so it can be used by attn module

        batch_size, text_total_length, max_num_images, max_num_tiles = cross_attention_mask.shape
        cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
        cross_attention_mask = self.reshape(cross_attention_mask, (
            batch_size, text_total_length, num_vision_tokens * max_num_images * max_num_tiles))
        dim1, dim2, dim3 = cross_attention_mask.shape
        cross_attention_mask = self.reshape(cross_attention_mask, (dim1, 1, dim2, dim3))

        # invert the mask
        cross_attention_mask = self.cast(self.sub(1.0, cross_attention_mask), dtype)
        full_text_row_masked_out_mask = self.reduce_any(self.not_equal1(cross_attention_mask, 1.0), -1)
        full_text_row_masked_out_mask = self.cast(full_text_row_masked_out_mask, cross_attention_mask.dtype)
        cross_attention_mask = self.mul1(cross_attention_mask, full_text_row_masked_out_mask)

        if self.use_flash_attention:
            cross_attention_mask = self.cast(cross_attention_mask, mstype.uint8)
        else:
            cross_attention_mask = self.mul2(cross_attention_mask, self.tensor_min_type)

        return cross_attention_mask, full_text_row_masked_out_mask

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, pixel_values=None, aspect_ratio_mask=None,
                  aspect_ratio_ids=None, cross_attention_mask=None, cross_attention_states=None,
                  input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  llm_boost_inputs=None, q_seq_lens=None, loss_mask=None, gather_index=None, seq_range=None):
        r"""
        MllamaForConditionalGeneration forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor, optional): the tokenized labels with datatype int32,
                Tensor of shape :math:`(batch, seq\_length)`. Default: ``None`` .
            pixel_values(Tensor, optional): the tensors corresponding to the input images.
                Tensor of shape :math:`(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)`.
            aspect_ratio_mask(Tensor, optional): mask to avoid performing attention on padding tiles. Mask values
                selected in `[0, 1]`. Tensor of shape :math:`(batch_size, max_num_images, max_num_tiles)`.
                Default: ``None``.
            aspect_ratio_ids(Tensor, optional): Aspect ratio ids used to select the appropriate precomputed tile
                embeddings based on the aspect ratio of each input image. These ids correspond to indices in the
                model's list of supported aspect ratios, offset by 1.
                Tensor of shape :math:`(batch_size, max_num_images)`. Default: ``None``.
            cross_attention_mask(Tensor, optional): Cross-attention mask to control the interaction between text
                tokens and image tiles. This 4D tensor defines which image tiles each text token should attend to.
                Tensor of shape :math:`(batch_size, seq_length, max_num_images, max_num_tiles)`. Default: ``None``.
            cross_attention_states(Tensor, optional): Output of the vision model, used for cross-attention. This
                tensor contains the processed image features that the language model will attend to. Default: ``None``.
            input_position(Tensor, optional): current position, used by model.predict. Default: ``None`` .
            position_ids(Tensor, optional): Reserved param, not used. Default: ``None`` .
            attention_mask(Tensor, optional): Reserved param, not used. Default: ``None`` .
            input_embeds(Tensor, optional): the input embedding Tensor of shape
                :math:`(batch, seq\_length, hidden_size)`. Default: ``None`` .
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction.  Default: ``Tensor([True])`` .
            batch_valid_length(Tensor, optional): the past calculated the index with datatype int32,
                used for incremental prediction. Tensor of shape :math:`(batch_size,)`.  Default: ``None`` .
            block_tables (Tensor[int64], optional): Store mapping tables for each sequence. Default: ``None`` .
            slot_mapping (Tensor[int32], optional): Store token cache physical slot index. Default: ``None`` .
            q_seq_lens (Tensor[int32], optional): In parallel decoding, the query may be flattened.
                The Paged Attention operator need `q_seq_lens` to obtain the length information. Default: ``None`` .
            loss_mask (Tensor, optional): Used to determine whether the corresponding token position participates
                in the loss calculation. If the value is :math:`(1)`, the loss of the position is calculated,
                and :math:`(0)` is not calculated. Default: ``None`` .

        Returns:
            Tensor, The loss or (logits, tokens, input_mask) of the network.
        """
        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if self.is_first_iteration:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            cross_attention_states = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask
            )
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )
            if not self.training:
                self.assign(self.cross_attention_states, cross_attention_states)
        else:
            cross_attention_states = self.cross_attention_states

        cross_attention_mask, full_text_row_masked_out_mask = self.prepare_cross_attention_mask(
            cross_attention_mask,
            num_vision_tokens=self.num_patches,
            dtype=self.dtype
        )

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))

        has_loss_mask = loss_mask is not None
        input_sliced_sig = self.input_sliced_sig
        if self.training and input_sliced_sig and labels is None:
            input_sliced_sig = False

        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if not input_sliced_sig and self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
            cross_attention_mask = self.slice1(cross_attention_mask,
                                               (0, 0, 0, 0),
                                               (bsz, 1, seqlen - 1, cross_attention_mask.shape[-1]),
                                               (1, 1, 1, 1))
            full_text_row_masked_out_mask = self.slice1(full_text_row_masked_out_mask,
                                                        (0, 0, 0, 0),
                                                        (bsz, 1, seqlen - 1, full_text_row_masked_out_mask.shape[-1]),
                                                        (1, 1, 1, 1))
            if has_loss_mask:
                loss_mask = self.slice(loss_mask, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        logits = self.language_model(tokens, cross_attention_mask, cross_attention_states,
                                     full_text_row_masked_out_mask, input_embeds, batch_valid_length,
                                     batch_index, zactivate_len, block_tables, slot_mapping, prefix_keys_values,
                                     attention_mask, position_ids, q_seq_lens, seq_range)

        input_mask = loss_mask if has_loss_mask \
            else self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if not input_sliced_sig and self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                if not has_loss_mask:
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
        loss = self.loss(logits, labels, input_mask)
        return loss
