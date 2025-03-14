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
"""test infer transformer core utils"""
import mindspore.nn as nn
from mindspore import Tensor

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.norm import get_norm
from mindformers.experimental.infer.core.rotary_embedding import RotaryEmbedding
from mindformers.experimental.infer.core.self_attention import SelfAttention, CoreAttention, SelfAttentionSubmodules
from mindformers.experimental.infer.core.flash_attention import FlashAttention
from mindformers.experimental.infer.core.gpt_model import LowerTriangularMaskWithDynamic
from mindformers.experimental.infer.core.transformer_block import TransformerBlock
from mindformers.experimental.infer.core.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindformers.experimental.infer.core.mlp import MLP, MLPSubmodules
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec, build_module


def get_transformer_layer_spec(config) -> ModuleSpec:
    """Generate module specification for a Transformer Layer based on transformer configuration."""
    self_attn = ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            core_attention=FlashAttention if config.use_flash_attention else CoreAttention,
            linear_proj=RowParallelLinear,
            linear_qkv=ColumnParallelLinear if config.qkv_concat else None,
            linear_q=ColumnParallelLinear if not config.qkv_concat else None,
            linear_k=ColumnParallelLinear if not config.qkv_concat else None,
            linear_v=ColumnParallelLinear if not config.qkv_concat else None
        )
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm(config),
            self_attention=self_attn,
            pre_mlp_layernorm=get_norm(config),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear
                )
            )
        )
    )


class MyTransformerLayerNet(nn.Cell):
    """A model class of new transform layer."""
    def __init__(self, config: TransformerConfig):
        super(MyTransformerLayerNet, self).__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            pad_token_id=config.pad_token_id,
        )
        self.rotary_embedding = RotaryEmbedding(kv_channels=config.hidden_size // config.num_attention_heads,
                                                rotary_cos_format=2,
                                                rotary_dtype=config.rotary_dtype)
        self.layer = build_module(get_transformer_layer_spec(config),
                                  config=config)

    def construct(self,
                  hidden_states: Tensor,
                  positions: Tensor,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  block_tables=None,
                  slot_mapping=None
                  ):
        """Construct for Transformer Layer network.

        Args:
            hidden_states (Tensor): Input tensor of shape (1, batch_size * seq_length, hidden_size)
            positions (Tensor): Position indices tensor of shape (seq_length,)
            batch_valid_length (Tensor): Valid sequence lengths tensor for padding mask
            context_lens_tensor (Tensor): Context length information tensor
            block_tables (Tensor): Block allocation tables for cache management
            slot_mapping (Tensor): Cache slot mapping information

        Returns:
            Tensor: Output hidden states after transformer layer processing
        """
        attention_mask = self.casual_mask(positions)
        rotary_pos_cos, rotary_pos_sin = \
            self.rotary_embedding.get_cos_sin_for_prefill(self.max_position_embeddings)
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )


class MyTransformerBlockNet(nn.Cell):
    """A model class of new transform block."""
    def __init__(self, config: TransformerConfig):
        super(MyTransformerBlockNet, self).__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            pad_token_id=config.pad_token_id,
        )
        self.rotary_embedding = RotaryEmbedding(kv_channels=config.hidden_size // config.num_attention_heads,
                                                rotary_cos_format=2,
                                                rotary_dtype=config.rotary_dtype)
        self.decode = TransformerBlock(config, spec=get_transformer_layer_spec(config))

    def construct(self,
                  hidden_states: Tensor,
                  positions: Tensor,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  block_tables=None,
                  slot_mapping=None
                  ):
        """Construct for Transformer Block network.

        Args:
            hidden_states (Tensor): Input tensor of shape (1, batch_size * seq_length, hidden_size)
            positions (Tensor): Position indices tensor of shape (seq_length,)
            batch_valid_length (Tensor): Valid sequence lengths tensor for dynamic masking
            context_lens_tensor (Tensor): Context length information tensor
            block_tables (Tensor): Block allocation tables for cache management
            slot_mapping (Tensor): Cache slot mapping information

        Returns:
            Tensor: Output hidden states after transformer block processing.
        """
        attention_mask = self.casual_mask(positions)
        rotary_pos_cos, rotary_pos_sin = \
            self.rotary_embedding.get_cos_sin_for_prefill(self.max_position_embeddings)
        return self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
