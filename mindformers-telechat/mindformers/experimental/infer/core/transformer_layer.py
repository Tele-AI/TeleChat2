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
"""Transformer Layer"""
from typing import Union

from mindspore import nn
from mindspore import ops as P

from mindformers.experimental.infer.core.norm import get_norm
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec, build_module
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig


class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
    """

    def __init__(self,
                 input_layernorm: Union[ModuleSpec, type, object] = nn.Identity(),
                 self_attention: Union[ModuleSpec, type] = nn.Identity(),
                 pre_cross_attn_layernorm: Union[ModuleSpec, type] = nn.Identity(),
                 cross_attention: Union[ModuleSpec, type] = nn.Identity(),
                 pre_mlp_layernorm: Union[ModuleSpec, type, object] = nn.Identity(),
                 mlp: Union[ModuleSpec, type] = nn.Identity(),
                 ):
        self.input_layernorm = input_layernorm
        self.self_attention = self_attention
        self.pre_cross_attn_layernorm = pre_cross_attn_layernorm
        self.cross_attention = cross_attention
        self.pre_mlp_layernorm = pre_mlp_layernorm
        self.mlp = mlp


class BaseTransformerLayer:
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(nn.Cell, BaseTransformerLayer):
    r"""
    A single transformer layer.

    Args:
        config (TransformerConfig): Configuration for transformer architecture.
        submodules (TransformerLayerSubmodules): Submodule specifications.
        layer_number (int): Number of the transformer layer.
        hidden_dropout (float): Default: None.

    Inputs:
        - hidden_states (Tensor): Input tensor of shape [B, S, H].
        - attention_mask (Tensor): Tensor of attention mask.
        - context (Tensor): Default: None.
        - context_mask (Tensor): Default: None.
        - rotary_pos_cos (Tensor): The cosine component of rotary position embedding.
        - rotary_pos_sin (Tensor): The sine component of rotary position embedding.
        - batch_valid_length (Tensor): Tensor of shape [batch_size] for valid seq length.
        - context_lens_tensor (Tensor): Tensor of context lengths.
        - block_tables (Tensor): Block tables for memory optimization.
        - slot_mapping (Tensor): Slot mapping for memory optimization.
        - prefix_keys_values (List[Tensor], optional): Prefix key-value pairs.
        - attention_bias (Tensor): Default: None.
        - packed_seq_params (Tensor): Default: None.
        - kv_cache (List[Tensor], optional): Key-value cache for incremental inference.

    Outputs:
        - output (Tensor) - Output tensor of transformer layer

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: TransformerLayerSubmodules,
                 layer_number: int = 1,
                 hidden_dropout: float = None,
                 ):
        super().__init__()
        if hidden_dropout:
            raise NotImplementedError(
                "For TransformerLayer in infer mode, `hidden_dropout` is not supported, "
                "but got `hidden_dropout={}`".format(hidden_dropout)
            )

        self.config = config
        self.layer_number = layer_number
        self.apply_residual_connection_post_norm = config.apply_residual_connection_post_layernorm
        self.add = P.Add()

        self.input_layernorm = get_norm(config)

        attention_optional_kwargs = {}
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        self.pre_mlp_layernorm = get_norm(config)
        self.mlp = build_module(submodules.mlp, config=self.config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            block_tables=None,
            slot_mapping=None,
            prefix_keys_values=None,
            attention_bias=None,
            packed_seq_params=None,
            kv_cache=None
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_cos (Tensor, optional): The cos of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): The sin of Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        has_context_or_mask = context is not None or context_mask is not None
        has_attention_or_packed_params = attention_bias is not None or packed_seq_params is not None
        if has_context_or_mask or has_attention_or_packed_params:
            raise ValueError(
                "context or context_mask or packed_seq_params"
                " or attention_bias is not None. Not support yet!!")
        # Layer norm at the beginning
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self-Attention
        attention_output = self.self_attention(
            x=input_layernorm_output,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attn_mask=attention_mask,
            actual_seq_qlen=batch_valid_length,
            actual_seq_kvlen=batch_valid_length,
            prefix_keys_values=prefix_keys_values,
            kv_cache=kv_cache
        )

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = input_layernorm_output
        else:
            residual = hidden_states

        norm_input = self.add(residual, attention_output)
        # # Layer norm post the self attention
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(norm_input)
        # # MLP
        mlp_output = self.mlp(pre_mlp_layernorm_output)
        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = pre_mlp_layernorm_output
        else:
            residual = norm_input

        output = self.add(residual, mlp_output)

        return output
