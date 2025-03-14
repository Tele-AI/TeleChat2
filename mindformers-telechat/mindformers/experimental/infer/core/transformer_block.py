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
"""Transformer Block"""
from typing import Union
from mindspore import nn, Tensor
from mindformers.experimental.infer.core.norm import get_norm
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec, build_module


class TransformerBlockSubmodules:
    """
    Class for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    def __init__(self, layer_specs=None, layer_norm=None):
        self.layer_specs = layer_specs if layer_specs is not None else []
        self.layer_norm = layer_norm


def _get_block_submodules(
        config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec]
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """
    from mindformers.experimental.infer.core.transformer_layer import BaseTransformerLayer
    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        if issubclass(spec.module, BaseTransformerLayer):
            num_layers = config.num_layers
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=get_norm(config)
            )
        raise ValueError(f"Unsupported module: {spec.module.__name__}.")

    raise TypeError(f"Invalid spec type: {type(spec).__name__}.")


class TransformerBlock(nn.Cell):
    r"""
    Transformer Block composed of multiple transformer layers.

    Args:
        config (TransformerConfig): Configuration for transformer architecture.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Submodule specifications.
        post_layer_norm (bool): Whether to apply layer norm at the end of transformer block. Default: True.
        pre_process (bool): Default: False.
        post_process (bool): Default: False.

    Inputs:
        - hidden_states (Tensor): Input tensor of shape [B, S, H].
        - attention_mask (Tensor): Tensor of attention mask.
        - rotary_pos_cos (Tensor): The cosine component of rotary position embedding.
        - rotary_pos_sin (Tensor): The sine component of rotary position embedding.
        - batch_valid_length (Tensor): Tensor of shape [batch_size] for valid seq length.
        - context_lens_tensor (Tensor): Tensor of context lengths.
        - block_tables (Tensor): Block tables for memory optimization.
        - slot_mapping (Tensor): Slot mapping for memory optimization.
        - prefix_keys_values (List[Tensor], optional): Prefix key-value pairs.
        - kv_cache (List[Tensor], optional): Key-value cache for incremental inference.

    Outputs:
        - hidden_states (Tensor) - Output tensor of transformer block

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process=False,
            post_process=False,
    ):
        super(TransformerBlock, self).__init__()

        if pre_process:
            raise NotImplementedError("For TransformerBlock, `pre_process=True` is not supported.")
        if post_process:
            raise NotImplementedError("For TransformerBlock, `post_process=True` is not supported.")

        self.config = config
        self.submodules = _get_block_submodules(config, spec)
        self.post_norm = post_layer_norm
        self.num_layers = config.num_layers

        self._build_layers()

    def _build_layers(self):
        '''The method to construct transformer layers'''
        # Transformer layers.
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number)

        self.layers = nn.CellList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        if self.post_norm:
            self.final_norm = self.submodules.layer_norm
        else:
            self.final_norm = None

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(self, hidden_states: Tensor, attention_mask: Tensor, rotary_pos_cos: Tensor = None,
                  rotary_pos_sin: Tensor = None, batch_valid_length=None, context_lens_tensor=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, kv_cache=None):
        """ Construct function of transformer. """
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            prefix_kv = prefix_keys_values[index] if prefix_keys_values is not None else None
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_cos=rotary_pos_cos,
                                  rotary_pos_sin=rotary_pos_sin, batch_valid_length=batch_valid_length,
                                  context_lens_tensor=context_lens_tensor, kv_cache=kv_cache,
                                  block_tables=block_tables, slot_mapping=slot_mapping, prefix_keys_values=prefix_kv)

        # final layernorm.
        if self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states
