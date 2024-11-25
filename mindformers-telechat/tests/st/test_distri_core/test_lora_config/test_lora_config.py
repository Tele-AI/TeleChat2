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
"""test lora config"""

import pytest

from mindformers.experimental.parallel_core.pynative.config import LoraConfig, ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.utils import valid_lora_config


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_init_lora_configs():
    """
    Feature: init lora config
    Description: Test to init lora config from target_cells
    Expectation: success
    """
    # right case 1:
    target_cells = [
        {'target_cells': [
            '.*.out_proj',
            '.*.mapping',
            '.*.projection',
            'backbone.layers.layers.0.attention.qkv_proj',
        ]},
        {'cell': 'backbone.layers.layers.0.attention.qkv_proj', 'rank': 4, 'alpha': 16}
    ]
    _ = LoraConfig(use_lora=True, target_cells=target_cells)

    # right case 2:
    target_cells = [
        {'target_cells': [
            '.*.qkv_proj',
        ]},
    ]
    _ = LoraConfig(use_lora=True, target_cells=target_cells)

    # wrong case 1:
    target_cells = []
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except ValueError as err:
        assert str(err) == "'target_cells' cannot not be empty."

    # wrong case 2:
    target_cells = [{}]
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except ValueError as err:
        assert str(err) == "for 'target_cells', the list of target_cells name must be set."

    # wrong case 3:
    target_cells = [
        {'target_cells': [
            '.*.qkv_proj',
        ]},
        {'target_cells': [
            '.*.mapping',
        ]}
    ]
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except ValueError as err:
        assert str(err) == "'target_cells' cannot not be defined more than once."

    # wrong case 4:
    target_cells = [
        {'target_cells': []},
    ]
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except ValueError as err:
        assert str(err) == "for 'target_cells', the list of target_cells name must be set."

        # wrong case 5:
    target_cells = [
        {'target_cells': [
            '.*.qkv_proj',
        ]},
        {'cell': 'not_a_valid_model_name', 'rank': 4, 'alpha': 16},
    ]
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except ValueError as err:
        assert str(
            err) == ("The cell need to set rank or alpha should be in the range defined by target_cells, but got name "
                     "'not_a_valid_model_name'.")

    # wrong case 6:
    target_cells = {}
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except TypeError as err:
        assert str(err) == "The type of 'target_cells' should be 'list', but got type 'dict'."

    # wrong case 7:
    target_cells = [
        {'target_cells': {}},
    ]
    try:
        _ = LoraConfig(use_lora=True, target_cells=target_cells)
    except TypeError as err:
        assert str(err) == "The type of 'target_cells' should be 'list', but got type 'dict'."


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_update_lora_configs():
    """
    Feature: update lora config from ckpt
    Description: Test to update lora config from ckpt
    Expectation: success
    """
    pretrain_params = {
        'backbone.embedding.word_embedding.weight': 1,
        'backbone.embedding.position_embedding.weight': 1,
        'backbone.query_embedding.weight': 1,
        'backbone.layers.layers.0.input_norm.gamma': 1,
        'backbone.layers.layers.0.input_norm.beta': 1,
        'backbone.layers.layers.0.attention.qkv_proj.weight': 1,
        'backbone.layers.layers.0.attention.qkv_proj.bias': 1,
        'backbone.layers.layers.0.attention.out_proj.weight': 1,
        'backbone.layers.layers.0.attention.out_proj.bias': 1,
        'backbone.layers.layers.0.post_attention_norm.gamma': 1,
        'backbone.layers.layers.0.post_attention_norm.beta': 1,
        'backbone.layers.layers.0.mlp.mapping.weight': 1,
        'backbone.layers.layers.0.mlp.mapping.bias': 1,
        'backbone.layers.layers.0.mlp.projection.weight': 1,
        'backbone.layers.layers.0.mlp.projection.bias': 1,
        'backbone.layers.final_norm.gamma': 1,
        'backbone.layers.final_norm.beta': 1,
        'backbone.query_layer.input_norm.gamma': 1,
        'backbone.query_layer.input_norm.beta': 1,
        'backbone.query_layer.attention.q_proj.weight': 1,
        'backbone.query_layer.attention.q_proj.bias': 1,
        'backbone.query_layer.attention.kv_proj.weight': 1,
        'backbone.query_layer.attention.kv_proj.bias': 1,
        'backbone.query_layer.attention.out_proj.weight': 1,
        'backbone.query_layer.attention.out_proj.bias': 1,
        'backbone.query_layer.post_attention_norm.gamma': 1,
        'backbone.query_layer.post_attention_norm.beta': 1,
        'backbone.query_layer.mlp.mapping.weight': 1,
        'backbone.query_layer.mlp.mapping.bias': 1,
        'backbone.query_layer.mlp.projection.weight': 1,
        'backbone.query_layer.mlp.projection.bias': 1,
    }

    # right case 1
    target_cells = [
        {'target_cells': [
            '.*.out_proj',
            '.*.mapping',
            '.*.projection',
            'backbone.layers.layers.0.attention.qkv_proj',
        ]},
        {'cell': 'backbone.layers.layers.0.attention.qkv_proj', 'rank': 4, 'alpha': 16}
    ]
    lora_config = LoraConfig(use_lora=True, target_cells=target_cells)
    parallel_config = ModelParallelConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    model_config = TransformerConfig(vocab_size=50304,
                                     num_layers=2,
                                     num_attention_heads=32,
                                     hidden_size=16,
                                     ffn_hidden_size=4 * 16,
                                     seq_length=1024,
                                     lora_config=lora_config,
                                     parallel_config=parallel_config,
                                     )
    _ = valid_lora_config(model_config, pretrain_params)

    # wrong case 1:
    target_cells = [
        {'target_cells': [
            '.*.position_embedding',
        ]},
        {'cell': '.*.position_embedding', 'rank': 4, 'alpha': 16}
    ]
    lora_config = LoraConfig(use_lora=True, target_cells=target_cells)
    parallel_config = ModelParallelConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    model_config = TransformerConfig(vocab_size=50304,
                                     num_layers=2,
                                     num_attention_heads=32,
                                     hidden_size=16,
                                     ffn_hidden_size=4 * 16,
                                     seq_length=1024,
                                     lora_config=lora_config,
                                     parallel_config=parallel_config,
                                     )
    try:
        _ = valid_lora_config(model_config, pretrain_params)
    except ValueError as err:
        assert str(err) == "target_cells in your lora config is invalid, please check your target_cells."

    # wrong case 2:
    target_cells = [
        {'target_cells': [
            '.*.position_embedding',
            'invalid_cell_name'
        ]},
        {'cell': 'invalid_cell_name', 'rank': 4, 'alpha': 16}
    ]
    lora_config = LoraConfig(use_lora=True, target_cells=target_cells)
    parallel_config = ModelParallelConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    model_config = TransformerConfig(vocab_size=50304,
                                     num_layers=2,
                                     num_attention_heads=32,
                                     hidden_size=16,
                                     ffn_hidden_size=4 * 16,
                                     seq_length=1024,
                                     lora_config=lora_config,
                                     parallel_config=parallel_config,
                                     )
    try:
        _ = valid_lora_config(model_config, pretrain_params)
    except ValueError as err:
        assert str(err) == "target_cells in your lora config is invalid, please check your target_cells."
