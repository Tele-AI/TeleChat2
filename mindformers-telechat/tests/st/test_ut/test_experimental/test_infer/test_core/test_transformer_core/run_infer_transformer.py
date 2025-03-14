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
"""run transformer block in infer mode"""
import argparse

import numpy as np

import mindspore as ms
from mindspore import Tensor

from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.models.llama import LlamaConfig
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

from tests.st.test_ut.test_experimental.test_infer.test_core.test_transformer_core.utils import (
    MyTransformerBlockNet,
    MyTransformerLayerNet,
)


def set_config():
    """Set config for mlp ut test."""
    config_ = LlamaConfig()
    config_.batch_size = 1
    config_.seq_length = 32
    config_.num_heads = 2
    config_.hidden_size = 64
    config_.intermediate_size = 128
    config_.num_layers = 2
    config_.hidden_act = 'silu'
    config_.num_blocks = 16
    config_.block_size = 64
    config_.mlp_has_bias = False
    config_.mlp_has_gate = True
    config_.ffn_concat = False
    config_.out_proj_has_bias = False
    config_.attn_proj_has_bias = False
    config_.use_flash_attention = True
    config_.apply_residual_connection_post_layernorm = False
    config_.compute_dtype = "float16"
    config_.param_init_type = "float16"
    config_.layernorm_compute_type = "float32"
    config_.normalization = 'RMSNorm'
    config_.pad_token_id = 151643
    transformer_config = TransformerConfig()
    return convert_to_transformer_config(config_, transformer_config)


def generate_inputs(config):
    """Generate input tensors for transformer block or layer inference test."""
    bs = config.batch_size
    seq_len = config.seq_length
    hidden = config.hidden_size
    num_blocks = config.num_blocks

    input_shape = (1, bs * seq_len, hidden)
    hidden_states = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    positions = Tensor(np.arange(0, seq_len).astype(np.int32))
    batch_valid_length = Tensor(np.ones((bs,)).astype(np.int32))
    context_lens_tensor = Tensor(np.zeros((bs,)).astype(np.int32))
    block_tables = Tensor(np.ones((bs, num_blocks)).astype(np.int64))
    slot_mapping = Tensor(np.ones((bs * seq_len,)).astype(np.int32))
    return hidden_states, positions, batch_valid_length, context_lens_tensor, block_tables, slot_mapping

def _test_transformer_block(config: TransformerConfig):
    """Test the Transformer Block module in inference mode."""
    (
        hidden_states, positions,
        batch_valid_length, context_lens_tensor,
        block_tables, slot_mapping
    ) = generate_inputs(config)
    my_net = MyTransformerBlockNet(config)

    block_output = my_net(
        hidden_states=hidden_states,
        positions=positions,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
    )

    assert block_output.shape == (config.batch_size, config.seq_length, config.hidden_size)


def _test_transformer_layer(config: TransformerConfig):
    """Test the Transformer Layer module in inference mode."""
    (
        hidden_states, positions,
        batch_valid_length, context_lens_tensor,
        block_tables, slot_mapping
    ) = generate_inputs(config)
    my_net = MyTransformerLayerNet(config)

    layer_output = my_net(
        hidden_states=hidden_states,
        positions=positions,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
    )

    assert layer_output.shape == (config.batch_size, config.seq_length, config.hidden_size)


def _test_transformer_core_module(module):
    """Set up environment and execute specified transformer module test."""
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, full_batch=False)

    seed_value = 42
    ms.set_seed(seed_value)
    np.random.seed(seed_value)

    config = set_config()
    TEST_FUNC[module](config)


TEST_FUNC = {
    'transformerlayer': _test_transformer_layer,
    'transformerblock': _test_transformer_block,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help='test module of parallel transformer')

    args = parser.parse_args()
    _test_transformer_core_module(args.module)
