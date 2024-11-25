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
"""base module."""

from mindspore import nn
from mindspore.common import dtype as mstype

from mindformers import LlamaConfig, MindFormerConfig
from mindformers.experimental.infer.core.transformer import (ParallelAttention, ParallelMLP, ParallelTransformer,
                                                             ParallelTransformerLayer)
from mindformers.experimental.infer.models.llama.utils import convert_model_config


class AttentionNet(nn.Cell):
    """testcase of ParallelAttention"""

    def __init__(self, config):
        super().__init__()
        self.attention = ParallelAttention(config=config, layer_number=0)

    def construct(self, x, batch_valid_length, block_tables, slot_mapping, freqs_cis=None, attn_mask=None):
        output = self.attention(x, batch_valid_length, block_tables, slot_mapping, freqs_cis=freqs_cis,
                                attn_mask=attn_mask, alibi_mask=None, prefix_keys_values=None, encoder_output=None,)
        return output


class MLPNet(nn.Cell):
    """testcase of ParallelMLP"""

    def __init__(self, config):
        super().__init__()
        self.mlp = ParallelMLP(config=config)

    def construct(self, x):
        output = self.mlp(x)
        return output


class TransformerLayerNet(nn.Cell):
    """testcase of ParallelTransformerLayer"""

    def __init__(self, config):
        super().__init__()
        self.layer = ParallelTransformerLayer(config=config,
                                              layer_number=1)

    def construct(self, x, freqs_cis=None, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None):
        output = self.layer(x, freqs_cis=freqs_cis, mask=mask, batch_valid_length=batch_valid_length,
                            block_tables=block_tables, slot_mapping=slot_mapping,
                            prefix_keys_values=prefix_keys_values)
        return output


class TransformerNet(nn.Cell):
    """testcase of Transformer"""

    def __init__(self, config):
        super().__init__()
        self.model = ParallelTransformer(config=config)

    def construct(self, tokens, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        output = self.model(tokens, batch_valid_length=batch_valid_length, batch_index=batch_index,
                            zactivate_len=zactivate_len, block_tables=block_tables,
                            slot_mapping=slot_mapping, prefix_keys_values=prefix_keys_values)
        return output


MODULES = {
    "attention": AttentionNet,
    "mlp": MLPNet,
    "transformerlayer": TransformerLayerNet,
    "transformer": TransformerNet,
}


def get_config():
    """get config of testcase"""
    base_config = LlamaConfig(
        param_init_dtype=mstype.float16,
        compute_dtype=mstype.float16,
        use_past=True,
        qkv_concat=True,
        num_heads=16,
        hidden_size=1024,
        use_flash_attention=True,
        qkv_has_bias=False,
        rotary_dtype=mstype.float16,
        num_blocks=16,
        block_size=256,
        out_proj_has_bias=False,
        vocab_size=1000,
        num_layers=2,
        seq_length=512,
        mlp_has_gate=True,
        ffn_concat=True,
        intermediate_size=4096,
        batch_size=2,
    )
    parallel_config = MindFormerConfig(
        tensor_parallel=2,
        context_parallel=1,
        vocab_emb_dp=False
    )
    base_config.parallel_config = parallel_config

    base_config = convert_model_config(base_config)

    return base_config


def get_module(module):
    """get module of testcase"""
    base_config = get_config()

    return MODULES[module](base_config)
