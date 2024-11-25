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
"""Run test module for testing router for graph or pynative"""
import argparse
from types import SimpleNamespace

import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.numpy import array_equal
from mindspore.common.parameter import Parameter
from mindspore.communication import init

from mindformers.models.llama import LlamaConfig
from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.tensor_parallel.layers import (RowParallelLinear as Graph_RP_Linear,
                                                                   ColumnParallelLinear as Graph_CP_Linear,
                                                                   VocabParallelEmbedding as Graph_VP_Embedding)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.\
    layers import (RowParallelLinear as Pynative_RP_Linear,
                   ColumnParallelLinear as Pynative_CP_Linear,
                   VocabParallelEmbedding as Pynative_VP_Embedding)
from mindformers.experimental.graph.transformer. \
    rotary_pos_embedding import (RotaryEmbedding as Graph_RotaryEmbedding,
                                 apply_rotary_pos_emb as graph_apply_rotary_pos_emb)
from mindformers.experimental.parallel_core.pynative.transformer.\
    rotary_pos_embedding import (RotaryEmbedding as Pynative_RotaryEmbedding,
                                 apply_rotary_pos_emb as pynative_apply_rotary_pos_emb)
from mindformers.experimental.graph.transformer.\
    transformer import (ParallelTransformer as Graph_ParallelTransformer,
                        ParallelTransformerLayer as Graph_ParallelTransformerLayer,
                        ParallelAttention as Graph_ParallelAttention,
                        ParallelMLP as Graph_ParallelMLP)
from mindformers.experimental.parallel_core.pynative.transformer.\
    transformer import (ParallelTransformer as Pynative_ParallelTransformer,
                        ParallelTransformerLayer as Pynative_ParallelTransformerLayer,
                        ParallelAttention as Pynative_ParallelAttention,
                        ParallelMLP as Pynative_ParallelMLP)
from mindformers.experimental.graph.transformer.language_model import (
    TransformerLanguageModel as Graph_TransformerLanguageModel)
from mindformers.experimental.parallel_core.pynative.transformer.language_model import (
    TransformerLanguageModel as Pynative_TransformerLanguageModel)
from mindformers.experimental.graph.transformer.language_model import Embedding as Graph_Embedding
from mindformers.experimental.parallel_core.pynative.transformer.language_model import Embedding as Pynative_Embedding
from mindformers.experimental.graph.optimizer.adamw import AdamW as Graph_AdamW
from mindformers.experimental.parallel_core.pynative.optimizer.zero.adamw_zero import AdamW as Pynative_AdamW
from mindformers.experimental.parallel_core import (ParallelTransformer,
                                                    ParallelTransformerLayer,
                                                    ParallelAttention,
                                                    ParallelMLP,
                                                    ColumnParallelLinear,
                                                    RowParallelLinear,
                                                    RotaryEmbedding,
                                                    apply_rotary_pos_emb,
                                                    VocabParallelEmbedding,
                                                    AdamW,
                                                    Embedding,
                                                    get_language_model)
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.utils import init_method_normal
from mindformers.core.context import build_context

from tests.utils.model_tester import ModelTester

parser = argparse.ArgumentParser()
parser.add_argument(
    '--graph',
    action='store_true',
    help='has_bias'
)
parser.add_argument(
    '--pynative',
    action='store_true',
    help='has_bias'
)
args_, _ = parser.parse_known_args()

BASE_CONFIG = {
    'num_heads': 8,
    'num_layers': 1,
    'hidden_size': 64,
    'multiple_of': 1,
    'seq_length': 8,
    'compute_dtype': 'float16',
    'layernorm_compute_dtype': 'float32',
    'softmax_compute_type': 'float16',
    'vocab_size': 32000,
    'use_flash_attention': True,
    'pad_token_id': 0,
    'ignore_token_id': -100,
    'type': 'LlamaConfig',
    'rms_norm_eps': 1e-05,
    'normalization': 'FusedRMSNorm',
    'mlp_has_gate': True,
    'hidden_act': 'silu',
    'params_dtype': 'float16',
    'hidden_dropout': 0.0
}


def get_config():
    """get instanced model config."""
    transformer_config = TransformerConfig()
    convert_to_transformer_config(LlamaConfig(**BASE_CONFIG), transformer_config)
    return transformer_config


def set_route_mode(mode):
    """set route mode."""
    runner = ModelTester(run_mode='train', batch_size=1)
    runner.args.mode = mode
    build_context(runner.args)


def add_attr_for_pynative(config):
    """add attributes for pynative."""
    config.parallel_config.standalone_embedding_stage = True
    config.lora_config = SimpleNamespace(use_lora=False)
    config.apply_residual_connection_post_norm = False
    config.residual_connection_dtype = None
    config.param_init_dtype = config.params_dtype
    config.use_gqa = False
    config.num_heads = config.num_attention_heads
    config.parallel_config.use_sequence_parallel = False
    config.use_flash_attention = False
    config.qkv_has_bias = False
    config.bias_init = 'zeros'
    config.parallel_config.zero_level = None
    config.init_method = 'normal'
    config.parallel_config.gradient_accumulation_fusion = False
    config.attention_dropout_rate = config.attention_dropout
    config.hidden_dropout_rate = config.attention_dropout
    config.out_proj_has_bias = False
    config.mlp_has_bias = config.add_bias_linear
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None
    config.moe_config.num_experts = 1
    config.out_hidden_size = 128
    config.use_retriever = False
    config.use_final_norm = False
    config.vocab_size = 32000
    config.embedding_init_dtype = config.embedding_init_type
    config.parallel_position_embedding = False
    config.rotary_interleaved = False
    config.rotary_base = 10000


def test_graph_router():
    """test graph router."""
    set_route_mode(mode=0)
    config = get_config()
    assert isinstance(ParallelTransformer(config=config), Graph_ParallelTransformer)
    assert isinstance(ParallelTransformerLayer(config=config, layer_number=0), Graph_ParallelTransformerLayer)
    assert isinstance(ParallelAttention(config=config, layer_number=0), Graph_ParallelAttention)
    assert isinstance(ParallelMLP(config=config), Graph_ParallelMLP)
    assert isinstance(
        ColumnParallelLinear(input_size=8, output_size=16, skip_weight_param_allocation=True, config=config),
        Graph_CP_Linear)
    assert isinstance(
        RowParallelLinear(input_size=8, output_size=16, init_method=init_method_normal(), config=config),
        Graph_RP_Linear)
    assert isinstance(RotaryEmbedding(kv_channels=8), Graph_RotaryEmbedding)
    x = initializer('normal', (1, 8, 4096, 64), ms.dtype.bfloat16)
    freqs = initializer('normal', (1, 1, 4096, 64), ms.dtype.bfloat16)
    assert array_equal(apply_rotary_pos_emb(x, freqs, config), graph_apply_rotary_pos_emb(x, freqs, config))
    assert isinstance(
        VocabParallelEmbedding(num_embeddings=123, embedding_dim=16, parallel_config=config,
                               init_method=init_method_normal()),
        Graph_VP_Embedding)
    assert isinstance(AdamW(params=[Parameter(initializer('zeros', (16, 16)), name='weight')]), Graph_AdamW)
    assert isinstance(get_language_model(config=config, num_tokentypes=0, add_pooler=False,
                                         encoder_attn_mask_type=None, decoder_attn_mask_type=None)[0],
                      Graph_TransformerLanguageModel)
    assert isinstance(Embedding(hidden_size=16, vocab_size=1600, config=config,
                                max_sequence_length=64, embedding_dropout_prob=0.),
                      Graph_Embedding)


def test_pynative_router():
    """test pynative router."""
    set_route_mode(mode=1)
    config = get_config()
    add_attr_for_pynative(config)
    assert isinstance(ParallelTransformer(config=config), Pynative_ParallelTransformer)
    assert isinstance(ParallelTransformerLayer(config=config, layer_number=0), Pynative_ParallelTransformerLayer)
    assert isinstance(ParallelAttention(config=config, layer_number=0), Pynative_ParallelAttention)
    assert isinstance(ParallelMLP(config=config), Pynative_ParallelMLP)
    assert isinstance(
        ColumnParallelLinear(input_size=8, output_size=16, init_method='normal', config=config),
        Pynative_CP_Linear)
    assert isinstance(
        RowParallelLinear(input_size=8, output_size=16, skip_bias_add=None, input_is_parallel=True,
                          init_method='normal', bias=False, config=config),
        Pynative_RP_Linear)
    assert isinstance(RotaryEmbedding(kv_channels=8), Pynative_RotaryEmbedding)
    x = initializer('normal', (1, 8, 4096, 64), ms.dtype.bfloat16)
    freqs = initializer('normal', (1, 1, 4096, 64), ms.dtype.bfloat16)
    assert array_equal(apply_rotary_pos_emb(x, freqs, None), pynative_apply_rotary_pos_emb(x, freqs, None))
    assert isinstance(
        VocabParallelEmbedding(num_embeddings=123, embedding_dim=16, config=config,
                               init_method='zeros'),
        Pynative_VP_Embedding)
    assert isinstance(AdamW(network=None, params=[Parameter(initializer('zeros', (16, 16)), name='weight')]),
                      Pynative_AdamW)
    assert isinstance(get_language_model(config=config, num_tokentypes=0, add_pooler=False,
                                         encoder_attn_mask_type=None, decoder_attn_mask_type=None)[0],
                      Pynative_TransformerLanguageModel)
    assert isinstance(Embedding(hidden_size=16, vocab_size=1600, config=config,
                                max_sequence_length=64, embedding_dropout_prob=0.),
                      Pynative_Embedding)


if args_.graph:
    ms.set_context(mode=ms.GRAPH_MODE)
    init()
    print("Running graph router")
    test_graph_router()

if args_.pynative:
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    initialize_model_parallel()
    print("Running pynative router")
    test_pynative_router()
