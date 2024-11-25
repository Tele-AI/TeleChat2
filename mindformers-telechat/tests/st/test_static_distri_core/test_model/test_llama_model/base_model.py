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
"""Llama Base Model."""
from mindformers.models.llama import LlamaConfig
from mindformers.experimental.model.llama.llama import LlamaForCausalLM

BASE_CONFIG = {
    'num_heads': 40,
    'num_layers': 2,
    'hidden_size': 5120,
    'multiple_of': 256,
    'seq_length': 4096,
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
    return LlamaConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return LlamaForCausalLM(config)
