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
"""Llama2 Base Model."""
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig

# copy from finetune_llama2_13b.yaml
BASE_CONFIG = {
    'batch_size': 1,
    'bos_token_id': 1,
    'compute_dtype': 'float16',
    'do_sample': False,
    'eos_token_id': 2,
    'extend_method': 'None',
    'hidden_size': 5120,
    'ignore_token_id': -100,
    'layernorm_compute_type': 'float32',
    'max_decode_length': 512,
    'multiple_of': 256,
    'num_heads': 40,
    'num_layers': 2,  # 40
    'offset': 0,
    'pad_token_id': 0,
    'param_init_type': 'float16',
    'repetition_penalty': 1,
    'rms_norm_eps': 1e-05,
    'rotary_dtype': 'float16',
    'scaling_factor': 1.0,
    'seq_length': 4096,
    'softmax_compute_type': 'float16',
    'top_k': 3,
    'top_p': 1,
    'type': 'LlamaConfig',
    'use_flash_attention': True,
    'use_past': True,
    'vocab_size': 32000
}


def get_config():
    """get instanced model config."""
    return LlamaConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return LlamaForCausalLM(config)
