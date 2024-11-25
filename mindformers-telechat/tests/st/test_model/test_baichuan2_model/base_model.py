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
"""Baichuan2 Base Model."""
from mindformers.models.llama import LlamaConfig
from research.baichuan2.baichuan2_13b import Baichuan13BV2ForCausalLM

# copy from finetune_baichuan2_13b.yaml
BASE_CONFIG = {
    'type': 'LlamaConfig',
    'batch_size': 1,
    'seq_length': 4096,
    'hidden_size': 5120,
    'num_layers': 2,  # default is 40
    'num_heads': 40,
    'vocab_size': 125696,
    'multiple_of': 128,
    'rms_norm_eps': 1.0e-6,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
    'ignore_token_id': -100,
    'compute_dtype': 'float16',
    'layernorm_compute_type': 'float32',
    'softmax_compute_type': 'float16',
    'param_init_type': 'float16',
    'use_past': True,
    'pretrain_seqlen': 2048,
    'extend_method': 'None',
    # 'compute_in_2d': True,  # deprecated
    'use_flash_attention': True,
    'block_size': 16,
    'num_blocks': 512,
    'is_dynamic': False,
    'offset': 0,
    # 'use_past_shard': False,  # deprecated
    'repetition_penalty': 1,
    'temperature': 1.0,
    'max_decode_length': 512,
    'top_k': 3,
    'top_p': 1,
    'do_sample': False
}


def get_config():
    """get instanced model config."""
    return LlamaConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return Baichuan13BV2ForCausalLM(config)
