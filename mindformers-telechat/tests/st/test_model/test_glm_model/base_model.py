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
"""GLM Base Model."""
from mindformers import GLMForPreTraining, GLMConfig

# copy from run_glm_6b_finetune.yaml
BASE_CONFIG = {
    'activation_func': 'GELU',
    'attention_dropout_rate': 0.0,
    'bos_token_id': 130004,
    'compute_dtype': 'float16',
    'do_sample': False,
    'embedding_dropout_prob': 0.0,
    'eos_token_id': 130005,
    'gmask_token_id': 130001,
    'hidden_dropout_rate': 0.0,
    'hidden_size': 4096,
    'hidden_size_per_attention_head': None,
    'inner_hidden_size': 16384,
    'is_enhanced_encoder': True,
    'is_sample_acceleration': False,
    'layernorm_compute_type': 'float32',
    'layernorm_epsilon': 1e-05,
    'layernorm_order': 'post',
    'mask_token_id': 130000,
    'max_decode_length': 2048,
    'num_heads': 32,
    'num_layers': 2,  # 28
    'pad_token_id': 3,
    'param_init_type': 'float16',
    'position_encoding_2d': True,
    'repetition_penalty': 1,
    'seq_length': 512,
    'softmax_compute_type': 'float32',
    'top_k': 1,
    'top_p': 1,
    'type': 'GLMConfig',
    'use_final_layernorm': True,
    'use_past': False,
    'vocab_size': 130528
}


def get_config():
    """get instanced model config."""
    return GLMConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return GLMForPreTraining(config)
