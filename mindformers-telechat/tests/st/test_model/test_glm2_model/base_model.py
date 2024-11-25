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
"""GLM2 Base Model."""
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration

# copy from finetune_glm2_6b_fp16.yaml
BASE_CONFIG = {
    'add_bias_linear': False,
    'add_qkv_bias': True,
    'apply_query_key_layer_scaling': True,
    'apply_residual_connection_post_layernorm': False,
    'attention_dropout': 0.0,
    'attention_softmax_in_fp32': True,
    'bias_dropout_fusion': True,
    'compute_dtype': 'bfloat16',
    'do_sample': False,
    'eos_token_id': 2,
    'ffn_hidden_size': 13696,
    'fp32_residual_connection': False,
    'hidden_dropout': 0.0,
    'hidden_size': 4096,
    'kv_channels': 128,
    'layernorm_compute_type': 'float32',
    'layernorm_epsilon': '1e-5',
    'max_decode_length': 256,
    'multi_query_attention': True,
    'multi_query_group_num': 2,
    'num_attention_heads': 32,
    'num_layers': 2,  # 28
    'pad_token_id': 0,
    'padded_vocab_size': 65024,
    'param_init_type': 'bfloat16',
    'post_layer_norm': True,
    'pre_seq_len': 'None',
    'prefix_projection': False,
    'quantization_bit': 0,
    'repetition_penalty': 1.0,
    'rmsnorm': True,
    'seq_length': 2048,
    'top_k': 1,
    'top_p': 1,
    'type': 'ChatGLM2Config',
    'use_flash_attention': True,
    'use_past': False
}


def get_config():
    """get instanced model config."""
    return ChatGLM2Config(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return ChatGLM2ForConditionalGeneration(config)
