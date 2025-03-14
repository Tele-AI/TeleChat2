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
from mindformers.models.llama import LlamaConfig
from mindformers.models.glm2 import ChatGLM2Config

GLM_CONFIG = {
    "batch_size": 1,
    "num_layers": 28,
    "padded_vocab_size": 65024,
    "hidden_size": 4096,
    "ffn_hidden_size": 13696,
    "kv_channels": 128,
    "num_attention_heads": 32,
    "seq_length": 2048,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "rmsnorm": True,
    "layernorm_epsilon": 0.00001,
    "apply_residual_connection_post_layernorm": False,
    "post_layer_norm": True,
    "add_bias_linear": False,
    "add_qkv_bias": True,
    "bias_dropout_fusion": True,
    "multi_query_attention": True,
    "multi_query_group_num": 2,
    "apply_query_key_layer_scaling": True,
    "attention_softmax_in_fp32": True,
    "fp32_residual_connection": False,
    "quantization_bit": 0,
    "pre_seq_len": None,
    "prefix_projection": False,
    "param_init_type": "float16",
    "compute_dtype": "float16",
    "layernorm_compute_type": "float32",
    "use_past": True,
    "use_flash_attention": True,
    "max_length": 256,
    "block_size": 16,
    "num_blocks": 128,
    "is_dynamic": True,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "repetition_penalty": 1.0,
    "max_decode_length": 256,
    "top_k": 1,
    "top_p": 1,
    "do_sample": False,
}

QWEN_CONFIG = {
    'batch_size': 1,
    'bos_token_id': 151643,
    'compute_dtype': 'bfloat16',
    'do_sample': False,
    'eos_token_id': [151643, 151645],
    'extend_method': 'None',
    'hidden_size': 896,
    'ignore_token_id': -100,
    'layernorm_compute_type': 'float32',
    'max_decode_length': 512,
    'multiple_of': 256,
    'num_heads': 14,
    'n_kv_heads': 2,
    'num_layers': 24,
    'offset': 0,
    'pad_token_id': 151643,
    'param_init_type': 'bfloat16',
    'repetition_penalty': 1,
    'rms_norm_eps': 1.0e-6,
    'rotary_dtype': 'bfloat16',
    'scaling_factor': 1.0,
    'seq_length': 8192,
    'emb_dropout_prob': 0.0,
    'theta': 1000000.0,
    'intermediate_size': 4864,
    'qkv_has_bias': True,
    'max_position_embeddings': 32768,
    'softmax_compute_type': 'float32',
    'top_k': 3,
    'top_p': 1,
    'type': 'LlamaConfig',
    'use_flash_attention': True,
    'use_past': True,
    'vocab_size': 151936,
    'block_size': 32,
    'num_blocks': 1024,
    'is_dynamic': True,
    'qkv_concat': True,
    'tie_word_embeddings': True,
}


def get_glm_config():
    """get instanced model config."""
    return ChatGLM2Config(**GLM_CONFIG)


def get_qwen_config():
    """get instanced model config."""
    return LlamaConfig(**QWEN_CONFIG)
