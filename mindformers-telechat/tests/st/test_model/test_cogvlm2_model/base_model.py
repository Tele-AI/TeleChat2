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
"""CogVLM2-Video Base Model."""
from mindformers.tools.register import MindFormerConfig
from mindformers.models.cogvlm2.cogvlm2_config import CogVLM2Config
from mindformers.models.cogvlm2.cogvlm2 import CogVLM2ForCausalLM, CogVLM2ImageForCausalLM

# copy from predict_cogvlm2_video_llama3_chat_13b.yaml
BASE_CONFIG = {
    'batch_size': 1,
    'num_queries': 66,
    'block_size': 16,
    'is_dynamic': True,
    'llm_model': {
        'arch': {'type': 'CogVLM2VideoLM'},
        'model_config': {'batch_size': 1,
                         'bos_token_id': 128000,
                         'compute_dtype': 'float16',
                         'do_sample': False,
                         'embedding_init_type': 'float16',
                         'eos_token_id': 128001,
                         'extend_method': 'None',
                         'fine_grain_interleave': 1,
                         'hidden_size': 4096,
                         'ignore_token_id': -100,
                         'intermediate_size': 14336,
                         'layernorm_compute_type': 'float32',
                         'max_decode_length': 2048,
                         'n_kv_heads': 8,
                         'num_heads': 32,
                         'num_layers': 2,
                         'offset': 0,
                         'pad_token_id': 128002,
                         'param_init_type': 'float16',
                         'repetition_penalty': 1,
                         'rms_norm_eps': 1e-05,
                         'rotary_dtype': 'float32',
                         'scaling_factor': 1.0,
                         'seq_length': 2048,
                         'softmax_compute_type': 'float32',
                         'theta': 500000,
                         'top_k': 3,
                         'top_p': 1,
                         'type': 'LlamaConfig',
                         'use_flash_attention': True,
                         'vocab_size': 128256}},
    'num_blocks': 512,
    'type': 'CogVLM2Config',
    'use_past': True,
    'vision_model': {
        'arch': {'type': 'EVAModel'},
        'model_config': {'class_token': True,
                         'compute_dtype': 'float16',
                         'hidden_size': 1792,
                         'image_size': 224,
                         'intermediate_size': 15360,
                         'layer_norm_eps': '1e-6',
                         'layer_norm_type': 'float32',
                         'num_attention_heads': 16,
                         'num_hidden_layers': 2,
                         'param_init_type': 'float16',
                         'patch_size': 14,
                         'post_norm': True,
                         'rotary_emb_type': 'float32',
                         'type': 'EVA02Config',
                         'use_abs_pos_emb': True,
                         'use_attn_norm': False,
                         'use_post_norm': True,
                         'use_qkv_fused': True,
                         'use_qkv_simple': True,
                         'use_rot_pos_emb': False,
                         'use_scale_mlp': False,
                         'use_swiglu': False,
                         'with_cls_token': False}}
}

# copy from predict_cogvlm2_image_llama3_chat_19b.yaml
IMAGE_BASE_CONFIG = {
    'block_size': 16,
    'is_dynamic': False,
    'llm_model': {
        'arch': {'type': 'LlamaForCausalLMForCogVLM2Image'},
        'model_config': {'batch_size': 1,
                         'bos_token_id': 128000,
                         'compute_dtype': 'float16',
                         'do_sample': False,
                         'embedding_init_type': 'float16',
                         'eos_token_id': 128001,
                         'extend_method': 'None',
                         'fine_grain_interleave': 1,
                         'hidden_size': 4096,
                         'ignore_token_id': -100,
                         'intermediate_size': 14336,
                         'layernorm_compute_type': 'float32',
                         'max_decode_length': 2048,
                         'n_kv_heads': 8,
                         'num_heads': 32,
                         'num_layers': 1,
                         'offset': 0,
                         'pad_token_id': 128002,
                         'param_init_type': 'float16',
                         'repetition_penalty': 1,
                         'rms_norm_eps': 1e-05,
                         'rotary_dtype': 'float32',
                         'scaling_factor': 1.0,
                         'seq_length': 4096,
                         'softmax_compute_type': 'float32',
                         'theta': 500000,
                         'top_k': 3,
                         'top_p': 1,
                         'type': 'LlamaConfig',
                         'use_flash_attention': True,
                         'vocab_size': 128256}},
    'num_blocks': 512,
    'type': 'CogVLM2Config',
    'use_past': True,
    'vision_model': {
        'arch': {'type': 'EVAModel'},
        'model_config': {'class_token': True,
                         'compute_dtype': 'float16',
                         'hidden_size': 1792,
                         'image_size': 1344,
                         'intermediate_size': 15360,
                         'layer_norm_eps': '1e-6',
                         'layer_norm_type': 'float32',
                         'num_attention_heads': 16,
                         'num_hidden_layers': 1,
                         'param_init_type': 'float16',
                         'patch_size': 14,
                         'post_norm': True,
                         'rotary_emb_type': 'float32',
                         'type': 'EVA02Config',
                         'use_abs_pos_emb': True,
                         'use_attn_norm': False,
                         'use_post_norm': True,
                         'use_qkv_fused': True,
                         'use_qkv_simple': True,
                         'use_rot_pos_emb': False,
                         'use_scale_mlp': False,
                         'use_swiglu': False,
                         'with_cls_token': False}}
}


def get_config():
    """get instanced model config."""
    model_config = MindFormerConfig(**BASE_CONFIG)
    return CogVLM2Config(**model_config)


def get_image_config():
    """get instanced model config."""
    model_config = MindFormerConfig(**IMAGE_BASE_CONFIG)
    return CogVLM2Config(**model_config)


def get_model(config):
    """get instanced model."""
    return CogVLM2ForCausalLM(config)


def get_image_model(config):
    """get instanced model."""
    return CogVLM2ImageForCausalLM(config)
