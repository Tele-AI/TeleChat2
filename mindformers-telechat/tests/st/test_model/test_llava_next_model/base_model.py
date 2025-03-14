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
"""Llava Next Base Model."""
from mindformers import MindFormerModuleType, MindFormerRegister
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.register import MindFormerConfig
from research.llava_next.llava_next import LlavaNextVlm
from research.llava_next.llava_next_config import LlavaNextConfig
from research.llava_next.llava_next_vision_tower import LlavaVisionEncoder

VIDEO_CONFIG = {
    "stage": 3,
    "add_newline": False,
    "video_contains_pooler": True,
    "spatial_pool_mode": "average",
    "spatial_pool_out_channels": 1024,
    "spatial_pool_stride": 2,
    "img_dynamic_batch": False,
    "construct_args_key": ["input_ids", "images", "image_context_pos"]
}

IMAGE_CONFIG = {
    "stage": 2,
    "video_contains_pooler": False,
    "add_newline": True,
    "img_dynamic_batch": False,
    "construct_args_key": ["input_ids", "images", "image_patches", "image_context_pos"]
}

# copy from finetune_llava_next_video_v1_7b_stage2.yaml
BASE_CONFIG = {
    'batch_size': 1,
    'num_queries': 576,
    'block_size': 16,
    'is_dynamic': True,
    'ignore_token_id': -100,
    'num_blocks': 512,
    'type': 'LlavaNextConfig',
    'use_past': True,
    'top_k': 3,
    'top_p': 1,
    "do_sample": False,
    "seq_length": 4096,
    "max_decode_length": 4096,
    'compute_dtype': 'bfloat16',
    'layernorm_compute_type': 'float32',
    'softmax_compute_type': 'float32',
    'rotary_dtype': 'float32',
    'param_init_type': 'float32',
    'text_model': {
        'arch': {'type': 'LlamaForCausalLM'},
        'model_config': {'bos_token_id': 1,
                         'do_sample': False,
                         'eos_token_id': 2,
                         'hidden_size': 4096,
                         'intermediate_size': 11008,
                         'num_heads': 32,
                         'num_layers': 2,
                         'pad_token_id': 0,
                         'rms_norm_eps': 1e-05,
                         'type': 'LlamaConfig',
                         'use_flash_attention': True,
                         'vocab_size': 32064}},
    'vision_model': {
        'arch': {'type': 'LlavaVisionEncoder'},
        'model_config': {
            'hidden_size': 1024,
            'image_size': 336,
            'intermediate_size': 4096,
            'num_attention_heads': 16,
            'num_hidden_layers': 2,
            "hidden_act": "quick_gelu",
            'patch_size': 14,
            'num_queries': 576,
            'type': 'LlavaNextVisionConfig',
            'use_flash_attention': True,
            "vision_feature_layer": -2,
            "vision_feature_select_strategy": "default"
        }}
}


def get_config(video_config=True):
    """get instanced model config."""
    if video_config:
        model_config = MindFormerConfig(**BASE_CONFIG, **VIDEO_CONFIG)
    else:
        model_config = MindFormerConfig(**BASE_CONFIG, **IMAGE_CONFIG)
    return LlavaNextConfig(**model_config)


def get_model(config):
    """get instanced model."""
    MindFormerRegister.register_cls(LlavaVisionEncoder, MindFormerModuleType.MODELS)

    class DataOrderedCell(PreTrainedModel):
        def __init__(self, config, **kwargs):
            super(DataOrderedCell, self).__init__(config, **kwargs)
            self.network = LlavaNextVlm(config)
            self.construct_args_key = config.construct_args_key

        def construct(self, *inputs):
            """The construct processes of inputs in lexicographical order."""
            key_inputs = {key: val for key, val in zip(self.construct_args_key, inputs)}
            return self.network(**key_inputs)

    return DataOrderedCell(config)
