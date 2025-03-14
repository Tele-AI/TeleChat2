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
"""telechat2 Base Model."""
import os
import sys

for path in sys.path:
    if path.endswith('/testcases'):
        new_path = os.path.join(path, 'research')
        if new_path not in sys.path:
            sys.path.append(new_path)
    if path.endswith('/research'):
        new_path = os.path.join(path, 'telechat2')
        if new_path not in sys.path:
            sys.path.append(new_path)
research_path = os.path.join('/root', 'mindformers', 'research', 'telechat2')
if research_path not in sys.path:
    sys.path.append(research_path)
# pylint: disable=C0413
from research.telechat2.telechat import TelechatForCausalLM
from research.telechat2.telechat_config import TelechatConfig

# copy from finetune_telechat_115b.yaml
BASE_CONFIG = {
    'type': 'TelechatConfig',
    'batch_size': 1,
    'seq_length': 8192,
    'hidden_size': 5120,  # default is 8192
    'num_layers': 4,  # default is 96
    'num_heads': 40,
    'n_kv_heads': 8,
    'vocab_size': 131072,
    'rms_norm_eps': 1.0e-5,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 3,
    'pp_interleave_num': 3,
    'ignore_token_id': -100,
    'embed_dropout_prob': 0.,
    'hidden_dropout_prob': 0.,
    'attention_dropout_prob': 0.,
    'intermediate_size': 40960,
    'res_dtype': "float32",
    'compute_dtype': "bfloat16",
    'layernorm_compute_type': "float32",
    'softmax_compute_type': "float32",
    'rotary_dtype': "float32",
    'param_init_type': "float32",
    'use_past': False,
    'parallel_optimizer': True,
    'pretrain_seqlen': 8192,  # seqlen of the pretrain checkpoint
    'extend_method': "None", # support "None", "PI", "NTK"
    'use_flash_attention': True,  # FA can accelerate training or finetune
    'offset': 0,
    'fine_grain_interleave': 2,
    'use_past_shard': False,
    'repetition_penalty': 1,
    'max_decode_length': 512,
    'top_k': 3,
    'top_p': 1,
    'do_sample': False,

}


def get_config():
    """get instanced model config."""
    return TelechatConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return TelechatForCausalLM(config)
