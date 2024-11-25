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
"""LLaMA models' utils."""


def convert_model_config(configs):
    """convert model config to dynamic-infer style"""
    ffn_hidden_size = configs.hidden_size * 4
    if configs.intermediate_size is not None:
        ffn_hidden_size = configs.intermediate_size
    else:
        if configs.ffn_dim_multiplier is not None:
            ffn_hidden_size = int((configs.ffn_dim_multiplier + 0.01) * ffn_hidden_size)
        ffn_hidden_size = int(2 * ffn_hidden_size / 3)
        ffn_hidden_size = configs.multiple_of * ((ffn_hidden_size + configs.multiple_of - 1) // configs.multiple_of)

    configs.apply_query_key_layer_scaling = False
    configs.apply_residual_connection_post_norm = False
    configs.attention_dropout_rate = 0.0
    configs.attention_type = 'self_attn'
    configs.ffn_hidden_size = ffn_hidden_size
    configs.hidden_act = "silu"
    configs.hidden_dropout_rate = 0.0
    configs.kv_num_heads = configs.num_heads if configs.n_kv_heads is None else configs.n_kv_heads
    configs.layernorm_epsilon = configs.rms_norm_eps
    configs.mask_func_type = "attn_mask_add"
    configs.mlp_has_bias = False
    configs.normalization = "RMSNorm"
    configs.num_experst = None
    configs.out_proj_has_bias = False
    configs.param_init_dtype = configs.param_init_type
    configs.layernorm_compute_dtype = configs.layernorm_compute_type
    configs.residual_connection_dtype = configs.softmax_compute_type
    configs.share_embedding_weight = False
    configs.softmax_compute_dtype = configs.softmax_compute_type
    configs.use_gqa = False
    configs.mlp_has_gate = True
    configs.post_norm = True
    configs.recompute_granularity = None
    configs.ffn_concat = configs.qkv_concat

    parallel_config = configs.parallel_config
    parallel_config.tensor_parallel = parallel_config.model_parallel
    parallel_config.expert_parallel = 1
    parallel_config.use_sequence_parallel = False
    parallel_config.use_zero3 = False
    configs.parallel_config = parallel_config

    return configs
