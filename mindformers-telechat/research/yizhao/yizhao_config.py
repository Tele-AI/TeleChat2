# Copyright 2023 Huawei Technologies Co., Ltd
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
"""YiZhao config"""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['YiZhaoConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class YiZhaoConfig(PretrainedConfig):
    """
    YiZhao model config class.
    """

    model_type = "YiZhao"

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 batch_size=1,  # only for incremental infer
                 num_layers=28,
                 padded_vocab_size=65024,
                 hidden_size=4096,
                 ffn_hidden_size=13696,
                 kv_channels=128,
                 num_attention_heads=32,
                 seq_length=2048,
                 hidden_dropout=0.0,
                 attention_dropout=0.0,
                 layernorm_epsilon=1e-5,
                 rope_ratio=1,
                 rmsnorm=True,
                 apply_residual_connection_post_layernorm=False,
                 post_layer_norm=True,
                 add_bias_linear=False,
                 add_qkv_bias=True,
                 bias_dropout_fusion=True,
                 multi_query_attention=True,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=True,
                 attention_softmax_in_fp32=True,
                 fp32_residual_connection=False,
                 quantization_bit=0,
                 pre_seq_len=None,
                 prefix_projection=False,
                 param_init_type: str = "float16",
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 rotary_dtype: str = "float32",
                 use_past=False,
                 use_flash_attention=False,
                 use_prompt_flash_attention=False,
                 use_incre_flash_attention=False,
                 block_size=16,
                 num_blocks=128,
                 is_dynamic=False,
                 eos_token_id=2,
                 pad_token_id=0,
                 gmask_token_id=None,
                 bos_token_id=None,
                 repetition_penalty=1.0,
                 checkpoint_name_or_path=None,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 no_recompute_layers=None,
                 mlp_concat=True,
                 qkv_concat=True,
                 use_llama_rope=False,
                 offset: int = 0,
                 pp_interleave_num: int = 1,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 mask_generate=None,
                 rl_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rope_ratio = rope_ratio
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.use_incre_flash_attention = use_incre_flash_attention
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty = repetition_penalty
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.gmask_token_id = gmask_token_id
        self.bos_token_id = bos_token_id
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.is_dynamic = is_dynamic
        self.num_heads = self.num_attention_heads
        self.n_kv_heads = self.multi_query_group_num if self.multi_query_attention else None
        self.no_recompute_layers = no_recompute_layers
        self.mlp_concat = mlp_concat
        self.qkv_concat = qkv_concat
        self.use_llama_rope = use_llama_rope
        self.offset = offset
        self.pp_interleave_num = pp_interleave_num
        self.alpha = alpha  # dpo loss coef
        self.beta = beta    # sft loss coef
        self.mask_generate = mask_generate
        self.rl_config = rl_config
