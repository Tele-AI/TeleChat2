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
"""Wizardcoder Config API."""

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig

__all__ = ['WizardCoderConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class WizardCoderConfig(PretrainedConfig):
    """
    wizardcoder config class which defines the model size
    """

    model_type = "wizardcoder"

    def __init__(self,
                 dropout_prob: float = 0.1,
                 batch_size: int = None,
                 seq_length: int = 1024,
                 n_position: int = 8192,
                 vocab_size: int = 50257,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 expand_ratio: int = 4,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 eos_token: int = 0,
                 pad_token: int = 49152,
                 param_init_type: str = "float16",
                 layernorm_dtype: str = "float32",
                 softmax_dtype: str = "float16",
                 compute_dtype: str = "float16",
                 hidden_act: str = 'gelu',
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 use_past: bool = False,
                 use_seq_parallel: bool = False,
                 use_select_recompute: bool = False,
                 checkpoint_name_or_path: str = "",
                 moe_config: MoEConfig = default_moe_config,
                 top_p=0.95,
                 top_k=1,
                 repetition_penalty=1,
                 max_length=20,
                 do_sample=False,
                 start_token_id=1,
                 eos_token_id=2,
                 is_encoder_decoder=False,
                 use_flash_attention=False,
                 **kwargs):
        super(WizardCoderConfig, self).__init__(**kwargs)
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_position = n_position
        self.vocab_size = vocab_size
        self.embedding_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_dtype = convert_mstype(layernorm_dtype)
        self.softmax_dtype = convert_mstype(softmax_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.use_past = use_past
        self.use_seq_parallel = use_seq_parallel
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.hidden_act = hidden_act
        self.use_select_recompute = use_select_recompute
        # Basic the configuration for the generation
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.do_sample = do_sample
        self.use_flash_attention = use_flash_attention
