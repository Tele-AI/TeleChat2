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
"""PanGuAlpha Config API."""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config
from mindformers.modules.transformer.transformer import default_moe_config
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

__all__ = ['PanguAlphaConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class PanguAlphaConfig(PretrainedConfig):
    """
    PanGuAlpha config class which defines the model size
    """

    model_type = "pangualpha"
    _support_list = MindFormerBook.get_config_support_list()['pangualpha']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 vocab_size: int = 40000,
                 hidden_size: int = 2560,
                 ffn_hidden_size: int = 2560 * 4,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 pad_token_id: int = 6,
                 eos_token_id: int = 8,
                 post_layernorm_residual: bool = False,
                 param_init_type: str = 'float32',
                 compute_dtype: str = 'float16',
                 softmax_compute_type: str = 'float16',
                 embedding_dropout_prob: float = 0.1,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 hidden_act: str = 'fast_gelu',
                 use_past: bool = False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_moe: bool = False,
                 expert_num: int = 1,
                 per_token_num_experts_chosen: int = 1,
                 checkpoint_name_or_path: str = '',
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        super(PanguAlphaConfig, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.hidden_act = hidden_act
        self.use_past = use_past
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.use_moe = bool(use_moe)
        self.expert_num = expert_num
        self.per_token_num_experts_chosen = per_token_num_experts_chosen
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
