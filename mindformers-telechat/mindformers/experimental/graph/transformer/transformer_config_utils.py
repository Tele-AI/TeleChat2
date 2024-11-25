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
"""TransformerConfig utils"""
import mindspore.common.dtype as mstype

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.experimental.utils import init_method_normal, init_method_zero

_CONFIG_MAPPING = {
    'vocab_size': 'padded_vocab_size',
    'num_heads': 'num_attention_heads',
    'n_kv_heads': 'num_query_groups',
    'hidden_dropout_rate': 'hidden_dropout',
    'attention_dropout_rate': 'attention_dropout',
    'softmax_compute_type': 'softmax_compute_dtype',
    'param_init_type': 'params_dtype',
    'qkv_has_bias': 'add_qkv_bias',
    'use_gqa': 'group_query_attention',
    'use_flash_attention': 'use_flash_attn',
    'max_position_embedding': 'max_position_embeddings',
    'weight_init': 'init_method'
}

_INIT_ATTRIBUTE = {
    '_name_or_path',
    '_commit_hash',
    'checkpoint_name_or_path',
    'mindformers_version',
    'tokenizer_class',
    'architectures',
    'is_encoder_decoder',
    'is_sample_acceleration'
}


def convert_to_transformer_config(config, transformer_config):
    r"""
    Convert a configuration object to a TransformerConfig.

    This function checks the type of the given configuration object and
    converts it to the appropriate TransformerConfig using the relevant
    conversion function.

    Args:
        config (Union[MindFormerConfig, PretrainedConfig]): The configuration object
            to be converted. It should be an instance of either `MindFormerConfig`
            or `PretrainedConfig`.

        transformer_config (TransformerConfig): An existing TransformerConfig instance
            to which the attributes from the provided config will be applied.

    Returns:
        TransformerConfig: The converted TransformerConfig object.

    Raises:
        Exception: If the provided config object is not of a supported type.

    Examples:
        >>> from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
        >>> from mindformers.models.llama.llama_config import LlamaConfig\

        >>> config = LlamaConfig()
        >>> transformer_config = TransformerConfig()
        >>> transformer_config = convert_to_transformer_config(config, transformer_config)
    """
    if isinstance(config, PretrainedConfig):
        return convert_pretrained_config(config, transformer_config)

    raise Exception(f"unsupported config type '{config}'.")


def convert_pretrained_config(config: PretrainedConfig, transformer_config: TransformerConfig):
    """Convert the PretrainedConfig class to TransformerConfig class."""
    flag = 1
    for attr in vars(config):
        value = getattr(config, attr)
        if attr.endswith('type'):
            value = convert_mstype(value)
        if attr == 'weight_init':
            if str(value).lower() == 'zero' or str(value).lower() == 'zeros':
                flag = 0
                value = None
            elif str(value).lower() == 'normal':
                flag = 1
                value = None
        if attr in _CONFIG_MAPPING:
            set_attr = _CONFIG_MAPPING[attr]
            setattr(transformer_config, set_attr, value)
        elif attr not in _INIT_ATTRIBUTE:
            setattr(transformer_config, attr, value)

    transformer_config.update()
    transformer_config.post_init_checks()

    transformer_config.data_parallel = config.parallel_config.data_parallel
    transformer_config.tensor_parallel = config.parallel_config.model_parallel
    transformer_config.context_parallel = config.parallel_config.context_parallel
    transformer_config.vocab_emb_dp = config.parallel_config.vocab_emb_dp
    if hasattr(config, 'n_kv_heads'):
        if config.n_kv_heads is not None:
            transformer_config.group_query_attention = True
    if flag == 1:
        transformer_config.init_method = init_method_normal(transformer_config.init_method_std,
                                                            transformer_config.params_dtype)
    else:
        transformer_config.init_method = init_method_zero(transformer_config.params_dtype)
    return transformer_config


def convert_mstype(ms_dtype: str = "float16"):
    """Convert the string type to MindSpore type."""
    ms_type = str(ms_dtype).lower()
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    return ms_dtype
