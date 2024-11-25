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
"""
Note: The config module of Parameter Efficient Tuning module.
"""
from typing import List
from mindformers.models.utils import convert_mstype
from mindformers.tools import DictConfig
from mindformers.pet.constants import PetType


__all__ = ['PetConfig', 'LoraConfig', 'Ptuning2Config', 'PrefixTuningConfig']


class PetConfig(DictConfig):
    """
    The configuration base class for Parameter-Efficient Tuning (Pet) algorithms.

    Args:
        pet_type (str, optional): The Pet method type. Default: ``None``.

    Returns:
        An instance of PetConfig.

    Examples:
        >>> from mindformers.pet.pet_config import LoraConfig
        >>> config = LoraConfig(target_modules='.*wq|.*wk|.*wv|.*wo')
        >>> print(config)
        {'pet_type': 'lora', 'lora_rank': 8, 'lora_alpha': 16,
        'lora_dropout': 0.01, 'lora_a_init': 'normal', 'lora_b_init'
        : 'zero', 'param_init_type': mindspore.float16, 'compute_dtype':
        mindspore.float16, 'target_modules': '.*wq|.*wk|.*wv|.*wo', 'exclude_layers': None
        , 'freeze_include': None, 'freeze_exclude': None}
    """
    def __init__(self,
                 pet_type: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pet_type = pet_type


class LoraConfig(PetConfig):
    """
    LoRA algorithm config.
    Used to set parameters for LoRA model runtime.

    Args:
        lora_rank (int, optional): The number of rows(columns) in LoRA matrices. Default: ``8``.
        lora_alpha (int, optional): A constant in lora_rank. Default: ``16``.
        lora_dropout (float, optional): The dropout rate, greater equal than 0 and less than 1. Default: ``0.01``.
        lora_a_init (str, optional): The initialization strategy of LoRA A matrix.
            Refers to (https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html). Default: ``normal``.
        lora_b_init (str, optional): The initialization strategy of LoRA B matrix.
            Refers to (https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html). Default: ``zero``.
        param_init_type (str, optional): The type of data in initialized tensor. Default: ``float16``.
        compute_dtype (str, optional): The compute type of data. Default: ``float16``.
        target_modules (str, optional): The layers that require replacement with LoRA algorithm. Default: ``None``.
        exclude_layers (str, optional): The layers that do not require
            replacement with the LoRA algorithm. Default: ``None``.
        freeze_include (List[str], optional): List of modules to be frozen. Default: ``None``.
        freeze_exclude (List[str], optional): List of modules that do not need
            to be frozen. When an item in the freeze_include and freeze_exclude list
            conflicts, the module that matches this item is not processed. Default: ``None``.

    Returns:
        An instance of LoraConfig.

    Examples:
        >>> from mindformers.pet.pet_config import LoraConfig
        >>> config = LoraConfig(target_modules='.*wq|.*wk|.*wv|.*wo')
        >>> print(config)
        {'pet_type': 'lora', 'lora_rank': 8, 'lora_alpha': 16,
        'lora_dropout': 0.01, 'lora_a_init': 'normal', 'lora_b_init'
        : 'zero', 'param_init_type': mindspore.float16, 'compute_dtype':
        mindspore.float16, 'target_modules': '.*wq|.*wk|.*wv|.*wo', 'exclude_layers': None
        , 'freeze_include': None, 'freeze_exclude': None}
    """
    def __init__(self,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.01,
                 lora_a_init: str = 'normal',
                 lora_b_init: str = 'zero',
                 param_init_type: str = 'float16',
                 compute_dtype: str = 'float16',
                 target_modules: str = None,
                 exclude_layers: str = None,
                 freeze_include: List[str] = None,
                 freeze_exclude: List[str] = None,
                 **kwargs):
        super().__init__(pet_type=PetType.LORA.value, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_a_init = lora_a_init
        self.lora_b_init = lora_b_init
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.target_modules = target_modules
        self.exclude_layers = exclude_layers
        self.freeze_include = freeze_include
        self.freeze_exclude = freeze_exclude


class Ptuning2Config(PetConfig):
    """
    p tuning v2 tuning algorithm config.
    """
    def __init__(self,
                 pre_seq_len: int = 128,
                 prefix_projection: bool = False,
                 projection_dim: int = 128,
                 dropout_prob: float = 0.01,
                 **kwargs):
        super().__init__(pet_type=PetType.P_TUNING_V2.value, **kwargs)
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.projection_dim = projection_dim
        self.dropout_prob = dropout_prob


class PrefixTuningConfig(PetConfig):
    """
    PrefixTuning algorithm config.
    """
    def __init__(self,
                 prefix_token_num: int = 128,
                 mid_dim: int = 512,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(pet_type=PetType.PREFIX_TUNING.value, **kwargs)
        self.prefix_token_num = prefix_token_num
        self.mid_dim = mid_dim
        self.dropout_rate = dropout_rate


class SLoraConfig(PetConfig):
    """
    Multi-LoRA algorithm config.

    Args:
        target_modules (`str`, *optional*, defaults to None):
            The Layers that require replacement with LoRa algorithm.
        lora_num ('int', *optional*, default to 1):
            The number of LoRA weights used in LoRA matrices
        lora_rank (`int`, *optional*, defaults to 8):
            The number of rows(columns) in LoRA matrices.
        lora_alpha (`int`, *optional*, defaults to 16):
            A constant in lora_rank.
        lora_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate, greater equal than 0 and less than 1.
        lora_extra_vocab_size (`int`, *optional*, defaults to 0):
            The number of extra vocabulary size introduced by LoRA
        lora_dtype (`str`, *optional*, defaults to 'float16'):
            The type of data in initialized tensor.

    Returns:
        Class, SLoraConfig.
    """
    def __init__(self,
                 target_modules: str = None,
                 lora_num: int = 1,
                 lora_rank: int = 8,
                 lora_alpha: int = 8,
                 lora_dropout: float = 0.0,
                 lora_extra_vocab_size: int = 0,
                 lora_dtype: str = "float16",
                 **kwargs):
        super().__init__(pet_type="slora", **kwargs)
        self.target_modules = target_modules
        self.lora_num = lora_num
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_extra_vocab_size = lora_extra_vocab_size
        self.lora_dtype = convert_mstype(lora_dtype)
