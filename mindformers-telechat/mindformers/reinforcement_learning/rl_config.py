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
"""
Note: The config module of Parameter Efficient Tuning module.
"""
from mindformers.tools import DictConfig


__all__ = ['RlConfig', 'DPOConfig']


class RlConfig(DictConfig):
    """
    The configuration base class for Reinforcement learning (RL) algorithms.

    Args:
        rl_type (str, optional): The Rl method type. Default: ``None``.

    Returns:
        An instance of RlConfig.

    Examples:
        >>> from mindformers.rl_preprocess.rl_config import DPOConfig
        >>> config = DPOConfig(dpo_alpha='0.5')
        >>> print(config)
        {'rl_type': 'dpo', 'dpo_alpha': 1.0, 'dpo_beta': 1.0}
    """
    def __init__(self,
                 rl_type: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rl_type = rl_type


class DPOConfig(RlConfig):
    """
    DPO algorithm config.
    Used to set parameters for DPO model runtime.

    Args:
        dpo_alpha (float, optional): coef for dpo loss. Default: ``0.5``.
        dpo_beta (float, optional): coef for sft loss. Default: ``1.0``.

    Returns:
        An instance of DPOConfig.

    Examples:
        >>> from mindformers.rl_preprocess.rl_config import DPOConfig
        >>> config = RlConfig(dpo_alpha='0.5')
        >>> print(config)
        {'rl_type': 'dpo', 'dpo_alpha': 1.0, 'dpo_beta': 1.0}
    """
    def __init__(self,
                 dpo_alpha: float = 0.5,
                 dpo_beta: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.dpo_alpha = dpo_alpha
        self.dpo_beta = dpo_beta
