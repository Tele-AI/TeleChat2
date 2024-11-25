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
"""Build Optimizer API."""
import inspect

from mindspore import nn
from mindspore.nn.optim import AdaFactor, AdamWeightDecay, SGD, Adagrad, Adam

from mindformers.core.lr import build_lr
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from .optim import FusedAdamWeightDecay, FP32StateAdamWeightDecay


def build_optim(
        config: dict = None, default_args: dict = None,
        module_type: str = 'optimizer', class_name: str = None, **kwargs):
    r"""Build optim For MindFormer.
    Instantiate the optim from MindFormerRegister's registry.

    Args:
        config (dict): The task optim's config. Default: None.
        default_args (dict): The default argument of optim API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'optim'.
        class_name (str): The class name of optim API. Default: None.

    Return:
        The function instance of optim API.

    Examples:
        >>> from mindformers.core import build_optim
        >>> from mindspore.common.parameter import Parameter
        >>> from mindspore.common import Tensor
        >>> params = [{"params": [Parameter(Tensor([1]), requires_grad=True, name=f"param_{i}") for i in range(2)]}]
        >>> # 1) use config dict to build optim
        >>> optim_config = {'type': 'AdamWeightDecay', 'weight_decay':0.05, 'params':params}
        >>> optim_from_config = build_optim(optim_config)
        >>> optim_from_config
        AdamWeightDecay<>
        >>> # 2) use class name to build optim
        >>> optim_from_class_name = build_optim(class_name='AdamWeightDecay', weight_decay=0.05, params=params)
        >>> optim_from_class_name
        AdamWeightDecay<>
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        if config.learning_rate is not None and isinstance(config.learning_rate, dict):
            if config.learning_rate.type is None:
                raise ValueError("optimizer's learning rate must be LearningRateSchedule type, "
                                 "but the type is not specified, it is None")
            lr_schedule = build_lr(config.learning_rate)
            config.learning_rate = lr_schedule
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.OPTIMIZER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_optim():
    """ register MindSpore builtin optimizer class. """
    for module_name in dir(nn.optim):
        if module_name.startswith('__'):
            continue
        optim = getattr(nn.optim, module_name)
        if inspect.isclass(optim):
            MindFormerRegister.register_cls(
                optim, MindFormerModuleType.OPTIMIZER)


def register_mf_optim():
    """ register MindFormers builtin optimizer class. """
    # adapt huggingface
    MindFormerRegister.register_cls(
        AdamWeightDecay, module_type=MindFormerModuleType.OPTIMIZER, alias="adamw")

    MindFormerRegister.register_cls(
        AdaFactor, module_type=MindFormerModuleType.OPTIMIZER, alias="adafactor")

    MindFormerRegister.register_cls(
        SGD, module_type=MindFormerModuleType.OPTIMIZER, alias="sgd")

    MindFormerRegister.register_cls(
        Adam, module_type=MindFormerModuleType.OPTIMIZER, alias="adam")

    MindFormerRegister.register_cls(
        Adagrad, module_type=MindFormerModuleType.OPTIMIZER, alias="adagrad")

    MindFormerRegister.register_cls(
        FusedAdamWeightDecay, module_type=MindFormerModuleType.OPTIMIZER, alias="fused_adamw")

    MindFormerRegister.register_cls(
        FP32StateAdamWeightDecay, module_type=MindFormerModuleType.OPTIMIZER, alias="fp32_adamw")


register_ms_optim()
register_mf_optim()
