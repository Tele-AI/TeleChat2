# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Build LR Schedule API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from mindformers.core.lr.lr_schedule import ConstantWarmUpLR, CosineWithWarmUpLR, \
    LinearWithWarmUpLR, CosineWithRestartsAndWarmUpLR, PolynomialWithWarmUpLR, CosineAnnealingLR, \
    CosineAnnealingWarmRestarts


def build_lr(
        config: dict = None, default_args: dict = None,
        module_type: str = 'lr', class_name: str = None, **kwargs):
    r"""Build lr For MindFormer.
    Instantiate the lr from MindFormerRegister's registry.

    Args:
        config (dict): The task lr's config. Default: None.
        default_args (dict): The default argument of lr API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'lr'.
        class_name (str): The class name of lr API. Default: None.

    Return:
        The function instance of lr API.

    Examples:
        >>> from mindformers.core.lr import build_lr
        >>> lr_config = {'type': 'CosineDecayLR', 'max_lr': 0.001,
        ...              'min_lr': 0., 'decay_steps':10}
        >>> # 1) use config dict to build lr
        >>> lr_from_config = build_lr(lr_config)
        >>> print(type(lr_from_config))
        <class 'mindspore.nn.learning_rate_schedule.CosineDecayLR'>
        >>> # 2) use class name to build lr
        >>> lr_class_name = build_lr(class_name='CosineDecayLR', max_lr=0.001, min_lr=0., decay_steps=10)
        >>> print(type(lr_class_name))
        <class 'mindspore.nn.learning_rate_schedule.CosineDecayLR'>
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.LR, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_lr():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.learning_rate_schedule):
        if module_name.startswith('__'):
            continue
        lr_schedule = getattr(nn.learning_rate_schedule, module_name)
        if inspect.isclass(lr_schedule):
            MindFormerRegister.register_cls(
                lr_schedule, MindFormerModuleType.LR)


def register_mf_lr():
    """ register MindFormers builtin LR class. """
    # adapt huggingface
    MindFormerRegister.register_cls(
        ConstantWarmUpLR, module_type=MindFormerModuleType.LR, alias="constant_with_warmup")

    MindFormerRegister.register_cls(
        LinearWithWarmUpLR, module_type=MindFormerModuleType.LR, alias="linear")

    MindFormerRegister.register_cls(
        CosineWithWarmUpLR, module_type=MindFormerModuleType.LR, alias="cosine")

    MindFormerRegister.register_cls(
        CosineWithRestartsAndWarmUpLR, module_type=MindFormerModuleType.LR, alias="cosine_with_restarts")

    MindFormerRegister.register_cls(
        PolynomialWithWarmUpLR, module_type=MindFormerModuleType.LR, alias="polynomial")

    MindFormerRegister.register_cls(
        CosineAnnealingLR, module_type=MindFormerModuleType.LR, alias="cosine_annealing")

    MindFormerRegister.register_cls(
        CosineAnnealingWarmRestarts, module_type=MindFormerModuleType.LR, alias="cosine_annealing_warm_restarts")


register_ms_lr()
register_mf_lr()
