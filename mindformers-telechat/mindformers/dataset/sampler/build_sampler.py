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
"""Build Sampler API."""
import inspect

from mindspore.dataset import samplers as sp

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_sampler(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset_sampler', class_name: str = None, **kwargs):
    r"""Build sampler For MindFormer.
    Instantiate the sampler from MindFormerRegister's registry.

    Args:
        config (dict): The task sampler's config. Default: None.
        default_args (dict): The default argument of sampler API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'sampler'.
        class_name (str): The class name of sampler API. Default: None.

    Return:
        The function instance of sampler API.

    Examples:
        >>> from mindformers import build_sampler
        >>> sampler_config = {'type': 'RandomSampler', 'replacement': False}
        >>> # 1) use config dict to build sampler
        >>> sampler_from_config = build_sampler(sampler_config)
        >>> # 2) use class name to build sampler
        >>> sampler_class_name = build_sampler(class_name='RandomSampler', replacement=False)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET_SAMPLER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_samplers():
    """ register MindSpore builtin transforms class. """
    for module_name in dir(sp):
        if module_name.startswith('__'):
            continue

        samplers = getattr(sp, module_name)
        if inspect.isclass(samplers):
            MindFormerRegister.register_cls(samplers, MindFormerModuleType.DATASET_SAMPLER)


register_ms_samplers()
