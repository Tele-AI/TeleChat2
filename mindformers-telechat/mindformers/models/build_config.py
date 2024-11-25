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

"""
build model config modules
"""
import copy
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_model_config(
        config: dict = None, default_args: dict = None,
        module_type: str = 'config', class_name: str = None,
        **kwargs):
    r"""Build model config For MindFormer.
    Instantiate the model config from MindFormerRegister's registry.

    Args:
        config (dict): The task model config's config. Default: None.
        default_args (dict): The default argument of model config API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'config'.
        class_name (str): The class name of model config API. Default: None.

    Return:
        The function instance of model config API.

    Examples:
        >>> from mindformers import build_model_config
        >>> model_config = {'type': 'ViTConfig'}
        >>> # 1) use config dict to build model
        >>> model_config_from_config = build_model_config(model_config)
        >>> # 2) use class name to build model
        >>> model_config_class_name = build_model_config(class_name='ViTConfig')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        config = copy.deepcopy(config)
        if config.text_config is not None:
            config.text_config = build_model_config(config.text_config)
        if config.vision_config is not None:
            config.vision_config = build_model_config(config.vision_config)
        if config.head_config is not None:
            config.head_config = build_model_config(config.head_config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.CONFIG, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
