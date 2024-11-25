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
"""Build Transforms API."""
import inspect

from mindspore.dataset import transforms as tf
from mindspore.dataset import vision as vs

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_transforms(
        config: [dict, list] = None, default_args: dict = None,
        module_type: str = 'transforms', class_name: str = None, **kwargs):
    r"""Build transform For MindFormer.
    Instantiate the transform from MindFormerRegister's registry.

    Args:
        config (dict, list): The task transform's config. Default: None.
        default_args (dict): The default argument of transform API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'transform'.
        class_name (str): The class name of transform API. Default: None.

    Return:
        The function instance of transform API.

    Examples:
        >>> from mindformers import build_transforms
        >>> transform_config = [{'type': 'Decode'}, {'type': 'Resize', 'size': 256}]
        >>> # 1) use config dict to build transform
        >>> transform_from_config = build_transforms(transform_config)
        >>> # 2) use class name to build transform
        >>> transform_class_name = build_transforms(class_name='Decode')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        cfg_transforms = config
        if not isinstance(cfg_transforms, list):
            cfg_transforms = [cfg_transforms]

        if len(cfg_transforms) > 1:
            default_args = None

        transforms = []
        for transform in cfg_transforms:
            if hasattr(transform, "model_transform_template") and \
                    getattr(transform, "model_transform_template") is not None:
                transform.model_transform_template = build_transforms(transform.model_transform_template,
                                                                      default_args=default_args)
            transform_op = MindFormerRegister.get_instance_from_cfg(
                transform, MindFormerModuleType.TRANSFORMS, default_args=default_args)
            transforms.append(transform_op)
        return transforms if len(transforms) > 1 else transforms[0]
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_transforms():
    """register MindSpore builtin transforms class."""
    for module_name in set(dir(vs.transforms) + dir(tf.transforms)):
        if module_name.startswith('__'):
            continue

        transforms = getattr(vs.transforms, module_name, None) \
            if getattr(vs.transforms, module_name, None)\
            else getattr(tf.transforms, module_name)
        if inspect.isclass(transforms):
            MindFormerRegister.register_cls(transforms,
                                            MindFormerModuleType.TRANSFORMS)


register_ms_transforms()
