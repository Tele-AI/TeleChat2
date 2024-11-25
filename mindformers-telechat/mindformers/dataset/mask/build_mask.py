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
"""Build Mask API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_mask(
        config: dict = None, default_args: dict = None,
        module_type: str = 'mask_policy', class_name: str = None, **kwargs):
    r"""Build mask For MindFormer.
    Instantiate the mask from MindFormerRegister's registry.

    Args:
        config (dict): The task mask's config. Default: None.
        default_args (dict): The default argument of mask API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'mask'.
        class_name (str): The class name of mask API. Default: None.

    Return:
        The function instance of mask API.

    Examples:
        >>> from mindformers import build_mask
        >>> mask_config = {'type': 'SimMask', 'input_size': 224}
        >>> # 1) use config dict to build mask
        >>> mask_from_config = build_mask(mask_config)
        >>> # 2) use class name to build mask
        >>> mask_class_name = build_mask(class_name='SimMask', input_size=224)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.MASK_POLICY, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
