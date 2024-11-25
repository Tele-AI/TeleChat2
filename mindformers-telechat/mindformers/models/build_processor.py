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

"""Build Processor API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from .build_tokenizer import build_tokenizer


def build_processor(
        config: dict = None, default_args: dict = None,
        module_type: str = 'processor', class_name: str = None, **kwargs):
    r"""Build processor For MindFormer.
    Instantiate the processor from MindFormerRegister's registry.

    Args:
        config (dict): The task processor's config. Default: None.
        default_args (dict): The default argument of processor API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'processor'.
        class_name (str): The class name of processor API. Default: None.

    Return:
        The function instance of processor API.

    Examples:
        >>> from mindformers import build_processor
        >>> from mindformers.tools.register import MindFormerConfig
        >>> config = MindFormerConfig('configs/vit/run_vit_base_p16_224_100ep.yaml')
        >>> processor_from_config = build_processor(config.processor)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        if config.image_processor is not None:
            config.image_processor = build_processor(config.image_processor)

        if config.tokenizer is not None:
            config.tokenizer = build_tokenizer(config.tokenizer, **kwargs)

        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.PROCESSOR, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
