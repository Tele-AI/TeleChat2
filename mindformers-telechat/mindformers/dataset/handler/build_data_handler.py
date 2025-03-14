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
"""Build Data Handler API."""

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_data_handler(config, module_type: str = 'data_handler', class_name: str = None, **kwargs):
    r"""Build data handler For MindFormer.
    Instantiate the data handler from MindFormerRegister's registry.

    Args:
        config (dict): The task data handler's config. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'data handler'.
        class_name (str): The class name of data handler API. Default: None.

    Return:
        The function instance of data handler API.

    Examples:
        >>> from mindformers.dataset.handler import build_data_handler
        >>> handler_config = {'type': 'AlpacaInstructDataHandler', 'seq_length': 4096, 'tokenizer_name': 'llama2_7b'}
        >>> # 1) use config dict to build data handler
        >>> data_handler_from_config = build_data_handler(handler_config)
        >>> # 2) use class name to build data handler
        >>> data_handler_class_name = build_data_handler(
        ...     class_name='AlpacaInstructDataHandler', ...)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_cls(
            MindFormerModuleType.DATA_HANDLER, config.type)(config, **kwargs)
    return MindFormerRegister.get_cls(module_type, class_name)(**kwargs)
