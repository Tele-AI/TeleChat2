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
"""Build Callback API."""
import inspect

from mindspore.train import callback

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_callback(
        config: [dict, list] = None, default_args: dict = None,
        module_type: str = 'callback', class_name: str = None, **kwargs):
    r"""Build callback For MindFormer.
    Instantiate the callback from MindFormerRegister's registry.

    Args:
        config (dict, list): The task callback's config. Default: None.
        default_args (dict): The default argument of callback API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'callback'.
        class_name (str): The class name of callback API. Default: None.

    Return:
        The function instance of callback API.

    Examples:
        >>> from mindformers import build_callback
        >>> callback_config = [{'type': 'MFLossMonitor'}, {'type': 'SummaryMonitor'}]
        >>> # 1) use config dict to build callback
        >>> callback_from_config = build_callback(callback_config)
        >>> # 2) use class name to build callback
        >>> callback_class_name = build_callback(class_name='MFLossMonitor')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        cfg_callback = config
        if not isinstance(cfg_callback, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_callback, MindFormerModuleType.CALLBACK, default_args=default_args)
        callbacks = []
        for callback_type in cfg_callback:
            if callback_type.get('type') == 'MFLossMonitor':
                callback_op = MindFormerRegister.get_instance_from_cfg(
                    callback_type, MindFormerModuleType.CALLBACK, default_args)
            else:
                callback_op = MindFormerRegister.get_instance_from_cfg(
                    callback_type, MindFormerModuleType.CALLBACK)
            callbacks.append(callback_op)
        return callbacks
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_cb():
    """ register MindSpore builtin LR class. """
    for module_name in dir(callback):
        if module_name.startswith('__'):
            continue
        monitor = getattr(callback, module_name)
        if inspect.isclass(monitor):
            MindFormerRegister.register_cls(
                monitor, MindFormerModuleType.CALLBACK)


register_ms_cb()
