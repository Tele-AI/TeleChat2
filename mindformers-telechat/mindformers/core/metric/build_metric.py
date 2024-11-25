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
"""Build Metric API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_metric(
        config: dict = None, default_args: dict = None,
        module_type: str = 'metric', class_name: str = None, **kwargs):
    r"""Build metric For MindFormer.
    Instantiate the metric from MindFormerRegister's registry.

    Args:
        config (dict, list): The task metric's config. Default: None.
        default_args (dict): The default argument of metric API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'metric'.
        class_name (str): The class name of metric API. Default: None.

    Return:
        The function instance of metric API.

    Examples:
        >>> from mindformers import build_metric
        >>> metric_config = {'type': 'Accuracy', 'eval_type': 'classification'}
        >>> # 1) use config dict to build metric
        >>> metric_from_config = build_metric(metric_config)
        >>> # 2) use class name to build metric
        >>> metric_class_name = build_metric(class_name='Accuracy', eval_type='classification')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.METRIC, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_mt():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.metrics):
        if module_name.startswith('__'):
            continue
        ms_metric = getattr(nn.metrics, module_name)
        if inspect.isclass(ms_metric):
            MindFormerRegister.register_cls(
                ms_metric, MindFormerModuleType.METRIC)


register_ms_mt()
