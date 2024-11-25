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
"""Build Wrapper."""

import inspect

import mindspore as ms
from mindspore import nn, Tensor

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from .wrapper import MFTrainOneStepCell, MFPipelineWithLossScaleCell


WRAPPERS_MINDFORMERS_DEFINED = ["MFTrainOneStepCell", "MFPipelineWithLossScaleCell"]


def build_wrapper(config: dict = None, default_args: dict = None,
                  module_type: str = 'wrapper', class_name: str = None, **kwargs):
    r"""Build Wrapper For MindFormer.
    Instantiate the wrapper from MindFormerRegister's registry.

    Args:
        config (dict): The task wrapper's config. Default: None.
        default_args (dict): The default argument of wrapper API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'wrapper'.
        class_name (str): The class name of wrapper API. Default: None.

    Return:
        The function instance of wrapper API.

    Examples:
        >>> from mindformers import build_wrapper
        >>> wrapper_config = {'type': 'TrainOneStepCell', 'sens': 1024}
        >>> # 1) use config dict to build wrapper
        >>> wrapper_from_config = build_wrapper(wrapper_config)
        >>> # 2) use class name to build wrapper
        >>> wrapper_class_name = build_wrapper(class_name='TrainOneStepCell', sens=1024)
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        if config.type not in WRAPPERS_MINDFORMERS_DEFINED:
            if default_args and "parallel_config" in default_args:
                del default_args["parallel_config"]
        if config.scale_sense is not None:
            if isinstance(config.scale_sense, dict) and config.scale_sense.type is not None:
                config.scale_sense = build_wrapper(config.scale_sense)
            elif isinstance(config.scale_sense, int):
                config.scale_sense = Tensor(config.scale_sense, ms.float32)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.WRAPPER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_wrap():
    """ register MindSpore builtin wrapper class. """
    for module_name in dir(nn.wrap):
        if module_name.startswith('__'):
            continue
        wrap = getattr(nn.wrap, module_name)
        if inspect.isclass(wrap):
            MindFormerRegister.register_cls(
                wrap, MindFormerModuleType.WRAPPER)


def register_mf_wrapper():
    """ register MindFormers builtin wrapper class. """
    # Support built-in model wrapper of MindFormers.
    MindFormerRegister.register_cls(
        nn.wrap.TrainOneStepCell,
        module_type=MindFormerModuleType.WRAPPER, alias="wrapper")

    MindFormerRegister.register_cls(
        nn.wrap.TrainOneStepWithLossScaleCell,
        module_type=MindFormerModuleType.WRAPPER, alias="loss_scale_wrapper")

    MindFormerRegister.register_cls(
        MFTrainOneStepCell,
        module_type=MindFormerModuleType.WRAPPER, alias="mf_wrapper")

    MindFormerRegister.register_cls(
        MFPipelineWithLossScaleCell,
        module_type=MindFormerModuleType.WRAPPER, alias="pipeline_wrapper")


register_ms_wrap()
register_mf_wrapper()
