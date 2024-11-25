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
"""Build Model API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from .build_config import build_model_config


def build_model(
        config: dict = None, default_args: dict = None,
        module_type: str = 'models', class_name: str = None, **kwargs):
    r"""Build model For MindFormer.
    Instantiate the model from MindFormerRegister's registry.

    Args:
        config (dict): The task model's config. Default: None.
        default_args (dict): The default argument of model API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'model'.
        class_name (str): The class name of model API. Default: None.

    Return:
        The function instance of model API.

    Examples:
        >>> from mindformers import build_model
        >>> from mindformers.tools.register import MindFormerConfig
        >>> config = MindFormerConfig('configs/vit/run_vit_base_p16_224_100ep.yaml')
        >>> model_from_config = build_model(config.model)
    """
    if config is None and class_name is None:
        return None

    if class_name:
        kwargs["config"] = config
        return MindFormerRegister.get_instance(module_type, class_name, **kwargs)

    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        if default_args is None:
            default_args = {}

        if isinstance(config.model_config, MindFormerConfig):
            model_config = build_model_config(config.model_config, default_args=default_args)
        else:
            model_config = config.model_config

        if model_config is not None:
            if default_args is not None:
                for key, value in default_args.items():
                    model_config.__setattr__(key, value)
                default_args = {}
            default_args.setdefault('config', model_config)

            return MindFormerRegister.get_instance_from_cfg(
                config.arch, MindFormerModuleType.MODELS, default_args=default_args)
        return None
    return None


def build_network(
        config: dict = None, default_args: dict = None):
    """Create the quant/pet network For MindFormer"""
    ckpt_cfg = config.model_config.checkpoint_name_or_path
    pet_config = config.model_config.pet_config
    quant_config = config.model_config.quantization_config
    network = build_model(config, default_args=default_args)
    if quant_config:
        from mindformers.modules.quantizers import AutoQuantizer
        quantizer = AutoQuantizer.from_config(quant_config)
        network = quantizer.preprocess_model(network, config=config)
    if pet_config:
        from mindformers.pet import get_pet_model, is_supported_pet_type
        if is_supported_pet_type(pet_config.pet_type):
            config.model_config.checkpoint_name_or_path = None
        network.checkpoint_name_or_path = ckpt_cfg
        network = get_pet_model(network, pet_config)
    return network


def build_encoder(
        config: dict = None, default_args: dict = None,
        module_type: str = 'encoder', class_name: str = None, **kwargs):
    """Build encoder API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_encoders = config
        if not isinstance(cfg_encoders, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_encoders, MindFormerModuleType.ENCODER, default_args=default_args)
        encoders = []
        for encoder in cfg_encoders:
            encoder_op = MindFormerRegister.get_instance_from_cfg(
                encoder, MindFormerModuleType.ENCODER)
            encoders.append(encoder_op)
        return encoders
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def build_head(
        config: dict = None, default_args: dict = None,
        module_type: str = 'head', class_name: str = None, **kwargs):
    """Build head API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_heads = config
        if not isinstance(cfg_heads, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_heads, MindFormerModuleType.HEAD, default_args=default_args)
        heads = []
        for head in cfg_heads:
            head_op = MindFormerRegister.get_instance_from_cfg(
                head, MindFormerModuleType.HEAD)
            heads.append(head_op)
        return heads
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
