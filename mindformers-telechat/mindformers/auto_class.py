# Copyright 2024 Huawei Technologies Co., Ltd
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
AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
"""
import os
import shutil

from mindformers.tools.utils import try_sync_file

from .mindformer_book import MindFormerBook, print_dict
from .models.build_processor import build_processor
from .models.base_config import BaseConfig
from .models.build_model import build_network
from .models.build_config import build_model_config
from .tools.logger import logger
from .tools.register.config import MindFormerConfig


__all__ = ['AutoConfig', 'AutoModel', 'AutoProcessor', 'AutoTokenizer']


class AutoConfig:
    """
    AutoConfig class,
    helps instantiates a config by yaml model name or path.
    If using a model name, the config yaml will be downloaded from obs to ./checkpoint_download dir

    Examples:
        >>> from mindformers.auto_class import AutoConfig
        >>>
        >>> # 1)  instantiates a config by yaml model name
        >>> config_a = AutoConfig.from_pretrained('clip_vit_b_32')
        >>> # 2)  instantiates a config by yaml model path
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        ...                            'configs', 'clip', 'run_clip_vit_b_32_pretrain_flickr8k.yaml')
        >>> config_b = AutoConfig.from_pretrained(config_path)
    """
    _support_list = MindFormerBook.get_config_support_list()
    _model_type = 0
    _model_name = 1

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(yaml_name_or_path)` method."
        )

    @classmethod
    def invalid_yaml_name(cls, yaml_name_or_path):
        """Check whether it is a valid yaml name"""
        if yaml_name_or_path.startswith('mindspore'):
            # Adaptation the name of yaml at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            yaml_name_or_path = yaml_name_or_path.split('/')[cls._model_name]

        if not yaml_name_or_path.split('_')[cls._model_type] in cls._support_list.keys():
            return True

        local_model_type = yaml_name_or_path.split('_')[cls._model_type]
        local_model_list = cls._support_list[local_model_type]
        if not isinstance(local_model_list, dict):
            if yaml_name_or_path in local_model_list:
                return False
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')

        local_model_names = local_model_list.keys()
        if len(yaml_name_or_path.split('_')) <= cls._model_name or \
            not yaml_name_or_path.split('_')[cls._model_name] in local_model_names:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')
        local_model_name = yaml_name_or_path.split('_')[cls._model_name]
        if yaml_name_or_path not in local_model_list[local_model_name]:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}_{local_model_name}\', '
                             f'please select from {local_model_list[local_model_name]}')
        return False

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a config by yaml model name or path.

        Args:
            yaml_name_or_path (str): A supported model name or a path to model config (.yaml),
                the supported model name could be selected from AutoConfig.show_support_list().
                If yaml_name_or_path is model name, it supports model names beginning with mindspore or
                the model name itself, such as "mindspore/vit_base_p16" or "vit_base_p16".
            pretrained_model_name_or_path (Optional[str]): Equal to "yaml_name_or_path",
                if "pretrained_model_name_or_path" is set, "yaml_name_or_path" is useless.

        Returns:
            A model config, which inherited from BaseConfig.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str,"
                            f" but got {type(yaml_name_or_path)}.")

        if os.path.exists(yaml_name_or_path):
            if not yaml_name_or_path.endswith(".yaml"):
                raise ValueError(f"{yaml_name_or_path} should be a .yaml file for model config.")
            config_args = MindFormerConfig(yaml_name_or_path)
            logger.info("the content in %s is used for config building.", yaml_name_or_path)

        elif cls.invalid_yaml_name(yaml_name_or_path):
            raise ValueError(f"{yaml_name_or_path} is not a supported"
                             f" model type or a valid path to model config."
                             f" supported model could be selected from {cls._support_list}.")
        else:
            yaml_name = yaml_name_or_path
            if yaml_name_or_path.startswith('mindspore'):
                # Adaptation the name of yaml at the beginning of mindspore,
                # the relevant file will be downloaded from the Xihe platform.
                # such as "mindspore/vit_base_p16"
                yaml_name = yaml_name_or_path.split('/')[cls._model_name]
                checkpoint_path = os.path.join(MindFormerBook.get_xihe_checkpoint_download_folder(),
                                               yaml_name.split('_')[cls._model_type])
            else:
                # Default the name of yaml,
                # the relevant file will be downloaded from the Obs platform.
                # such as "vit_base_p16"
                checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                               yaml_name_or_path.split('_')[cls._model_type])

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            yaml_file = os.path.join(checkpoint_path, yaml_name + ".yaml")

            def get_default_yaml_file(model_name):
                default_yaml_file = ""
                for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                    if model_name in model_dict:
                        default_yaml_file = model_dict.get(model_name)
                        break
                return default_yaml_file

            if not os.path.exists(yaml_file):
                default_yaml_file = get_default_yaml_file(yaml_name)
                if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
                    shutil.copy(default_yaml_file, yaml_file)
                    logger.info("default yaml config in %s is used.", yaml_file)
                else:
                    raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')
            config_args = MindFormerConfig(yaml_file)
        config_args.model.model_config.update(**kwargs)
        config = build_model_config(config_args.model.model_config)
        MindFormerBook.set_model_config_to_name(id(config), config_args.model.arch.type)
        return config

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list


class AutoModel:
    """
    AutoModel class
    helps instantiates a model by yaml model name, path or config.
    If using a model name,
    the config yaml and checkpoint file will be downloaded from obs to ./checkpoint_download dir

    Examples:
        >>> from mindformers.auto_class import AutoModel
        >>>
        >>> # 1)  input model name, load model and weights
        >>> model_a = AutoModel.from_pretrained('clip_vit_b_32')
        >>> # 2)  input model directory, load model and weights
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(), 'clip')
        >>> model_b = AutoModel.from_pretrained(checkpoint_dir)
        >>> # 3)  input yaml path, load model without weights
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        ...                            'configs', 'clip', 'run_clip_vit_b_32_pretrain_flickr8k.yaml')
        >>> model_c = AutoModel.from_config(config_path)
        >>> # 4)  input config, load model without weights
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('clip_vit_b_32')
        >>> model_d = AutoModel.from_config(config)
    """
    _support_list = MindFormerBook.get_model_support_list()
    _model_type = 0
    _model_name = 1

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_dir)` method "
            "or `AutoModel.from_config(config)` method."
        )

    @classmethod
    def invalid_model_name(cls, pretrained_model_name_or_dir):
        """Check whether it is a valid model name"""
        if pretrained_model_name_or_dir.startswith('mindspore'):
            # Adaptation the name of model at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            pretrained_model_name_or_dir = pretrained_model_name_or_dir.split('/')[cls._model_name]

        if not pretrained_model_name_or_dir.split('_')[cls._model_type] in cls._support_list.keys():
            return True

        local_model_type = pretrained_model_name_or_dir.split('_')[cls._model_type]
        local_model_list = cls._support_list[local_model_type]
        if not isinstance(local_model_list, dict):
            if pretrained_model_name_or_dir in local_model_list:
                return False
            raise ValueError(f'\'{pretrained_model_name_or_dir}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')

        local_model_names = local_model_list.keys()
        if len(pretrained_model_name_or_dir.split('_')) <= cls._model_name or \
            not pretrained_model_name_or_dir.split('_')[cls._model_name] in local_model_names:
            raise ValueError(f'\'{pretrained_model_name_or_dir}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')
        local_model_name = pretrained_model_name_or_dir.split('_')[cls._model_name]
        if pretrained_model_name_or_dir not in local_model_list[local_model_name]:
            raise ValueError(f'\'{pretrained_model_name_or_dir}\' is not supported by '
                             f'\'{local_model_type}_{local_model_name}\', please select from '
                             f'{local_model_list[local_model_name]}')
        return False

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        From config method, which instantiates a Model by config.

        Args:
            config (str, BaseConfig, MindFormerConfig): A model config inherited from BaseConfig,
                or a path to .yaml file for model config, or a model config inherited from MindFormerConfig.

        Returns:
            A model, which inherited from BaseModel.
        """
        if config is None:
            raise ValueError("a model cannot be built from config with config is None.")

        download_checkpoint = kwargs.pop("download_checkpoint", True)

        if isinstance(config, MindFormerConfig):
            config_args = config
        elif isinstance(config, BaseConfig):
            inversed_config = cls._inverse_parse_config(config)
            config_args = cls._wrap_config(inversed_config)
        elif os.path.exists(config) and config.endswith(".yaml"):
            config_args = MindFormerConfig(config)
        else:
            raise ValueError("config should be inherited from BaseConfig,"
                             " or a path to .yaml file for model config.")

        config_args.model.model_config.update(**kwargs)
        if not download_checkpoint:
            config_args.model.model_config.checkpoint_name_or_path = None
        model = build_network(config_args.model)
        logger.info("model built successfully!")
        return model

    @classmethod
    def _inverse_parse_config(cls, config):
        """
        Inverse parse config method, which builds yaml file content for model config.

        Args:
            config (BaseConfig): A model config inherited from BaseConfig.

        Returns:
            A model config, which follows the yaml content.
        """
        if not isinstance(config, BaseConfig):
            return config

        class_name = config.__class__.__name__
        config.update({"type": class_name})

        for key, val in config.items():
            new_val = cls._inverse_parse_config(val)
            config.update({key: new_val})

        return config

    @classmethod
    def _wrap_config(cls, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (BaseConfig): A config processed by _inverse_parse_config function.

        Returns:
            A model config, which has the same content as a yaml file.
        """
        model_name = config.pop("model_name", None)
        if model_name is None:
            model_name = MindFormerBook.get_model_config_to_name().get(id(config), None)

        arch = BaseConfig(type=model_name)
        model = BaseConfig(model_config=config, arch=arch)
        return BaseConfig(model=model)

    @classmethod
    def _get_config_args(cls, pretrained_model_name_or_dir, **kwargs):
        """build config args."""
        is_exist = os.path.exists(pretrained_model_name_or_dir)
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if is_exist:
            if not is_dir:
                raise ValueError(f"{pretrained_model_name_or_dir} is not a directory.")
        else:
            if cls.invalid_model_name(pretrained_model_name_or_dir):
                raise ValueError(f"{pretrained_model_name_or_dir} is not a supported model"
                                 f" type or a valid path to model config. supported model"
                                 f" could be selected from {cls._support_list}.")

        if is_dir:
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".yaml")]
            ckpt_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".ckpt")]
            if not yaml_list or not ckpt_list:
                raise FileNotFoundError(f"there is no yaml file for model config or ckpt file"
                                        f" for model weights in {pretrained_model_name_or_dir}")

            yaml_file = os.path.join(pretrained_model_name_or_dir, yaml_list[cls._model_type])
            ckpt_file = os.path.join(pretrained_model_name_or_dir, ckpt_list[cls._model_type])

            config_args = MindFormerConfig(yaml_file)
            kwargs["checkpoint_name_or_path"] = kwargs.get("checkpoint_name_or_path") \
                if "checkpoint_name_or_path" in kwargs.keys() else ckpt_file
            config_args.model.model_config.update(**kwargs)
            logger.info("model config: %s and checkpoint_name_or_path: %s are used for "
                        "model building.", yaml_file, config_args.model.model_config.checkpoint_name_or_path)
        else:
            pretrained_checkpoint_name = pretrained_model_name_or_dir
            if pretrained_model_name_or_dir.startswith('mindspore'):
                # Adaptation the name of model at the beginning of mindspore,
                # the relevant file will be downloaded from the Xihe platform.
                # such as "mindspore/vit_base_p16"
                pretrained_checkpoint_name = pretrained_model_name_or_dir.split('/')[cls._model_name]
                checkpoint_path = os.path.join(
                    MindFormerBook.get_xihe_checkpoint_download_folder(),
                    pretrained_checkpoint_name.split('_')[cls._model_type])
            else:
                # Default the name of model,
                # the relevant file will be downloaded from the Obs platform.
                # such as "vit_base_p16"
                checkpoint_path = os.path.join(
                    MindFormerBook.get_default_checkpoint_download_folder(),
                    pretrained_model_name_or_dir.split("_")[cls._model_type])

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            yaml_file = os.path.join(checkpoint_path, pretrained_checkpoint_name + ".yaml")

            def get_default_yaml_file(model_name):
                default_yaml_file = ""
                for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                    if model_name in model_dict:
                        default_yaml_file = model_dict.get(model_name)
                        break
                return default_yaml_file

            if not os.path.exists(yaml_file):
                default_yaml_file = get_default_yaml_file(pretrained_checkpoint_name)
                if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
                    shutil.copy(default_yaml_file, yaml_file)
                    logger.info("default yaml config in %s is used.", yaml_file)
                else:
                    raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')
            try_sync_file(yaml_file)
            config_args = MindFormerConfig(yaml_file)
            kwargs["checkpoint_name_or_path"] = kwargs.get("checkpoint_name_or_path") \
                if "checkpoint_name_or_path" in kwargs.keys() else pretrained_model_name_or_dir
            config_args.model.model_config.update(**kwargs)
        return config_args

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir, **kwargs):
        """
        From pretrain method, which instantiates a Model by pretrained model name or path.

        Args:
            pretrained_model_name_or_dir (str): A supported model name or a directory to model checkpoint
                (including .yaml file for config and .ckpt file for weights), the supported model name could be
                selected from AutoModel.show_support_list().
                If pretrained_model_name_or_dir is model name, it supports model names beginning with mindspore or
                the model name itself, such as "mindspore/vit_base_p16" or "vit_base_p16".
            pretrained_model_name_or_path (Optional[str]): Equal to "pretrained_model_name_or_dir",
                if "pretrained_model_name_or_path" is set, "pretrained_model_name_or_dir" is useless.

        Returns:
            A model, which inherited from BaseModel.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        download_checkpoint = kwargs.pop("download_checkpoint", True)
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_dir = pretrained_model_name_or_path

        if not isinstance(pretrained_model_name_or_dir, str):
            raise TypeError(f"pretrained_model_name_or_dir should be a str,"
                            f" but got {type(pretrained_model_name_or_dir)}")
        pretrained_model_name_or_dir = os.path.realpath(pretrained_model_name_or_dir)
        config_args = cls._get_config_args(pretrained_model_name_or_dir, **kwargs)
        if not download_checkpoint:
            config_args.model.model_config.checkpoint_name_or_path = None
        model = build_network(config_args.model)
        cls.default_checkpoint_download_path = model.default_checkpoint_download_path

        logger.info("model built successfully!")
        return model

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list


class AutoProcessor:
    """
    AutoProcessor
    helps instantiates a processor by yaml model name or path.
    If using a model name, the config yaml will be downloaded from obs to ./checkpoint_download dir

    Examples:
        >>> from mindformers.auto_class import AutoProcessor
        >>>
        >>> # 1)  instantiates a processor by yaml model name
        >>> pro_a = AutoProcessor.from_pretrained('clip_vit_b_32')
        >>> # 2)  instantiates a processor by yaml model path
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        ...                            'configs', 'clip', 'run_clip_vit_b_32_pretrain_flickr8k.yaml')
        >>> pro_b = AutoProcessor.from_pretrained(config_path)
    """
    _support_list = MindFormerBook.get_processor_support_list()
    _model_type = 0
    _model_name = 1

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(yaml_name_or_path)` method."
        )

    @classmethod
    def invalid_yaml_name(cls, yaml_name_or_path):
        """Check whether it is a valid yaml name"""

        if yaml_name_or_path.startswith('mindspore'):
            # Adaptation the name of yaml at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            yaml_name_or_path = yaml_name_or_path.split('/')[cls._model_name]

        local_model_type = yaml_name_or_path.split('_')[cls._model_type]
        local_model_list = cls._support_list[local_model_type]
        if not isinstance(local_model_list, dict):
            if yaml_name_or_path in local_model_list:
                return False
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')

        local_model_names = local_model_list.keys()
        if len(yaml_name_or_path.split('_')) <= cls._model_name or \
            not yaml_name_or_path.split('_')[cls._model_name] in local_model_names:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')
        local_model_name = yaml_name_or_path.split('_')[cls._model_name]
        if yaml_name_or_path not in local_model_list[local_model_name]:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}_{local_model_name}\', '
                             f'please select from {local_model_list[local_model_name]}')
        return False

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiated a processor by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported yaml name or a path to .yaml file,
                the supported model name could be selected from .show_support_list().
                If yaml_name_or_path is model name, it supports model names beginning with mindspore or
                the model name itself, such as "mindspore/vit_base_p16" or "vit_base_p16".
            pretrained_model_name_or_path (Optional[str]): Equal to "yaml_name_or_path",
                if "pretrained_model_name_or_path" is set, "yaml_name_or_path" is useless.

        Returns:
            A processor which inherited from ProcessorMixin.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str, but got {type(yaml_name_or_path)}")

        is_exist = os.path.exists(yaml_name_or_path)
        model_name = yaml_name_or_path.split('/')[cls._model_name].split("_")[cls._model_type] \
            if yaml_name_or_path.startswith('mindspore') else yaml_name_or_path.split("_")[cls._model_type]
        if not is_exist and model_name not in cls._support_list.keys():
            raise ValueError(f'{yaml_name_or_path} does not exist,'
                             f' and it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')

        if is_exist:
            yaml_name_or_path = os.path.realpath(yaml_name_or_path)
            logger.info("config in %s is used for auto processor building.", yaml_name_or_path)
            if os.path.isdir(yaml_name_or_path):
                yaml_list = [file for file in os.listdir(yaml_name_or_path) if file.endswith(".yaml")]
                yaml_name = os.path.join(yaml_name_or_path, yaml_list[cls._model_type])
                config_args = MindFormerConfig(yaml_name)
            else:
                config_args = MindFormerConfig(yaml_name_or_path)
        else:
            yaml_name = yaml_name_or_path
            if not cls.invalid_yaml_name(yaml_name_or_path):
                if yaml_name_or_path.startswith('mindspore'):
                    # Adaptation the name of yaml at the beginning of mindspore,
                    # the relevant file will be downloaded from the Xihe platform.
                    # such as "mindspore/vit_base_p16"
                    yaml_name = yaml_name_or_path.split('/')[cls._model_name]
                    checkpoint_path = os.path.join(MindFormerBook.get_xihe_checkpoint_download_folder(),
                                                   yaml_name.split('_')[cls._model_type])
                else:
                    # Default the name of yaml,
                    # the relevant file will be downloaded from the Obs platform.
                    # such as "vit_base_p16"
                    checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                                   yaml_name_or_path.split('_')[cls._model_type])
            else:
                raise ValueError(f'{yaml_name_or_path} does not exist,'
                                 f' or it is not supported by {cls.__name__}.'
                                 f' please select from {cls._support_list}.')

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            yaml_file = os.path.join(checkpoint_path, yaml_name + ".yaml")

            def get_default_yaml_file(model_name):
                default_yaml_file = ""
                for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                    if model_name in model_dict:
                        default_yaml_file = model_dict.get(model_name)
                        break
                return default_yaml_file

            if not os.path.exists(yaml_file):
                default_yaml_file = get_default_yaml_file(yaml_name)
                if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
                    shutil.copy(default_yaml_file, yaml_file)
                    logger.info("default yaml config in %s is used.", yaml_file)
                else:
                    raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')
            config_args = MindFormerConfig(yaml_file)

        lib_path = yaml_name_or_path
        if not os.path.isdir(lib_path):
            lib_path = None
        processor = build_processor(config_args.processor, lib_path=lib_path)
        logger.info("processor built successfully!")
        return processor

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list


class AutoTokenizer:
    """
    Load the tokenizer according to the `yaml_name_or_path`. It supports the following situations
    1. `yaml_name_or_path` is the model name.
    2. `yaml_name_or_path` is the path to the downloaded files.

    Examples:
        >>> from mindformers.auto_class import AutoTokenizer
        >>>
        >>> # 1)  instantiates a tokenizer by the model name
        >>> tokenizer_a = AutoTokenizer.from_pretrained("clip_vit_b_32")
        >>> # 2)  instantiates a tokenizer by the path to the downloaded files.
        >>> from mindformers.models.clip.clip_tokenizer import CLIPTokenizer
        >>> clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        >>> clip_tokenizer.save_pretrained(path_saved)
        >>> restore_tokenizer = AutoTokenizer.from_pretrained(path_saved)
    """
    _support_list = MindFormerBook.get_tokenizer_support_list()
    _model_type = 0
    _model_name = 1

    @classmethod
    def invalid_yaml_name(cls, yaml_name_or_path):
        """Check whether it is a valid yaml name"""
        if yaml_name_or_path.startswith('mindspore'):
            # Adaptation the name of yaml at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            yaml_name_or_path = yaml_name_or_path.split('/')[cls._model_name]

        if not yaml_name_or_path.split('_')[cls._model_type] in cls._support_list.keys():
            return True

        local_model_type = yaml_name_or_path.split('_')[cls._model_type]
        local_model_list = cls._support_list[local_model_type]
        if not isinstance(local_model_list, dict):
            if yaml_name_or_path in local_model_list:
                return False
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')

        local_model_names = local_model_list.keys()
        if len(yaml_name_or_path.split('_')) <= cls._model_name or \
            not yaml_name_or_path.split('_')[cls._model_name] in local_model_names:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')
        local_model_name = yaml_name_or_path.split('_')[cls._model_name]
        if yaml_name_or_path not in local_model_list[local_model_name]:
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}_{local_model_name}\', '
                             f'please select from {local_model_list[local_model_name]}')
        return False

    @classmethod
    def _get_class_name_from_yaml(cls, yaml_name_or_path):
        """
        Try to find the yaml from the given path
        Args:
            yaml_name_or_path (str): The directory of the config yaml

        Returns:
            The class name of the tokenizer in the config yaml.
        """
        is_exist = os.path.exists(yaml_name_or_path)
        is_dir = os.path.isdir(yaml_name_or_path)
        is_file = os.path.isfile(yaml_name_or_path)
        if not is_file:
            if not is_exist:
                raise ValueError(f"{yaml_name_or_path} does not exist, Please pass a valid the directory.")
            if not is_dir:
                raise ValueError(f"{yaml_name_or_path} is not a directory. You should pass the directory.")
            # If passed a directory, load the file from the yaml files
            yaml_list = [file for file in os.listdir(yaml_name_or_path) if file.endswith(".yaml")]
            if not yaml_list:
                return None
            yaml_file = os.path.join(yaml_name_or_path, yaml_list[cls._model_type])
        else:
            yaml_file = yaml_name_or_path
        logger.info("Config in the yaml file %s are used for tokenizer building.", yaml_file)
        config = MindFormerConfig(yaml_file)

        class_name = None
        if config and 'processor' in config and 'tokenizer' in config['processor'] \
                and 'type' in config['processor']['tokenizer']:
            class_name = config['processor']['tokenizer'].pop('type', None)
            logger.info("Load the tokenizer name %s from the %s", class_name, yaml_name_or_path)

        return class_name

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a tokenizer by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported yaml name or a path to .yaml file,
                the supported model name could be selected from .show_support_list().
                If yaml_name_or_path is model name, it supports model names beginning with mindspore or
                the model name itself, such as "mindspore/clip_vit_b_32" or "clip_vit_b_32".
            pretrained_model_name_or_path (Optional[str]): Equal to "yaml_name_or_path",
                if "pretrained_model_name_or_path" is set, "yaml_name_or_path" is useless.

        Returns:
            A tokenizer which inherited from PretrainedTokenizer.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        from . import MindFormerRegister
        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str, but got {type(yaml_name_or_path)}")

        # Try to load from the remote
        if not cls.invalid_yaml_name(yaml_name_or_path):
            # Should download the files from the remote storage
            yaml_name = yaml_name_or_path
            if yaml_name_or_path.startswith('mindspore'):
                # Adaptation the name of yaml at the beginning of mindspore,
                # the relevant file will be downloaded from the Xihe platform.
                # such as "mindspore/vit_base_p16"
                yaml_name = yaml_name_or_path.split('/')[cls._model_name]
                checkpoint_path = os.path.join(MindFormerBook.get_xihe_checkpoint_download_folder(),
                                               yaml_name.split('_')[cls._model_type])
            else:
                # Default the name of yaml,
                # the relevant file will be downloaded from the Obs platform.
                # such as "vit_base_p16"
                checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                               yaml_name_or_path.split('_')[cls._model_type])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            yaml_file = os.path.join(checkpoint_path, yaml_name + ".yaml")

            def get_default_yaml_file(model_name):
                default_yaml_file = ""
                for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                    if model_name in model_dict:
                        default_yaml_file = model_dict.get(model_name)
                        break
                return default_yaml_file

            if not os.path.exists(yaml_file):
                default_yaml_file = get_default_yaml_file(yaml_name)
                if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
                    shutil.copy(default_yaml_file, yaml_file)
                    logger.info("default yaml config in %s is used.", yaml_file)
                else:
                    raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')
            class_name = cls._get_class_name_from_yaml(yaml_file)
        elif os.path.isdir(yaml_name_or_path):
            yaml_name_or_path = os.path.realpath(yaml_name_or_path)
            class_name = cls._get_class_name_from_yaml(yaml_name_or_path)
            if not class_name:
                raise ValueError(f"The file `model_name.yaml` should exist in the path "
                                 f"{yaml_name_or_path}/model_name.yaml and should have `processor` configs like "
                                 f"configs/gpt2/run_gpt2.yaml, but not found.")
        else:
            raise FileNotFoundError(f"Tokenizer type `{yaml_name_or_path}` does not exist. "
                                    f"Use `{cls.__name__}.show_support_list()` to check the supported tokenizer. "
                                    f"Or make sure the `{yaml_name_or_path}` is a directory.")

        dynamic_class = MindFormerRegister.get_cls(module_type='tokenizer', class_name=class_name)
        instanced_class = dynamic_class.from_pretrained(yaml_name_or_path, **kwargs)
        logger.info("%s Tokenizer built successfully!", instanced_class.__class__.__name__)
        return instanced_class

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list
