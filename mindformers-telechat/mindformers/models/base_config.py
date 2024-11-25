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
BaseConfig class,
which is all model configs' base class
"""
import os
import shutil

import yaml

from ..mindformer_book import MindFormerBook
from ..mindformer_book import print_path_or_list
from ..tools.logger import logger
from ..tools.register.config import MindFormerConfig
from ..models.build_config import build_model_config


class BaseConfig(dict):
    """
    Base Config for all models' config

    Examples:
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> from mindformers.models.base_config import BaseConfig
        >>> class MyConfig(BaseConfig):
        ...     _support_list = MindFormerBook.get_config_support_list()['my_model']
        ...
        ...     def __init__(self,
        ...                  data_size: int = 32,
        ...                  net_size: list = [1, 2, 3],
        ...                  loss_type: str = "MyLoss",
        ...                  checkpoint_name_or_path: str = '',
        ...                  **kwargs):
        ...         self.data_size = data_size
        ...         self.net_size = net_size
        ...         loss_type = loss_type
        ...         self.checkpoint_name_or_path = checkpoint_name_or_path
        ...         super(MyConfig, self).__init__(**kwargs)
        ...
        >>> mynet = MyModel(MyConfig)
        >>> output = mynet(input)
    """
    _support_list = []
    _model_type = 0
    _model_name = 1

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to_dict(self):
        """
        for yaml dump,
        transform from Config to a strict dict class
        """
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, BaseConfig):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a config by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported model name or a path to model config (.yaml),
                the supported model name could be selected from AutoConfig.show_support_list().
                If yaml_name_or_path is model name,
                it supports model names beginning with mindspore or the model name itself,
                such as "mindspore/vit_base_p16" or "vit_base_p16".
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
                raise ValueError(f"{yaml_name_or_path} should be a .yaml file for model"
                                 " config.")

            config_args = MindFormerConfig(yaml_name_or_path)
            logger.info("the content in %s is used for"
                        " config building.", yaml_name_or_path)
        elif yaml_name_or_path not in cls._support_list:
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

    def save_pretrained(self, save_directory=None, save_name="mindspore_model"):
        """
        Save_pretrained.

        Args:
            save_directory (str): a directory to save config yaml

            save_name (str): the name of save files.
        """
        if save_directory is None:
            save_directory = MindFormerBook.get_default_checkpoint_save_folder()

        if not isinstance(save_directory, str) or not isinstance(save_name, str):
            raise TypeError(f"save_directory and save_name should be a str,"
                            f" but got {type(save_directory)} and {type(save_name)}.")

        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        save_path = os.path.join(save_directory, save_name + ".yaml")

        parsed_config, removed_list = self._inverse_parse_config()
        wraped_config = self._wrap_config(parsed_config)
        for key, val in removed_list:
            self[key] = val
        self.remove_type()

        meraged_dict = {}
        if os.path.exists(save_path):
            with open(save_path, 'r') as file_reader:
                meraged_dict = yaml.safe_load(file_reader.read())
            file_reader.close()
        meraged_dict.update(wraped_config)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(save_path, flags_, 0o750), 'w') as file_pointer:
            file_pointer.write(yaml.dump(meraged_dict))
        logger.info("config saved successfully!")

    def remove_type(self):
        """remove type caused by save’"""
        if isinstance(self, BaseConfig):
            self.pop("type")

        for key, val in self.items():
            if isinstance(val, BaseConfig):
                val.pop("type")
                self.update({key: val})

    def inverse_parse_config(self):
        """inverse_parse_config"""
        val, _ = self._inverse_parse_config()
        return val

    def _inverse_parse_config(self):
        """
        Inverse parse config method, which builds yaml file content for model config.

        Returns:
            A model config, which follows the yaml content.
        """
        self.update({"type": self.__class__.__name__})
        removed_list = []

        for key, val in self.items():
            if isinstance(val, BaseConfig):
                val = val.inverse_parse_config()
            elif not isinstance(val, (str, int, float, bool)):
                removed_list.append((key, val))
                continue
            self.update({key: val})

        for key, _ in removed_list:
            self.pop(key)
        return self, removed_list

    def _wrap_config(self, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (BaseConfig): a config processed by _inverse_parse_config function.

        Returns:
            A (config) dict for yaml.dump.
        """
        model_name = self.pop("model_name", None)
        if model_name is None:
            model_name = MindFormerBook.get_model_config_to_name().get(id(config), None)

        return {"model": {"model_config": config.to_dict(), "arch": {"type": model_name}}}

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list
