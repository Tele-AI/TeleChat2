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
BaseModel
"""
from typing import Optional
import os
import shutil
import yaml

import mindspore as ms
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .build_model import build_network
from ..generation import GenerationMixin
from ..mindformer_book import MindFormerBook, print_path_or_list
from .configuration_utils import PretrainedConfig
from ..tools.register import MindFormerConfig, DictConfig
from ..tools.download_tools import download_with_progress_bar
from ..tools.logger import logger
from ..tools.utils import try_sync_file, replace_tk_to_mindpet

IGNORE_KEYS = ["_name_or_path"]


class BaseModel(nn.Cell, GenerationMixin):
    """
    The base model that contains the class method `from_pretained` and `save_pretrained`, any new model that should
    inherit the class.

    Note:
        GenerationMixin provides the method `generate` that enable the generation for nlp models.

    Args:
        config(PretrainedConfig): The model configuration that inherits the `PretrainedConfig`.
    """
    _support_list = []
    _model_type = 0
    _model_name = 1

    def __init__(self, config: PretrainedConfig, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.config = config
        self.default_checkpoint_download_path = None

    def load_checkpoint(self, config):
        """
        load checkpoint for models.

        Args:
            config (ModelConfig): a model config instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        if checkpoint_name_or_path:
            if not isinstance(checkpoint_name_or_path, str):
                raise TypeError(f"checkpoint_name_or_path should be a str,"
                                f" but got {type(checkpoint_name_or_path)}")

            if os.path.exists(checkpoint_name_or_path):
                param = load_checkpoint(checkpoint_name_or_path)
                ckpt_file = checkpoint_name_or_path
            elif checkpoint_name_or_path not in self._support_list:
                raise ValueError(f"{checkpoint_name_or_path} is not a supported default model"
                                 f" or a valid path to checkpoint,"
                                 f" please select from {self._support_list}.")
            else:
                checkpoint_name = checkpoint_name_or_path
                if checkpoint_name_or_path.startswith('mindspore'):
                    # Adaptation the name of checkpoint at the beginning of mindspore,
                    # the relevant file will be downloaded from the Xihe platform.
                    # such as "mindspore/vit_base_p16"
                    checkpoint_name = checkpoint_name_or_path.split('/')[self._model_name]
                    default_checkpoint_download_folder = os.path.join(
                        MindFormerBook.get_xihe_checkpoint_download_folder(),
                        checkpoint_name.split('_')[self._model_type])
                else:
                    # Default the name of checkpoint,
                    # the relevant file will be downloaded from the Obs platform.
                    # such as "vit_base_p16"
                    default_checkpoint_download_folder = os.path.join(
                        MindFormerBook.get_default_checkpoint_download_folder(),
                        checkpoint_name_or_path.split("_")[self._model_type])

                if not os.path.exists(default_checkpoint_download_folder):
                    os.makedirs(default_checkpoint_download_folder, exist_ok=True)

                ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name + ".ckpt")
                if not os.path.exists(ckpt_file):
                    url = MindFormerBook.get_model_ckpt_url_list()[checkpoint_name_or_path][self._model_type]
                    succeed = download_with_progress_bar(url, ckpt_file)
                    if not succeed:
                        logger.info("checkpoint download failed, and pretrained weights are unloaded.")
                        return
                try_sync_file(ckpt_file)
                self.default_checkpoint_download_path = ckpt_file
                logger.info("start to read the ckpt file: %s", os.path.getsize(ckpt_file))
                param = load_checkpoint(ckpt_file)

            param = self.fuse_weight_from_ckpt(param)
            param = replace_tk_to_mindpet(param)
            load_param_into_net(self, param)
            logger.info("weights in %s are loaded", ckpt_file)
        else:
            logger.info("model built, but weights is unloaded, since the config has no"
                        " checkpoint_name_or_path attribute or"
                        " checkpoint_name_or_path is None.")

    def save_pretrained(self,
                        save_directory: Optional[str] = None,
                        save_name: str = "mindspore_model"):
        """
        Save the model weight and configuration file.
        (only supports standalone mode, and distribute mode waits for developing)

        Args:
            save_directory(str): a directory to save the model weight and configuration.
                If None, the directory will be  `./checkpoint_save`, which can be obtained by the
                `MindFormerBook.get_default_checkpoint_save_folder()`. If set, the directory will be what is set.
            save_name(str): the name of saved files, including model weight and configuration file.
                Default mindspore_model.

        Examples:
            >>> import os
            >>> from mindformers import T5ForConditionalGeneration, MindFormerBook
            >>> net = T5ForConditionalGeneration.from_pretrained('t5_small')
            >>> net.save_pretrained()
            >>> output_path = MindFormerBook.get_default_checkpoint_save_folder()
            >>> print(os.listdir(output_path))
            ['mindspore_model.yaml', 'mindspore_model.ckpt']

        """
        if save_directory is None:
            save_directory = MindFormerBook.get_default_checkpoint_save_folder()

        if not isinstance(save_directory, str) or not isinstance(save_name, str):
            raise TypeError(f"save_directory and save_name should be a str,"
                            f" but got {type(save_directory)} and {type(save_name)}.")

        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        checkpoint_path = os.path.join(save_directory, save_name + '.ckpt')
        config_path = os.path.join(save_directory, save_name + '.yaml')

        ms.save_checkpoint(self, checkpoint_path)
        if self.config is None:
            # A model should have "config" attribute for model save.
            raise AttributeError("the model has no config attribute.")

        parsed_config, remove_list = self._inverse_parse_config(self.config)
        wraped_config = self._wrap_config(parsed_config)
        for key, val in remove_list:
            self.config.__dict__[key] = val
        self.remove_type(self.config)

        meraged_dict = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file_reader:
                meraged_dict = yaml.safe_load(file_reader.read())
            file_reader.close()
        meraged_dict.update(wraped_config)

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(config_path, flags_, 0o750), 'w') as file_pointer:
            file_pointer.write(yaml.dump(meraged_dict))
        file_pointer.close()
        logger.info("model saved successfully!")

    def remove_type(self, config):
        """remove type caused by save"""
        if isinstance(config, PretrainedConfig):
            config.__dict__.pop("type")

        for key, val in config.__dict__.items():
            if isinstance(val, PretrainedConfig):
                val.__dict__.pop("type")
                config.__dict__.update({key: val})

    def _inverse_parse_config(self, config):
        """
        Inverse parse config method, which builds yaml file content for model config.

        Args:
            config (PretrainedConfig): a model config inherited from PretrainedConfig.

        Returns:
            A model config, which follows the yaml content.
        """
        config.__dict__.update({"type": config.__class__.__name__})
        removed_list = []

        for key, val in config.__dict__.items():
            if isinstance(val, PretrainedConfig):
                val = val.inverse_parse_config()
            elif not isinstance(val, (str, int, float, bool, DictConfig)) or key in IGNORE_KEYS:
                removed_list.append((key, val))
                continue
            config.__dict__.update({key: val})

        for key, _ in removed_list:
            config.__dict__.pop(key)
        return config, removed_list

    def _wrap_config(self, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (PretrainedConfig): a config processed by _inverse_parse_config function.

        Returns:
            A (config) dict for yaml.dump.
        """
        model_name = self.__class__.__name__
        return {"model": {"model_config": config.to_dict(), "arch": {"type": model_name}}}

    def fuse_weight_from_ckpt(self, ckpt_dict):
        """
        Fuse weight function, which fuse the weight and update parameter dictionary before loading weight into network.

        Args:
            ckpt_dict (dict): dictionary containing weights.

        Returns:
            ckpt_dict (dict) after fusing.
        """
        return ckpt_dict

    @classmethod
    def _get_config_args(cls, pretrained_model_name_or_dir, **kwargs):
        """build config args."""
        is_exist = os.path.exists(pretrained_model_name_or_dir)
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if not is_exist and (pretrained_model_name_or_dir not in cls._support_list):
            raise ValueError(f'{pretrained_model_name_or_dir} does not exist,'
                             f' or it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')

        if is_exist and not is_dir:
            raise ValueError(f"{pretrained_model_name_or_dir} is not a directory.")

        if is_dir:
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".yaml")]
            ckpt_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".ckpt")]
            if not yaml_list or not ckpt_list:
                raise FileNotFoundError(f"there is no yaml file for model config or ckpt file "
                                        f"for model weights in {pretrained_model_name_or_dir}.")
            yaml_file = os.path.join(pretrained_model_name_or_dir, yaml_list[cls._model_type])
            ckpt_file = os.path.join(pretrained_model_name_or_dir, ckpt_list[cls._model_type])

            config_args = MindFormerConfig(yaml_file)
            kwargs["checkpoint_name_or_path"] = kwargs.get("checkpoint_name_or_path") \
                if "checkpoint_name_or_path" in kwargs.keys() else ckpt_file
            config_args.model.model_config.update(**kwargs)
            logger.info("model config: %s and checkpoint_name_or_path: %s are used for "
                        "model building.", yaml_file, config_args.model.model_config.checkpoint_name_or_path)
        else:
            pretrained_model_name = pretrained_model_name_or_dir
            if pretrained_model_name_or_dir.startswith('mindspore'):
                # Adaptation the name of pretrained model at the beginning of mindspore,
                # the relevant file will be downloaded from the Xihe platform.
                # such as "mindspore/vit_base_p16"
                pretrained_model_name = pretrained_model_name.split('/')[cls._model_name]
                checkpoint_path = os.path.join(MindFormerBook.get_xihe_checkpoint_download_folder(),
                                               pretrained_model_name.split('_')[cls._model_type])
            else:
                # Default the name of pretrained model,
                # the relevant file will be downloaded from the Obs platform.
                # such as "vit_base_p16"
                checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                               pretrained_model_name.split('_')[cls._model_type])

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            yaml_file = os.path.join(checkpoint_path, pretrained_model_name + ".yaml")

            def get_default_yaml_file(model_name):
                default_yaml_file = ""
                for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                    if model_name in model_dict:
                        default_yaml_file = model_dict.get(model_name)
                        break
                return default_yaml_file

            if not os.path.exists(yaml_file):
                default_yaml_file = get_default_yaml_file(pretrained_model_name)
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
    def from_pretrained(cls, pretrained_model_name_or_dir: str, **kwargs):
        """
        Instantiates a model by the pretrained_model_name_or_dir. It download the model weights if the user pass
        a model name, or load the weight from the given directory if given the path.
        (only support standalone mode, and distribute mode waits for developing!)

        Args:
            pretrained_model_name_or_dir (str): It supports the following two input types.
                If `pretrained_model_name_or_dir` is a supported model name, for example, `vit_base_p16` and `t5_small`,
                it will download the necessary files from the cloud. User can pass one from the support list by call
                `MindFormerBook.get_model_support_list()`. If `pretrained_model_name_or_dir` is a path to the local
                directory where there should have model weights ended with `.ckpt` and configuration file ended
                with `yaml`.
            pretrained_model_name_or_path (Optional[str]): Equal to "pretrained_model_name_or_dir",
                if "pretrained_model_name_or_path" is set, "pretrained_model_name_or_dir" is useless.

        Examples:
            >>> from mindformers import T5ForConditionalGeneration
            >>> net = T5ForConditionalGeneration.from_pretrained('t5_small')

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
        config_args = cls._get_config_args(pretrained_model_name_or_dir, **kwargs)
        if not download_checkpoint:
            config_args.model.model_config.checkpoint_name_or_path = None
        model = build_network(config_args.model)
        logger.info("model built successfully!")
        return model

    @classmethod
    def show_support_list(cls):
        """show_support_list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get_support_list method"""
        return cls._support_list
