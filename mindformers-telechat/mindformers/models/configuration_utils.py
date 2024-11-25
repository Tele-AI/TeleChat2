# coding=utf-8
# Copyright 2024-2024 Huawei Technologies Co., Ltd
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration base class and utilities."""

import os
import shutil
import re
import json
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from packaging import version
from mindspore._c_expression.typing import Float, BFloat

from mindformers import __version__
from mindformers.tools import MindFormerConfig
from mindformers.tools.generic import experimental_mode_func_checker, is_experimental_mode
from mindformers.tools.logger import logger
from mindformers.models.build_config import build_model_config
from mindformers.models.utils import CONFIG_NAME, ms_type_to_str
from mindformers.mindformer_book import MindFormerBook, print_path_or_list
from mindformers.tools import (
    PushToHubMixin,
    DictConfig,
    custom_object_save,
    add_model_info_to_auto_map,
    cached_file,
    download_url,
    extract_commit_hash,
    is_remote_url,
)

__all__ = ["PretrainedConfig"]

_re_configuration_file = re.compile(r"config\.(.*)\.json")

IGNORE_KEYS = ["_name_or_path"]


def get_configuration_file(configuration_files: List[str]) -> str:
    """
    Get the configuration file to use for this version of mindformers.

    Args:
        configuration_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`, the configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        search = _re_configuration_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            configuration_files_map[v] = file_name
    available_versions = sorted(configuration_files_map.keys())

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME
    mindformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= mindformers_version:
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break

    return configuration_file


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.
    """
    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        obj_value = getattr(config_obj, str(key), None)
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            if not diff_value:
                diff[key] = diff_value
        elif key not in dict_b or value != dict_b.get(key) or key not in default or value != default.get(key):
            diff[key] = value
    return diff


class PretrainedConfig(PushToHubMixin):
    """
    Base class for all configuration classes. Handles a few parameters common to all models'
    configurations as well as methods for loading/downloading/saving configurations.

    Note:
        A configuration file can be loaded and saved to disk. Loading the configuration file and using
        this file to initialize a model does **not** load the model weights. It only affects the model's
        configuration.

    Args:
        name_or_path (str, optional):
            Store the string that was passed to :func:`mindformers.models.PreTrainedModel.from_pretrained`
            as `pretrained_model_name_or_path` if the configuration was created with such a method.
            Default: ``""`` .
        checkpoint_name_or_path (str, optional):
            The path or name of the checkpoint file. Default: ``None`` .
        mindformers_version (str, optional):
            The version of MindSpore Transformers. Default: ``None`` .

    Returns:
        PretrainedConfig, a PretrainedConfig instance.

    Examples:
        >>> from mindformers.models import LlamaConfig
        >>> config = LlamaConfig(num_layers=2, seq_length=1024)
        >>> print(config)
        LlamaConfig {
            "batch_size": 1,
            "block_size": 16,
            "bos_token_id": 1,
            "checkpoint_name_or_path": "",
            "compute_dtype": "float16",
            "do_sample": true,
            "embedding_init_type": "float16",
            "eos_token_id": 2,
            "extend_method": "None",
            "ffn_dim_multiplier": null,
            "fine_grain_interleave": 1,
            "hidden_size": 4096,
            "ignore_token_id": -100,
            "intermediate_size": null,
            "is_dynamic": false,
            "layernorm_compute_type": "float32",
            "llm_backend": "",
            "max_decode_length": 1024,
            "max_position_embedding": 1024,
            "mindformers_version": "dev",
            "model_type": "llama",
            "multiple_of": 256,
            "n_kv_heads": null,
            "num_blocks": 512,
            "num_heads": 32,
            "num_layers": 2,
            "offset": 0,
            "pad_token_id": 0,
            "parallel_decoding_params": null,
            "parallel_optimizer": false,
            "param_init_type": "float16",
            "pp_interleave_num": 1,
            "qkv_concat": false,
            "qkv_has_bias": false,
            "quant_config": null,
            "repetition_penalty": 1.0,
            "rms_norm_eps": 1e-05,
            "rotary_dtype": "float32",
            "scaling_factor": 1.0,
            "seq_length": 1024,
            "softmax_compute_type": "float32",
            "theta": 10000.0,
            "tie_word_embeddings": false,
            "top_k": 5,
            "top_p": 1.0,
            "use_attn_mask_compression": false,
            "use_flash_attention": false,
            "use_past": false,
            "use_ring_attention": false,
            "use_rope_slice": false,
            "vocab_size": 32000
            }
    """
    model_type: str = ""
    is_composition: bool = False
    attribute_map: Dict[str, str] = {}
    _auto_class: Optional[str] = None

    _support_list = []
    _model_type = 0
    _model_name = 1

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        self.checkpoint_name_or_path = kwargs.pop("checkpoint_name_or_path", None)

        # version info
        self.mindformers_version = kwargs.pop("mindformers_version", None)
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)

        # general config
        self.architectures = kwargs.pop("architectures", None)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", None)

        # generation config
        self.is_sample_acceleration = kwargs.pop("is_sample_acceleration", None)

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)

    def _to_dict_helper(self, output):
        if "parallel_config" in output:
            output["parallel_config"] = output["parallel_config"].to_dict()
        if "moe_config" in output:
            output["moe_config"] = output["moe_config"].to_dict()
        if "op_parallel_config" in output:
            output["op_parallel_config"] = output["op_parallel_config"].to_dict()
        if "embed_parallel_config" in output:
            output["embed_parallel_config"] = output["embed_parallel_config"].to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            dict[str, Any], dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        self._to_dict_helper(output)

        # Mindformers version when serializing the model
        output["mindformers_version"] = __version__

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["mindformers_version"]
            if isinstance(value, DictConfig):
                value = value.to_dict()

            output[key] = value

        self.dict_ms_dtype_to_str(output)
        return output

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs) -> "PretrainedConfig":
        """
        From pretrain method, which instantiates a config by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported model name or a path to model config (.yaml),
                the supported model name could be selected from :func:`mindformers.AutoConfig.show_support_list` .
                If yaml_name_or_path is model name,
                it supports model names beginning with mindspore or the model name itself,
                such as "mindspore/vit_base_p16" or "vit_base_p16".
            pretrained_model_name_or_path (str, optional):
                Equal to "yaml_name_or_path", if "pretrained_model_name_or_path" is set,
                "yaml_name_or_path" is useless. Default: ``None`` .

        Returns:
            A model config, which inherited from PretrainedConfig.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")

        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        config = cls.get_config_experimental_mode(
            pretrained_model_name_or_path=yaml_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs
        ) if is_experimental_mode(yaml_name_or_path) else cls.get_config_origin_mode(
            yaml_name_or_path, **kwargs)

        return config

    @classmethod
    @experimental_mode_func_checker()
    def get_config_experimental_mode(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs):
        """Get config object by from_pretrained with experimental mode

        :param cache_dir: local path for caching file
        :param force_download: whether to download from hub by force
        :param kwargs: kwargs params
        :param local_files_only: whether to load local files only
        :param pretrained_model_name_or_path: model file name or path
        :param revision: revision information
        :param token: token information
        :return: config object
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["token"] = token
        kwargs["revision"] = revision

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        config = cls.from_dict(config_dict, **kwargs)

        return config

    @classmethod
    def get_config_origin_mode(cls, yaml_name_or_path, **kwargs):
        """Get config object by from_pretrained with original mode

        :param yaml_name_or_path: yaml file name or corresponding path
        :param kwargs: kwargs params
        :return: config object
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

    def save_pretrained(self, save_directory=None, save_name="mindspore_model", **kwargs):
        """
        Saves the pre-trained configuration to the specified directory

        Args:
            save_directory (str, optional): a directory to save config yaml. Default: ``None`` .
            save_name (str, optional): the name of save files. Default: ``"mindspore_model"`` .
        """
        save_json = kwargs.pop("save_json", False)

        if save_json:
            push_to_hub = kwargs.get("push_to_hub", False)
            self.save_config_experimental_mode(save_directory, push_to_hub, **kwargs)
        else:
            self.save_config_origin_mode(save_directory, save_name)

    @experimental_mode_func_checker()
    def save_config_experimental_mode(self, save_directory, push_to_hub, **kwargs):
        """Save config to local directory with json format in experimental mode

        :param save_directory: local directory for saving json config file
        :param push_to_hub: whether push config json file to remote hub
        :param kwargs: kwargs params
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)

        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def save_config_origin_mode(self, save_directory, save_name):
        """Save config to local directory with yaml format in original mode

        :param save_directory: local directory for saving yaml config file
        :param save_name: yaml config file name
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
            self.__dict__[key] = val

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
        if isinstance(self, PretrainedConfig):
            self.__dict__.pop("type")

        for key, val in self.__dict__.items():
            if isinstance(val, PretrainedConfig):
                val.__dict__.pop("type")
                self.__dict__.update({key: val})

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
        self.__dict__.update({"type": self.__class__.__name__})
        removed_list = []

        for key, val in self.__dict__.items():
            if isinstance(val, PretrainedConfig):
                val = val.inverse_parse_config()
            elif not isinstance(val, (str, int, float, bool, DictConfig)) or key in IGNORE_KEYS:
                removed_list.append((key, val))
                continue
            self.__dict__.update({key: val})

        for key, _ in removed_list:
            self.__dict__.pop(key)
        return self, removed_list

    def _wrap_config(self, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (PretrainedConfig): a config processed by _inverse_parse_config function.

        Returns:
            A (config) dict for yaml.dump.
        """
        model_name = self.__dict__.pop("model_name", None)
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

    @classmethod
    def get_config_dict(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        `PretrainedConfig` using :func:`mindformers.models.PretrainedConfig.from_dict`.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): The identifier of the pre-trained checkpoint
                from which we want the dictionary of parameters.

        Returns:
            Tuple[dict, dict], the dictionary(ies) that will be used to instantiate the configuration object.

        """
        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """get config dict"""
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'xxxxxxxxxxxxxxxxxxxxx', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                ) from e

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            ) from e

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if "auto_map" in config_dict and not is_local:
            config_dict["auto_map"] = add_model_info_to_auto_map(
                config_dict["auto_map"], pretrained_model_name_or_path
            )
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a `PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (Dict[str, Any]): Dictionary that will be used to instantiate the configuration object.
                Such a dictionary can be retrieved from a pretrained checkpoint by leveraging
                the :func:`mindformers.models.PretrainedConfig.get_config_dict` method.

        Returns:
            `PretrainedConfig`, the configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        config = cls(**config_dict)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a PretrainedConfig from the path to a JSON file of parameters.

        Args:
            json_file (Union[str, os.PathLike]): Path to the JSON file containing the parameters.

        Returns:
            PretrainedConfig, the configuration object instantiated from that JSON file.
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (Union[str, os.PathLike]): Path to the JSON file in which this configuration instance's
                parameters will be saved.
            use_diff (bool, optional): If set to `True`, only the difference between the config instance and
                the default :class:`mindformers.models.PretrainedConfig` is serialized to JSON file.
                Default: ``True`` .
        """
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(json_file_path, flags_, 0o750), 'w', encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (bool, optional): If set to `True`, only the difference between the config instance and
                the default `PretrainedConfig()` is serialized to JSON string. Default: ``True`` .

        Returns:
            str, string containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def _to_diff_dict_helper(self, serializable_config_dict):
        attributes = ["parallel_config", "moe_config", "op_parallel_config", "embed_parallel_config"]
        for attr in attributes:
            if attr in serializable_config_dict:
                diff_parallel_config = getattr(self, attr).to_diff_dict()
                if not diff_parallel_config:
                    del serializable_config_dict[attr]
                else:
                    serializable_config_dict[attr] = diff_parallel_config

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            dict[str, Any], dictionary of all the attributes that make up this configuration instance.
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                    isinstance(getattr(self, key, None), PretrainedConfig)
                    and key in class_config_dict
                    and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if diff:
                    serializable_config_dict[key] = diff
            elif (
                    key not in default_config_dict
                    or key == "mindformers_version"
                    or value != default_config_dict.get(key, None)
                    or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        self.dict_ms_dtype_to_str(serializable_config_dict)
        self._to_diff_dict_helper(serializable_config_dict)
        return serializable_config_dict

    def dict_ms_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *ms_dtype* key and if it's not None,
        converts ms.dtype to a string of just the type. For example, `ms.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        for k, v in d.items():
            if isinstance(v, (Float, BFloat)):
                d[k] = ms_type_to_str.get(v)
        for value in d.values():
            if isinstance(value, dict):
                self.dict_ms_dtype_to_str(value)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        """
        Register this class with a given auto class. This should only be used for custom configurations as the ones in
        the library are already mapped with `AutoConfig`.

        .. warning::
            This API is experimental and may have some slight breaking changes in the next releases.

        Args:
            auto_class (Union[str, type], optional): The auto class to register this new configuration with.
                Default: ``"AutoConfig"`` .
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        cls._auto_class = auto_class
