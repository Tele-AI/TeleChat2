# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2024-2024 Huawei Technologies Co., Ltd
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

"""Auto Config class"""

import os
import re
import shutil
import warnings
import importlib
from collections import OrderedDict
from typing import List, Union

from mindformers.models.utils import CONFIG_NAME
from mindformers.mindformer_book import print_dict, MindFormerBook
from mindformers.models.build_config import build_model_config
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.logger import logger
from mindformers.tools.hub import resolve_trust_remote_code, get_class_from_dynamic_module
from mindformers.tools.generic import experimental_mode_func_checker, is_experimental_mode
from mindformers.tools.register import MindFormerConfig

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertConfig"),
        ("blip2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("clip", "CLIPConfig"),
        ("glm", "GLMConfig"),
        ("glm2", "ChatGLM2Config"),
        ("gpt2", "GPT2Config"),
        ("llama", "LlamaConfig"),
        ("mae", "ViTMAEConfig"),
        ("pangualpha", "PanguAlphaConfig"),
        ("sam", "SamConfig"),
        ("swin", "SwinConfig"),
        ("t5", "T5Config"),
        ("vit", "ViTConfig")
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        ("bert", "BertModel"),
        ("blip2", "Blip2Llm"),
        ("bloom", "BloomModel"),
        ("clip", "CLIPModel"),
        ("glm", "GLMChatModel"),
        ("glm2", "ChatGLM2Model"),
        ("gpt2", "GPT2Model"),
        ("llama", "LlamaModel"),
        ("mae", "ViTMAEModel"),
        ("pangualpha", "PanguAlphaModel"),
        ("sam", "SamModel"),
        ("swin", "SwinModel"),
        ("t5", "T5ForConditionalGeneration"),
        ("vit", "ViTModel")
    ]
)

EXP_ERROR_MSG = "The input yaml_name_or_path should be a path to yaml file, e.g. " \
                "'run_xxx_model.yaml', or a model name supported, e.g. llama2_7b."


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():  # pylint: disable=W0212
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    # pylint: disable=W0231
    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        """return module attributes based on module name"""
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        if key not in self._modules:
            self._modules[key] = importlib.import_module(f".{key}", "mindformers.models")
        if hasattr(self._modules[key], value):
            return getattr(self._modules[key], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        mindformers_module = importlib.import_module("mindformers")
        return getattr(mindformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Mindformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    """doc"""
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    """doc"""
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).

    Examples:
        >>> from mindformers import AutoConfig
        >>> # 1)  instantiates a config by yaml model name
        >>> config_a = AutoConfig.from_pretrained('configs/llama2/predict_llama2_7b.yaml')
        >>> # 2)  instantiates a config by json file
        >>> config_b = AutoConfig.from_pretrained('./config.json')
        >>> # 3)  instantiates a config by directory containing a configuration json file
        >>> config_c = AutoConfig.from_pretrained('./dir/')
        >>> # 4)  instantiates a config by model_id from modelers.cn
        >>> config_d = AutoConfig.from_pretrained('MindSpore-Lab/glm2_6b')
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
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a config by YAML, JSON, directory or model_id from modelers.cn.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            yaml_name_or_path (str): YAML file path, JSON file path, a folder containing a config.json file,
                or a model_id from modelers.cn. The last three are experimental features.
            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the loaded values.

        Returns:
            A model config, which inherited from PretrainedConfig.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        config = cls.get_config_experimental_mode(yaml_name_or_path, **kwargs) if is_experimental_mode(
            yaml_name_or_path) else cls.get_config_origin_mode(yaml_name_or_path, **kwargs)
        return config

    @classmethod
    def get_config_origin_mode(cls, yaml_name_or_path, **kwargs):
        """Get config object by from_pretrained with original mode

        :param yaml_name_or_path: yaml file name or corresponding path
        :param kwargs: kwargs params
        :return: config object
        """
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
    @experimental_mode_func_checker(EXP_ERROR_MSG)
    def get_config_experimental_mode(cls, pretrained_model_name_or_path, **kwargs):
        """Get config object by from_pretrained with experimental mode

        :param pretrained_model_name_or_path: model file name or corresponding path
        :return: config object
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
                "Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **unused_kwargs)
        # Fallback: use pattern matching on the string.
        # We go from longer names to shorter names to catch roberta before bert (for instance)
        for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
            if pattern in str(pretrained_model_name_or_path):
                return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list

    @staticmethod
    def register(model_type, config, exist_ok=False):
        r"""
        Register a new configuration for this class.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            model_type (str): The model type like "bert" or "gpt".
            config (PretrainedConfig): The config to register.
            exist_ok (bool, optional): If set to True, no error will be raised even if model_type already exists.
                Default: ``False`` .
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
