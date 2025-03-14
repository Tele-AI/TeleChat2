# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""AutoProcessor class"""
import os
import shutil
import importlib
import inspect
import json
from collections import OrderedDict

from .tokenization_auto import AutoTokenizer
from ..configuration_utils import PretrainedConfig
from ..tokenization_utils_base import TOKENIZER_CONFIG_FILE
from .image_processing_auto import AutoImageProcessor
from ..image_processing_utils import ImageProcessingMixin
from ..processing_utils import ProcessorMixin
from ..utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, AutoConfig
from ...tools.hub.hub import get_file_from_repo
from ...tools.generic import experimental_mode_func_checker
from ...tools import get_class_from_dynamic_module, resolve_trust_remote_code, logger
from ...tools.register.config import MindFormerConfig
from ...mindformer_book import MindFormerBook, print_dict
from ..build_processor import build_processor


EXP_ERROR_MSG = "The input yaml_name_or_path should be a path to yaml file, or a " \
                "path to directory which has yaml file, or a model name supported, e.g. llama2_7b."

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertProcessor"),
        ("blip2", "Blip2Processor"),
        ("clip", "CLIPProcessor"),
        ("glm", "GLMProcessor"),
        ("gpt2", "GPT2Processor"),
        ("llama", "LlamaProcessor"),
        ("mae", "ViTMAEProcessor"),
        ("pangualpha", "PanguAlphaProcessor"),
        ("sam", "SamProcessor"),
        ("swin", "SwinProcessor"),
        ("t5", "T5Processor"),
        ("vit", "ViTProcessor")
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def is_experimental_mode(path):
    """Check if the experimental is used based on yaml name or path"""
    experimental_mode = False
    is_exists = os.path.exists(path)
    is_dir = os.path.isdir(path)
    if is_exists:
        if is_dir:
            yaml_list = [file for file in os.listdir(path)
                         if file.endswith(".yaml")]
            if not yaml_list:
                experimental_mode = True
    else:  # repo
        if "/" in path and path.split("/")[0] != "mindspore":
            experimental_mode = True

    return experimental_mode


def processor_class_from_name(class_name: str):
    """Import procrossor class based on module_name"""
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in processors:
            module = importlib.import_module(f".{module_name}", "mindformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for processor in PROCESSOR_MAPPING._extra_content.values():  # pylint: disable=W0212
        if getattr(processor, "__name__", None) == class_name:
            return processor

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("mindformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


class AutoProcessor:
    r"""
    This is a generic processor class that will be instantiated as one of the processor
    classes of the library when created with the ``from_pretrained()`` class method.
    This class cannot be instantiated directly using \_\_init\_\_() (throws an error).

    Examples:
        >>> from mindformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("glm3_6b")
    """
    _support_list = MindFormerBook.get_processor_support_list()
    _model_type = 0
    _model_name = 1

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
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
        From pretrain method, which instantiates a processor by YAML, directory or model_id from modelers.cn.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            yaml_name_or_path (str): YAML file path, a folder containing JSON file,
                or a model_id from modelers.cn. The last three are experimental features.
            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the loaded values.

        Returns:
            A processor which inherited from ProcessorMixin.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if is_experimental_mode(yaml_name_or_path):
            processor = cls.from_pretrained_experimental(yaml_name_or_path, **kwargs)
        else:
            processor = cls.from_pretrained_origin(yaml_name_or_path, **kwargs)

        return processor

    @classmethod
    def from_pretrained_origin(cls, yaml_name_or_path, **kwargs):
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
            raise TypeError(f"yaml_name_or_path should be a str,"
                            f" but got {type(yaml_name_or_path)}")

        is_exist = os.path.exists(yaml_name_or_path)
        model_name = yaml_name_or_path.split('/')[cls._model_name].split("_")[cls._model_type] \
            if yaml_name_or_path.startswith('mindspore') else yaml_name_or_path.split("_")[cls._model_type]
        if not is_exist and model_name not in cls._support_list.keys():
            raise ValueError(f'{yaml_name_or_path} does not exist,'
                             f' and it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')

        if is_exist:
            logger.info("config in %s is used for auto processor"
                        " building.", yaml_name_or_path)
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
    @experimental_mode_func_checker(EXP_ERROR_MSG)
    def from_pretrained_experimental(cls, pretrained_model_name_or_path, **kwargs):
        """Experimental features."""
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        processor_class = None
        processor_auto_map = None

        # First, let's see if we have a preprocessor config.
        # Filter the kwargs for `get_file_from_repo`.
        get_file_from_repo_kwargs = {
            key: kwargs.get(key) for key in inspect.signature(get_file_from_repo).parameters.keys() if key in kwargs
        }

        # Let's start by checking whether the processor class is saved in an processor config
        processor_config_file = get_file_from_repo(
            pretrained_model_name_or_path, PROCESSOR_NAME, **get_file_from_repo_kwargs
        )
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class", None)
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict.get("auto_map").get("AutoProcessor")

        if processor_class is None:
            preprocessor_config_file = get_file_from_repo(
                pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, **get_file_from_repo_kwargs
            )
            if preprocessor_config_file is not None:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict.get("auto_map").get("AutoProcessor")

        if processor_class is None:
            # Next, let's check whether the processor class is saved in a tokenizer
            tokenizer_config_file = get_file_from_repo(
                pretrained_model_name_or_path, TOKENIZER_CONFIG_FILE, **get_file_from_repo_kwargs
            )
            if tokenizer_config_file is not None:
                with open(tokenizer_config_file, encoding="utf-8") as reader:
                    config_dict = json.load(reader)

                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict.get("auto_map").get("AutoProcessor")

        if processor_class is None:
            # Otherwise, load config, if it can be loaded.
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
                if isinstance(config, tuple):
                    config = config[0]

            # And check if the config contains the processor class.
            processor_class = getattr(config, "processor_class", None)
            if hasattr(config, "auto_map") and "AutoProcessor" in config.auto_map:
                processor_auto_map = config.auto_map["AutoProcessor"]

        if processor_class is not None:
            processor_class = processor_class_from_name(processor_class)

        has_remote_code = processor_auto_map is not None
        has_local_code = processor_class is not None or type(config) in PROCESSOR_MAPPING  # pylint: disable=C0123
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:  # pylint: disable=R1705
            processor_class = get_class_from_dynamic_module(
                processor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)  # pylint: disable=C0303
            if os.path.isdir(pretrained_model_name_or_path):
                processor_class.register_for_auto_class()
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        elif processor_class is not None:
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        # Last try: we use the PROCESSOR_MAPPING.
        elif type(config) in PROCESSOR_MAPPING:  # pylint: disable=C0123
            return PROCESSOR_MAPPING[type(config)].from_pretrained(pretrained_model_name_or_path, **kwargs)

        # At this stage, there doesn't seem to be a `Processor` class available for this model, so let's try a
        # tokenizer.
        try:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:  # pylint: disable=W0703
            try:
                return AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception as e:  # pylint: disable=W0703
                raise ValueError(
                    f"Unrecognized processing class in {pretrained_model_name_or_path}. "
                    f"Can't instantiate a processor, a tokenizer, an image processor for this model. "
                    f"Make sure the repository contains the files of at least one of those processing classes."
                ) from e

    @staticmethod
    def register(config_class, processor_class, exist_ok=False):
        r"""
        Register a new processor for this class.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            config_class (PretrainedConfig): The model config class.
            processor_class (ProcessorMixin): The processor class.
            exist_ok (bool, optional): If set to True, no error will be raised even if config_class already exists.
                Default: ``False``.
        """
        PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list
