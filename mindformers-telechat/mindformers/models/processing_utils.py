# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
 Processing saving/loading class for common processors.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Union
import json
import yaml

from ..mindformer_book import print_path_or_list, MindFormerBook
from .build_processor import build_processor
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import PreTrainedTokenizerBase
from .image_processing_utils import ImageProcessingMixin
from .utils import PROCESSOR_NAME
from ..tools.logger import logger
from ..tools.generic import add_model_info_to_auto_map
from ..tools.register import MindFormerConfig
from ..tools.hub.dynamic_module_utils import custom_object_save
from ..tools.hub.hub import (
    PushToHubMixin,
    is_offline_mode,
    is_remote_url,
    download_url,
    cached_file
)
from ..tools.generic import experimental_mode_func_checker
from ..utils import direct_mindformers_import


AUTO_TO_BASE_CLASS_MAPPING = {
    "AutoTokenizer": "PreTrainedTokenizerBase",
    "AutoFeatureExtractor": "FeatureExtractionMixin",
    "AutoImageProcessor": "ImageProcessingMixin",
}


def is_experimental_mode(path):
    """Check if the experimental is used based on yaml name or path"""
    experimental_mode = False
    is_exists = os.path.exists(path)
    is_dir = os.path.isdir(path)
    if is_exists:
        if is_dir:
            experimental_mode = True
        else:  # file
            if not (path.endswith(".yaml") or path.endswith(".yml")):
                experimental_mode = True
    else:  # repo
        if "/" in path and path.split("/")[0] != "mindspore":
            experimental_mode = True

    return experimental_mode


class ProcessorMixin(PushToHubMixin):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    attributes = ["feature_extractor", "tokenizer"]
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    _support_list = []
    _model_type = 0
    _model_name = 1

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        self.config = {}
        self.config.update(kwargs)
        self.max_length = kwargs.pop("max_length", None)
        self.padding = kwargs.pop("padding", False)
        self.return_tensors = kwargs.pop("return_tensors", None)

        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:  # pylint: disable=R1720
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # this feature depends on lazy module, and will be implemented in future
        # Dynamically import the Mindformers module to grab the attribute classes of the processor form their names.
        mindformers_module = direct_mindformers_import(Path(__file__).parent.parent)

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            if isinstance(class_name, tuple):
                proper_class = tuple(getattr(mindformers_module, n) for n in class_name if n is not None)
            else:
                proper_class = getattr(mindformers_module, class_name)

            if not isinstance(arg, proper_class):
                raise ValueError(
                    f"Received a {type(arg).__name__} for argument {attribute_name}, but a {class_name} was expected."
                )

            setattr(self, attribute_name, arg)

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr_str = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr_str}"

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a processor by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported yaml name or a path to .yaml file,
                the supported model name could be selected from .show_support_list().
                If yaml_name_or_path is model name, it supports model names beginning with mindspore
                or the model name itself, such as "mindspore/vit_base_p16" or "vit_base_p16".
            pretrained_model_name_or_path (Optional[str]): Equal to "yaml_name_or_path",
                if "pretrained_model_name_or_path" is set, "yaml_name_or_path" is useless.

        Returns:
            A processor which inherited from ProcessorMixin.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")

        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if is_experimental_mode(yaml_name_or_path):
            processor = cls.from_pretrained_experimental(
                pretrained_model_name_or_path=yaml_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs
            )
        else:
            processor = cls.from_pretrained_origin(yaml_name_or_path, **kwargs)

        return processor

    @classmethod
    def from_pretrained_origin(cls, yaml_name_or_path, **kwargs):
        """
        From pretrain method, which instantiates a processor by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported yaml name or a path to .yaml file,
                the supported model name could be selected from .show_support_list().
                If yaml_name_or_path is model name, it supports model names beginning with mindspore
                or the model name itself, such as "mindspore/vit_base_p16" or "vit_base_p16".
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
        if not is_exist and yaml_name_or_path not in cls._support_list:
            raise ValueError(f'{yaml_name_or_path} does not exist,'
                             f' or it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')
        if is_exist:
            logger.info("config in %s is used for processor"
                        " building.", yaml_name_or_path)

            config_args = MindFormerConfig(yaml_name_or_path)
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

        processor = build_processor(config_args.processor)
        logger.info("processor built successfully!")
        return processor

    @classmethod
    @experimental_mode_func_checker()
    def from_pretrained_experimental(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs):
        """
        Instantiate a processor associated with a pretrained model.
        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    @classmethod
    def get_processor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on openmind.cn, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
        else:
            processor_file = PROCESSOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://openmind.cn/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                ) from e

        if resolved_processor_file is None:
            return {}, kwargs

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            processor_dict = json.loads(text)

        except json.JSONDecodeError as e:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            ) from e

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if "auto_map" in processor_dict and not is_local:
            processor_dict["auto_map"] = add_model_info_to_auto_map(
                processor_dict["auto_map"], pretrained_model_name_or_path
            )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict, **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Unlike image processors or feature extractors whose `__init__` accept `kwargs`, processor don't have `kwargs`.
        # We have to pop up some unused (but specific) arguments to make it work.
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

        processor = cls(*args, **processor_dict)

        # Update processor with kwargs if needed
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs

        return processor

    def save_pretrained(self, save_directory=None, save_name="mindspore_model", **kwargs):
        """
        Save_pretrained.

        Args:
            save_directory (str): a directory to save config yaml

            save_name (str): the name of save files.
        """
        save_json = kwargs.get("save_json", False)

        if save_json:
            push_to_hub = kwargs.pop("push_to_hub", False)
            self.save_pretrained_experimental(save_directory, push_to_hub, **kwargs)
        else:
            self.save_pretrained_origin(save_directory, save_name)

    def save_pretrained_origin(self, save_directory=None, save_name="mindspore_model"):
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

        parsed_config = self._inverse_parse_config(self.config)
        wraped_config = self._wrap_config(parsed_config)

        config_path = os.path.join(save_directory, save_name + '.yaml')
        meraged_dict = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file_reader:
                meraged_dict = yaml.safe_load(file_reader.read())
            file_reader.close()
        meraged_dict.update(wraped_config)

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(config_path, flags_, 0o750), 'w') as file_pointer:
            file_pointer.write(yaml.dump(meraged_dict))
        logger.info("processor saved successfully!")

    def save_pretrained_experimental(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizer) else a) for a in attrs]
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)  # pylint: disable=W0212
            attribute.save_pretrained(save_directory, save_json=True)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def show_support_list(cls):
        """show_support_list"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get_support_list method"""
        return cls._support_list

    def _inverse_parse_config(self, config):
        """
        Inverse parse config method, which builds yaml file content for feature extractor config.

        Args:
            Config (dict): a dict, which contains input parameters of feature extractor.

        Returns:
            A dict, which follows the yaml content.
        """
        parsed_config = {"type": self.__class__.__name__}

        for key, val in config.items():
            if isinstance(val, PreTrainedTokenizerBase):
                parsed_sub_config = {"type": val.__class__.__name__}
                parsed_sub_config.update(val.init_kwargs)
                parsed_config.update({key: parsed_sub_config})
            elif isinstance(val, ImageProcessingMixin):
                parsed_sub_config = {"type": val.__class__.__name__}
                parsed_sub_config.update(val.config)
                parsed_config.update({key: parsed_sub_config})
            else:
                parsed_config.update({key: val})
        return parsed_config

    def _wrap_config(self, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (dict): a dict processed by _inverse_parse_config function.

        Returns:
            A dict for yaml.dump.
        """
        return {"processor": config}

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import mindformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """verify and get args from pretrained"""
        args = []

        # can be extracted from method in future (wait for lazy module)
        mindformers_module = direct_mindformers_import(Path(__file__).parent.parent)
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(getattr(mindformers_module, n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = getattr(mindformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

    @property
    def model_input_names(self):
        first_attribute = getattr(self, self.attributes[0])
        return getattr(first_attribute, "model_input_names", None)
