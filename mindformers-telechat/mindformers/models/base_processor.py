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
BaseProcessor
"""
import os
import shutil

import yaml

from ..mindformer_book import print_path_or_list, MindFormerBook
from .build_processor import build_processor
from .tokenization_utils_base import PreTrainedTokenizerBase
from ..tools.logger import logger
from ..tools.register import MindFormerConfig


class BaseAudioProcessor:
    """
    BaseAudioProcessor for all audio preprocess.

    Examples:
        >>> from mindspore.dataset.audio import AllpassBiquad
        >>> from mindformers.models.base_processor import BaseAudioProcessor
        >>> sample_rate = 44100
        >>> central_freq = 200.0
        >>> class MyAudioProcessor(BaseAudioProcessor):
        ...     def __init__(self, audio_property):
        ...         super(MyAudioProcessor, self).__init__(sample_rate=sample_rate, central_freq=central_freq )
        ...         self.all_pass_biquad = AllpassBiquad(44100, 200.0)
        ...
        ...     def preprocess(self, audio_data, **kwargs):
        ...         res = []
        ...         for audio in audio_data:
        ...             audio = self.all_pass_biquad(audio)
        ...             res.append(audio)
        ...         return res
        ...
        >>> my_audio_processor = MyAudioProcessor(sample_rate, central_freq)
        >>> output = my_audio_processor(audio)
    """
    def __init__(self, **kwargs):
        self.config = {}
        self.config.update(kwargs)

    def __call__(self, audio_data, **kwargs):
        """forward process"""
        return self.preprocess(audio_data, **kwargs)

    def preprocess(self, audio_data, **kwargs):
        """preprocess method"""
        raise NotImplementedError("Each audio processor must implement its own preprocess method")


class BaseProcessor:
    """
    Base processor

    Examples:
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> from mindformers.models.base_processor import BaseProcessor
        >>> class MyProcessor(BaseProcessor):
        ...     _support_list = MindFormerBook.get_processor_support_list()['my_model']
        ...
        ...     def __init__(self, image_processor=None, audio_processor=None, tokenizer=None, return_tensors='ms'):
        ...         super(MyProcessor, self).__init__(
        ...             image_processor=image_processor,
        ...             audio_processor=audio_processor,
        ...             tokenizer=tokenizer,
        ...             return_tensors=return_tensors)
        ...
        >>> myprocessor = MyProcessor(image_processor, audio_processor, tokenizer)
        >>> output = mynet(image, audio, text)
    """
    _support_list = []
    _model_type = 0
    _model_name = 1

    def __init__(self, **kwargs):
        self.config = {}
        self.config.update(kwargs)
        self.image_processor = kwargs.pop("image_processor", None)
        self.audio_processor = kwargs.pop("audio_processor", None)
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.max_length = kwargs.pop("max_length", None)
        self.padding = kwargs.pop("padding", False)
        self.return_tensors = kwargs.pop("return_tensors", None)

    def __call__(self, image_input=None, text_input=None):
        """call function"""
        output = {}

        if image_input is not None and self.image_processor:
            if not isinstance(self.image_processor, BaseImageProcessor):
                raise TypeError(f"feature_extractor should inherit from the BaseImageProcessor,"
                                f" but got {type(self.image_processor)}.")

            image_output = self.image_processor(image_input)
            output['image'] = image_output

        if text_input is not None and self.tokenizer:
            if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"tokenizer should inherited from the PreTrainedTokenizerBase,"
                                f" but got {type(self.tokenizer)}.")
            # Format the input into a batch
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input, return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            output['text'] = text_output

        return output

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
            elif isinstance(val, (BaseImageProcessor, BaseAudioProcessor)):
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
            A processor which inherited from BaseProcessor.
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
    def show_support_list(cls):
        """show_support_list"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get_support_list method"""
        return cls._support_list
