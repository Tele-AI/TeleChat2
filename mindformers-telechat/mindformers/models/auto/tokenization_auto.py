# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Tokenizer class."""

import importlib
import json
import os
import warnings
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Union
from mindformers.tools.generic import experimental_mode_func_checker
from ..tokenization_utils import PreTrainedTokenizer
from ..tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...tools import (
    cached_file,
    extract_commit_hash,
)
from ...tools.hub import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils.import_utils import is_sentencepiece_available, is_tokenizers_available
from ...tools.logger import logger
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    PretrainedConfig,
    AutoConfig,
    config_class_to_model_type
)
from .auto_factory import _LazyAutoMapping
from ...mindformer_book import MindFormerBook, print_dict

TOKENIZER_SUPPORT_LIST = MindFormerBook.get_tokenizer_support_list()

EXP_ERROR_MSG = "The input yaml_name_or_path should be a path to directory which has " \
                "yaml file, or a model name supported, e.g. llama2_7b."


def is_experimental_mode(path):
    """Check whether AutoTokenizer.from_pretrained() should go into original or experimental mode

    :param path: (str) path to AutoTokenizer.from_pretrained()
    :return: (bool) whether AutoTokenizer.from_pretrained() should go into original or experimental mode
    """
    experimental_mode = False

    is_exist = os.path.exists(path)
    is_dir = os.path.isdir(path)
    if is_dir:
        yaml_list = [file for file in os.listdir(path) if file.endswith(".yaml")]
        if not yaml_list:
            experimental_mode = True
    else:
        if (path.split("_")[0] not in TOKENIZER_SUPPORT_LIST and not path.startswith('mindspore')) or is_exist:
            experimental_mode = True

    return experimental_mode


# pylint: disable=C0103
if is_tokenizers_available():
    from ..tokenization_utils_fast import PreTrainedTokenizerFast
else:
    PreTrainedTokenizerFast = None

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        (
            "clip",
            (
                "CLIPTokenizer",
                None if is_tokenizers_available() else None,
            ),
        ),
        ("glm", ("ChatGLMTokenizer", None)),
        ("glm2", ("ChatGLM2Tokenizer", None)),
        ("glm3", ("ChatGLM3Tokenizer", None)),
        ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
        (
            "llama",
            (
                "LlamaTokenizer" if is_sentencepiece_available() else None,
                "LlamaTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "t5",
            (
                "T5Tokenizer" if is_sentencepiece_available() else None,
                "T5TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("pangualpha", ("PanguAlphaTokenizer", None)),
    ]
)

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def tokenizer_class_from_name(class_name: str):
    """tokenizer_class_from_name"""
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module = importlib.import_module(f".{module_name}", "mindformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    # pylint: disable=W0212
    for _, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init, and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("mindformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_tokenizer_config(
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        subfolder: str = "",
        **kwargs,
):
    """Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from user-uploaded and cache.
    tokenizer_config = get_tokenizer_config("mindformersinfra/test_auto_tokenizer_gpt2_ms")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from mindformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained("tokenizer-test", save_json=True)
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```
    """
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
            "Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    commit_hash = kwargs.get("_commit_hash", None)
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    result["_commit_hash"] = commit_hash
    return result


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the from_pretrained class method.
    This class cannot be instantiated directly using `\_\_init\_\_()` (throws an error).

    Examples:
        >>> from mindformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("llama2_7b")
    """

    _model_type = 0
    _model_name = 1

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def invalid_yaml_name(cls, yaml_name_or_path):
        """Check whether it is a valid yaml name"""
        if yaml_name_or_path.startswith('mindspore'):
            # Adaptation the name of yaml at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            yaml_name_or_path = yaml_name_or_path.split('/')[cls._model_name]

        if not yaml_name_or_path.split('_')[cls._model_type] in TOKENIZER_SUPPORT_LIST.keys():
            return True

        local_model_type = yaml_name_or_path.split('_')[cls._model_type]
        local_model_list = TOKENIZER_SUPPORT_LIST[local_model_type]
        if not isinstance(local_model_list, dict):
            if yaml_name_or_path in local_model_list:
                return False
            raise ValueError(f'\'{yaml_name_or_path}\' is not supported by \'{local_model_type}\', '
                             f'please select from {local_model_list}')

        local_model_names = local_model_list.keys()
        if len(yaml_name_or_path.split('_')) <= cls._model_name or \
                yaml_name_or_path.split('_')[cls._model_name] not in local_model_names:
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
        from ...tools import MindFormerConfig
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
            class_name = config['processor']['tokenizer'].get('type', None)
            logger.info("Load the tokenizer name %s from the %s", class_name, yaml_name_or_path)

        return class_name, config

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, *args, **kwargs):
        r"""
        From pretrain method, which instantiates a tokenizer by a directory or model_id from modelers.cn.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            yaml_name_or_path (str): a folder containing YAML file, a folder containing JSON file,
                or a model_id from modelers.cn. The last two are experimental features.
            args (Any, optional): Will be passed along to the underlying tokenizer \_\_init\_\_() method.
                Only works in experimental mode.
            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the loaded values.

        Returns:
            A tokenizer.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            yaml_name_or_path = pretrained_model_name_or_path

        if not is_experimental_mode(yaml_name_or_path):
            instanced_class = cls.get_class_from_origin_mode(yaml_name_or_path, **kwargs)
        else:
            instanced_class = cls.get_class_from_experimental_mode(yaml_name_or_path, *args, **kwargs)

        return instanced_class

    @classmethod
    def get_class_from_origin_mode(cls, yaml_name_or_path, **kwargs):
        """original logic: from yaml."""
        from ...tools import MindFormerRegister

        if not isinstance(yaml_name_or_path, str):
            raise TypeError(f"yaml_name_or_path should be a str,"
                            f" but got {type(yaml_name_or_path)}")

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
            class_name, config = cls._get_class_name_from_yaml(yaml_file)
        elif os.path.isdir(yaml_name_or_path):
            class_name, config = cls._get_class_name_from_yaml(yaml_name_or_path)
            if not class_name:
                raise ValueError(f"The file `model_name.yaml` should exist in the path "
                                 f"{yaml_name_or_path}/model_name.yaml and should have `processor` configs like "
                                 f"configs/gpt2/run_gpt2.yaml, but not found.")
        else:
            raise FileNotFoundError(f"Tokenizer type `{yaml_name_or_path}` does not exist. "
                                    f"Use `{cls.__name__}.show_support_list()` to check the supported tokenizer. "
                                    f"Or make sure the `{yaml_name_or_path}` is a directory.")
        if 'auto_register' in config.processor.tokenizer:
            MindFormerRegister.auto_register(
                class_reference=config.processor.tokenizer.pop('auto_register'), module_type='tokenizer')
            return MindFormerRegister.get_instance_from_cfg(
                config.processor.tokenizer, 'tokenizer')
        dynamic_class = MindFormerRegister.get_cls(module_type='tokenizer', class_name=class_name)
        instanced_class = dynamic_class.from_pretrained(yaml_name_or_path, **kwargs)
        logger.info("%s Tokenizer built successfully!", instanced_class.__class__.__name__)
        return instanced_class

    @classmethod
    @experimental_mode_func_checker(EXP_ERROR_MSG)
    def get_class_from_experimental_mode(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                  single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                  applicable to all derived classes)

            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PretrainedConfig`], *optional*)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (`bool`, *optional*, defaults to `True`):
                Use a [fast Rust-based tokenizer](https://huggingface.co/docs/tokenizers/index) if it is supported for
                a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer
                is returned instead.
            tokenizer_type (`str`, *optional*):
                Tokenizer type to be loaded.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
                `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__()` for more details.

        Examples:
            >>> from mindformers import AutoTokenizer
            >>> # Download vocabulary from mindformers obs.
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> # Download vocabulary from user-uploaded and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained("mindformersinfra/test_auto_tokenizer_gpt2_ms")
            >>> # If vocabulary files are in a directory
            >>> # (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
            >>> # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")
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

        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_tuple is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES.keys())}."
                )

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast:
                if tokenizer_fast_class_name is not None:
                    tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)
                else:
                    logger.warning(
                        "`use_fast` is set to `True` but the tokenizer class does not have a fast version. "
                        " Falling back to the slow version."
                    )
            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(f"Tokenizer class {tokenizer_class_name} is not currently imported.")

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in tokenizer_config:
            kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and config.auto_map is not None and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        has_remote_code = tokenizer_auto_map is not None
        # pylint: disable=C0123
        has_local_code = config_tokenizer_class is not None or type(config) in TOKENIZER_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            if use_fast and tokenizer_auto_map[1] is not None:
                class_ref = tokenizer_auto_map[1]
            else:
                class_ref = tokenizer_auto_map[0]
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                tokenizer_class.register_for_auto_class()
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        if config_tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]
            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            if tokenizer_class_py is not None:
                return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            raise ValueError(
                "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                "in order to use this tokenizer."
            )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING.keys())}."
        )

    @staticmethod
    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
        r"""
        Register new tokenizers for this class.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            config_class (PretrainedConfig): The model config class.
            slow_tokenizer_class (PreTrainedTokenizer, optional): The slow_tokenizer class. Default: ``None``.
            fast_tokenizer_class (PreTrainedTokenizerFast, optional): The fast_tokenizer class. Default: ``None``.
            exist_ok (bool, optional): If set to True, no error will be raised even if config_class already exists.
                Default: ``False``.
        """
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError("You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class")
        if slow_tokenizer_class is not None and issubclass(slow_tokenizer_class, PreTrainedTokenizerFast):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        if fast_tokenizer_class is not None and issubclass(fast_tokenizer_class, PreTrainedTokenizer):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        if (
                slow_tokenizer_class is not None
                and fast_tokenizer_class is not None
                and issubclass(fast_tokenizer_class, PreTrainedTokenizerFast)
                and fast_tokenizer_class.slow_tokenizer_class != slow_tokenizer_class
        ):
            raise ValueError(
                "The fast tokenizer class you are passing has a `slow_tokenizer_class` attribute that is not "
                "consistent with the slow tokenizer class you passed (fast tokenizer has "
                f"{fast_tokenizer_class.slow_tokenizer_class} and you passed {slow_tokenizer_class}. Fix one of those "
                "so they match!"
            )

        # Avoid resetting a set slow/fast tokenizer if we are passing just the other ones.
        # pylint: disable=W0212
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(TOKENIZER_SUPPORT_LIST)

    @classmethod
    @staticmethod
    def get_support_list(cls):
        """get support list method"""
        return TOKENIZER_SUPPORT_LIST
