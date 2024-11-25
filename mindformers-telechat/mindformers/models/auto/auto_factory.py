# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" auto factory """
import os
import copy
import shutil
import importlib
from collections import OrderedDict
import functools
import types

from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import try_sync_file
from mindformers.tools.hub import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
    cached_file,
    extract_commit_hash,
)
from mindformers.tools.generic import experimental_mode_func_checker
from mindformers.models.auto.configuration_auto import (
    AutoConfig,
    replace_list_option_in_docstrings,
)
from mindformers.models.utils import CONFIG_NAME
from mindformers.models.configuration_utils import PretrainedConfig
from ...mindformer_book import MindFormerBook, print_dict
from ..build_model import build_network

EXP_ERROR_MSG = "Please use AutoModel.from_pretrained(), and the input pretrained_model_name_or_dir " \
                "should be a path to directory which has yaml file, or a model name supported, e.g. llama2_7b."

CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the `BaseAutoModelClass.from_pretrained` class method or the `BaseAutoModelClass.from_config` class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
"""

FROM_CONFIG_DOCSTRING = """
        From config method, which instantiates one of the model classes of the library by model config or YAML.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use `BaseAutoModelClass.from_pretrained` to load the model weights.

        Args:
            config (Union[MindFormerConfig, PretrainedConfig, str]): MindFormerConfig, YAML file,
                or a model config inherited from PretrainedConfig (experimental feature).
                The model class to instantiate is selected based on the configuration class:

                List options

            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the config values.

        Returns:
            A model, which inherited from PreTrainedModel.

        Examples:
            >>> from mindformers import AutoConfig, BaseAutoModelClass
            >>> # Download configuration from openmind and cache.
            >>> config = AutoConfig.from_pretrained("checkpoint_placeholder")
            >>> model = BaseAutoModelClass.from_config(config)
"""

FROM_PRETRAINED_MINDFORMERS_DOCSTRING = r"""
        From pretrain method, which instantiates one of the model classes of the library by directory or model_id from modelers.cn.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_dir` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_dir`:

        List options

        The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with `model.train()`

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            pretrained_model_name_or_dir (str):
                A folder containing a YAML file and ckpt file, a folder containing config.json and ckpt file,
                or a model_id from modelers.cn. The last two are experimental features.
            model_args (Any, optional):
                Will be passed along to the underlying model \_\_init\_\_() method. Only works in experimental mode.
            kwargs (Dict[str, Any], optional): 
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). `**kwargs` will be passed to the underlying model's `__init__` method
                when `config` is provided or automatically loaded; otherwise `**kwargs` will be first passed to
                `PretrainedConfig.from_pretrained` to create a configuration, and the keys which do not correspond to 
                any configuration attribute will be passed to the underlying model's `__init__` function.
                Some of available keys are showed below:

                - config (PretrainedConfig, optional):
                  Configuration for the model to use instead of an automatically loaded configuration.
                  DeFault: ``None``. Configuration can be automatically loaded when:

                  - The model is provided by the library (loaded with the model_id string of a pretrained model).
                  - The model was saved using `PreTrainedModel.save_pretrained` and is reloaded by supplying the
                    save directory.
                  - The model is loaded by supplying a local directory as `pretrained_model_name_or_dir` and a
                    configuration JSON file named 'config.json' is found in the directory.

                - cache_dir (Union[str, os.PathLike], optional):
                  Path to a directory in which a downloaded pretrained model configuration should be cached if the
                  standard cache should not be used. DeFault: ``None``.
                - force_download (bool, optional):
                  Whether to force the (re-)download of the model weights and configuration files, overriding
                  the cached versions if they exist. DeFault: ``False``.
                - resume_download (bool, optional):
                  Whether to delete incompletely received files. Will attempt to resume the download if such a
                  file exists. DeFault: ``False``.
                - proxies (Dict[str, str], optional):
                  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                  'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request. DeFault: ``None``.
                - local_files_only (bool, optional):
                  Whether to only look at local files (i.e., not try downloading the model). DeFault: ``False``.
                - revision (str, optional):
                  The specific model version to use. It can be a branch name, a tag name, a commit id, or any
                  identifier allowed by git. DeFault: ``"main"``.
                - trust_remote_code (bool, optional):
                  Whether to allow for custom models defined on the Hub in their own modeling files. This option
                  should only be set to ``True`` for repositories you trust and in which you have read the code,
                  as it will execute code present on the Hub on your local machine. DeFault: ``False``.
                - code_revision (str, optional):
                  The specific revision to use for the code on the Hub, if the code leaves in a different repository
                  than the rest of the model. It can be a branch name, a tag name, a commit id, or any identifier
                  allowed by git. DeFault: ``"main"``.

        Returns:
            A model, which inherited from PreTrainedModel.

        Examples:
            >>> from mindformers import AutoConfig, BaseAutoModelClass
            >>> # Download model and configuration from openmind and cache.
            >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")
            >>> # Update configuration during loading
            >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)
            >>> model.config.output_attentions
            True
"""


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def _getattribute_from_module(module, attr):
    """get attributes from module"""
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(_getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    mindformers_module = importlib.import_module("mindformers")

    if module != mindformers_module:
        try:
            return _getattribute_from_module(mindformers_module, attr)
        except ValueError as e:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {mindformers_module}!") from e
    else:
        raise ValueError(f"Could not find {attr} in {mindformers_module}!")


def _get_model_class(config, model_mapping):
    """get model class"""
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoModelClass:
    """Base class for auto models."""
    _model_mapping = None

    _support_list = MindFormerBook.get_model_support_list()
    _model_type = 0
    _model_name = 1

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_dir)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
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
    def is_experimental_mode_from_config(cls, config):
        """Check whether AutoModel.from_config() should go into original or experimental mode."""
        if isinstance(config, PretrainedConfig):
            model_name = config.__dict__.pop("model_name", None)
            if model_name is None:
                model_name = MindFormerBook.get_model_config_to_name().get(id(config), None)
            return model_name is None
        return False

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        From config method, which instantiates a Model by model config or YAML.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            config (Union[MindFormerConfig, PretrainedConfig, str]): MindFormerConfig, YAML file,
                or a model config inherited from PretrainedConfig (experimental feature).
            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the config values.

        Returns:
            A model, which inherited from PreTrainedModel.
        """
        if cls.is_experimental_mode_from_config(config):
            return cls.from_config_experimental_mode(config, **kwargs)
        return cls.from_config_origin_mode(config, **kwargs)

    @classmethod
    def from_config_experimental_mode(cls, config, **kwargs):
        """get models from_config"""
        if not isinstance(config, PretrainedConfig):
            raise TypeError(f"config should be a model config, like LlamaConfig, but get {type(config)}")
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()  # pylint: disable=C0123
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, config._name_or_path, has_local_code, has_remote_code  # pylint: disable=W0212
        )

        # pylint: disable=R1705
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):  # pylint: disable=W0212
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop("code_revision", None)
            return model_class._from_config(config, **kwargs)  # pylint: disable=W0212
        elif type(config) in cls._model_mapping.keys():  # pylint: disable=C0123
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)  # pylint: disable=W0212

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_config_origin_mode(cls, config, **kwargs):
        """
        From config method, which instantiates a Model by config.

        Args:
            config (str, PretrainedConfig, MindFormerConfig): A model config inherited from PretrainedConfig,
            or a path to .yaml file for model config, or a model config inherited from MindFormerConfig.

        Returns:
            A model, which inherited from PreTrainedModel.
        """
        if config is None:
            raise ValueError("a model cannot be built from config with config is None.")

        download_checkpoint = kwargs.pop("download_checkpoint", True)

        if isinstance(config, MindFormerConfig):
            config_args = config
        elif isinstance(config, str) and os.path.exists(config) and config.endswith(".yaml"):
            config_args = MindFormerConfig(config)
        elif isinstance(config, PretrainedConfig):
            inversed_config = cls._inverse_parse_config(config)
            config_args = cls._wrap_config(inversed_config)
        else:
            raise ValueError("config should be inherited from PretrainedConfig,"
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
            config (PretrainedConfig): A model config inherited from PretrainedConfig.

        Returns:
            A model config, which follows the yaml content.
        """
        if not isinstance(config, PretrainedConfig):
            return config

        class_name = config.__class__.__name__
        config.__dict__.update({"type": class_name})

        for key, val in config.__dict__.items():
            new_val = cls._inverse_parse_config(val)
            config.__dict__.update({key: new_val})

        return config

    @classmethod
    def _wrap_config(cls, config):
        """
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (PretrainedConfig): A config processed by _inverse_parse_config function.

        Returns:
            A model config, which has the same content as a yaml file.
        """
        model_name = config.__dict__.pop("model_name", None)
        if model_name is None:
            model_name = MindFormerBook.get_model_config_to_name().get(id(config), None)

        arch = MindFormerConfig(type=model_name)
        model = MindFormerConfig(model_config=config.to_dict(), arch=arch)
        return MindFormerConfig(model=model)

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
    def is_experimental_mode(cls, pretrained_model_name_or_dir):
        """Check whether AutoModel.from_pretrained() should go into original or experimental mode."""
        is_exist = os.path.exists(pretrained_model_name_or_dir)
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if is_exist:
            if not is_dir:
                return False
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".yaml")]
            config_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                           if file == CONFIG_NAME]
            if not yaml_list and config_list:
                return True
            return False

        if "/" in pretrained_model_name_or_dir and \
                pretrained_model_name_or_dir.split("/")[0] != "mindspore":
            return True
        return False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir, *model_args, **kwargs):
        r"""
        From pretrain method, which instantiates a model by directory or model_id from modelers.cn.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            pretrained_model_name_or_dir (str): A folder containing a YAML file and ckpt file, a folder
                containing config.json and ckpt file, or a model_id from modelers.cn.
                The last two are experimental features.
            model_args (Any, optional): Will be passed along to the underlying model \_\_init\_\_() method.
                Only works in experimental mode.
            kwargs (Dict[str, Any], optional): The values in kwargs of any keys which are configuration
                attributes will be used to override the loaded values.

        Returns:
            A model, which inherited from PreTrainedModel.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_dir = pretrained_model_name_or_path

        if not isinstance(pretrained_model_name_or_dir, str):
            raise TypeError(f"pretrained_model_name_or_dir should be a str,"
                            f" but got {type(pretrained_model_name_or_dir)}")

        if cls.is_experimental_mode(pretrained_model_name_or_dir):
            return cls.from_pretrained_experimental_mode(pretrained_model_name_or_dir, *model_args, **kwargs)
        return cls.from_pretrained_origin_mode(pretrained_model_name_or_dir, **kwargs)

    @classmethod
    @experimental_mode_func_checker(EXP_ERROR_MSG)
    def from_pretrained_experimental_mode(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """get models from_pretrained"""
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            hub_kwargs["token"] = token

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()  # pylint: disable=C0123
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        # pylint: disable=R1705
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
            )
            _ = hub_kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained_experimental_mode(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():  # pylint: disable=C0123
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained_experimental_mode(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained_origin_mode(cls, pretrained_model_name_or_dir, **kwargs):
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
            A model, which inherited from PreTrainedModel.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        download_checkpoint = kwargs.pop("download_checkpoint", True)
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_dir = pretrained_model_name_or_path

        if not isinstance(pretrained_model_name_or_dir, str):
            raise TypeError(f"pretrained_model_name_or_dir should be a str,"
                            f" but got {type(pretrained_model_name_or_dir)}")
        config_args = cls._get_config_args(pretrained_model_name_or_dir, **kwargs)
        config_args.model.model_config.parallel_config = config_args.parallel_config
        if not download_checkpoint:
            config_args.model.model_config.checkpoint_name_or_path = None
        model = build_network(config_args.model)
        cls.default_checkpoint_download_path = model.default_checkpoint_download_path

        logger.info("model built successfully!")
        return model

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        r"""
        Register a new model for this class.

        Warning:
            The API is experimental and may have some slight breaking changes in the next releases.

        Args:
            config_class (PretrainedConfig): The model config class.
            model_class (PretrainedModel): The model class.
            exist_ok (bool, optional): If set to True, no error will be raised even if config_class already exists.
                Default: ``False`` .
        """
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)

    @classmethod
    def show_support_list(cls):
        """show support list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get support list method"""
        return cls._support_list


def insert_head_doc(docstring, head_doc=""):
    if head_doc:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )


def auto_class_update(cls, checkpoint_for_example="bert-base-cased", head_doc=""):
    """Create a new class with the right name from the base class"""
    model_mapping = cls._model_mapping  # pylint: disable=W0212
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Now we need to copy and re-register `from_config` and `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(
        model_mapping._model_mapping,  # pylint: disable=W0212
        use_model_types=False)(from_config)
    cls.from_config = classmethod(from_config)

    from_pretrained_docstring = FROM_PRETRAINED_MINDFORMERS_DOCSTRING
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace("BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace("shortcut_placeholder", shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(
        model_mapping._model_mapping,  # pylint: disable=W0212
        use_model_types=False)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


class _LazyAutoMapping(OrderedDict):
    """
    A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model or processor(tokenizer、image_processor、feature_extacture or
        processor) class
    """

    # pylint: disable=W0231
    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        # pylint: disable=W0212
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        module_name = self._reverse_config_mapping[key.__name__]
        if module_name in self._model_mapping:
            model_name = self._model_mapping[module_name]
            return self._load_attr_from_module(module_name, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, module_name, attr):
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "mindformers.models")
        return _getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a mindformers model.")

        self._extra_content[key] = value
