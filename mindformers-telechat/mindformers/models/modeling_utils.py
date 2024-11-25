# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
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
""" Configuration PreTrainedModel."""
import os
import re
import json
import shutil
from functools import partial
from typing import Dict, Optional, Union
import yaml
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context, Model

from mindformers.tools.hub import (
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    is_offline_mode,
    is_remote_url,
    get_checkpoint_shard_files,
    convert_file_size_to_int,
    has_file
)
from mindformers.tools.hub.dynamic_module_utils import custom_object_save
from mindformers.generation import GenerationConfig, GenerationMixin
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig, DictConfig
from mindformers.tools.ckpt_transform import TransformCkpt, make_soft_link
from mindformers.tools.utils import (
    get_real_rank,
    get_output_root_path,
    clear_auto_trans_output,
    remake_folder,
    barrier_world
)
from ..mindformer_book import MindFormerBook, print_path_or_list
from ..tools.download_tools import download_with_progress_bar
from ..tools.utils import try_sync_file, replace_tk_to_mindpet
from .configuration_utils import PretrainedConfig
from .utils import CONFIG_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from .build_model import build_network

__all__ = ["PreTrainedModel"]

IGNORE_KEYS = ["_name_or_path"]


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Examples:
        >>> dtype_byte_size(mindspore.float32)
        4
    """
    if dtype == ms.bool_:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def save_checkpoint(save_obj, save_directory):
    ckpt_file_name = os.path.join(save_directory, WEIGHTS_NAME)
    ms.save_checkpoint(save_obj, ckpt_file_name)


def shard_checkpoint(
        state_dict: Dict[str, ms.Parameter], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, mindspore.Parameter]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"mindspore_model.ckpt"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dict_lists = [{}]
    last_block_size = 0
    total_size = 0

    for name, weight in state_dict.items():
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split, but only if we have put at least one
        # weight in the current shard.
        if last_block_size + weight_size > max_shard_size and sharded_state_dict_lists[-1]:
            sharded_state_dict_lists.append({})
            last_block_size = 0

        sharded_state_dict_lists[-1][name] = weight
        last_block_size += weight_size
        total_size += weight_size

    # If we only have one shard, we return it
    if len(sharded_state_dict_lists) == 1:
        return {weights_name: sharded_state_dict_lists[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dict_lists):
        shard_file = weights_name.replace(".ckpt", f"-{idx + 1:05d}-of-{len(sharded_state_dict_lists):05d}.ckpt")
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_sharded_checkpoint(model, folder, strict=True):
    """
    load checkpoint from a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_files = list(set(index["weight_map"].values()))
    shard_files = [os.path.join(folder, f) for f in shard_files]

    state_dict = {}
    for shard_file in shard_files:
        state_dict.update(load_checkpoint(shard_file))

    # load params into net
    not_load_network_params = load_param_into_net(model, state_dict, strict_load=strict)
    logger.info("Network parameters are not loaded: %s", str(not_load_network_params))
    return not_load_network_params


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `mindspore.nn.Cell`, to be used as a mixin.
    """


class PreTrainedModel(nn.Cell, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
    """
    Base class for all models. Takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models
    to resize the input embeddings and prune heads in the self-attention heads.

    Args:
        config (PretrainedConfig): configuration class for this model architecture.
        inputs (tuple, optional): A variable number of position parameters reserved
            for the position parameters to be expanded.
        kwargs (dict, optional): A variable number of keyword parameters reserved
            for the keyword parameters to be expanded.

    Examples:
        >>> from mindformers import AutoModel
        >>> import mindspore as ms
        >>> ms.set_context(mode=0)
        >>> network = AutoModel.from_pretrained('llama2_7b')
        >>> type(network)
        <class 'mindformers.models.llama.llama.LlamaForCausalLM'>
    """
    _support_list = []
    _model_type = 0
    _model_name = 1

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False

    # Flash Attention 2 support
    _supports_flash_attn_2 = False

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a Mindspore model.
        """
        return "ms"

    # pylint: disable=W0613
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(**kwargs)
        GenerationMixin.__init__(self)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.default_checkpoint_download_path = None
        self.name_or_path = config.name_or_path
        self.warning_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initializee under go here.

        Args:
            ms_dtype (`mindspore.dtype`, *optional*):
                Override the default `mindspore.dtype` and load the model under this dtype.
        """
        # ignore set default type
        model = cls(config, **kwargs)
        return model

    @property
    def base_model(self) -> nn.Cell:
        """
        `mindspore.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            Boolean type, Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GeneratorMixin" in str(cls.prepare_inputs_for_generation) and "GeneratorMixin" in str(cls.generate):
            return False
        return True

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            save_name: str = "mindspore_model",
            **kwargs
    ):
        """
        Save the model weight and configuration file.
        (only supports standalone mode, and distribute mode waits for developing)

        Args:
            save_directory (Union[str, os.PathLike]): A directory to save the model weight and configuration.
            save_name (str): The name of saved files, including model weight and configuration file.
                Default: ``mindspore_model`` .
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.

        Examples:
            >>> import os
            >>> from mindformers import AutoModel
            >>> import mindspore as ms
            >>> ms.set_context(mode=0)
            >>> net = AutoModel.from_pretrained('llama2_7b')
            >>> net.save_pretrained('./checkpoint_save')
        """
        is_main_process = kwargs.pop("is_main_process", True)
        state_dict = kwargs.pop("state_dict", None)
        push_to_hub = kwargs.pop("push_to_hub", False)
        max_shard_size = kwargs.pop("max_shard_size", "5GB")
        variant = kwargs.pop("variant", None)
        token = kwargs.pop("variant", None)

        save_json = kwargs.pop("save_json", False)
        if not save_json:
            self.save_pretrained_origin_mode(save_directory=save_directory, save_name=save_name)
        else:
            self.save_pretrained_experimental_mode(
                save_directory=save_directory,
                is_main_process=is_main_process,
                state_dict=state_dict,
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                variant=variant,
                token=token,
                **kwargs
            )

    def save_pretrained_experimental_mode(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,  # state_dict
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of mindspore.Parameter):
                The state dictionary of the model to save. Will default to `mindspore.load_checkpoint()`, but can be
                used to only save parts of the model or if special precautions need to be taken when recovering the
                state dictionary of a model (like when using model parallelism).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if token is not None:
            kwargs["token"] = token

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            kwargs["push_to_hub"] = push_to_hub
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        model_to_save = self
        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        if is_main_process:  # Save the config
            model_to_save.config.save_pretrained(save_directory, save_json=True)

        if state_dict is None:  # Save the model
            state_dict = {}
            for item in model_to_save.get_parameters():
                state_dict[item.name] = item.data

        # Handle the case where some param_list keys shouldn't be saved
        def choice_func(x, keys_to_ignore_on_save):
            if keys_to_ignore_on_save is not None:
                for k in keys_to_ignore_on_save:
                    if k in x:
                        return False
                return True
            return True

        weights_name = _add_variant(WEIGHTS_NAME, variant)

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".ckpt", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".ckpt", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                    filename.startswith(weights_no_suffix)
                    and os.path.isfile(full_filename)
                    and filename not in shards.keys()
                    and is_main_process
                    and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            ms.save_checkpoint(shard, os.path.join(save_directory, shard_file), \
                               choice_func=partial(choice_func, keys_to_ignore_on_save=self._keys_to_ignore_on_save))

        if index is None:
            path_to_weights = os.path.join(save_directory, _add_variant(WEIGHTS_NAME, variant))
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = os.path.join(save_directory, _add_variant(WEIGHTS_INDEX_NAME, variant))
            # Save the index as well
            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            with os.fdopen(os.open(save_index_file, flags_, 0o750), 'w', encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    def save_pretrained_origin_mode(self,
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


    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """
        prepare inputs for transform ckpt.
        """
        raise RuntimeError(
            "A model class needs to define a `prepare_inputs_for_predict_layout`"
            " method in order to use parallel predict."
        )

    # pylint: disable=W0613
    def set_dynamic_inputs(self, **kwargs):
        """
        Compile static graphs into dynamic shapes
        """

        raise RuntimeError(
            "A model class needs to define a `set_dynamic_inputs`"
            " method in order to use `model.set_inputs()`."
        )

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
    def from_pretrained(
            cls,
            pretrained_model_name_or_dir: str,
            *model_args,
            **kwargs
    ):
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
            model_args (str, optional): Model extension parameters. If included "pretrained_model_name_or_path",
                equal to "pretrained_model_name_or_dir", if "pretrained_model_name_or_path" is set,
                "pretrained_model_name_or_dir" is useless.
            kwargs (dict, optional): A variable number of keyword parameters reserved
                for the keyword parameters to be expanded.

        Returns:
            A model, which inherited from PreTrainedModel.
        """
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")

        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_dir = pretrained_model_name_or_path

        if not isinstance(pretrained_model_name_or_dir, str):
            raise TypeError(f"pretrained_model_name_or_dir should be a str,"
                            f" but got {type(pretrained_model_name_or_dir)}")

        if cls.is_experimental_mode(pretrained_model_name_or_dir):
            return cls.from_pretrained_experimental_mode(pretrained_model_name_or_dir,
                                                         config=config,
                                                         cache_dir=cache_dir,
                                                         ignore_mismatched_sizes=ignore_mismatched_sizes,
                                                         force_download=force_download,
                                                         local_files_only=local_files_only,
                                                         token=token,
                                                         revision=revision,
                                                         *model_args,
                                                         **kwargs)
        return cls.from_pretrained_origin_mode(pretrained_model_name_or_dir, **kwargs)

    @classmethod
    def from_pretrained_origin_mode(cls, pretrained_model_name_or_dir: str, **kwargs):
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

        Returns:
            A model, which inherited from PreTrainedModel.

        Examples:
            >>> from mindformers import T5ForConditionalGeneration
            >>> net = T5ForConditionalGeneration.from_pretrained('t5_small')
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
    def from_pretrained_experimental_mode(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs,
    ):
        r"""
        Instantiate a pretrained mindspore model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.set_train(False)` (Dropout modules are deactivated).
        To train the model, you should first set it back in training mode with `model.set_train(True)`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing model weights saved using
                  [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                  this case, `from_tf` should be set to `True` and a configuration object should be provided as
                  `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                  PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                  `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                  `True`.
                - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                  arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                - an instance of a class derived from [`PretrainedConfig`],
                - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                  model).
                - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                  save directory.
                - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                  configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, mindspore.Parameter]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model.
                Behaves differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                  underlying model's `__init__` method (we assume all relevant updates to the configuration have
                  already been done)
                - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                  initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                  corresponds to a configuration attribute will be used to override said attribute with the
                  supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                  will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](XXX) to
        use this method in a firewalled environment.

        </Tip>

        Examples:
            >>> from mindformers import GPT2Model
            >>> # Download model and configuration from openmind and cache.
            >>> model = GPT2Model.from_pretrained("XXX")
            >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
            >>> model = GPT2Model.from_pretrained("./test/saved_model/")
        """
        state_dict = kwargs.pop("state_dict", None)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        auto_trans_ckpt = kwargs.pop("auto_trans_ckpt", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        user_agent = {"file_type": "model", "framework": "mindspore", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if os.path.isfile(
                        os.path.join(pretrained_model_name_or_path, subfolder, \
                                     _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a Mindspore checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, \
                        _add_variant(WEIGHTS_NAME, variant)
                    )
                elif os.path.isfile(
                        os.path.join(pretrained_model_name_or_path, subfolder, \
                                     _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, \
                        _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}"
                        f" found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, We try to give a helpful error message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                        }
                        if variant is not None and has_file(
                                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                f" {variant}. Use `variant=None` to load this model from those weights."
                            )

                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}."
                        )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'xxx', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
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

        config.name_or_path = pretrained_model_name_or_path

        network = cls(config, *model_args, **model_kwargs)

        if ms.context.get_auto_parallel_context('parallel_mode') in \
                ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel'] and auto_trans_ckpt:
            logger.info(".........Remake strategy and transformed_checkpoint dir.........")
            clear_auto_trans_output()

            logger.info(".........Remake src_checkpoint dir.........")
            src_checkpoint_dir = os.path.join(get_output_root_path(), "src_checkpoint")
            remake_folder(src_checkpoint_dir, permissions=0o750)
            src_checkpoint = os.path.join(src_checkpoint_dir, "mindspore_model.ckpt")

            logger.info(".........Set strategy save dir.........")
            rank_id = get_real_rank()
            strategy_ckpt_save_dir = os.path.join(get_output_root_path(), "strategy")
            dst_strategy = os.path.join(strategy_ckpt_save_dir, f"ckpt_strategy_rank_{rank_id}.ckpt")
            context.set_auto_parallel_context(strategy_ckpt_save_file=dst_strategy)
            logger.info(f"set strategy path to '{dst_strategy}'")

            model = Model(network)
            if network.config:
                batch_size = network.config.batch_size
                seq_length = network.config.seq_length
            else:
                batch_size = config.model.model_config.batch_size
                seq_length = config.model.model_config.seq_length
            input_ids = np.ones(shape=tuple([batch_size, seq_length]))
            # pylint: disable=E1111
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            logger.info(".........Building model.........")
            model.infer_predict_layout(*infer_data)
            # assert os.path.exists(dst_strategy), f"{dst_strategy} is not found!"

            logger.info(".........Get src checkpoint.........")
            if rank_id == 0:
                src_checkpoint = cls._get_src_checkpoint(
                    state_dict=state_dict,
                    resolved_archive_file=resolved_archive_file,
                    src_checkpoint=src_checkpoint,
                )
            barrier_world("Get src checkpoint by rank 0.")

            logger.info(".........Transform checkpoint.........")
            # transform sharded checkpoint to one or state_dict to one checkpoint
            transform_ckpt = TransformCkpt(auto_trans_ckpt=True)
            transformed_checkpoint = transform_ckpt(
                src_checkpoint=src_checkpoint,
                src_strategy=None,
            )
            resolved_archive_file = os.path.join(transformed_checkpoint,
                                                 os.path.basename(src_checkpoint).split(".")[0],
                                                 f"rank_{get_real_rank()}",
                                                 f"checkpoint_{get_real_rank()}.ckpt")

        # load params into net
        (
            network,
            missing_keys,
            unexpected_keys,
            mismatched_keys
        ) = cls._load_pretrained_model(
            network,
            state_dict,
            resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes
        )

        # make sure we use the model's config since the __init__ call might have copied it
        config = network.config

        # Set model in evaluation mode to deactivate DropOut modules by default
        network.set_train(False)

        # If it is a model with generation capabilities, attempt to load the generation config
        if network.can_generate() and pretrained_model_name_or_path is not None:
            network.generation_config = GenerationConfig.from_model_config(
                config
            )

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
            }
            return network, loading_info

        return network

    @classmethod
    def _get_src_checkpoint(
            cls,
            state_dict,
            resolved_archive_file,
            src_checkpoint
    ):
        """Get src checkpoint for ckpt transform."""
        if state_dict is None:
            if isinstance(resolved_archive_file, str):
                if not os.path.exists(resolved_archive_file):
                    raise ValueError(f"resolved_archive_file:{resolved_archive_file} not found!")
                make_soft_link(src_checkpoint, resolved_archive_file)
                return src_checkpoint
            if isinstance(resolved_archive_file, (list, tuple)):
                state_dict = {}
                for resolved_archive_file_ in resolved_archive_file:
                    if not os.path.exists(resolved_archive_file_):
                        raise ValueError(f"resolved_archive_file_:{resolved_archive_file_} not found!")
                    state_dict.update(load_checkpoint(resolved_archive_file_))
            else:
                raise ValueError(f"`resolved_archive_file` should be str, list or tuple,"
                                 f" but get {type(resolved_archive_file)}.")
        ms.save_checkpoint(state_dict, src_checkpoint)
        return src_checkpoint

    @classmethod
    def _load_pretrained_model(
            cls,
            model,
            state_dict,
            resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=False
    ):
        """load pretrained model"""
        model_state_dict = {}
        for item in model.get_parameters():
            model_state_dict[item.name] = item.data
        expected_keys = list(model_state_dict.keys())

        if state_dict is None:
            if isinstance(resolved_archive_file, (list, tuple)):
                state_dict = {}
                for resolved_archive_file_ in resolved_archive_file:
                    if not os.path.exists(resolved_archive_file_):
                        raise ValueError(f"resolved_archive_file_:{resolved_archive_file_} not found!")
                    state_dict.update(load_checkpoint(resolved_archive_file_))
            elif isinstance(resolved_archive_file, str):
                if not os.path.exists(resolved_archive_file):
                    raise ValueError(f"resolved_archive_file:{resolved_archive_file} not found!")
                state_dict = load_checkpoint(resolved_archive_file)
            else:
                raise ValueError(f"`resolved_archive_file` should be str, list or tuple,"
                                 f" but get {type(resolved_archive_file)}.")
        loaded_keys = list(state_dict.keys())

        prefix = model.base_model_prefix
        if prefix:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            prefix_ = f"{prefix}."
            expected_keys = [s[len(prefix_):] if s.startswith(prefix_) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if model_key not in model_state_dict:
                        if remove_prefix_from_model:
                            model_key = f"{prefix}.{checkpoint_key}"
                        elif add_prefix_to_model:
                            model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        msgs = load_param_into_net(model, state_dict)
        missing_keys, unexpected_keys = msgs

        if unexpected_keys:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if missing_keys:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if mismatched_keys:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        .. warning::
            This API is experimental and may have some slight breaking changes in the next releases.

        Args:
            auto_class (Union[str, type], optional):
                The auto class to register this new model with. Default: ``AutoModel`` .
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import mindformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

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

    @classmethod
    def show_support_list(cls):
        """show_support_list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get_support_list method"""
        return cls._support_list

    def kvcache(self, layer_idx):
        raise RuntimeError("Override the kvcache method to get key cache value cache for swap.")

    def fuse_weight_from_ckpt(self, ckpt_dict):
        """
        Fuse weight function, which fuse the weight and update parameter dictionary before loading weight into network.

        Args:
            ckpt_dict (dict): dictionary containing weights.

        Returns:
            ckpt_dict (dict) after fusing.
        """
        return ckpt_dict

    def convert_name(self, weight_name):
        """convert HuggingFace weight name to MindFormers weight name"""
        raise RuntimeError(f"{self.__class__.__name__} does not implemented convert_name method.")

    def convert_weight_dict(self, source_dict):
        """convert HuggingFace weight dict to MindFormers weight dict"""
        raise RuntimeError(f"{self.__class__.__name__} does not implemented convert_weight_dict method.")

    def convert_map_dict(self, source_dict):
        """convert HuggingFace map dict to MindFormers map dict"""
        raise RuntimeError(f"{self.__class__.__name__} does not implemented convert_map_dict method.")
