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
"""Configuration."""
import ast
# pylint: disable=W0613
import inspect
import argparse
import copy
import os
import re
from typing import Union, List, Optional
from collections import deque, OrderedDict
from abc import ABCMeta, abstractmethod
import numbers
from functools import partial

import yaml
import mindspore.common.dtype as mstype
try:
    from mindspore._checkparam import Validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as Validator
    import mindspore._checkparam as Rel
from mindspore.common.initializer import _INITIALIZER_ALIAS

from mindformers.tools import DictConfig, logger
from mindformers.experimental.parallel_core.pynative.utils import load_yaml, DictWithValueError, divide

_SUPPORT_DTYPE_DICT = DictWithValueError(
    {"float16": mstype.float16, "float32": mstype.float32, "bfloat16": mstype.bfloat16}
)

_SUPPORT_INIT_METHOD = DictWithValueError(_INITIALIZER_ALIAS)

mapping_dict = {
    # model config
    'padded_vocab_size': 'model_config.vocab_size',
    'hidden_size': 'model_config.hidden_size',
    'seq_length': 'model_config.seq_length',
    'rotary_base': 'model_config.rotary_base',
    'num_layers': 'model_config.num_layers',
    'position_embedding_type': 'model_config.position_embedding_type',
    'use_rotary_position_embeddings': 'model_config.use_rotary_embedding',
    'init_method_std': 'model_config.init_method_std',
    'normalization': 'model_config.normalization',
    'norm_epsilon': 'model_config.norm_epsilon',
    'group_query_attention': 'model_config.group_query_attention',
    'num_attention_heads': 'model_config.num_attention_heads',
    'num_query_groups': 'model_config.num_query_groups',
    'attention_dropout': 'model_config.attention_dropout_rate',
    'ffn_hidden_size': 'model_config.ffn_hidden_size',
    'hidden_dropout': 'model_config.hidden_dropout_rate',
    'attention_softmax_in_fp32': 'model_config.attention_softmax_in_fp32',
    'use_flash_attn': 'model_config.use_flash_attention',
    'untie_embeddings_and_output_weights': 'model_config.untie_embeddings_and_output_weights',
    'transformer_impl': 'model_config.transformer_impl',
    'recompute_granularity': 'model_config.recompute_granularity',
    'recompute_method': 'model_config.recompute_method',
    'recompute_num_layers': 'model_config.recompute_num_layers',
    'fp16_lm_cross_entropy': 'model_config.fp16_lm_cross_entropy',
    'fp32_residual_connection': 'model_config.fp32_residual_connection',
    'add_qkv_bias': 'model_config.qkv_has_bias',
    'add_dense_bias': 'model_config.out_proj_has_bias',
    'add_bias_linear': 'model_config.add_bias_linear',
    'use_sandwich_norm': 'model_config.use_sandwich_norm',
    'pre_tockens': 'model_config.fa_config.pre_tokens',
    'next_tockens': 'model_config.fa_config.next_tokens',
    'shape_order': 'model_config.fa_config.input_layout',
    'attn_post_norm_scale': 'model_config.attn_post_norm_scale',
    'ffn_post_norm_scale': 'model_config.ffn_post_norm_scale',
    'params_dtype': 'model_config.params_dtype',
    'compute_dtype': 'model_config.compute_dtype',
    'embedding_dtype': 'model_config.embedding_init_dtype',
    'use_fused_swiglu': 'model_config.apply_swiglu_fusion',
    'apply_rope_fusion': 'model_config.apply_rope_fusion',
    'hidden_act': 'model_config.hidden_act',
    # training config
    'seed': 'training_config.seed',
    'log_interval': 'training_config.log_interval',
    'train_iters': 'training_config.training_iters',
    'save_interval': 'training_config.save_interval',
    'eval_interval': 'training_config.eval_interval',
    'accumulate_allreduce_grads_in_fp32': 'training_config.accumulate_allreduce_grads_in_fp32',
    'clip_grad': 'optimizer_config.clip_grad',
    'bf16': 'training_config.bf16',
    'fp16': 'training_config.fp16',
    'loss_scale': 'training_config.loss_scale',
    'initial_loss_scale': 'training_config.loss_scale_value',
    'loss_scale_window': 'training_config.loss_scale_window',
    'hysteresis': 'training_config.loss_scale_factor',
    'use_distributed_optimizer': 'training_config.use_distributed_optimizer',
    'resume_training': 'training_config.resume_training',
    'resume_crc_check': 'training_config.crc_check',
    'load_checkpoint': 'training_config.load_checkpoint',
    'save': 'training_config.output_dir',
    'ckpt_prefix': 'training_config.prefix',
    'ckpt_format': 'training_config.ckpt_format',
    'keep_checkpoint_max': 'training_config.keep_checkpoint_max',
    'wrap_with_ddp': 'training_config.wrap_with_ddp',
    'bucket_size': 'training_config.bucket_size',
    'enable_mem_align': 'training_config.enable_mem_align',
    'overlap_grad_reduce': 'training_config.overlap_grad_reduce',
    'delay_grad_reduce': 'training_config.delay_grad_reduce',
    # dataset config
    'reset_attention_mask': 'dataset_config.reset_attention_mask',
    'reset_position_ids': 'dataset_config.reset_position_ids',
    'eod_mask_loss': 'dataset_config.eod_mask_loss',
    'pad_token_id': 'dataset_config.pad_token_id',
    'eos_token_id': 'dataset_config.eos_token_id',
    'drop_remainder': 'dataset_config.drop_remainder',
    'micro_batch_size': 'dataset_config.batch_size',
    'micro_batch_num': 'dataset_config.micro_batch_num',
    # optimizer config
    'optimizer': 'optimizer_config.optimizer_type',
    'adam_beta1': 'optimizer_config.betas',
    'adam_beta2': 'optimizer_config.betas',
    'adam_eps': 'optimizer_config.eps',
    'lr_decay_style': 'optimizer_config.lr_decay_style',
    'lr': 'optimizer_config.learning_rate',
    'min_lr': 'optimizer_config.min_lr',
    'lr_warmup_iters': 'optimizer_config.lr_warmup_iters',
    'lr_decay_iters': 'optimizer_config.lr_decay_iters',
    'override_opt_param_scheduler': 'optimizer_config.override_opt_param_scheduler',
    'weight_decay': 'optimizer_config.weight_decay',
    'overlap_param_gather': 'optimizer_config.overlap_param_gather',
    # parallel config
    'tensor_model_parallel_size': 'parallel_config.tensor_model_parallel_size',
    'context_parallel_size': 'parallel_config.context_parallel_size',
    'expert_model_parallel_size': 'parallel_config.expert_model_parallel_size',
    'virtual_pipeline_model_parallel_size': 'parallel_config.virtual_pipeline_model_parallel_size',
    'num_layers_per_virtual_pipeline_stage': 'parallel_config.num_layers_per_virtual_pipeline_stage',
    'sequence_parallel': 'parallel_config.sequence_parallel',
    'pipeline_model_parallel_size': 'parallel_config.pipeline_model_parallel_size',
    'num_layer_list': 'parallel_config.num_layer_list',
    'recompute': 'parallel_config.recompute',
    'select_recompute': 'parallel_config.select_recompute',
    'select_comm_recompute': 'parallel_config.select_comm_recompute'
}


def config_to_str(cls, gap=2 * " "):
    """Return class attribute str for print."""
    attributes = vars(cls)
    print_str = "\n" + cls.__class__.__name__ + "\n"
    for name, val in attributes.items():
        new_str = str(val)
        new_str = new_str.replace("\n", "\n" + gap)
        print_str += f"{gap}{name}: {new_str}\n"

    return print_str


def dict_to_dictconfig(input_dict):
    """Convert dict to DictConfig."""
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            input_dict[key] = dict_to_dictconfig(value)
        return DictConfig(**input_dict)
    return input_dict


def parse_value(value: str):
    """
    Parse value to int or float if possible, otherwise return original value.

    Args:
        value (str): value str need to parse

    Returns:
        parsed value from value string.
    """
    try:
        # 尝试转换为整数
        return int(value)
    except ValueError:
        try:
            # 尝试转换为浮点数
            return float(value)
        except ValueError:
            # 无法转换为数字，返回原始字符串
            return value


def get_default_config():
    """Append default config to raw dict.

    Returns:
        dict: raw dict with default config.
    """
    default_param_dict = {}
    default_param_dict['dataset_config.drop_remainder'] = True
    default_param_dict['dataset_config.data_layout'] = "SBH"
    default_param_dict['optimizer_config.optimizer_type'] = "mint.AdamW"
    default_param_dict['training_config.grad_clip_kwargs.clip_value'] = 1.0
    default_param_dict['training_config.grad_clip_kwargs.grad_clip_type'] = "ClipGlobalNorm"
    default_param_dict['training_config.eval_metric'] = "perplexity"
    default_param_dict['training_config.loss_func_kwargs.loss_func_type'] = "VocabParallelCrossEntropy"
    default_param_dict['model_config.output_layer_init_method'] = "normal"
    default_param_dict['model_config.mask_func_type'] = "attn_mask_fill"
    default_param_dict['model_config.gated_linear_unit'] = True
    default_param_dict['model_config.mlp_has_bias'] = False
    default_param_dict['model_config.embedding_init_dtype'] = 'float32'
    default_param_dict['parallel_config.model_customize_staged'] = True

    return default_param_dict


def modify_flatten_dict(flatten_dict: dict, default_param_dict: dict):
    """Modify flatten dict.

    Args:
        flatten_dict (dict): flatten dict.
        default_param_dict (dict): default param dict.

    Returns:
        dict: modified flatten dict.
    """
    flatten_dict = modify_megatron_param(flatten_dict)

    if flatten_dict.get('use_fused_rmsnorm', False):
        flatten_dict.pop('use_fused_rmsnorm', False)
        flatten_dict['normalization'] = 'FusedRMSNorm'

    if flatten_dict.get('swiglu', False):
        flatten_dict['hidden_act'] = 'swiglu'

    if str(flatten_dict.get('params_dtype', 'float32')) == 'torch.bfloat16':
        flatten_dict['params_dtype'] = 'bfloat16'

    if flatten_dict.get('bf16', False):
        flatten_dict['compute_dtype'] = 'bfloat16'

    if flatten_dict.get('fp16', False):
        flatten_dict['compute_dtype'] = 'float16'

    if str(flatten_dict.get('embedding_dtype', 'float32')) == 'torch.bfloat16':
        flatten_dict['embedding_dtype'] = 'bfloat16'

    loss_scale_key = ['loss_scale', 'initial_loss_scale', 'hysteresis', 'loss_scale_window']
    for key in loss_scale_key:
        if flatten_dict.get(key, None):
            flatten_dict[key] = int(flatten_dict[key])

    for key, value in default_param_dict.items():
        if key not in flatten_dict:
            flatten_dict[key] = value
            mapping_dict[key] = key

    return flatten_dict


def modify_megatron_param(flatten_dict):
    """Modify megatron param.

    Args:
        flatten_dict (dict): flatten dict.

    Returns:
        dict: modified flatten dict.
    """
    if 'adam_beta1' in flatten_dict and 'adam_beta2' in flatten_dict:
        flatten_dict['optimizer_config.betas'] = [flatten_dict['adam_beta1'], flatten_dict['adam_beta2']]
        flatten_dict.pop('adam_beta1')
        flatten_dict.pop('adam_beta2')
        mapping_dict['optimizer_config.betas'] = 'optimizer_config.betas'

    if  'global_batch_size' in flatten_dict and 'micro_batch_size' in flatten_dict:
        data_parallel_size = flatten_dict.get('data_parallel_size', 1)
        micro_batch_num = flatten_dict['global_batch_size'] // (flatten_dict['micro_batch_size'] * data_parallel_size)
        flatten_dict['micro_batch_num'] = micro_batch_num

    if 'num_layers_per_virtual_pipeline_stage' in flatten_dict and not flatten_dict.get('num_layer_list', None):
        num_layers_per_pipeline_stage = flatten_dict['num_layers'] // flatten_dict['pipeline_model_parallel_size']
        flatten_dict['virtual_pipeline_model_parallel_size'] = \
            num_layers_per_pipeline_stage // flatten_dict['num_layers_per_virtual_pipeline_stage']

    return flatten_dict


def flatten_dict_to_raw(flatten_dict: dict, model_type: str):
    """Flatten dict to raw dict.

    Args:
        flatten_dict (dict): flatten dict.
        model_type (str) : model type.

    Returns:
        dict: raw dict.
    """
    raw_dict = {}
    not_mapping_key = []
    default_param_dict = get_default_config()
    flatten_dict = modify_flatten_dict(flatten_dict, default_param_dict)
    for key, value in flatten_dict.items():
        if key not in mapping_dict:
            not_mapping_key.append(key)
            continue
        convert_key = mapping_dict[key]
        convert_key_list = convert_key.split('.')
        cur_dict = raw_dict
        for nested_key in convert_key_list[:-1]:
            if nested_key not in cur_dict:
                if nested_key == "model_config":
                    nested_key = model_type
                cur_dict[nested_key] = OrderedDict()
            cur_dict = cur_dict[nested_key]
        cur_dict[convert_key_list[-1]] = value

    if not_mapping_key:
        print(f'not mapping keys is: {not_mapping_key}')
    return raw_dict


def init_configs_from_args(run_args: argparse.Namespace = None, model_type: str = "model_config", config_classes=None):
    """Initialize config class from run command.

        Args:
            run_args (argparse.Namespace): run command.
            config_classes (Union[list[BaseConfig], None]): Config classes to be initialized. When no config class
                is passed in, all known configs will be initialized as optional config of AllConfig. Default: None
            model_type (str) : model config name

        Returns:
            Union[list[BaseConfig], AllConfig]: Initialized config instances, when no config class is passed in,
                AllConfig will be returned.
        """
    if not isinstance(run_args, argparse.Namespace):
        raise ValueError("run_args should be argparse.Namespace.")

    flatten_dict = vars(run_args)
    raw_dict = flatten_dict_to_raw(flatten_dict, model_type)

    return init_configs_from_dict(raw_dict, config_classes)


class BaseConfig(metaclass=ABCMeta):
    """
    Base config class, which enables validator registration while attribute setting, and depended config registration.

    Notice:
        All dict attributes will be transformered to DictConfig Recursively.
    """

    _validation_func_dict = {}
    _depended_configs = {}  # {config_class: optional_flag}

    @abstractmethod
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validation_func_dict = {}
        cls._depended_configs = {}
        for base in cls.__bases__:
            if issubclass(base, BaseConfig):
                # pylint: disable=W0212
                cls._validation_func_dict.update(base._validation_func_dict)
                # pylint: disable=W0212
                cls._depended_configs.update(base._depended_configs)

        # check if config_name is set
        if not hasattr(cls, "config_name"):
            raise ValueError(f"Config {cls.__name__} should have a 'config_name' class attribute.")

    def __setattr__(self, name, value):
        """Set attribute with validation."""
        validator_func = self._validation_func_dict.get(name)
        if validator_func is not None:
            value = validator_func(self, value)
        if isinstance(value, dict):
            value = dict_to_dictconfig(value)
        super().__setattr__(name, value)

    def __str__(self):
        return config_to_str(self)

    def update_attrs(self, **kwargs):
        """
        Update the attributes of the object with the given key-value pairs. Dict attributes will be transformed
            to DictConfig recursively.

        Args:
            **kwargs: Key-value pairs where the key is the attribute name and the value is the new value.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_all_sub_configs(cls):
        """
        Returns a set of all subclasses of the given class, including subclasses of subclasses.

        Returns:
            visited_configs (set[BaseConfig]): A set containing all subclasses of the given class.

        """
        visited_configs = set(cls.__subclasses__())
        bfs_queue = deque(visited_configs)

        while bfs_queue:
            current_config = bfs_queue.popleft()
            for direct_sub_config in current_config.__subclasses__():
                if direct_sub_config not in visited_configs:
                    visited_configs.add(direct_sub_config)
                    bfs_queue.append(direct_sub_config)
        return visited_configs


    @classmethod
    def is_depended_config(cls, config_class, raise_error=False):
        """Check whether the config class is a depended config.

        Args:
            config_class (BaseConfig): The config class to be checked.
        """
        if not issubclass(config_class, BaseConfig):
            raise ValueError(f"{config_class} should be a subclass of BaseConfig.")
        if raise_error and config_class not in cls._depended_configs:
            raise ValueError(f"{config_class} is not a depended config of {cls}.")
        return config_class in cls._depended_configs

    @classmethod
    def _register_depended_config(cls, config_class, optional=False):
        """Register depended config class and add validation function for the depended config.

        Args:
            config_class (BaseConfig): The depended config class.
            optional (bool): Whether the depended config is optional.
        """
        if cls.is_depended_config(config_class):
            logger.warning(f"{config_class} is already a depended config of {cls}.")

        # add validation function for the depended config
        if optional:
            def validate_config(config_instance, config):
                if config is not None and not isinstance(config, config_class):
                    raise ValueError(f"{config} should be an instance of {config_class} or None.")
                return config
        else:
            def validate_config(config_instance, config):
                if not isinstance(config, config_class):
                    raise ValueError(f"{config} should be an instance of {config_class}.")
                return config

        cls._validation_func_dict[config_class.__name__] = validate_config

        cls._depended_configs[config_class] = optional

    @classmethod
    def _remove_depended_config(cls, config_class):
        """Remove depended config class and validation function for the depended config.

        Args:
            config_class (BaseConfig): The depended config class to be removed.

        Raises:
            ValueError: If config_config is not config.
            ValueError: If config_class is not a depended config of cls.
        """
        cls.is_depended_config(config_class, raise_error=True)

        # remove validation function for the depended config
        removed_class = cls._validation_func_dict.pop(config_class.__name__)

        removed_class_optional_flag = cls._depended_configs.pop(config_class)

        return removed_class, removed_class_optional_flag

    @classmethod
    def remove_depended_config(cls, config_class):
        """Remove depended config class(s) and validation function(s) for the depended config(s).

        Args:
            config_class (Union[BaseConfig, list[BaseConfig]]): The depended config
                class or a list of depended config classes.
        """
        if isinstance(config_class, list):
            for config in config_class:
                cls._remove_depended_config(config)
        else:
            cls._remove_depended_config(config_class)

    @classmethod
    def update_depended_config(cls, original_config, new_config=None, optional=None):
        """Update depended config class and validation function for the depended config.

        Args:
            original_config (BaseConfig): The original depended config class.
            new_config (BaseConfig, optional): The new depended config class. if this is set,
                the original depended config will be replaced by the new depended config. Default: None.
            optional (bool, optional): Whether the new depended config is optional. if this is set,
                the optional flag of the depended config will be updated. Default: None.
        """
        cls.is_depended_config(original_config, raise_error=True)

        if new_config is None and optional is None:
            logger.warning("No new depended config or optional flag is set.")
        else:
            _, original_optional_flag = cls._remove_depended_config(original_config)
            optional_flag = optional if optional is not None else original_optional_flag
            new_config = new_config if new_config is not None else original_config
            cls._register_depended_config(new_config, optional_flag)

    @classmethod
    def get_depended_configs(cls):
        """get depended configs

        Returns:
            Dict[BaseConfig, bool]: The depended config class and optional flag.
        """
        return copy.deepcopy(cls._depended_configs)

    @classmethod
    def register_depended_config(cls, config_class, optional=False):
        """Register depended config class(s) and add validation function(s) for the depended config(s).

        Args:
            config_class (Union[BaseConfig, list[BaseConfig]]): The depended config
                class or a list of depended config classes.
            optional (Union[bool, list[bool]], optional): Whether the depended config is optional.
                if is a list, the length of the list should be the same as the config_class. Default: False.
        """
        if isinstance(config_class, list):
            optional_flags = optional if isinstance(optional, list) else [optional] * len(config_class)
            if len(config_class) != len(optional_flags):
                raise ValueError("The length of config_class and optional should be the same.")
            for idx, config in enumerate(config_class):
                cls._register_depended_config(config, optional=optional_flags[idx])
        else:
            cls._register_depended_config(config_class, optional=optional)

    @classmethod
    def validator(cls, name):
        """Validator decorator, which registers validation function for attribute.
        Once the validation function is registered, the attribute will be validated while __setattr__ is called.

        Args:
            name (str): The name of the attribute to be validated.

        Returns:
            decorator: The decorator function.
        """

        def decorator(func):
            cls._validation_func_dict[name] = func
            return func

        return decorator

    def get_extra_params(self):
        """Get extra params that are not defined in the __init__ methods.

        Returns:
            dict: Extra params.
        """
        known_params = inspect.signature(self.__class__).parameters
        return {k: v for k, v in self.__dict__.items() if k not in known_params}

    def get_needed_params_for_class(self, a_class):
        """
        Returns a dictionary of the needed parameters for a given class.

        Args:
            a_class: The class for which the needed parameters are to be retrieved.

        Returns:
            A dictionary containing the needed parameters and their corresponding values from the current instance.
        """
        needed_parameters = inspect.signature(a_class).parameters.keys()
        return {k: v for k, v in self.__dict__.items() if k in needed_parameters}


class AllConfig(BaseConfig):
    """A Config that contains all other configs, which will be used in init_configs methods as the default config.

    Args:
        **kwargs: Other arguments.
    """

    # set config name for identifying while using init_configs methods
    config_name = "all_config"

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__()
        self.update_attrs(**kwargs)

    @classmethod
    def register_all_known_configs(cls):
        """Register all known configs. Known configs are the subclasses of BaseConfig.

        Notice:
            - This method should be called after all known configs are defined.
        """
        all_configs = BaseConfig.get_all_sub_configs()
        all_configs.remove(cls)

        cls.register_depended_config(list(all_configs), optional=True)


def build_dependency_graph_of_configs(config_classes):
    """
    Builds a dependency graph of configuration classes based on their dependencies.

    Args:
        config_classes (list[BaseConfig]): A list of configuration classes.

    Returns:
        dependency_graph dict[BaseConfig, list[BaseConfig]]: A dictionary representing the dependency graph,
        where the keys are configuration classes and the values are lists of configuration classes that
        they depend on.

    Raises:
        ValueError: If a configuration class is not a subclass of BaseConfig.
        ValueError: If a configuration class does not have the 'config_name' attribute set.

    """
    dependency_graph = {}
    for config_class in config_classes:
        # check if config_class is already in dependency_graph
        if config_class in dependency_graph:
            continue
        # check class
        if not issubclass(config_class, BaseConfig):
            raise ValueError(f"{config_class} should be a subclass of BaseConfig.")
        # check if config_name is set
        if not hasattr(config_class, "config_name"):
            raise ValueError("You should set config_name for config class before using init_configs methods.")

        depended_configs_dict = config_class.get_depended_configs()
        depended_configs = []

        # get depended configs
        for depended_config, optional_flag in depended_configs_dict.items():
            # filter out optional depended config which is not passed in
            if not optional_flag or depended_config in config_classes:
                depended_configs.append(depended_config)
                if depended_config not in config_classes:
                    config_classes.append(depended_config)
                    logger.warning(
                        f"Will initialize config {depended_config.config_name}"
                        + f"since it is required by {config_class.config_name}."
                    )

        dependency_graph[config_class] = depended_configs
    return dependency_graph


def _check_arguments(configs):
    """ check arguments. """
    training_config = configs.get("training_config", None)
    model_config = configs.get("model_config", None)
    transformer_config = None
    for key in configs:
        if isinstance(configs[key], TransformerConfig):
            transformer_config = configs[key]
            break
    if transformer_config and training_config:
        if training_config.fp16:
            logger.warning("Use fp16, 'params_dtype' and 'compute_dtype' will be set to 'float16' automatically.")
            transformer_config.params_dtype = 'float16'
            transformer_config.compute_dtype = 'float16'
        if training_config.bf16:
            logger.warning("Use bf16, 'params_dtype' and 'compute_dtype' will be set to 'bfloat16' automatically.")
            transformer_config.params_dtype = 'bfloat16'
            transformer_config.compute_dtype = 'bfloat16'

    if model_config:
        hidden_act = model_config.hidden_act
        if hidden_act == 'swiglu':
            logger.warning("Use swiglu, 'gated_linear_unit' will be set to 'True' automatically.")
            model_config.gated_linear_unit = True
        else:
            logger.warning(f"Use {hidden_act}, 'gated_linear_unit' will be set to 'False' automatically.")
            model_config.gated_linear_unit = False

        if model_config.parallel_config.recv_dtype != model_config.compute_dtype:
            logger.warning("Model_config.compute_dtype must as same as parallel_config.recv_dtype, "
                           f"but model_config.compute_dtype is {model_config.compute_dtype}, "
                           f"parallel_config.recv_dtype is {model_config.parallel_config.recv_dtype}, "
                           f"'parallel_config.recv_dtype' will be set to '{model_config.compute_dtype}' "
                           "automatically.")
            model_config.parallel_config.recv_dtype = model_config.compute_dtype


# pylint: disable=W0102
def init_configs_from_dict(raw_dict: dict, config_classes=None):
    """
    Initialize config class from configuration dictionary.

    Args:
        raw_dict (dict): Configuration dictionary.
        config_classes (Union[list[BaseConfig], None]): Config classes to be initialized. When no config class
            is passed in, all known configs will be initialized as optional config of AllConfig. Default: None

    Returns:
        Union[list[BaseConfig], AllConfig]: Initialized config instances, when no config class is passed in,
            AllConfig will be returned.

    Raises:
        ValueError: If a cycle is detected in the configuration dependencies.
    """
    # check if no config class is passed in
    no_passed_in_configs = config_classes is None

    # when no config class is passed in, all known configs will be initialized as optional config of AllConfig
    if no_passed_in_configs:
        config_classes = [AllConfig]
        # register here to ensure all known configs can be reached
        AllConfig.register_all_known_configs()
        # read configs to be initialized directly from raw_dict
        known_configs = AllConfig.get_depended_configs().keys()
        for known_config in known_configs:
            if known_config.config_name in raw_dict:
                config_classes.append(known_config)
        # make sure all_config is in raw_dict
        if AllConfig.config_name not in raw_dict:
            raw_dict[AllConfig.config_name] = {}
    else:
        returned_config_names = [config_class.config_name for config_class in config_classes]
        if AllConfig in config_classes:
            AllConfig.register_all_known_configs()

    # construct dependency graph
    dependency_graph = build_dependency_graph_of_configs(config_classes)

    # topological sort with cycle detection
    visited = {config_class: False for config_class in config_classes}
    on_path = {config_class: False for config_class in config_classes}  # Tracks nodes on the current path
    config_class_initialization_stack = deque()

    def dfs(config_class):
        visited[config_class] = True
        on_path[config_class] = True  # Mark as on the current path
        for dependency in dependency_graph.get(config_class):
            if on_path[dependency]:
                raise ValueError(
                    "Cycle detected in configuration dependencies:" +
                    f"{config_class.config_name} -> {dependency.config_name}"
                )
            if not visited[dependency]:
                dfs(dependency)

        on_path[config_class] = False  # Remove from the current path
        config_class_initialization_stack.append(config_class)

    for config_class in config_classes:
        if not visited[config_class]:
            dfs(config_class)

    # initialize configs
    initialized_configs = {}
    while config_class_initialization_stack:
        config_class = config_class_initialization_stack.popleft()
        if config_class.config_name not in raw_dict:
            raise ValueError(f"Config {config_class.config_name} not found.")
        kwargs = raw_dict[config_class.config_name]
        depened_config_instances = {
            depended_config.config_name: initialized_configs.get(depended_config.config_name)
            for depended_config in dependency_graph.get(config_class)
        }
        kwargs.update(depened_config_instances)
        config_instance = config_class(**kwargs)
        initialized_configs[config_class.config_name] = config_instance
        logger.warning(f"Initialized config {config_class.config_name}:")
        logger.warning(config_instance)

    # add some rules for arguments
    _check_arguments(initialized_configs)

    # if no passed in configs, add all other parameters to AllConfig as dict config
    if no_passed_in_configs:
        for config_name in raw_dict.keys():
            if config_name not in initialized_configs:
                setattr(initialized_configs.get(AllConfig.config_name), config_name, raw_dict.get(config_name))
        return initialized_configs.get(AllConfig.config_name)

    # return in order if config classes are passed in
    return [initialized_configs.get(config_name) for config_name in returned_config_names]


# pylint: disable=W0102
def init_configs_from_yaml(file_path: str, config_classes=None, **kwargs):
    """Initialize config class from configuration yaml file.

    Args:
        file_path (str): configuration yaml file.
        config_classes (Union[list[BaseConfig], None]): Config classes to be initialized. When no config class
            is passed in, all known configs will be initialized as optional config of AllConfig. Default: None
        kwargs (dict): extra arguments.

    Returns:
        Union[list[BaseConfig], AllConfig]: Initialized config instances, when no config class is passed in,
            AllConfig will be returned.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path should be a string.")
    if not file_path.endswith("yaml") and not file_path.endswith("yml"):
        raise ValueError("file_path should be a yaml file.")
    filepath = os.path.realpath(file_path)
    with open(filepath, encoding="utf-8") as fp:
        raw_dict = load_yaml(fp, yaml_loader=yaml.FullLoader)

    raw_dict.update(kwargs)

    return init_configs_from_dict(raw_dict, config_classes)


class GeneralConfig(BaseConfig):
    """A General Config

    Args:
        **kwargs: Arbitrary keyword arguments.
    """

    # set config name for identifying while using init_configs methods
    config_name = "general_config"

    def __init__(self, **kwargs):
        super().__init__()
        self.update_attrs(**kwargs)


class ModelParallelConfig(BaseConfig):
    """Parallel config class.

    Args:
        tensor_model_parallel_size (int): Dimensionality of tensor parallel. Default: 1.
        pipeline_model_parallel_size (int): Number of stages when using pipeline parallel. Default: 1.
        context_parallel_size (int): Dimensionality of context parallel. Default: 1.
        expert_model_parallel_size (int): Dimensionality of expert parallel. Default: 1.
        virtual_pipeline_model_parallel_size (int): Number of virtual stages when using pipeline parallel.
            Default: None.
        sequence_parallel (bool): Enable sequence parallel. Default: False.
        recv_dtype (str, optional): Communication data type of p2p communication when using pipeline
            parallel. Default: 'float32'.
        zero_level (str, optional): Zero level for ZeRO optimizer,
            if None, will not use ZeRO optimizer. Default: None.
        gradient_accumulation_fusion (bool): Enable gradient accumulation
            during linear backward execution. Default: False.
        overlap_p2p_comm (bool): Enable overlap p2p commucation in pipeline interleaved. Default: False.
        num_layer_list (list): User-defined pipeline parallel model layer division. Default: None.
        recompute_config (dict): Recompute strateges. Default: None.
    """

    # set config name for identifying while using init_configs methods
    config_name = "parallel_config"

    def __init__(
            self,
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            context_parallel_size: int = 1,
            expert_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: int = None,
            sequence_parallel: bool = False,
            recv_dtype: str = "float32",
            zero_level: str = None,
            standalone_embedding_stage: bool = False,
            overlap_grad_reduce: bool = False,
            gradient_accumulation_fusion: bool = False,
            overlap_p2p_comm: bool = True,
            use_cpu_initialization: bool = False,
            deterministic_mode: bool = False,
            num_layer_list: list = None,
            recompute_config: dict = None,
            recompute: str = None,
            select_recompute: str = None,
            select_comm_recompute: str = None,
            **kwargs,
    ):
        super().__init__()

        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.recv_dtype = recv_dtype
        self.zero_level = zero_level
        self.standalone_embedding_stage = standalone_embedding_stage
        self.overlap_grad_reduce = overlap_grad_reduce
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.overlap_p2p_comm = overlap_p2p_comm
        self.use_cpu_initialization = use_cpu_initialization
        self.deterministic_mode = deterministic_mode
        self.num_layer_list = num_layer_list
        self.recompute_config = recompute_config if recompute_config is not None else {}
        if recompute is not None:
            self.recompute_config['recompute'] = list(ast.literal_eval(recompute))
        if select_recompute is not None:
            self.recompute_config['select_recompute'] = list(ast.literal_eval(select_recompute))
        if select_comm_recompute is not None:
            self.recompute_config['select_comm_recompute'] = list(ast.literal_eval(select_comm_recompute))
        if not self.recompute_config:
            self.recompute_config = None

        self.update_attrs(**kwargs)


@ModelParallelConfig.validator("tensor_model_parallel_size")
def validate_tensor_model_parallel_size(config_instance, tensor_model_parallel_size):
    """Validate tensor_model_parallel_size."""
    Validator.check_positive_int(tensor_model_parallel_size, "tensor_model_parallel_size")
    return tensor_model_parallel_size


@ModelParallelConfig.validator("pipeline_model_parallel_size")
def validate_pipeline_model_parallel_size(config_instance, pipeline_model_parallel_size):
    """Validate pipeline_model_parallel_size."""
    Validator.check_positive_int(pipeline_model_parallel_size, "pipeline_model_parallel_size")
    return pipeline_model_parallel_size


@ModelParallelConfig.validator("context_parallel_size")
def validate_context_parallel_size(config_instance, context_parallel_size):
    """Validate context_parallel_size."""
    Validator.check_positive_int(context_parallel_size, "context_parallel_size")
    return context_parallel_size


@ModelParallelConfig.validator("expert_model_parallel_size")
def validate_expert_model_parallel_size(config_instance, expert_model_parallel_size):
    """Validate expert_model_parallel_size."""
    Validator.check_positive_int(expert_model_parallel_size, "expert_model_parallel_size")
    return expert_model_parallel_size


@ModelParallelConfig.validator("virtual_pipeline_model_parallel_size")
def validate_virtual_pipeline_model_parallel_size(config_instance, virtual_pipeline_model_parallel_size):
    """Validate virtual pipeline stage."""
    if virtual_pipeline_model_parallel_size is not None:
        Validator.check_positive_int(virtual_pipeline_model_parallel_size,
                                     "virtual_pipeline_model_parallel_size")
    return virtual_pipeline_model_parallel_size


@ModelParallelConfig.validator("sequence_parallel")
def validate_sequence_parallel(config_instance, sequence_parallel):
    """Validate sequence_parallel."""
    Validator.check_bool(sequence_parallel, "sequence_parallel")
    return sequence_parallel


@ModelParallelConfig.validator("recv_dtype")
def validate_recv_dtype(config_instance, recv_dtype):
    """Validate recv_dtype."""
    if recv_dtype in (mstype.float16, mstype.float32, mstype.bfloat16):
        return recv_dtype
    return _SUPPORT_DTYPE_DICT[recv_dtype]


@ModelParallelConfig.validator("zero_level")
def validate_zero_level(config_instance, zero_level):
    """Validate zero_level."""
    if zero_level is not None:
        Validator.check_string(zero_level, ["z1", "z2", "z3"], "zero_level")
        if (
                config_instance.sequence_parallel
                or config_instance.pipeline_model_parallel_size > 1
                or config_instance.expert_model_parallel_size > 1
                or config_instance.context_parallel_size > 1
        ):
            logger.warning(
                "Accuracy is not guaranteed when zero is used with parallel"
                + "strategies other than data parallel and tensor parallel."
            )
    return zero_level


@ModelParallelConfig.validator("gradient_accumulation_fusion")
def validate_gradient_accumulation_fusion(config_instance, gradient_accumulation_fusion):
    """Validate gradient_accumulation_fusion."""
    Validator.check_bool(gradient_accumulation_fusion, "gradient_accumulation_fusion")
    return gradient_accumulation_fusion

@ModelParallelConfig.validator("overlap_p2p_comm")
def validate_overlap_p2p_comm(config_instance, overlap_p2p_comm):
    """Validate if overlap_p2p_comm is bool."""
    Validator.check_bool(overlap_p2p_comm, "overlap_p2p_comm")
    return overlap_p2p_comm

@ModelParallelConfig.validator("use_cpu_initialization")
def validate_use_cpu_initialization(config_instance, use_cpu_initialization):
    """Validate use_cpu_initialization."""
    Validator.check_bool(use_cpu_initialization, "use_cpu_initialization")
    return use_cpu_initialization


@ModelParallelConfig.validator("deterministic_mode")
def validate_deterministic_mode(config_instance, deterministic_mode):
    """Validate deterministic_mode."""
    Validator.check_bool(deterministic_mode, "deterministic_mode")
    return deterministic_mode


@ModelParallelConfig.validator("num_layer_list")
def validate_num_layer_list(config_instance, num_layer_list):
    """Validate num_layer_list."""
    if num_layer_list is not None:
        if isinstance(num_layer_list, str):
            num_layer_list = list(ast.literal_eval(num_layer_list))
        Validator.check_value_type("num_layer_list", num_layer_list, list)
    return num_layer_list


@ModelParallelConfig.validator("recompute_config")
def validate_recompute_config(config_instance, recompute_config):
    """Validate recompute_config."""
    default_recompute_config = {'recompute': [],
                                'select_recompute': [],
                                'select_comm_recompute': []}
    check_keys = default_recompute_config.keys()

    if recompute_config is not None:
        if not isinstance(recompute_config, dict):
            raise TypeError("recompute_config should be a dict.")
        for key, value in recompute_config.items():
            if key not in check_keys:
                raise ValueError(f"Key '{key}' is not supported in recompute_config.")
            if value and not isinstance(value, list):
                raise TypeError(f"Key '{key}' should be list in recompute_config.")
    return recompute_config


class DatasetConfig(BaseConfig):
    r"""Dataset config class.

    Args:
        dataset_dir (str, optional): Dataset file directory. Default: './dataset'.
        shuffle (bool, optional): Shuffle dataset. Default: None.
        kwargs (dict, optional): Other dataset config arguments.
        batch_size (int, optional): batch size / micro_batch_size for training and evaluation. Default: 1.
        micro_batch_num (int, optional): Number of micro batch when using pipeline parallel or
            gradient accumulation. Defaults: 1.
        data_layout (str, optional): Input layout. Default: "BSH".
        train_samples (int, optional): Number of train samples for sample-based training. Default: 0.
    """

    # set config name for identifying while using init_configs methods
    config_name = "dataset_config"

    def __init__(
            self,
            dataset_dir: str = "./dataset",
            shuffle: bool = False,
            batch_size: int = 1,
            micro_batch_num: int = 1,
            train_samples: int = 0,
            data_layout: str = "BSH",
            **kwargs,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.micro_batch_num = micro_batch_num
        self.train_samples = train_samples
        self.data_layout = data_layout

        self.update_attrs(**kwargs)


@DatasetConfig.validator("dataset_dir")
def validate_dataset_dir(config_instance, dataset_dir):
    """Validate dataset_dir."""
    Validator.check_value_type("dataset_dir", dataset_dir, [str])
    return dataset_dir


@DatasetConfig.validator("shuffle")
def validate_shuffle(config_instance, shuffle):
    """Validate shuffle."""
    if shuffle is not None:
        Validator.check_bool(shuffle, "shuffle")
    return shuffle


@DatasetConfig.validator("batch_size")
def validate_batch_size(config_instance, batch_size):
    """Validate batch_size."""
    Validator.check_positive_int(batch_size, "batch_size")
    return batch_size


@DatasetConfig.validator("micro_batch_num")
def validate_micro_batch_num(config_instance, micro_batch_num):
    """Validate micro_batch_num."""
    Validator.check_positive_int(micro_batch_num, "micro_batch_num")
    return micro_batch_num


@DatasetConfig.validator("data_layout")
def validate_data_layout(config_instance, data_layout):
    Validator.check_string(data_layout, ["BSH", "SBH"], "data_layout")
    return data_layout


@DatasetConfig.validator("train_samples")
def validate_train_samples(config_instance, train_samples):
    """Validate train_samples."""
    Validator.check_non_negative_int(train_samples, "train_samples")
    return train_samples


class LoraConfig(BaseConfig):
    r"""LoRA config class.

    Args:
        use_lora (bool): Apply LoRA to the pretrain model. Default: False.
        lora_rank (int): The dimension for LoRA modules. Default: 8.
        lora_alpha (int): The alpha parameter for LoRA scaling. Default: 32.
        lora_dropout (float): the dropout rate for LoRA. Default: 0.0.
        target_cells (list[dict]): The names of the cells to build LoRA modules. If 'use_lora' is
            True, this argument should at least contains a dict with the key 'targets_cells' and
            the value of names of the cells to apply LoRA. In addition, if you want to set special
            rank or alpha for cells in target_cells, you can add dict to the list.
            For example:
            case 1:
                target_cells = [
                  {'target_cells':[
                      '.*.qkv_proj'
                  ]},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA.

            case 2:
                target_cells = [
                  {'target_cells':[
                      'backbone.layers.layers.0.attention.qkv_proj'
                  ]},
              ]
            In this case, the cell 'backbone.layers.layers.0.attention.qkv_proj' will be applied LoRA.

            case 3:
                [
                  {'target_cells':[
                      '.*.qkv_proj',
                  ]},
                  {'cell':'backbone.layers.layers.0.attention.qkv_proj', 'rank':4, 'alpha':16},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA. In addition, the rank
            and alpha of the cell 'backbone.layers.layers.0.attention.qkv_proj' is 4 and 32, the rank and
            alpha of other cells are set to 'lora_rank' and 'lora_alpha'.
    """
    config_name = "lora_config"

    def __init__(
            self,
            use_lora: bool = False,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.0,
            target_cells: List = None,
            **kwargs,
    ):
        super().__init__()
        self.use_lora = use_lora
        if use_lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.target_cells = target_cells
            self.lora_module = None

            self.update_attrs(**kwargs)


@LoraConfig.validator("use_lora")
def validate_use_lora(config_instance, use_lora):
    """Validate lora_rank."""
    Validator.check_bool(use_lora, "use_lora")
    return use_lora


@LoraConfig.validator("lora_rank")
def validate_lora_rank(config_instance, lora_rank):
    """Validate lora_rank."""
    Validator.check_positive_int(lora_rank, "lora_rank")
    return lora_rank


@LoraConfig.validator("lora_alpha")
def validate_lora_alpha(config_instance, lora_alpha):
    """Validate lora_alpha."""
    Validator.check_positive_int(lora_alpha, "lora_alpha")
    return lora_alpha


@LoraConfig.validator("lora_dropout")
def validate_lora_dropout(config_instance, lora_dropout):
    """Validate lora_dropout."""
    Validator.check_non_negative_float(lora_dropout, "lora_dropout")
    return lora_dropout


@LoraConfig.validator("target_cells")
def validate_target_cells(config_instance, target_cells):
    """Validate target_cells."""
    Validator.check_value_type("target_cells", target_cells, list)
    if not target_cells:
        raise ValueError("'target_cells' cannot not be empty.")

    # valid target_cells
    target_cells_defined = False
    for item in target_cells:
        if 'target_cells' in item.keys():
            if target_cells_defined:
                raise ValueError("'target_cells' cannot not be defined more than once.")
            target_cells_defined = True
            Validator.check_value_type("target_cells", item['target_cells'], list)
            target_cells_lst = item['target_cells']
            if not target_cells_lst:
                raise ValueError("for 'target_cells', the list of target_cells name must be set.")
    if not target_cells_defined:
        raise ValueError("for 'target_cells', the list of target_cells name must be set.")

    def _check_in_target_cells(cell_name):
        target_cell_found = False
        for target_key in target_cells_lst:
            match = re.match(target_key, cell_name)
            if match is not None and match.group() == cell_name:
                return target_key
        return target_cell_found

    # valid rank and alpha for specific cells
    specific_lora_cell = []
    for item in target_cells:
        if 'cell' in item.keys():
            cell_name = item['cell']
            if not _check_in_target_cells(cell_name):
                raise ValueError(
                    f"The cell need to set rank or alpha should be in the range defined by target_cells, but got name "
                    f"'{cell_name}'.")
            specific_lora_cell.append(item)
    return target_cells_lst, specific_lora_cell


class MoEConfig(BaseConfig):
    r"""MoE config class.

    Args:
        num_experts (int): The number of experts. Default: 1.
        moe_grouped_gemm (bool): Use grouped gemm or not.
        moe_router_topk (int): Router TopK number. Default: 2.
        moe_router_load_balancing_type (str): type of moe router load balancing algorithm. Choose from:
                                              ["aux_loss", "none"]. Default: "none".
        moe_token_dispatcher_type (str): type of moe token dispatcher algorithm. Choose from:
                                              ["alltoall"]. Default: "alltoall".
        use_self_defined_alltoall (bool): Use self-defined `alltoall` operators. Default: False.
        moe_expert_capacity_factor (float): The capacity factor for each expert. Default: None.
        moe_pad_expert_input_to_capacity (bool): Whether pads the input for each expert
                                                 to match the expert capacity length. Default: False.
        moe_token_drop_policy (str): The policy to drop tokens. Default: "probs".
        moe_aux_loss_coeff (float): Scaling coefficient for the aux loss. Default: 0.0.
        moe_z_loss_coeff (float): Scaling coefficient for the z-loss. Default: None.
        moe_input_jitter_eps (float): Add noise to the input tensor by
                                      applying jitter with a specified epsilon value. Default: None.
    """
    config_name = "moe_config"

    def __init__(
            self,
            num_experts: int = 1,
            moe_grouped_gemm: bool = False,
            moe_router_topk: int = 2,
            moe_router_load_balancing_type: str = "none",
            moe_token_dispatcher_type: str = 'alltoall',
            use_self_defined_alltoall: bool = False,
            moe_expert_capacity_factor: float = None,
            moe_pad_expert_input_to_capacity: bool = False,
            moe_token_drop_policy: str = "probs",
            moe_aux_loss_coeff: float = 0.0,
            moe_z_loss_coeff: float = None,
            moe_input_jitter_eps: float = None,
            **kwargs,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_router_topk = moe_router_topk
        self.moe_router_load_balancing_type = moe_router_load_balancing_type
        self.moe_token_dispatcher_type = moe_token_dispatcher_type
        self.use_self_defined_alltoall = use_self_defined_alltoall
        self.moe_expert_capacity_factor = moe_expert_capacity_factor
        self.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity
        self.moe_token_drop_policy = moe_token_drop_policy
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_input_jitter_eps = moe_input_jitter_eps
        self.update_attrs(**kwargs)


@MoEConfig.validator("moe_grouped_gemm")
def validate_moe_grouped_gemm(config_instance, moe_grouped_gemm):
    """Validate moe_grouped_gemm."""
    Validator.check_bool(moe_grouped_gemm, "moe_grouped_gemm")
    return moe_grouped_gemm


@MoEConfig.validator("moe_router_topk")
def validate_moe_router_topk(config_instance, moe_router_topk):
    """Validate moe_router_topk."""
    Validator.check_positive_int(moe_router_topk, "moe_router_topk")
    return moe_router_topk


@MoEConfig.validator("moe_router_load_balancing_type")
def validate_moe_router_load_balancing_type(config_instance, moe_router_load_balancing_type):
    """Validate moe_router_load_balancing_type."""
    Validator.check_string(moe_router_load_balancing_type, ["aux_loss", "none"], "moe_router_load_balancing_type")
    return moe_router_load_balancing_type


@MoEConfig.validator("moe_token_dispatcher_type")
def validate_moe_token_dispatcher_type(config_instance, moe_token_dispatcher_type):
    """Validate moe_router_load_balancing_type."""
    Validator.check_string(moe_token_dispatcher_type, ["alltoall"], "moe_token_dispatcher_type")
    return moe_token_dispatcher_type


@MoEConfig.validator("use_self_defined_alltoall")
def validate_use_self_defined_alltoall(config_instance, use_self_defined_alltoall):
    """Validate use_self_defined_alltoall."""
    Validator.check_bool(use_self_defined_alltoall, "use_self_defined_alltoall")
    return use_self_defined_alltoall


@MoEConfig.validator("moe_expert_capacity_factor")
def validate_moe_expert_capacity_factor(config_instance, moe_expert_capacity_factor):
    """Validate moe_expert_capacity_factor."""
    if moe_expert_capacity_factor is not None:
        if config_instance.moe_token_dispatcher_type != "alltoall":
            raise ValueError(
                "moe_expert_capacity_factor only works with alltoall token dispatcher."
            )
        if moe_expert_capacity_factor < 0:
            moe_expert_capacity_factor = None
        if config_instance.moe_router_load_balancing_type not in ["aux_loss", "none"]:
            raise ValueError(
                "moe_expert_capacity_factor only works with aux_loss or none load balancing."
            )
    return moe_expert_capacity_factor


@MoEConfig.validator("moe_pad_expert_input_to_capacity")
def validate_moe_pad_expert_input_to_capacity(config_instance, moe_pad_expert_input_to_capacity):
    """Validate moe_pad_expert_input_to_capacity."""
    Validator.check_bool(moe_pad_expert_input_to_capacity, "moe_pad_expert_input_to_capacity")
    if moe_pad_expert_input_to_capacity is None and \
        config_instance.moe_expert_capacity_factor is None:
        raise ValueError(
            "moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity."
        )
    return moe_pad_expert_input_to_capacity


class TrainingConfig(BaseConfig):
    r"""
    Training config.

    Args:
        parallel_config (ModelParallelConfig): Parallel config.
        dataset_config (DatasetConfig): Dataset config.
        seed (Union[int, None], optional): Random seed for initialization. Default: None.
        output_dir (str, optional): Output directory for saving checkpoints, logs and so on. Default: './output'.
        training_iters (int, optional) : Training iterations for training. Default: 0.
        epochs (Union[int, None], optional) : Epochs for training. Default: None.
        log_interval (Union[int, None], optional): Log interval for training. Default: None.
        eval_interval (Union[int, None], optional): Evaluation interval for training. Default: None.
        save_interval (Union[int, None], optional): Save interval for training. Default: None.
        best_metric_comparison (Union[str, None], optional): the method to compare best metric. Default: None.
        eval_metric(Union[str, None], optional): the name of evaluation metrics. Default: None
        grad_clip_kwargs (dict, optional): Gradient clip arguments. Default: None.
        loss_scale (Union[float, int, None], optional): Initial value of loss scale. If set,
            will use static loss scaler. Default: None.
        loss_scale_value (Union[float, int, None], optional): Initial value of dynamic loss scale. Default: None.
        loss_scale_factor (Union[int, None], optional): Factor of dynamic loss scale. Default: None.
        loss_scale_window (Union[int, None], optional): Window size of dynamic loss scale. Default: None.
        loss_reduction (str, optional): Loss reduction method. Default: 'mean'.
        calculate_per_token_loss (bool): Apply grad and loss calculation base on num of tokens. Default: False.
        wrap_with_ddp (bool): Using DistributedDataParallel to wrap model. Default: False.
        overlap_grad_reduce (bool): Enable gradient computing and synchronization communication overlap when using
            DistributedDataParallel. Default: False.
        delay_grad_reduce (bool): If set, delay grad reductions in all but first PP stage. Default: False.
        overlap_param_gather (bool): Enable forward computing and param gather communication overlap when using
            DistributedDataParallel. Default: False.
        use_distributed_optimizer (bool): Enable DistributedOptimizer when using DistributedDataParallel.
            Default: False.
        bucket_size (Optional[int]): Bucket size which is used to partition buffer into buckets when
            overlap_grad_reduce=True. Default: None.
        check_for_nan_in_grad (bool): If True, check gradients in buffer are finite after synchronization.
            Default: False.
        resume_training (bool): Resume training. Default: False.
        crc_check (bool): CRC check when save/load checkpoint. Enable this may cause low train performance.
            Default: False.
        load_checkpoint (str, optional): Where to load checkpoint. Default: ''.
        enable_compile_cache (bool): Save compile cache. Enable this may cause low train performance. Default: False.
        compile_cache_path (str, optional): Where to save compile cache. Default: './{output_dir}/compile_cache'.
        ckpt_format (str, optional): checkpoint save format. Default: 'ckpt'.
        prefix (str, optional): checkpoint save prefix. Default: 'network'.
        keep_checkpoint_max (str, optional): max saved checkpoint number. Default: 5.
        kwargs (dict, optional): Other dataset config arguments.
    """

    # set config name for identifying while using init_configs methods
    config_name = "training_config"

    def __init__(
            self,
            parallel_config: ModelParallelConfig,
            dataset_config: DatasetConfig = DatasetConfig(),
            lora_config: LoraConfig = LoraConfig(),
            seed: int = None,
            output_dir: str = "./output",
            training_iters: int = 0,
            epochs: int = None,
            log_interval: int = None,
            eval_interval: int = None,
            save_interval: int = None,
            best_metric_comparison: str = None,
            eval_metric: str = None,
            grad_clip_kwargs: dict = None,
            loss_scale: Union[float, int] = None,
            loss_scale_value: Union[float, int] = None,
            loss_scale_factor: int = None,
            loss_scale_window: int = None,
            loss_reduction: str = "mean",
            calculate_per_token_loss: bool = False,
            wrap_with_ddp: bool = False,
            accumulate_allreduce_grads_in_fp32: bool = False,
            overlap_grad_reduce: bool = False,
            delay_grad_reduce: bool = False,
            use_distributed_optimizer: bool = False,
            bucket_size: Optional[int] = None,
            check_for_nan_in_grad: bool = False,
            fp16: bool = False,
            bf16: bool = False,
            resume_training: bool = False,
            crc_check: bool = False,
            load_checkpoint: str = "",
            enable_compile_cache: bool = False,
            compile_cache_path: str = None,
            ckpt_format: str = "ckpt",
            prefix: str = "network",
            keep_checkpoint_max: int = 5,
            enable_mem_align: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.parallel_config = parallel_config
        self.dataset_config = dataset_config
        self.lora_config = lora_config
        self.seed = seed
        self.output_dir = output_dir
        self.training_iters = training_iters
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.best_metric_comparison = best_metric_comparison
        self.eval_metric = eval_metric
        self.loss_scale = loss_scale
        self.loss_scale_value = loss_scale_value
        self.loss_scale_factor = loss_scale_factor
        self.loss_scale_window = loss_scale_window
        self.grad_clip_kwargs = grad_clip_kwargs
        self.loss_reduction = loss_reduction
        self.calculate_per_token_loss = calculate_per_token_loss
        self.wrap_with_ddp = wrap_with_ddp
        self.accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
        self.overlap_grad_reduce = overlap_grad_reduce
        self.delay_grad_reduce = delay_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.bucket_size = bucket_size
        self.check_for_nan_in_grad = check_for_nan_in_grad
        self.fp16 = fp16
        self.bf16 = bf16
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 can not be set at the same time.")
        self.resume_training = resume_training
        self.crc_check = crc_check
        self.load_checkpoint = load_checkpoint
        self.enable_compile_cache = enable_compile_cache
        if compile_cache_path is not None:
            self.compile_cache_path = compile_cache_path
        else:
            self.compile_cache_path = os.path.join(self.output_dir, "compile_cache")
        self.ckpt_format = ckpt_format
        self.prefix = prefix
        self.keep_checkpoint_max = keep_checkpoint_max
        self.enable_mem_align = enable_mem_align
        self.update_attrs(**kwargs)


TrainingConfig.register_depended_config([ModelParallelConfig, DatasetConfig, LoraConfig], optional=[False, True, True])


@TrainingConfig.validator("seed")
def validate_seed(config_instance, seed):
    """Validate seed."""
    if seed is not None:
        Validator.check_positive_int(seed, "seed")
    return seed


@TrainingConfig.validator("output_dir")
def validate_output_dir(config_instance, output_dir):
    """Validate output_dir."""
    Validator.check_value_type("output_dir", output_dir, [str])
    return output_dir


@TrainingConfig.validator("training_iters")
def validate_training_iters(config_instance, training_iters):
    """Validate training_iters."""
    Validator.check_non_negative_int(training_iters, "training_iters")
    return training_iters


@TrainingConfig.validator("epochs")
def validate_epochs(config_instance, epochs):
    """Validate epochs."""
    if epochs is not None:
        Validator.check_positive_int(epochs, "epochs")
    return epochs


@TrainingConfig.validator("log_interval")
def validate_log_interval(config_instance, log_interval):
    """Validate log_interval."""
    if log_interval is not None:
        Validator.check_positive_int(log_interval, "log_interval")
    return log_interval


@TrainingConfig.validator("eval_interval")
def validate_eval_interval(config_instance, eval_interval):
    """Validate eval_interval."""
    if eval_interval is not None:
        Validator.check_positive_int(eval_interval, "eval_interval")
    return eval_interval


@TrainingConfig.validator("save_interval")
def validate_save_interval(config_instance, save_interval):
    """Validate save_interval."""
    if save_interval is not None:
        Validator.check_positive_int(save_interval, "save_interval")
    return save_interval


@TrainingConfig.validator("best_metric_comparison")
def validate_best_metric_comparison(config_instance, best_metric_comparison):
    """Validate best_metric_comparison."""
    if best_metric_comparison is not None:
        if config_instance.eval_interval is None:
            logger.warning("eval_interval should be set when best_metric_comparison is set.")
        Validator.check_string(
            best_metric_comparison, ["less_equal", "greater_equal", "less", "greater"], "best_metric_comparison"
        )
    return best_metric_comparison


@TrainingConfig.validator("eval_metric")
def validate_eval_metric(config_instance, eval_metric):
    """Validate eval_metric."""
    if eval_metric is not None:
        if config_instance.best_metric_comparison is None:
            logger.warning("best_metric_comparison should be set when eval_metric is set.")
        Validator.check_value_type("eval_metric", eval_metric, [str])
    return eval_metric


@TrainingConfig.validator("loss_scale")
def validate_loss_scale(config_instance, loss_scale):
    """Validate loss_cale."""
    if loss_scale is not None:
        if isinstance(loss_scale, int):
            Validator.check_positive_int(loss_scale, "loss_scale")
        elif isinstance(loss_scale, float):
            Validator.check_positive_float(loss_scale, "loss_scale")
    return loss_scale


@TrainingConfig.validator("loss_scale_value")
def validate_loss_scale_value(config_instance, loss_scale_value):
    """Validate loss_scale_value."""
    if loss_scale_value is not None:
        # check int and float
        if isinstance(loss_scale_value, int):
            Validator.check_positive_int(loss_scale_value, "loss_scale_value")
        elif isinstance(loss_scale_value, float):
            Validator.check_positive_float(loss_scale_value, "loss_scale_value")
        else:
            raise ValueError("loss_scale_value should be int or float.")
        if config_instance.loss_scale is not None:
            logger.warning(
                "loss_scale and loss_scale_value should not be set together. "
                "Will use static loss scaler with loss_scale."
            )
    return loss_scale_value


@TrainingConfig.validator("loss_scale_factor")
def validate_loss_scale_factor(config_instance, loss_scale_factor):
    """Validate loss_scale_factor."""
    if loss_scale_factor is not None:
        Validator.check_positive_int(loss_scale_factor, "loss_scale_factor")
        if config_instance.loss_scale_value is None:
            raise ValueError("loss_scale_value, loss_scale_factor, and loss_scale_window should be set together.")
    return loss_scale_factor


@TrainingConfig.validator("loss_scale_window")
def validate_loss_scale_window(config_instance, loss_scale_window):
    """Validate loss_scale_window."""
    if loss_scale_window is not None:
        Validator.check_positive_int(loss_scale_window, "loss_scale_window")
        if config_instance.loss_scale_value is None or config_instance.loss_scale_factor is None:
            raise ValueError("loss_scale_value, loss_scale_factor, and loss_scale_window should be set together.")
    return loss_scale_window


@TrainingConfig.validator("loss_reduction")
def validate_loss_reduction(config_instance, loss_reduction):
    """Validate loss_reduction."""
    Validator.check_string(loss_reduction, ["mean", "sum"], "loss_reduction")
    return loss_reduction


@TrainingConfig.validator("wrap_with_ddp")
def validate_wrap_with_ddp(config_instance, wrap_with_ddp):
    """Validate wrap_with_ddp."""
    Validator.check_bool(wrap_with_ddp, "wrap_with_ddp")
    return wrap_with_ddp


@TrainingConfig.validator("accumulate_allreduce_grads_in_fp32")
def validate_accumulate_allreduce_grads_in_fp32(config_instance, accumulate_allreduce_grads_in_fp32):
    """Validate accumulate_allreduce_grads_in_fp32."""
    Validator.check_bool(accumulate_allreduce_grads_in_fp32, "accumulate_allreduce_grads_in_fp32")
    return accumulate_allreduce_grads_in_fp32


@TrainingConfig.validator("overlap_grad_reduce")
def validate_overlap_grad_reduce(config_instance, overlap_grad_reduce):
    """Validate overlap_grad_reduce."""
    if not config_instance.wrap_with_ddp and overlap_grad_reduce:
        logger.warning("For training config, overlap_grad_reduce only take effect when `wrap_with_ddp=True`"
                       "overlap_grad_reduce has been set to `False`.")
        overlap_grad_reduce = False
    Validator.check_bool(overlap_grad_reduce, "overlap_grad_reduce")
    return overlap_grad_reduce


@TrainingConfig.validator("delay_grad_reduce")
def validate_delay_grad_reduce(config_instance, delay_grad_reduce):
    """Validate overlap_grad_reduce."""
    if not config_instance.overlap_grad_reduce and delay_grad_reduce:
        logger.warning("For training config, delay_grad_reduce only take effect when `overlap_grad_reduce=True`"
                       "delay_grad_reduce has been set to `False`.")
        delay_grad_reduce = False
    Validator.check_bool(delay_grad_reduce, "delay_grad_reduce")
    return delay_grad_reduce


@TrainingConfig.validator("use_distributed_optimizer")
def validate_use_distributed_optimizer(config_instance, use_distributed_optimizer):
    """Validate use_distributed_optimizer."""
    if not config_instance.wrap_with_ddp and use_distributed_optimizer:
        logger.warning("For training config, use_distributed_optimizer only take effect when `wrap_with_ddp=True`"
                       "use_distributed_optimizer has been set to `False`.")
        use_distributed_optimizer = False
    Validator.check_bool(use_distributed_optimizer, "use_distributed_optimizer")
    return use_distributed_optimizer


@TrainingConfig.validator("bucket_size")
def validate_bucket_size(config_instance, bucket_size):
    """Validate bucket_size."""
    if bucket_size is not None:
        Validator.check_positive_int(bucket_size, "bucket_size")
        if not config_instance.wrap_with_ddp:
            raise ValueError("bucket_size can only be set when `wrap_with_ddp=True`.")
    return bucket_size


@TrainingConfig.validator("check_for_nan_in_grad")
def validate_check_for_nan_in_grad(config_instance, check_for_nan_in_grad):
    """Validate check_for_nan_in_grad."""
    if not config_instance.wrap_with_ddp and check_for_nan_in_grad:
        logger.warning("For training config, check_for_nan_in_grad only take effect when `wrap_with_ddp=True`"
                       "overlap_grad_reduce has been set to `False`.")
        check_for_nan_in_grad = False
    Validator.check_bool(check_for_nan_in_grad, "check_for_nan_in_grad")
    return check_for_nan_in_grad


@TrainingConfig.validator("fp16")
def validate_fp16(config_instance, fp16):
    """Validate fp16."""
    Validator.check_bool(fp16, "fp16")
    return fp16


@TrainingConfig.validator("bf16")
def validate_bf16(config_instance, bf16):
    """Validate bf16."""
    Validator.check_bool(bf16, "bf16")
    return bf16

@TrainingConfig.validator("resume_training")
def validate_resume_training(config_instance, resume_training):
    """Validate resume_training is bool."""
    Validator.check_bool(resume_training, "resume_training")
    return resume_training

@TrainingConfig.validator("crc_check")
def validate_crc_check(config_instance, crc_check):
    """Validate crc_check is bool."""
    Validator.check_bool(crc_check, "crc_check")
    return crc_check

@TrainingConfig.validator("load_checkpoint")
def validate_load_checkpoint(config_instance, load_checkpoint):
    """Validate load_checkpoint is str."""
    Validator.check_value_type("load_checkpoint", load_checkpoint, [str])
    return load_checkpoint

@TrainingConfig.validator("enable_compile_cache")
def validate_enable_compile_cache(config_instance, enable_compile_cache):
    """Validate enable_compile_cache is bool."""
    Validator.check_bool(enable_compile_cache, "enable_compile_cache")
    return enable_compile_cache

@TrainingConfig.validator("compile_cache_path")
def validate_compile_cache_path(config_instance, compile_cache_path):
    """Validate compile_cache_path is str."""
    Validator.check_value_type("compile_cache_path", compile_cache_path, [str])
    return compile_cache_path

@TrainingConfig.validator("ckpt_format")
def validate_ckpt_format(config_instance, ckpt_format):
    """Validate ckpt_format is str."""
    Validator.check_value_type("ckpt_format", ckpt_format, [str])
    if config_instance.crc_check and ckpt_format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    return ckpt_format

@TrainingConfig.validator("keep_checkpoint_max")
def validate_keep_checkpoint_max(config_instance, keep_checkpoint_max):
    """Validate keep_checkpoint_max is int"""
    if keep_checkpoint_max is not None:
        Validator.check_positive_int(keep_checkpoint_max, "keep_checkpoint_max")
    return keep_checkpoint_max

def check_fa_config(**kwargs):
    """ check flash attention config validation. """

    def _check_sparse_mode(sparse_mode):
        support_sparse_mode = (0, 1, 2, 3, 4)
        if sparse_mode not in support_sparse_mode:
            raise NotImplementedError("For flash attention, sparse_mode only support"
                                      "[0, 1, 2, 3, 4] for now, but got {}".format(str(sparse_mode)))

    args_and_check_map = {
        'keep_prob': partial(Validator.check_float_range, lower_limit=0, upper_limit=1,
                             rel=Rel.INC_BOTH, arg_name='keep_prob'),
        'pre_tokens': partial(Validator.check_int_range, lower_limit=-2147483647, upper_limit=2147483647,
                              rel=Rel.INC_BOTH, arg_name='pre_tokens'),
        'next_tokens': partial(Validator.check_int_range, lower_limit=-2147483647, upper_limit=2147483647,
                               rel=Rel.INC_BOTH, arg_name='next_tokens'),
        'input_layout': partial(Validator.check_string, valid_values=('BNSD'), arg_name='input_layout'),
        'sparse_mode': _check_sparse_mode
    }
    for arg_name, value in kwargs.items():
        if arg_name not in args_and_check_map.keys():
            raise ValueError("For FAConfig, only `keep_prob`, `pre_tokens`, `next_tokens`, `input_layout`, "
                             "and `sparse_mode` are configuable, but got {}".format(arg_name))
        args_and_check_map[arg_name](value)


class TransformerConfig(BaseConfig):
    """Transformer config class.

    Args:
        vocab_size (int): Vocabulary size.
        num_layers (int): Number of model layers.
        num_attention_heads (int): Number of heads for MultiHeadAttention.
        hidden_size (int): Dimensionality of the encoder layers.
        ffn_hidden_size (int): Dimensionality the FeedForward block project to.
        parallel_config (ModelParallelConfig): Parallel config.
        lora_config (LoraConfig): Lora config.
        attention_type (str): Attention type. Default: 'self_attn'.
        position_embedding_type (str): Position embedding type. Default: 'absolute'
        parallel_position_embedding (bool): Apply parallel vocab embedding layer when using
            absolute position embedding. Default: False
        rotary_config (dict): Rotary config. Default: None
        use_query_layer (bool): Using query layer after transformer. Default: False.
        use_visual_encoder (bool): Using visual encoder. Default: False.
        use_retriever (bool): Using retriever. Default: False
        group_query_attention (bool): Enable group query attention. Default: False.
        num_query_groups (int): Number of heads for key and value when using group query attention.
            Default: 32.
        qkv_has_bias (bool): Linears apply on query, key and value in Attention block has bias
            parameter. Default: True.
        out_proj_has_bias (bool): Linear applies on output of core attention block has bias
            parameter. Default: True.
        head_skip_weight_param_allocation (bool): If Head will skip weight allocation and use word
            as weights. Default: False.
        apply_query_key_layer_scaling (bool): Apply query key scaling in core attention block.
            Default: False.
        use_flash_attention (bool): Enable flash attention. Default: False.
        mask_func_type (str): Attention mask compute method. Default: 'attn_mask_add'.
        mlp_has_bis (bool): Linears in MLP block have bias parameters. Default: True.
        hidden_act (str): Activation used in MLP block. Default: 'gelu'.
        normalization (str): Normalization used in transformerlayer block. Default: 'LayerNorm'.
        norm_epsilon (float): Epsilon of normalization. Default: 1.e-5.
        apply_residual_connection_post_norm (bool): Apply residual connection after normalization.
            Default: False.
        use_final_norm (bool): Apply final norm after transformer. Default: True.
        residual_connection_dtype (str): Compute data type of residual connection. Default: 'float32'.
        init_method_std (float): Init method std value. Default: 0.01
        params_dtype (str): Parameter initialize data type. Default: 'float32'.
        embedding_init_dtype (str): Embedding parameter initialize data type. Default: 'float32'.
        compute_dtype (str): Compute data type of linear module. Default: 'float16'.
        softmax_compute_dtype (str): Compute data type of softmax layer. Default: 'float32'.
        fp16_lm_cross_entropy (bool): Apply float16 when calculating cross entropy. Default: False.
        hidden_dropout (float): Dropout rate for output of attention block and mlp block in transformerlayer.
            Default: 0.0.
        attention_dropout (float): Dropout rate for attention socre. Default: 0.0.
        num_experts (int, optional): Number of experts. Default: None.
        untie_embeddings_and_output_weights (bool): If false, share embedding with head layer. Default: False.
        flatten_labels_and_input_mask (bool): flatten labels and input mask in public layer. Default: True.
        recompute_method (str, optional): Recompute method. Default: None.
        recompute_num_layers (int, optional): Number of layers to recompute. Default: None.
        recompute_granularity (str, optional): Recompute granularity. Default: None.
        moe_config (MoEConfig, optional): MoE config. Default: None.
        dataset_config (dict): dataset config. Default: None.
    """

    # set config name for identifying while using init_configs methods
    config_name = "model_config"

    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_attention_heads: int,
            hidden_size: int,
            ffn_hidden_size: int,
            parallel_config: ModelParallelConfig,
            lora_config: LoraConfig = LoraConfig(),
            dataset_config: DatasetConfig = DatasetConfig(),
            moe_config: MoEConfig = MoEConfig(),
            attention_type: str = "self_attn",
            position_embedding_type: str = 'absolute',
            parallel_position_embedding: bool = False,
            rotary_config: dict = None,
            use_query_layer: bool = False,
            use_visual_encoder: bool = False,
            use_retriever: bool = False,
            group_query_attention: bool = False,
            num_query_groups: int = 32,
            qkv_has_bias: bool = True,
            out_proj_has_bias: bool = True,
            head_skip_weight_param_allocation: bool = True,
            apply_query_key_layer_scaling: bool = False,
            use_flash_attention: bool = False,
            fa_config=None,
            enable_flash_sp: bool = False,
            mask_func_type: str = "attn_mask_add",
            mlp_has_bias: bool = True,
            hidden_act: str = "gelu",
            normalization: str = "LayerNorm",
            norm_epsilon: float = 1.0e-5,
            apply_residual_connection_post_norm: bool = False,
            use_final_norm: bool = True,
            residual_connection_dtype: str = "float32",
            init_method_std: float = 0.01,
            params_dtype: str = "float32",
            embedding_init_dtype: str = "float32",
            compute_dtype: str = "float32",
            softmax_compute_dtype: str = "float32",
            init_method: str = 'normal',
            bias_init: str = 'zeros',
            fp16_lm_cross_entropy: bool = False,
            attention_dropout: float = 0.0,
            out_hidden_size: int = None,
            num_experts: int = None,
            untie_embeddings_and_output_weights: bool = False,
            flatten_labels_and_input_mask: bool = True,
            recompute_method: str = None,
            recompute_num_layers: int = None,
            recompute_granularity: str = None,
            fp32_residual_connection: bool = False,
            kv_channels: int = None,
            hidden_dropout: float = 0.0,
            bias_dropout_fusion: bool = False,
            fp8_format: str = None,
            clone_scatter_output_in_embedding: bool = False,
            add_bias_linear: bool = False,
            attention_softmax_in_fp32: bool = True,
            masked_softmax_fusion: bool = False,
            distribute_saved_activations: bool = False,
            retro_add_retriever: bool = False,
            transformer_impl: str = 'local',
            encoder_num_layers: int = None,
            decoder_num_layers: int = None,
            model_type: str = "encoder_or_decoder",
            select_comm_recompute: bool = False,
            select_recompute: bool = False,
            apply_rope_fusion: bool = False,

            **kwargs,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.lora_config = lora_config
        self.dataset_config = dataset_config
        self.vocab_size = vocab_size
        self.moe_config = moe_config
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.attention_type = attention_type
        self.position_embedding_type = position_embedding_type
        self.parallel_position_embedding = parallel_position_embedding
        self.rotary_config = rotary_config
        self.use_query_layer = use_query_layer
        self.use_visual_encoder = use_visual_encoder
        self.use_retriever = use_retriever
        self.group_query_attention = group_query_attention
        self.num_query_groups = num_query_groups
        self.qkv_has_bias = qkv_has_bias
        self.out_proj_has_bias = out_proj_has_bias
        self.head_skip_weight_param_allocation = head_skip_weight_param_allocation
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.use_flash_attention = use_flash_attention
        if self.use_flash_attention and fa_config:
            check_fa_config(**fa_config)
            self.fa_config = fa_config
        self.enable_flash_sp = enable_flash_sp
        self.mask_func_type = mask_func_type
        self.mlp_has_bias = mlp_has_bias
        self.hidden_act = hidden_act
        self.normalization = normalization
        self.norm_epsilon = norm_epsilon
        self.apply_residual_connection_post_norm = apply_residual_connection_post_norm
        self.use_final_norm = use_final_norm
        self.residual_connection_dtype = residual_connection_dtype
        self.init_method_std = init_method_std
        self.params_dtype = params_dtype
        self.embedding_init_dtype = embedding_init_dtype
        self.compute_dtype = compute_dtype
        self.softmax_compute_dtype = softmax_compute_dtype
        self.init_method = init_method
        self.bias_init = bias_init
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.attention_dropout = attention_dropout
        self.out_hidden_size = out_hidden_size
        self.num_experts = num_experts
        self.untie_embeddings_and_output_weights = untie_embeddings_and_output_weights
        self.flatten_labels_and_input_mask = flatten_labels_and_input_mask
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers
        self.recompute_granularity = recompute_granularity
        self.fp32_residual_connection = fp32_residual_connection
        self.kv_channels = kv_channels
        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.fp8 = fp8_format
        self.clone_scatter_output_in_embedding = clone_scatter_output_in_embedding
        self.add_bias_linear = add_bias_linear
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.masked_softmax_fusion = masked_softmax_fusion
        self.distribute_saved_activations = distribute_saved_activations
        self.retro_add_retriever = retro_add_retriever
        self.transformer_impl = transformer_impl
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.model_type = model_type
        self.select_comm_recompute = select_comm_recompute
        self.select_recompute = select_recompute
        self.apply_rope_fusion = apply_rope_fusion
        self.gated_linear_unit = False

        if "recompute_activations" in kwargs:
            if kwargs["recompute_activations"]:
                self.recompute_granularity = "selective"
            kwargs.pop("recompute_activations")

        self.update_attrs(**self.rotary_config)
        self.update_attrs(**kwargs)

    def update_lora_config(self, cell_name):
        lora_module = self.lora_config.lora_module
        self.lora_config.lora_module = None if lora_module is None else lora_module.get(cell_name, None)


TransformerConfig.register_depended_config([ModelParallelConfig,
                                            LoraConfig,
                                            DatasetConfig,
                                            MoEConfig],
                                           optional=[False, True, True, True])


@TransformerConfig.validator("vocab_size")
def validate_vocab_size(config_instance, vocab_size):
    """Validate vocab_size."""
    Validator.check_positive_int(vocab_size, "vocab_size")
    return vocab_size


@TransformerConfig.validator("num_layers")
def validate_num_layers(config_instance, num_layers):
    """Validate num_layers."""
    Validator.check_positive_int(num_layers, "num_layers")
    return num_layers


@TransformerConfig.validator("num_attention_heads")
def validate_num_attention_heads(config_instance, num_attention_heads):
    """Validate num_attention_heads."""
    Validator.check_positive_int(num_attention_heads, "num_attention_heads")
    return num_attention_heads


@TransformerConfig.validator("hidden_size")
def validate_hidden_size(config_instance, hidden_size):
    """Validate hidden_size."""
    Validator.check_positive_int(hidden_size, "hidden_size")
    return hidden_size


@TransformerConfig.validator("ffn_hidden_size")
def validate_ffn_hidden_size(config_instance, ffn_hidden_size):
    """Validate ffn_hidden_size."""
    Validator.check_positive_int(ffn_hidden_size, "ffn_hidden_size")
    return ffn_hidden_size


@TransformerConfig.validator("attention_type")
def validate_attention_type(config_instance, attention_type):
    """Validate attention_type."""
    Validator.check_value_type("attention_type", attention_type, [str])
    return attention_type


@TransformerConfig.validator("group_query_attention")
def validate_group_query_attention(config_instance, group_query_attention):
    """Validate group_query_attention."""
    Validator.check_bool(group_query_attention, "group_query_attention")
    return group_query_attention


@TransformerConfig.validator("num_query_groups")
def validate_num_query_groups(config_instance, num_query_groups):
    """Validate num_query_groups."""
    Validator.check_positive_int(num_query_groups, "num_query_groups")
    return num_query_groups


@TransformerConfig.validator("qkv_has_bias")
def validate_qkv_has_bias(config_instance, qkv_has_bias):
    """Validate qkv_has_bias."""
    Validator.check_bool(qkv_has_bias, "qkv_has_bias")
    return qkv_has_bias


@TransformerConfig.validator("out_proj_has_bias")
def validate_out_proj_has_bias(config_instance, out_proj_has_bias):
    """Validate out_proj_has_bias."""
    Validator.check_bool(out_proj_has_bias, "out_proj_has_bias")
    return out_proj_has_bias


@TransformerConfig.validator("head_skip_weight_param_allocation")
def validate_head_skip_weight_param_allocation(config_instance, head_skip_weight_param_allocation):
    """Validate head_skip_weight_param_allocation."""
    Validator.check_bool(head_skip_weight_param_allocation, "head_skip_weight_param_allocation")
    return head_skip_weight_param_allocation


@TransformerConfig.validator("apply_query_key_layer_scaling")
def validate_apply_query_key_layer_scaling(config_instance, apply_query_key_layer_scaling):
    """Validate apply_query_key_layer_scaling."""
    Validator.check_bool(apply_query_key_layer_scaling, "apply_query_key_layer_scaling")
    return apply_query_key_layer_scaling


@TransformerConfig.validator("use_flash_attention")
def validate_use_flash_attention(config_instance, use_flash_attention):
    """Validate use_flash_attention."""
    Validator.check_bool(use_flash_attention, "use_flash_attention")
    return use_flash_attention


@TransformerConfig.validator("rotary_config")
def validate_rotary_config(config_instance, rotary_config):
    """Validate fa_config."""
    default_rotary_config = {
        'rotary_percent': 1.0,
        'rotary_interleaved': False,
        'seq_len_interpolation_factor': None,
        'rotary_base': 10000
    }
    check_keys = default_rotary_config.keys()

    if rotary_config is not None:
        if not isinstance(rotary_config, dict):
            raise TypeError("rotary_config should be a dict.")

        for key, value in rotary_config.items():
            # pylint: disable=R1720
            if key not in check_keys:
                raise ValueError(f"Key '{key}' is not supported in rotary_config.")
            elif key in ('rotary_interleaved', 'seq_len_interpolation_factor'):
                if not isinstance(value, bool):
                    raise TypeError(f"Key '{key}' should be bool in rotary_config.")
            elif key in 'rotary_percent':
                if not isinstance(value, float):
                    raise TypeError(f"Key '{key}' should be float in rotary_config.")
            elif key in 'rotary_base':
                if not isinstance(value, int):
                    raise TypeError(f"Key '{key}' should be int in rotary_config.")
            default_rotary_config[key] = value
        rotary_config = default_rotary_config
    else:
        rotary_config = default_rotary_config
    return rotary_config


@TransformerConfig.validator("mask_func_type")
def validate_mask_func_type(config_instance, mask_func_type):
    """Validate mask_func_type."""
    Validator.check_value_type("mask_func_type", mask_func_type, [str])
    return mask_func_type


@TransformerConfig.validator("mlp_has_bias")
def validate_mlp_has_bias(config_instance, mlp_has_bias):
    """Validate mlp_has_bias."""
    Validator.check_bool(mlp_has_bias, "mlp_has_bias")
    return mlp_has_bias


@TransformerConfig.validator("hidden_act")
def validate_hidden_act(config_instance, hidden_act):
    """Validate hidden_act."""
    Validator.check_value_type("hidden_act", hidden_act, [str])
    return hidden_act


@TransformerConfig.validator("normalization")
def validate_normalization(config_instance, normalization):
    """Validate normalization."""
    Validator.check_value_type("normalization", normalization, [str])
    return normalization


@TransformerConfig.validator("norm_epsilon")
def validate_layernorm_epsilon(config_instance, norm_epsilon):
    """Validate norm_epsilon."""
    Validator.check_positive_float(norm_epsilon, "norm_epsilon")
    return norm_epsilon


@TransformerConfig.validator("apply_residual_connection_post_norm")
def validate_apply_residual_connection_post_norm(config_instance, apply_residual_connection_post_norm):
    """Validate apply_residual_connection_post_norm."""
    Validator.check_bool(apply_residual_connection_post_norm, "apply_residual_connection_post_norm")
    return apply_residual_connection_post_norm


@TransformerConfig.validator("residual_connection_dtype")
def validate_residual_connection_dtype(config_instance, residual_connection_dtype):
    """Validate residual_connection_dtype."""
    return _SUPPORT_DTYPE_DICT[residual_connection_dtype]


@TransformerConfig.validator("params_dtype")
def validate_params_dtype(config_instance, params_dtype):
    """Validate params_dtype."""
    return _SUPPORT_DTYPE_DICT[params_dtype]


@TransformerConfig.validator("embedding_init_dtype")
def validate_embedding_init_dtype(config_instance, embedding_init_dtype):
    """Validate embedding_init_dtype."""
    return _SUPPORT_DTYPE_DICT[embedding_init_dtype]


@TransformerConfig.validator("compute_dtype")
def validate_compute_dtype(config_instance, compute_dtype):
    """Validate compute_dtype."""
    return _SUPPORT_DTYPE_DICT[compute_dtype]


@TransformerConfig.validator("softmax_compute_dtype")
def validate_softmax_compute_dtype(config_instance, softmax_compute_dtype):
    """Validate softmax_compute_dtype."""
    return _SUPPORT_DTYPE_DICT[softmax_compute_dtype]


@TransformerConfig.validator("init_method")
def validate_init_method(config_instance, init_method):
    """Validate init_method."""
    if isinstance(init_method, numbers.Number):
        return init_method
    return _SUPPORT_INIT_METHOD[init_method]()


@TransformerConfig.validator("bias_init")
def validate_bias_init(config_instance, bias_init):
    """Validate bias_init."""
    if isinstance(bias_init, numbers.Number):
        return bias_init
    return _SUPPORT_INIT_METHOD[bias_init]()


@TransformerConfig.validator("hidden_dropout")
def validate_hidden_dropout(config_instance, hidden_dropout):
    """Validate hidden_dropout."""
    Validator.check_float_range(hidden_dropout, 0, 1, Rel.INC_BOTH, "hidden_dropout")
    return hidden_dropout


@TransformerConfig.validator("attention_dropout")
def validate_attention_dropout(config_instance, attention_dropout):
    """Validate attention_dropout."""
    Validator.check_float_range(attention_dropout, 0, 1, Rel.INC_BOTH, "attention_dropout")
    return attention_dropout


@TransformerConfig.validator("num_experts")
def validate_num_experts(config_instance, num_experts):
    """Validate num_experts."""
    Validator.check_value_type("num_experts", num_experts, [int, type(None)])
    return num_experts


@TransformerConfig.validator("share_embedding_weight")
def validate_share_embedding_weight(config_instance, share_embedding_weight):
    """Validate share_embedding_weight."""
    Validator.check_bool(share_embedding_weight, "share_embedding_weight")
    return share_embedding_weight


@TransformerConfig.validator("recompute_method")
def validate_recompute_method(config_instance, recompute_method):
    if recompute_method is not None:
        Validator.check_string(recompute_method, ["uniform", "block"], "recompute_method")
    return recompute_method


@TransformerConfig.validator("recompute_num_layers")
def validate_recompute_num_layers(config_instance, recompute_num_layers):
    """Validate recompute_num_layers."""
    if recompute_num_layers is not None:
        Validator.check_int_range(
            recompute_num_layers, 1, config_instance.num_layers, Rel.INC_BOTH, "recompute_num_layers"
        )
        if config_instance.recompute_method is None:
            logger.warning("recompute_method, recompute_num_layers should be set together.")
    return recompute_num_layers


@TransformerConfig.validator("recompute_granularity")
def validate_recompute_granularity(config_instance, recompute_granularity):
    """Validate recompute_granularity."""
    if recompute_granularity is not None:
        Validator.check_string(recompute_granularity, ["selective", "full"], "recompute_granularity")
        if recompute_granularity == "full":
            if config_instance.recompute_method is None:
                logger.warning("recompute_method should be set when recompute_granularity is set to 'full'.")
            if config_instance.recompute_num_layers is None:
                logger.warning("recompute_num_layers should be set when recompute_granularity is set to 'full'.")
    return recompute_granularity


@TransformerConfig.validator("kv_channels")
def validate_kv_channels(config_instance, kv_channels):
    """Validate kv_channels."""
    if kv_channels is None:
        kv_channels = divide(config_instance.hidden_size, config_instance.num_attention_heads)
    return kv_channels


@TransformerConfig.validator("fp32_residual_connection")
def validate_fp32_residual_connection(config_instance, fp32_residual_connection):
    """Validate fp32_residual_connection."""
    Validator.check_bool(fp32_residual_connection, "fp32_residual_connection")
    return fp32_residual_connection


@TransformerConfig.validator("retro_add_retriever")
def validate_retro_add_retriever(config_instance, retro_add_retriever):
    """Validate retro_add_retriever."""
    Validator.check_bool(retro_add_retriever, "retro_add_retriever")
    return retro_add_retriever


@TransformerConfig.validator("transformer_impl")
def validate_transformer_impl(config_instance, transformer_impl):
    if transformer_impl is not None:
        Validator.check_string(transformer_impl, ["local", "transformer_engine"], "transformer_impl")
    return transformer_impl


@TransformerConfig.validator("fp8_format")
def validate_fp8_format(config_instance, fp8_format):
    if fp8_format is not None:
        Validator.check_string(fp8_format, ['e4m3', 'hybrid'], "fp8_format")
    return fp8_format


@TransformerConfig.validator("encoder_num_layers")
def validate_encoder_num_layers(config_instance, encoder_num_layers):
    """Validate encoder_num_layers."""
    if encoder_num_layers is not None:
        Validator.check_positive_int(encoder_num_layers, "encoder_num_layers")
    return encoder_num_layers


@TransformerConfig.validator("decoder_num_layers")
def validate_decoder_num_layers(config_instance, decoder_num_layers):
    """Validate decoder_num_layers."""
    if decoder_num_layers is not None:
        Validator.check_positive_int(decoder_num_layers, "decoder_num_layers")
    return decoder_num_layers


@TransformerConfig.validator("add_bias_linear")
def validate_add_bias_linear(config_instance, add_bias_linear):
    """Validate add_bias_linear."""
    Validator.check_bool(add_bias_linear, "add_bias_linear")
    return add_bias_linear


@TransformerConfig.validator("clone_scatter_output_in_embedding")
def validate_clone_scatter_output_in_embedding(config_instance, clone_scatter_output_in_embedding):
    """Validate clone_scatter_output_in_embedding."""
    Validator.check_bool(clone_scatter_output_in_embedding, "clone_scatter_output_in_embedding")
    return clone_scatter_output_in_embedding


@TransformerConfig.validator("attention_softmax_in_fp32")
def validate_attention_softmax_in_fp32(config_instance, attention_softmax_in_fp32):
    """Validate attention_softmax_in_fp32."""
    Validator.check_bool(attention_softmax_in_fp32, "attention_softmax_in_fp32")
    return attention_softmax_in_fp32


@TransformerConfig.validator("masked_softmax_fusion")
def validate_masked_softmax_fusion(config_instance, masked_softmax_fusion):
    """Validate masked_softmax_fusion."""
    Validator.check_bool(masked_softmax_fusion, "masked_softmax_fusion")
    return masked_softmax_fusion


@TransformerConfig.validator("distribute_saved_activations")
def validate_distribute_saved_activations(config_instance, distribute_saved_activations):
    """Validate distribute_saved_activations."""
    Validator.check_bool(distribute_saved_activations, "distribute_saved_activations")
    return distribute_saved_activations


@TransformerConfig.validator("select_comm_recompute")
def validate_select_comm_recompute(config_instance, select_comm_recompute):
    """Validate select_comm_recompute."""
    Validator.check_bool(select_comm_recompute, "select_comm_recompute")
    return select_comm_recompute


@TransformerConfig.validator("select_recompute")
def validate_select_recompute(config_instance, select_recompute):
    """Validate select_recompute."""
    Validator.check_bool(select_recompute, "select_recompute")
    return select_recompute


@TransformerConfig.validator("apply_rope_fusion")
def validate_apply_rope_fusion(config_instance, apply_rope_fusion):
    """Validate apply_rope_fusion."""
    Validator.check_bool(apply_rope_fusion, "apply_rope_fusion")
    return apply_rope_fusion


class OptimizerConfig(BaseConfig):
    r"""Optimizer config class.

    Args:
        parallel_config (ModelParallelConfig): Parallel config.
        optimizer_type (str): Optimizer type. Default: 'AdamWeightDecay'.
        learning_rate (float): Learning rate. Default: 0.01.
        learning_rate_scheduler_kwargs (dict, optional): Learning rate scheduler kwargs.
        weight_decay (float): Weight decay. Default: 0.0.
        weight_decay_kwargs (dict, optional): Weight decay kwargs.
        zero_config (dict, optional): ZeRO optimizer config.
        - param_resident (bool): After the forward propagation, the parameters are resident and not split.
          Default: Flase.

        - allreduce_after_grad_accumulation (bool): Use allreduce in optimizer after gradient accumulation.
          Default: Flase.

        - grad_allreduce_op (str): Gradient allreduce operator. like `sum`, `mean`. Default: sum.

        - opt_parallel_group (str): Name of communication group used by optimizer parallel. Default: None.

        - cpu_offload (bool): The process of optimizer will be offload to host. The gradients, parameters and
          optimizer status will be offload to host. Default: Flase.
        end_weight_decay (float): End weight decay. Default: 0.0.
        lr_decay_iters (int): Number of iterations to decay learning rate. Default: None.
        lr_decay_samples (int): Number of samples to decay learning rate. Default: None.
        lr_wsd_decay_iters (int): Number of iterations for the annealing phase in the wsd schedule. Default: None.
        lr_wsd_decay_samples (int): Number of samples for the annealing phase in the wsd schedule. Default: None.
        lr_warmup_iters (int): Number of iterations to linearly warmup learning rate over. Default: 0.
        lr_warmup_samples (int): Number of samples to linearly warmup learning rate over.. Default: 0.
        lr_warmup_init (float): Initial value for learning rate warmup. Default: 0.0.
        lr_warmup_fraction (float): Fraction of lr-warmup-(iters/samples) to use for warmup. Default: None.
        min_lr (float): Minimum value for learning rate. Default: 0.0.
        lr_decay_style (str): Learning rate decay function. Default: "constant".
        weight_decay_incr_style (str): Weight decay increment function. Default: "constant".
        lr_wsd_decay_style (str): Decay style for the annealing phase of WSD. Default: "linear".
        use_checkpoint_opt_param_scheduler (bool): Use checkpoint to set the values of the scheduler. Default: True.
        override_opt_param_scheduler (bool): Reset the values of the scheduler. Default: False.
    """

    # set config name for identifying while using init_configs methods
    config_name = "optimizer_config"

    def __init__(
            self,
            parallel_config: ModelParallelConfig,
            optimizer_type: str = "AdamWeightDecay",
            learning_rate: float = 1e-3,
            learning_rate_scheduler_kwargs: dict = None,
            weight_decay: float = 0.0,  # start_weight_decay
            end_weight_decay: float = 0.0,
            weight_decay_kwargs: dict = None,
            zero_config: dict = None,
            clip_grad: float = 0.0,
            # scheduler config
            lr_decay_iters: int = None,
            lr_decay_samples: int = None,
            lr_wsd_decay_iters: int = None,
            lr_wsd_decay_samples: int = None,
            lr_warmup_iters: int = 0,
            lr_warmup_samples: int = 0,
            lr_warmup_init: float = 0.0,
            lr_warmup_fraction: float = None,
            min_lr: float = 0.0,
            lr_decay_style: str = 'constant',
            weight_decay_incr_style: str = 'constant',
            lr_wsd_decay_style: str = 'linear',
            use_checkpoint_opt_param_scheduler: bool = True,
            override_opt_param_scheduler: bool = False,
            overlap_param_gather: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.parallel_config = parallel_config
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.learning_rate_scheduler_kwargs = learning_rate_scheduler_kwargs
        self.weight_decay = weight_decay
        self.end_weight_decay = end_weight_decay
        self.weight_decay_kwargs = weight_decay_kwargs
        self.zero_config = zero_config
        self.clip_grad = clip_grad
        # scheduler config
        self.lr_decay_iters = lr_decay_iters
        self.lr_decay_samples = lr_decay_samples
        self.lr_wsd_decay_iters = lr_wsd_decay_iters
        self.lr_wsd_decay_samples = lr_wsd_decay_samples
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_warmup_samples = lr_warmup_samples
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_fraction = lr_warmup_fraction
        self.min_lr = min_lr
        self.lr_decay_style = lr_decay_style
        self.weight_decay_incr_style = weight_decay_incr_style
        self.lr_wsd_decay_style = lr_wsd_decay_style
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.overlap_param_gather = overlap_param_gather
        self.update_attrs(**kwargs)


OptimizerConfig.register_depended_config(ModelParallelConfig)


@OptimizerConfig.validator("optimizer_type")
def validate_type(config_instance, optimizer_type):
    """Validate type."""
    Validator.check_value_type("optimizer_type", optimizer_type, [str])
    return optimizer_type


@OptimizerConfig.validator("learning_rate")
def validate_learning_rate(config_instance, learning_rate):
    """Validate learning_rate."""
    Validator.check_positive_float(learning_rate, "learning_rate")
    return learning_rate


@OptimizerConfig.validator("learning_rate_scheduler_kwargs")
def validate_learning_rate_scheduler_kwargs(config_instance, learning_rate_scheduler_kwargs):
    """Validate learning_rate_scheduler_kwargs."""
    if learning_rate_scheduler_kwargs is not None:
        Validator.check_value_type("learning_rate_scheduler_kwargs", learning_rate_scheduler_kwargs, [dict])
    return learning_rate_scheduler_kwargs


@OptimizerConfig.validator("weight_decay")
def validate_weight_decay(config_instance, weight_decay):
    """Validate weight_decay."""
    Validator.check_non_negative_float(weight_decay, "weight_decay")
    return weight_decay


@OptimizerConfig.validator("weight_decay_kwargs")
def validate_weight_decay_kwargs(config_instance, weight_decay_kwargs):
    """Validate weight_decay_kwargs."""
    if weight_decay_kwargs is not None:
        Validator.check_value_type("weight_decay_kwargs", weight_decay_kwargs, [dict])
    return weight_decay_kwargs


@OptimizerConfig.validator("clip_grad")
def validate_clip_grad(config_instance, clip_grad):
    """Validate learning_rate."""
    Validator.check_non_negative_float(clip_grad, "clip_grad")
    return clip_grad


@OptimizerConfig.validator("end_weight_decay")
def validate_end_weight_decay(config_instance, end_weight_decay):
    """Validate end_weight_decay."""
    Validator.check_non_negative_float(end_weight_decay, "end_weight_decay")
    return end_weight_decay


@OptimizerConfig.validator("lr_decay_iters")
def validate_lr_decay_iters(config_instance, lr_decay_iters):
    """Validate lr_decay_iters."""
    Validator.check_value_type("lr_decay_iters", lr_decay_iters, [int, type(None)])
    return lr_decay_iters


@OptimizerConfig.validator("lr_decay_samples")
def validate_lr_decay_samples(config_instance, lr_decay_samples):
    """Validate lr_decay_samples."""
    Validator.check_value_type("lr_decay_samples", lr_decay_samples, [int, type(None)])
    return lr_decay_samples


@OptimizerConfig.validator("lr_wsd_decay_iters")
def validate_lr_wsd_decay_iters(config_instance, lr_wsd_decay_iters):
    """Validate lr_wsd_decay_iters."""
    Validator.check_value_type("lr_wsd_decay_iters", lr_wsd_decay_iters, [int, type(None)])
    return lr_wsd_decay_iters


@OptimizerConfig.validator("lr_wsd_decay_samples")
def validate_lr_wsd_decay_samples(config_instance, lr_wsd_decay_samples):
    """Validate lr_wsd_decay_samples."""
    Validator.check_value_type("lr_wsd_decay_samples", lr_wsd_decay_samples, [int, type(None)])
    return lr_wsd_decay_samples


@OptimizerConfig.validator("lr_warmup_iters")
def validate_lr_warmup_iters(config_instance, lr_warmup_iters):
    """Validate lr_warmup_iters."""
    Validator.check_non_negative_int(lr_warmup_iters, "lr_warmup_iters")
    return lr_warmup_iters


@OptimizerConfig.validator("lr_warmup_samples")
def validate_lr_warmup_samples(config_instance, lr_warmup_samples):
    """Validate lr_warmup_samples."""
    Validator.check_non_negative_int(lr_warmup_samples, "lr_warmup_samples")
    return lr_warmup_samples


@OptimizerConfig.validator("lr_warmup_init")
def validate_lr_warmup_init(config_instance, lr_warmup_init):
    """Validate lr_warmup_init."""
    Validator.check_non_negative_float(lr_warmup_init, "lr_warmup_init")
    return lr_warmup_init

@OptimizerConfig.validator("lr_warmup_fraction")
def validate_lr_warmup_fraction(config_instance, lr_warmup_fraction):
    """Validate lr_warmup_fraction."""
    Validator.check_value_type("lr_warmup_fraction", lr_warmup_fraction, [float, type(None)])
    return lr_warmup_fraction


@OptimizerConfig.validator("min_lr")
def validate_min_lr(config_instance, min_lr):
    """Validate min_lr."""
    Validator.check_non_negative_float(min_lr, "min_lr")
    return min_lr


@OptimizerConfig.validator("lr_decay_style")
def validate_lr_decay_style(config_instance, lr_decay_style):
    """Validate lr_decay_style."""
    Validator.check_string(lr_decay_style,
                           ["constant", "WSD", "linear", "cosine", "inverse-square-root"], "lr_decay_style")
    return lr_decay_style


@OptimizerConfig.validator("weight_decay_incr_style")
def validate_weight_decay_incr_style(config_instance, weight_decay_incr_style):
    """Validate weight_decay_incr_style."""
    Validator.check_string(weight_decay_incr_style, ["constant", "linear", "cosine"], "weight_decay_incr_style")
    return weight_decay_incr_style


@OptimizerConfig.validator("lr_wsd_decay_style")
def validate_lr_wsd_decay_style(config_instance, lr_wsd_decay_style):
    """Validate lr_wsd_decay_style."""
    Validator.check_string(lr_wsd_decay_style, ["linear", "cosine", "exponential"], "lr_wsd_decay_style")
    return lr_wsd_decay_style


@OptimizerConfig.validator("use_checkpoint_opt_param_scheduler")
def validate_use_checkpoint_opt_param_scheduler(config_instance, use_checkpoint_opt_param_scheduler):
    """Validate use_checkpoint_opt_param_scheduler."""
    Validator.check_bool(use_checkpoint_opt_param_scheduler, "use_checkpoint_opt_param_scheduler")
    return use_checkpoint_opt_param_scheduler


@OptimizerConfig.validator("overlap_param_gather")
def validate_overlap_param_gather(config_instance, overlap_param_gather):
    """Validate overlap_param_gather."""
    Validator.check_bool(overlap_param_gather, "overlap_param_gather")
    return overlap_param_gather


@OptimizerConfig.validator("override_opt_param_scheduler")
def validate_override_opt_param_scheduler(config_instance, override_opt_param_scheduler):
    """Validate override_opt_param_scheduler."""
    Validator.check_bool(override_opt_param_scheduler, "override_opt_param_scheduler")
    return override_opt_param_scheduler
