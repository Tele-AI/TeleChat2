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
# pylint: disable=unused-variable
"""utils"""

import re
import math

import mindspore.ops as P
import mindspore.nn as nn
from mindspore.communication import get_group_size
from mindspore.nn.optim.optimizer import Optimizer

from mindformers.tools import logger
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    is_pipeline_last_stage
)


class DictWithValueError(dict):
    """
    A dictionary subclass that raises a custom error with a helpful message when a key is not found.
    """

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise ValueError(f"'{key}' is not supported, please select one of {list(self.keys())}") from e


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError("{} is not divisible by {}".format(numerator, denominator))


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def calculate_dividable_vocab_size(vocab_size, denominator=128):
    """
    Calculate the nearest dividable vocab size by the denominator.

    Args:
        vocab_size (int): The original vocab size.
        denominator (int): The denominator to divide the vocab size.

    Returns:
        padded_vocab_size (int): The nearest dividable vocab size.
    """
    padded_vocab_size = math.ceil(vocab_size / denominator) * denominator
    if padded_vocab_size != vocab_size:
        logger.warning(
            f"Add {padded_vocab_size - vocab_size} padded tokens to the "
            f"vocab size {vocab_size} to make it dividable by {denominator}"
        )
    return padded_vocab_size


def add_attr_for_shared_weight(layer, weight_name='weight'):
    """ add 'share' attr for embedding or head layer weight """
    if get_pipeline_model_parallel_world_size() == 1:
        return

    cur_layer = layer
    param_name = weight_name
    if "." in weight_name:
        sub_module = [item for item in weight_name.split(".")]
        param_name = sub_module[-1]
        sub_module = sub_module[:-1]
        for next_layer in sub_module:
            cur_layer = getattr(cur_layer, next_layer)
    if hasattr(cur_layer, param_name):
        param_instance = getattr(cur_layer, param_name)
        if param_instance is not None:
            setattr(param_instance, 'share', True)
            weight_sum = param_instance.value().sum()
            if is_pipeline_last_stage() and weight_sum != 0.0:
                P.assign(param_instance, P.zeros_like(param_instance, dtype=param_instance.value().dtype))
        else:
            raise RuntimeError(f"For 'add_attr_for_shared_weight' function, class '{type(layer).__name__}' "
                               f"weight is None, so the 'share' attribute cannot be added.")
    else:
        raise RuntimeError(f"For 'add_attr_for_shared_weight' function, "
                           f"class '{type(layer).__name__}' have no weight for adding 'share' attribute")


def get_default_dict_for_optimizer(optimizer, model_sharded_state_dict):
    """get default sharded state dict for the optimizer with models' shard and no opt shard"""
    state_dict = {}
    for model_param in optimizer.parameters:
        model_name = model_param.name
        if model_name in model_sharded_state_dict and "shard" in model_sharded_state_dict[model_name]:
            shard = list(model_sharded_state_dict[model_name]["shard"])
        else:
            raise Exception(f"the input dict has no shard info for '{model_name}'.")

        for optim_param in optimizer.get_parameters():
            optim_name = optim_param.name
            if optim_name.endswith(model_name) and optim_name != model_name:
                state_dict[optim_name] = {
                    "shape": model_param.shape,
                    "shard": tuple(shard),
                    "opt_weight_shard_step": 0,
                    "opt_weight_shard_size": 0,
                }
    return state_dict


def generate_state_dict(network: Module, optimizer: Optimizer):
    r"""
    Generete the sharded stated dict for the network and optimizer.

    The `network` should be of type Module which has inherited or overridden method sharded_state_dict,
    The `optimizer` is the corresponding optimizer.

    Args:
        network (Module): the integral model to be used
        optimizer (Optimizer): the corresponding optimizer

    Returns:
        Dict, which contains the necessary sharded info for checkpoint transformation, e.g. 'total_rank',
    'stage_rank_size', 'stage' for pipeline, etc.

    Supported Platforms:
        ``Ascend``
    """
    try:
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()
        ep_size = get_expert_model_parallel_world_size()
        ep_rank = get_expert_model_parallel_rank()
    except AssertionError:
        pp_size = 1
        pp_rank = 0
        ep_size = 1
        ep_rank = 0
    state_dict = {
        'total_rank': get_group_size(),
        'stage_rank_size': get_group_size() // pp_size,
        'stage': pp_rank,
        'expert_world_size': ep_size,
        'expert_rank': ep_rank,
    }
    state_dict['model'] = {}
    if isinstance(network, (nn.SequentialCell, nn.CellList)):
        for model_chunk in network:
            state_dict['model'].update(model_chunk.sharded_state_dict())
    else:
        state_dict['model'].update(network.sharded_state_dict())
    state_dict['optimizer'] = {}
    if optimizer is not None:
        if hasattr(optimizer, 'sharded_state_dict'):
            state_dict['optimizer'] = optimizer.sharded_state_dict(state_dict['model'])
        else:
            logger.warning(f"The optimizer {type(optimizer).__name__} has no sharded_state_dict overridden")
            state_dict['optimizer'] = get_default_dict_for_optimizer(optimizer, state_dict['model'])

    return state_dict


def save_strategy_file(state_dict, strategy_file_name):
    r"""
    Save the strategy file according to the state_dict and strategy_file_name

    Args:
        state_dict (Dict): dict with sharding metainfo
        strategy_file_name (String): the name of the target saving file

    Supported Platforms:
        ``Ascend``
    """
    import os
    import stat
    from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy

    stra = ckpt_strategy()

    # pylint: disable=W0612
    total_rank = state_dict["total_rank"]
    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    optimizer_param = state_dict["optimizer"]
    stra.current_stage = 0
    model_param.update(optimizer_param)
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = item["opt_weight_shard_step"] if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] if "opt_weight_shard_size" in item.keys() else 0
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        repeat_calc_num = 1
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(
                f"For {name}, the shard{shard} requires {shard_mul} devices, "
                f"but the device number of this stage is {stage_rank_size}, "
                f"it can not be divisible by {shard_mul}"
            )
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1
        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(strategy_file_name[: strategy_file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(strategy_file_name, flags, modes), 'wb') as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(
            f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
            f"the permission to write files, or the disk space is insufficient and so on."
        )
        raise e


def valid_lora_config(model_config, params):
    """valid target_cells in lora_config by params of the pretrain checkpoint.

    Args:
        model_config (TransformerConfig): Config of the model.
        params (dict): parameters of the pretrain model.
    """
    lora_config = model_config.lora_config
    target_cells_flag = {}
    target_cells_lst = lora_config.target_cells[0]
    specific_lora_cell = lora_config.target_cells[1]
    for cell_name in target_cells_lst:
        target_cells_flag.update({f'{cell_name}': False})

    lora_module = {}
    key_list = _get_cell_name_from_params(params)

    for key in key_list:
        match_result = _check_target_cell_exists(key, target_cells_lst)
        if not match_result:
            continue
        target_cells_flag[match_result] = True

        if _check_target_linear(key):
            # find key in specific_rank and specific_alpha
            rank, alpha = _get_specific_rank_alpha(key, specific_lora_cell, lora_config.lora_rank,
                                                   lora_config.lora_alpha)
            module_dict = _create_module_dict(key, (rank, alpha), lora_config.lora_dropout)
            if not lora_module:
                lora_module = module_dict
            else:
                _update_dic(lora_module, module_dict)

    # valid if all target_cells are found
    invalid_target_cell = [k for k, v in target_cells_flag.items() if v is False]

    if invalid_target_cell:
        logger.warning(
            f"target_cells should be in parameters name of the model, but got '{invalid_target_cell}', "
            f"these cells will be ignored.")
    if not lora_module:
        raise ValueError("target_cells in your lora config is invalid, please check your target_cells.")

    model_config.lora_config.lora_module = lora_module
    return model_config


def _check_target_linear(key):
    """check the cell is the Linear module"""
    if any(p in key for p in ['norm', 'embedding']):
        logger.warning(f"target_cells type should be Linear layers, but got '{key}', this cell will be ignored.")
        return False
    return True


def _update_dic(total_dic, item_dic):
    """update lora module dict"""
    for item in item_dic.keys():
        total_value = total_dic.get(item)
        item_value = item_dic.get(item)

        if total_value is None:
            total_dic.update({item: item_value})
        else:
            _update_dic(total_value, item_value)
            total_dic.update({item: total_value})
    return total_dic


def _get_cell_name_from_params(params):
    """get cell name list from parameter dict"""
    cell_name = []
    for key in params.keys():
        modules = key.split('.')
        split = '.'
        cell_name.append(split.join(modules[:-1]))
    return list(set(cell_name))


def _create_module_dict(key, val, dropout):
    """create lora module dict by cell name and rank/alpha"""
    key_lst = key.split('.')
    rank, alpha = val
    key_num = len(key_lst)
    final_dict = tmp_dict = {}
    for index, k in enumerate(key_lst):
        tmp_dict.setdefault(k, {})
        tmp_dict = tmp_dict.get(k)
        if index == key_num - 1:
            tmp_dict['rank'] = rank
            tmp_dict['alpha'] = alpha
            tmp_dict['dropout'] = dropout
    return final_dict


def _get_specific_rank_alpha(key, specific_lora_cell, rank, alpha):
    """get rank and alpha value from target_cells config"""
    if not specific_lora_cell:
        return rank, alpha
    default_rank, defult_alpha = rank, alpha
    for rank_alpha_dict in specific_lora_cell:
        cell_name = rank_alpha_dict['cell']
        match = re.match(cell_name, key)
        if match is not None and match.group() == key:
            rank = rank_alpha_dict.get('rank', default_rank)
            alpha = rank_alpha_dict.get('alpha', defult_alpha)
    return rank, alpha


def _check_target_cell_exists(key, target_cells_lst):
    """check target_cell is in pretrain parameters"""
    target_cell_found = False
    for target_key in target_cells_lst:
        match = re.match(target_key, key)
        if match is not None and match.group() == key:
            return target_key
    return target_cell_found
