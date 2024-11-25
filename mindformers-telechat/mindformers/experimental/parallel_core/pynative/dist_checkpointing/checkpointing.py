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

"""Model and parameters serialization."""

__all__ = [
    "get_checkpoint_name",
    "save_rng_state",
    "load_rng_state",
    'save_pre_process',
    'load_post_process',
    "save_checkpoint",
    "load_checkpoint"
]

import os
import glob
import json
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import _checkparam as validator
from mindspore.ops import operations as P
import mindspore.communication.comm_func as comm_func
try:
    from mindspore import default_generator, set_rng_state
except ImportError:
    from mindspore.nn.generator import default_generator, set_rng_state
from mindspore.communication import get_rank
from mindformers.tools.logger import logger
from mindformers.experimental.parallel_core.pynative.utils import generate_state_dict, save_strategy_file
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_expert_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import (
    get_rng_tracer,
    CANDIDATE_MODES
)
from mindformers.experimental.parallel_core.pynative.optimizer.optimizer import MixedPrecisionOptimizer

# Distribution configurations.
_STRATEGY_DIR = "strategy"
_FORMAT = "ckpt"


# pylint: disable=W0622
def get_checkpoint_name(ckpt_path, format=_FORMAT, get_name_from_file=False,
                        prefix: str = "network", epoch_num: int = None, step_num: int = None):
    """
    Get checkpoint file name of model and optimizer.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── rank_0
    │   ├── network_rank_0-0_0.ckpt
    │   └── network_rank_0-0_1.ckpt
    └── rank_1
        ├── network_rank_1-0_0.ckpt
        └── network_rank_1-0_1.ckpt
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    validator.check_value_type("ckpt_path", ckpt_path, [str])
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    os.makedirs(ckpt_local_path, exist_ok=True)
    # get default strategy file name
    strategy_local_path = os.path.join(ckpt_path, _STRATEGY_DIR)
    strategy_file = os.path.join(strategy_local_path, f"stratey{rank}.ckpt")
    # read ckpt name according to the ckpt path or return default name
    if get_name_from_file:
        rank_ckpts = glob.glob(os.path.join(ckpt_local_path, "*." + format))
        if not rank_ckpts:
            raise RuntimeError(f"{ckpt_local_path} has no .{format} ckpt file found")
        for checkpoint_file in rank_ckpts:
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            ckpt_file = checkpoint_file
    else:
        ckpt_file = os.path.join(ckpt_local_path, f"{prefix}_rank_{rank}-{epoch_num}_{step_num}.{format}")
    return ckpt_file, strategy_file


def save_rng_state():
    """ save random number generator state. """
    rng_state_dict = get_rng_tracer().get_state()
    rng_state_dict["default_generator"] = default_generator
    return rng_state_dict


def load_rng_state(param_dict):
    """ load random number generator state. """
    # set default rng tracer state
    target_state = {
        mode: param_dict.pop(mode)
        for mode in CANDIDATE_MODES
        if mode in param_dict
    }
    get_rng_tracer().set_state(target_state)
    # set default generator state
    if "default_generator" in param_dict:
        default_generator_loaded = param_dict.pop("default_generator")
        set_rng_state(default_generator_loaded)


def _update_zero(params_dict, shard_info, param, group):
    """ allgather param among dp region when using zero optimizer. """
    tensor_concat = comm_func.all_gather_into_tensor(param, group=group)[0]
    params_dict[param.name] = ms.Parameter(tensor_concat, name=param.name)
    shard_info[param.name]['opt_weight_shard_size'] = 0
    shard_info[param.name]['opt_weight_shard_step'] = 0


def _get_params_dict(model, optimizer):
    """ get params dict for saving checkpoint. """
    params_dict = None
    if optimizer is None:
        params_dict = model.parameters_dict()
    elif isinstance(optimizer, MixedPrecisionOptimizer):
        params_dict = optimizer.state_dict()
    else:
        params_dict = optimizer.parameters_dict()
    if not params_dict:
        raise ValueError("None of params dict has been extract from model and optimizer.")
    return params_dict


# pylint: disable=W0212
def save_pre_process(shard_info, model, optimizer, config):
    """ preprocess before saving, split qkv and handle pp embedding share """
    model_shard_info = shard_info["model"]
    optimizer_shard_info = shard_info["optimizer"]
    params_dict = _get_params_dict(model, optimizer)
    # ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        dp_tp_group = get_data_parallel_group(with_context_parallel=optimizer.with_context_parallel)
        for idx, param in enumerate(optimizer._parameters):
            if optimizer._status_splited[idx] or optimizer._parameter_splited[idx]:
                _update_zero(params_dict, optimizer_shard_info, optimizer.moments1[idx], dp_tp_group)
                _update_zero(params_dict, optimizer_shard_info, optimizer.moments2[idx], dp_tp_group)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                _update_zero(params_dict, model_shard_info, param, dp_tp_group)

    # process qkv/moe/pp-share
    for name, param in list(params_dict.items()):
        target_shard_info = model_shard_info if name in model_shard_info else optimizer_shard_info
        ### moe layer
        if config.moe_config is not None and config.moe_config.num_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.moe_config.num_experts // get_expert_model_parallel_world_size()
            local_experts_list = []
            for idx in range(local_expert_num):
                local_expert_name = name.replace("local_experts.0", f"local_experts.{idx}")
                local_expert_param = params_dict.pop(local_expert_name).asnumpy()
                local_experts_list.append(local_expert_param)
                shard_dict = target_shard_info.pop(local_expert_name)
            local_experts_concat = np.stack(local_experts_list, axis=0)
            params_dict[name] = ms.Parameter(ms.Tensor(local_experts_concat))
            shard_dict['shape'] = local_experts_concat.shape
            shard_dict['shard'] = (get_expert_model_parallel_world_size(),) + shard_dict['shard']
            target_shard_info[name] = shard_dict

        ### handle pipeline head sharing
        if get_pipeline_model_parallel_world_size() == 1 and not config.untie_embeddings_and_output_weights:
            language_model_embedding = "language_model.embedding.word_embeddings.weight"
            language_model_head = "language_model.output_layer.weight"
            if language_model_embedding in name:
                new_name = name.replace(language_model_embedding, language_model_head)
                params_dict[new_name] = ms.Parameter(param, name=new_name)
                target_shard_info[new_name] = target_shard_info[name]

    return shard_info, params_dict


# pylint: disable=W0212
def load_post_process(config, params_dict, optimizer=None):
    """ load post processing, concat qkv """
    for name, param in list(params_dict.items()):
        ### moe layer
        if config.moe_config is not None and config.moe_config.num_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.moe_config.num_experts // get_expert_model_parallel_world_size()
            params_dict.pop(name)
            for shard_id in range(local_expert_num):
                new_name = name.replace("local_experts.0", f"local_experts.{shard_id}")
                params_dict[new_name] = ms.Parameter(ms.Tensor(param[shard_id]))

    ### ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        shard_id = get_data_parallel_rank()
        split = P.Split(0, get_data_parallel_world_size())
        for idx, param in enumerate(optimizer._parameters):
            if optimizer._status_splited[idx] or optimizer._parameter_splited[idx]:
                # moments1
                moments1_name = optimizer.moments1[idx].name
                moments1 = params_dict[moments1_name]
                splited_tensor = split(moments1)[shard_id]
                params_dict[moments1_name] = ms.Parameter(splited_tensor, name=moments1_name)
                # moments2
                moments2_name = optimizer.moments2[idx].name
                moments2 = params_dict[moments2_name]
                splited_tensor = split(moments2)[shard_id]
                params_dict[moments2_name] = ms.Parameter(splited_tensor, name=moments2_name)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                # param
                if "norm" in param.name or "embedding" in param.name:
                    continue
                cell_param = params_dict[param.name]
                splited_tensor = split(cell_param)[shard_id]
                params_dict[param.name] = ms.Parameter(splited_tensor, name=param.name)

    return params_dict


# pylint: disable=W0622
def save_checkpoint(config, model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    only_save_strategy=False, prefix: str = 'network', epoch_num: int = 0, step_num: int = 0,
                    crc_check: bool = False, keep_checkpoint_max: int = 5, **kwargs):
    """
    Save checkpoint of distributed network to a specified file in the process of specified rank.

    Args:
        model (Cell): The network to be saved.
        ckpt_path (str): Checkpoint file path. Default: ``"./"``.
        optimizer (Cell): The optimizer to be saved. Default: ``None``.
        kwargs (dict): Configuration options dictionary.

    Raises:
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell.
        TypeError: If the type of parameter `ckpt_path` is not str.
    """
    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    if keep_checkpoint_max < 1:
        raise ValueError(f"expect keep_checkpoint_max >= 1, but got {keep_checkpoint_max}")
    # validator check
    validator.check_value_type("model", model, [nn.Cell], "save_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "save_checkpoint")
    rank_path = os.path.join(ckpt_path, f"rank_{get_rank()}")
    ckpt_file, strategy_file = get_checkpoint_name(ckpt_path, format=format, prefix=prefix, epoch_num=epoch_num,
                                                   step_num=step_num)
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in save_checkpoint.")
    logger.info(f"Saving model to {ckpt_file}")

    # generate sharded info
    shard_info = generate_state_dict(model, optimizer)
    shard_info, params_dict = save_pre_process(shard_info, model, optimizer, config)

    # saving
    save_strategy_file(shard_info, strategy_file)
    if not only_save_strategy:
        rng_state_dict = save_rng_state()
        append_dict = rng_state_dict.copy()
        if opt_param_scheduler is not None:
            opt_state_dict = opt_param_scheduler.state_dict()
            append_dict.update(opt_state_dict)
        append_dict.update({"epoch_num": epoch_num, "step_num": step_num})
        # ensure ckpt number is less than `keep_checkpoint_max` after saving,
        # so make 1 free space for incoming ckpt
        ensure_total_ckpt_is_less_than_limit(ckpt_path=rank_path, limit=keep_checkpoint_max - 1, format=format)
        ms.save_checkpoint(params_dict, ckpt_file, append_dict=append_dict, format=format, crc_check=crc_check)
        record_last_ckpt_to_json(epoch=epoch_num, step=step_num, ckpt_file=os.path.basename(ckpt_file),
                                 meta_json=os.path.join(rank_path, 'meta.json'))
    logger.info("ckpt saved")

def ensure_total_ckpt_is_less_than_limit(ckpt_path: str, limit: int = 5, format: str = _FORMAT):
    """
    make sure the provided path contain less than limited number of checkpoint file
    Args:
        ckpt_path (str): Checkpoint file path.
        limit (int): limited number of checkpoint file. Default: 5
        format (str): checkpoint format. Default: '_format'
    """
    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format}')
    ]
    # ckpt_list: [oldest, ..., newest]
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    ckpt_num = len(ckpt_list)
    if ckpt_num > limit:
        for rm_ckpt_name in ckpt_list[: (ckpt_num - limit)]:
            logger.debug(f"Current checkpoint file exceed keep_checkpoint_max, removing {rm_ckpt_name}")
            rm_ckpt_path = os.path.join(ckpt_path, rm_ckpt_name)
            os.remove(rm_ckpt_path)

# pylint: disable=W0622
def load_checkpoint(config, model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    crc_check=False, **kwargs):
    """
    Load checkpoint info from a specified file in process of rank 0.

    Args:
        ckpt_path (str): Checkpoint file path.
        model (Cell): The network where the parameters will be loaded.
        optimizer (Cell): The optimizer where the parameters will be loaded.

    Raises:
        TypeError: If the type of parameter `ckpt_path` is not str.
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell. Default: ``None``.
    """
    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    validator.check_value_type("model", model, [nn.Cell], "load_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "load_checkpoint")
    logger.info("ckpt loading")
    if os.path.isdir(ckpt_path):
        src_ckpt_file = get_last_checkpoint(os.path.join(ckpt_path, f"rank_{get_rank()}"), format=format)
    elif os.path.isfile(ckpt_path):
        src_ckpt_file = ckpt_path
    else:
        raise ValueError(f"There is no *.{format} in {ckpt_path}, load failed.")
    logger.info(f"using latest checkpoint: {src_ckpt_file}")
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in load_checkpoint.")

    param_dict = ms.load_checkpoint(src_ckpt_file, format=format, crc_check=crc_check)
    resume_dict = {
        "epoch_num": int(param_dict.pop("epoch_num", 0)),
        "step_num": int(param_dict.pop("step_num", 0))
    }

    load_rng_state(param_dict)
    if opt_param_scheduler is not None:
        opt_param_scheduler.load_state_dict(param_dict)
    if isinstance(optimizer, MixedPrecisionOptimizer):
        # restore distributed optimizer
        optimizer.load_state_dict(param_dict)
        # synchronize parameters in optimizer to model
        optimizer.reload_main_params()
    else:
        # restore native optimizer/model
        param_dict = load_post_process(config, param_dict, optimizer)
        target = optimizer if optimizer is not None else model
        param_not_load, ckpt_not_load = ms.load_param_into_net(target, param_dict)
        if param_not_load:
            logger.warning(f"param_not_load:{param_not_load}")
        if ckpt_not_load:
            logger.warning(f"ckpt_not_load:{ckpt_not_load}")
    logger.info("ckpt loaded")

    return resume_dict

def get_last_checkpoint(ckpt_path: str, format: str = _FORMAT):
    """Get last timestamp checkpoint under ckpt_path."""
    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format}')
    ]
    if not ckpt_list:
        return None
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    return os.path.join(ckpt_path, ckpt_list[-1])

def record_last_ckpt_to_json(epoch: int, step: int, ckpt_file: str, meta_json: str):
    """record last ckpt info to json"""
    meta_data = {
        "last_epoch": epoch,
        "last_step": step,
        "last_ckpt_file": ckpt_file
    }
    with open(meta_json, 'w', encoding="utf-8") as fp:
        json.dump(meta_data, fp)
