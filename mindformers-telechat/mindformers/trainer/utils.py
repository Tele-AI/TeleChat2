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
"""Trainer Utils."""
import os
import json
import sys
import random
import re
from enum import Enum

import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net, load_checkpoint_async
from mindspore import set_seed as ms_set_seed
from mindspore import Parameter
from mindspore import ops, mint

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank
from mindformers.utils.load_checkpoint_utils import CkptFormat, load_checkpoint_with_safetensors, build_model
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import (
    replace_tk_to_mindpet,
    check_shared_disk,
    get_device_num_per_node,
    format_path
)
from mindformers.tools.ckpt_transform import TransformCkpt
from mindformers.models.base_model import BaseModel
from mindformers.models.modeling_utils import PreTrainedModel


class BaseEnum(str, Enum):
    """
    Base Enum for MindFormers.
    """

    @classmethod
    def _missing_(cls, value):
        """Enum with more explicit error message for missing values."""
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class IntervalStrategy(BaseEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(BaseEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class LoggingIntervalStrategy(BaseEnum):
    STEPS = "steps"
    EPOCH = "epoch"


class SaveIntervalStrategy(BaseEnum):
    """
    Stores the acceptable string identifiers for save checkpoint monitor.
    """
    NO = "no"
    STEPS = "steps"
    SECONDS = "seconds"


class LrSchedulerType(BaseEnum):
    """
    Stores the acceptable string identifiers for learning rate schedule.
    """
    # will be support item for future.
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType(BaseEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    # supported item for test, will be deleted in the future.
    ADAMWEIGHTDECAY = 'AdamWeightDecay'

    # will be support item for future.
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAFACTOR = "adafactor"
    FUSED_ADAMW = "fused_adamw"
    FP32_ADAMW = "fp32_adamw"


class WrapperType(BaseEnum):
    """
    Stores the acceptable string identifiers for training wrapper.
    """
    # will be support item for future.
    MFWRAPPER = 'mf_wrapper'
    TRAINONESTEP = 'wrapper'
    TRAINONESTEPWITHLOSSSCALE = 'loss_scale_wrapper'


def set_seed(seed: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `MindSpore`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms_set_seed(seed)


def check_keywords_in_name(name, keywords=()):
    """ Check keywords in name. """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_runner_config(config, dataset):
    """ Check runner config. """
    data_size = dataset.get_dataset_size()
    new_epochs = config.runner_config.epochs
    config.runner_config.origin_epochs = new_epochs
    if config.runner_config.gradient_accumulation_steps is None:
        config.runner_config.gradient_accumulation_steps = 1
    if config.runner_config.initial_epoch is None:
        config.runner_config.initial_epoch = 0
    if config.runner_config.initial_step is None:
        config.runner_config.initial_step = 0
    if config.runner_config.sink_mode:
        if config.runner_config.sink_size != -1:
            if config.runner_config.sink_size <= 0:
                raise ValueError("per epoch size must be more than 0 or equal to -1, "
                                 f"but get {config.runner_config.sink_size}")
            if data_size < config.runner_config.sink_size:
                logger.warning("The data size %s (get from dataset.get_dataset_size()) is smaller "
                               "than the sink_size %s (get from config.runner_config.sink_size), "
                               "you should set the config.runner_config.sink_size to %s",
                               data_size, config.runner_config.sink_size, data_size)
            config.runner_config.epochs = int((data_size / config.runner_config.sink_size) * new_epochs)
        else:
            config.runner_config.sink_size = data_size
    else:
        logger.warning("Sink mode is False, per epoch size is invalid, it will reset -1.")
        config.runner_config.sink_size = -1

    config.data_size = data_size
    logger.info("Will be Training epochs:%d, sink_size:%d",
                config.runner_config.origin_epochs, config.runner_config.sink_size)
    logger.info("Create training dataset finish, dataset size:%d", data_size)


def check_train_data_loader_type(new_config, old_config):
    """Check train data loader config type."""
    if new_config.train_dataset is None:
        return None
    if new_config.train_dataset.get('data_loader') is None:
        return None
    train_data_loader_type = new_config.train_dataset.get('data_loader').get('type')
    if old_config.train_dataset is not None and train_data_loader_type is not None:
        default_train_data_loader_type = old_config.train_dataset.data_loader.type
        if train_data_loader_type != default_train_data_loader_type:
            logger.warning("train dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           train_data_loader_type)
            old_config.train_dataset.data_loader = {}
    return None


def check_eval_data_loader_type(new_config, old_config):
    """Check eval data loader config type."""
    if new_config.eval_dataset is None:
        return None
    if new_config.eval_dataset.get('data_loader') is None:
        return None
    eval_data_loader_type = new_config.eval_dataset.get('data_loader').get('type')
    if old_config.eval_dataset is not None and eval_data_loader_type is not None:
        default_eval_data_loader_type = old_config.eval_dataset.data_loader.type
        if eval_data_loader_type != default_eval_data_loader_type:
            logger.warning("eval dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           eval_data_loader_type)
            old_config.eval_dataset.data_loader = {}
    return None


def check_optimizer_and_lr_type(new_config, old_config):
    """Check optimizer and lr schedule config type."""
    optimizer_type = new_config.optimizer.get('type')
    if old_config.optimizer is not None and optimizer_type is not None:
        default_optimizer_type = old_config.optimizer.type
        if optimizer_type != default_optimizer_type:
            logger.warning(
                "optimizer type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually except (params).",
                optimizer_type)
            old_config.optimizer = {}

    if hasattr(new_config.optimizer, 'learning_rate'):
        lr_type = new_config.optimizer.learning_rate.get('type')
        if old_config.lr_schedule is not None and lr_type is not None:
            default_lr_type = old_config.lr_schedule.type
            if lr_type != default_lr_type:
                logger.warning(
                    "lr schedule type is changed to %s."
                    "The default parameters will be cleared."
                    "Please make sure to input the corresponding parameter values manually.",
                    lr_type)
                old_config.lr_schedule = None


def check_wrapper_config(new_config, old_config):
    """Check wrapper config."""
    wrapper_type = new_config.runner_wrapper.get('type')
    if old_config.runner_wrapper is not None and wrapper_type is not None:
        default_wrapper_type = old_config.runner_wrapper.type
        if wrapper_type != default_wrapper_type:
            logger.warning(
                "wrapper type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually.",
                wrapper_type)
            old_config.runner_wrapper = {}


def config2dict(config):
    """MindFormerConfig Type Convert to Dict."""
    if not isinstance(config, (dict, MindFormerConfig)):
        return config
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, MindFormerConfig):
            value = config2dict(value)
        new_dict.setdefault(key, value)
    return new_dict


def get_distribute_checkpoint_path(checkpoint_dir, rank_id=None, ckpt_format='ckpt'):
    """Helper function to get the checkpoint path based on rank_id and checkpoint directory."""
    if os.path.isdir(checkpoint_dir):
        logger.info(
            "When distributed loads are sliced weights,"
            "load_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/**.ckpt")
        rank_id = rank_id if rank_id is not None else get_real_rank()
        distribute_checkpoint_dir = os.path.join(checkpoint_dir, "rank_{}".format(rank_id))
        distribute_checkpoint_path = get_last_checkpoint(distribute_checkpoint_dir, ckpt_format)
        logger.info("distribute checkpoint dir: %s", distribute_checkpoint_dir)
    elif os.path.isfile(checkpoint_dir):
        logger.info("Your load_checkpoint is file, it will be load in network.")
        distribute_checkpoint_path = checkpoint_dir
    else:
        raise FileNotFoundError(f"{checkpoint_dir} is not found.")
    return distribute_checkpoint_path


def load_distributed_checkpoint(checkpoint_dir, choice_func=None, rank_id=None, ckpt_format='ckpt',
                                remove_redundancy=False):
    """Load Checkpoint in Parallel Mode."""
    distribute_checkpoint_path = get_distribute_checkpoint_path(checkpoint_dir, rank_id, ckpt_format)
    checkpoint_dict = load_checkpoint(distribute_checkpoint_path,
                                      choice_func=choice_func,
                                      format=ckpt_format,
                                      remove_redundancy=remove_redundancy)
    logger.info("Distribute load is success.")
    return checkpoint_dict


def load_distributed_checkpoint_async(checkpoint_dir, choice_func=None, rank_id=None):
    """Load Checkpoint with async in Parallel Mode"""
    distribute_checkpoint_path = get_distribute_checkpoint_path(checkpoint_dir, rank_id)
    async_checkpoint_future = load_checkpoint_async(distribute_checkpoint_path, choice_func=choice_func)
    logger.info("Starting loading distributed checkpoints asynchronously.")
    return async_checkpoint_future


def load_resume_context_from_checkpoint(config, dataset):
    """resume training, load training info from checkpoint to config"""
    if not os.path.realpath(config.load_checkpoint) or \
            not os.path.exists(config.load_checkpoint):
        raise FileNotFoundError(f"The load_checkpoint must be correct, but get {config.load_checkpoint}")

    if os.path.isdir(config.load_checkpoint):
        if isinstance(config.resume_training, bool):
            # if load checkpoint is complete safetensors, get resume param from hyper_param.safetensors
            if is_hyper_param_existed_in_sf_dir(config.load_checkpoint, config.load_ckpt_format):
                hyper_param_file = os.path.join(config.load_checkpoint, 'hyper_param.safetensors')
                resume_param = load_checkpoint(hyper_param_file, format='safetensors')
                resume_dict = {'loss_scale': resume_param['loss_scale'],
                               'epoch_num': resume_param['epoch_num'],
                               'step_num': resume_param['step_num'],
                               'global_batch_size': resume_param['global_batch_size']}
            else:
                if config.use_graceful_exit:
                    rank_id = get_real_rank()
                else:
                    rank_id = 0
                resume_dict = load_distributed_checkpoint(config.load_checkpoint,
                                                          choice_func=lambda x: x in ["loss_scale", "epoch_num",
                                                                                      "step_num", "global_batch_size"],
                                                          rank_id=rank_id, ckpt_format=config.load_ckpt_format,
                                                          remove_redundancy=config.remove_redundancy)
        else:
            checkpoint_tmp = os.path.join(config.load_checkpoint, f"rank_{config.rank_id}", config.resume_training)
            resume_dict = load_checkpoint(
                checkpoint_tmp,
                choice_func=lambda x: x in ["loss_scale", "epoch_num", "step_num", "global_batch_size"],
                format=config.load_ckpt_format, remove_redundancy=config.remove_redundancy)

    else:
        resume_dict = load_checkpoint(
            config.load_checkpoint,
            choice_func=lambda x: x in ["loss_scale", "epoch_num", "step_num", "global_batch_size"],
            format=config.load_ckpt_format, remove_redundancy=config.remove_redundancy)

    step_scale = 1.0
    if "global_batch_size" in resume_dict:
        last_global_batch_size = int(resume_dict["global_batch_size"])
        step_scale = last_global_batch_size / config.runner_config.global_batch_size
        logger.info("Detect global batch size is changed from %d to %d, scale: %f",
                    last_global_batch_size, config.runner_config.global_batch_size, step_scale)
    config.runner_config.step_scale = step_scale

    if "step_num" in resume_dict:
        config.runner_config.initial_step = int(resume_dict["step_num"] * step_scale)
    else:
        config.runner_config.initial_step = 0

    if "epoch_num" in resume_dict:
        if config.runner_config.sink_mode:
            config.runner_config.initial_epoch = int(resume_dict["epoch_num"] * step_scale)
        else:
            data_size = dataset.get_dataset_size()
            not_last_step_in_epoch = int(config.runner_config.initial_step % data_size != 0)
            config.runner_config.initial_epoch = int(resume_dict["epoch_num"]) - not_last_step_in_epoch
    else:
        config.runner_config.initial_epoch = 0

    for callback in config.callbacks:
        if "type" in callback and callback["type"] == "CheckpointMonitor":
            if config.runner_wrapper.scale_sense is not None and "loss_scale" in resume_dict:
                if hasattr(config.runner_wrapper.scale_sense, "loss_scale_value"):
                    config.runner_wrapper.scale_sense.loss_scale_value = resume_dict["loss_scale"]
                else:
                    config.runner_wrapper.scale_sense = resume_dict["loss_scale"]
            break
    logger.info("initial epoch: %d", config.runner_config.initial_epoch)
    logger.info("initial step: %d", config.runner_config.initial_step)
    logger.info("step scale: %f", config.runner_config.step_scale)


def is_hyper_param_existed_in_sf_dir(checkpoint_dir, load_ckpt_format):
    """return True if load_ckpt_format is True and given dir contains hyper_param.safetensors file used for resume training"""
    return load_ckpt_format == 'safetensors' and os.path.exists(os.path.join(checkpoint_dir, 'hyper_param.safetensors'))


def transform_and_load_checkpoint(config, model, network, dataset, optimizer=None, do_eval=False, do_predict=False):
    """
    load checkpoint into net, transform checkpoint if transform is True
    1. build net if parallel mode is auto_parallel
    2. get strategy
    3. make softlink of input path
    4. transform checkpoint if need
    5. load ckpt
    """
    # check checkpoint config valid
    check_checkpoint_config_valid(config)

    # load safetensors process
    if not config.only_save_strategy and config.load_ckpt_format == CkptFormat.SAFETENSORS.value:
        load_checkpoint_with_safetensors(config, model, network, dataset, do_eval=do_eval,
                                         do_predict=do_predict, optimizer=optimizer)
        return

    if (not config.auto_trans_ckpt and not config.only_save_strategy and
            check_path_include_total_ckpt(config.load_checkpoint)):
        load_ckpt(config, network, optimizer=optimizer)
        return

    logger.warning(".ckpt file loading mode will be offline in June 2025.Recommend loading .safetensors file")
    # load ckpt process
    checkpoint_future = None
    # judge whether to enable load checkpoint and model build in parallel
    if config.load_ckpt_async:
        if config.auto_trans_ckpt:
            logger.info("The configuration of 'load_ckpt_async=True' will not "
                        "take effect since 'auto_trans_ckpt=True'.")
        elif config.only_save_strategy:
            logger.info("Only_save_strategy is True, load ckpt async will not be performed.")
        else:
            logger.info(".........Start loading checkpoint async.........")
            checkpoint_future = get_load_checkpoint_result(config)

    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        # build net if parallel mode is auto_parallel
        logger.info(".........Building model.........")
        build_model(config, model, dataset, do_eval=do_eval, do_predict=do_predict)
        if config.only_save_strategy:
            logger.info("Only_save_strategy is True, model.compile() finished, system exit! ")
            sys.exit(0)

    elif config.only_save_strategy:
        logger.info("only_save_strategy is True, "
                    "but stand_alone and data_parallel mode do not have strategy file, system exit! ")
        sys.exit(0)

    if config.auto_trans_ckpt:
        is_share_disk = check_shared_disk(config.output_dir)
        logger.info("%s is_share_disk: %r", os.path.abspath(config.output_dir), is_share_disk)
        transform_process_num = config.transform_process_num if config.transform_process_num else 1
        transform_by_rank = config.transform_by_rank if config.transform_by_rank else False
        npu_num_per_node = config.npu_num_per_node \
            if config.npu_num_per_node else get_device_num_per_node()
        transform_ckpt = TransformCkpt(auto_trans_ckpt=True,
                                       transform_process_num=transform_process_num,
                                       transform_by_rank=transform_by_rank,
                                       npu_num_per_node=npu_num_per_node)
        config.load_checkpoint = transform_ckpt(
            src_checkpoint=config.load_checkpoint,
            src_strategy=config.src_strategy_path_or_dir
        )

    # load ckpt
    load_ckpt(config, network, optimizer=optimizer, model=model, future=checkpoint_future)


def check_checkpoint_config_valid(config):
    # check valid load checkpoint path
    if not config.only_save_strategy and (not os.path.realpath(config.load_checkpoint) or
                                          not os.path.exists(config.load_checkpoint)):
        raise FileNotFoundError(f"The load_checkpoint must be correct, but get {config.load_checkpoint}")

    # check valid format
    if config.load_ckpt_format is not None and config.load_ckpt_format not in CkptFormat.support_type():
        raise ValueError(
            f"config.load_ckpt_format only support for 'ckpt' or 'safetensors', but got {config.load_ckpt_format}.")


def check_rank_folders(path, rank_id):
    """check if the folders in path are correct"""
    folder_name = "rank_{}".format(rank_id)
    if not os.path.exists(os.path.join(path, folder_name)):
        return False
    return True


def check_ckpt_file_exist(path):
    """check if the files in path endswith .ckpt"""
    for file_name in os.listdir(path):
        if file_name.endswith('.ckpt'):
            return True
    return False


def check_path_include_total_ckpt(path):
    """check if the input path is total, not split."""
    if path is None:
        return False
    if os.path.isdir(path):
        if check_ckpt_file_exist(path):
            return True
    elif path.endswith('.ckpt'):
        pattern = r'rank_\d+'
        if re.search(pattern, path):
            return False
        return True
    return False


def load_slora_ckpt(checkpoint_dict, config, network):
    """
    1. Adjust the weight name of Linear to the weight name of the SLoraLinear according to the target_modules.
    2. Load SLora weight Parameter from lora_ckpt_path.

    Args:
        checkpoint_dict (Dict[str, Parameter]): contains the base model Parameter dict.
        config: config with SLora adapter config json path.
        network: SLora network.

    Returns:
         Dict[str, Parameter]: contain all the Parameter weights required by the SLora network.
    """
    # 1. replace slora layer in checkpoint_dict.keys(); 2. load slora param
    pet_config = config.model.model_config.get("pet_config")
    if not pet_config or pet_config.pet_type != "slora" or not network.lora_adapter.registered_loras:
        return checkpoint_dict

    logger.info("............Start load slora checkpoint ............")
    adapter_path = os.path.join(pet_config.adapter_path, "lora_adapter.json")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"The adapter_path must be correct, but get {adapter_path}")
    with open(adapter_path, 'r') as file:
        path_dict = json.load(file)
    adapter_list = []
    config_list = []
    for adapter_name in network.lora_adapter.adapter_names[1:]:
        if adapter_name in path_dict.keys():
            adapter_model = load_checkpoint(os.path.join(path_dict[adapter_name], "adapter_model.ckpt"))
            with open(os.path.join(path_dict[adapter_name], "adapter_config.json"), 'r') as file:
                adapter_config = json.load(file)
        else:
            adapter_model = {}
            adapter_config = {}
        adapter_list.append(adapter_model)
        config_list.append(adapter_config)

    # collect lora weights
    slora_params = {}
    for param_name, param_shape in network.lora_adapter.registered_loras.items():
        lora_shape = tuple(param_shape[1:])
        slora_param = ops.zeros(lora_shape)
        for lora_params, lora_config in zip(adapter_list, config_list):
            if param_name in lora_params.keys():
                lora_param = lora_params[param_name]
                if re.match('.*lora_b.*', param_name):
                    # transpose lora_B shape from (n, r) to (r, n)
                    lora_param = ops.transpose(lora_param, (1, 0))
                    lora_param = mint.mul(lora_param, lora_config.get("lora_alpha") / lora_config.get("r"))
                if lora_param.shape != lora_shape:
                    pad_a = lora_shape[0] - lora_param.shape[0]
                    pad_b = lora_shape[1] - lora_param.shape[1]
                    lora_param = mint.nn.functional.pad(lora_param, (0, pad_b, 0, pad_a), mode='constant')
            else:
                lora_param = ops.zeros(lora_shape)
            slora_param = ops.cast(slora_param, lora_param.dtype)
            slora_param = ops.cat((slora_param, lora_param))
        slora_params[param_name] = Parameter(slora_param.reshape(param_shape))

    dst_checkpoint_dir = pet_config.adapter_path
    if config.auto_trans_ckpt:
        # Save collected lora weights as single ckpt
        from mindspore import save_checkpoint
        src_checkpoint_dir = os.path.join(config.output_dir, "slora_checkpoint")
        os.makedirs(src_checkpoint_dir, exist_ok=True)
        src_checkpoint_dir = os.path.join(src_checkpoint_dir, "slora.ckpt")
        save_checkpoint(slora_params, src_checkpoint_dir)

        logger.info("............Start transform slora checkpoint ............")
        transform_process_num = config.transform_process_num if config.transform_process_num else 1
        transform_by_rank = config.transform_by_rank if config.transform_by_rank else False
        npu_num_per_node = config.npu_num_per_node \
            if config.npu_num_per_node else get_device_num_per_node()
        transform_ckpt = TransformCkpt(auto_trans_ckpt=True,
                                       transform_process_num=transform_process_num,
                                       transform_by_rank=transform_by_rank,
                                       npu_num_per_node=npu_num_per_node)
        dst_checkpoint_dir = transform_ckpt(
            src_checkpoint=src_checkpoint_dir,
            src_strategy=config.src_strategy_path_or_dir
        )

    if config.use_parallel:
        dst_checkpoint_dir = os.path.join(dst_checkpoint_dir, "slora")
        slora_params = load_distributed_checkpoint(dst_checkpoint_dir)

    checkpoint_dict.update(slora_params)
    return checkpoint_dict


def get_load_checkpoint_result(config):
    """
    Get load checkpoint result:
    when loading checkpoint and model build are done in series, get the checkpoint dict
    when loading checkpoint and model build are done in parallel, get mindspore custom future object
    """
    load_ckpt_async = config.load_ckpt_async
    checkpoint_dict = {}
    # Define an object variable for an asynchronous task
    checkpoint_future = None
    rank_id = get_real_rank() if get_real_rank() else 0
    if config.auto_trans_ckpt:
        for checkpoint_name in os.listdir(config.load_checkpoint):
            checkpoint_path = os.path.join(config.load_checkpoint, checkpoint_name)
            # auto_trans_ckpt is set to true, loading ckpt with async is not supported
            checkpoint_dict.update(load_distributed_checkpoint(checkpoint_path))
            logger.info("loaded checkpoint: %s", str(checkpoint_path))
    else:
        if os.path.isdir(config.load_checkpoint) and check_ckpt_file_exist(config.load_checkpoint):
            for ckpt_file in os.listdir(config.load_checkpoint):
                if ckpt_file.endswith('.ckpt'):
                    checkpoint_path = os.path.join(config.load_checkpoint, ckpt_file)
                    if load_ckpt_async:
                        checkpoint_future = load_checkpoint_async(checkpoint_path)
                    else:
                        checkpoint_dict.update(load_checkpoint(checkpoint_path))
        elif os.path.isfile(config.load_checkpoint) and config.load_checkpoint.endswith('.ckpt'):
            if load_ckpt_async:
                checkpoint_future = load_checkpoint_async(config.load_checkpoint)
            else:
                checkpoint_dict = load_checkpoint(config.load_checkpoint)
        elif os.path.isdir(config.load_checkpoint) and check_rank_folders(config.load_checkpoint, rank_id):
            if isinstance(config.resume_training, str):
                checkpoint_tmp = os.path.join(config.load_checkpoint, f"rank_{config.rank_id}",
                                              config.resume_training)
                if load_ckpt_async:
                    checkpoint_future = load_checkpoint_async(checkpoint_tmp)
                else:
                    checkpoint_dict = load_checkpoint(checkpoint_tmp)
            else:
                if load_ckpt_async:
                    checkpoint_future = load_distributed_checkpoint_async(config.load_checkpoint)
                else:
                    checkpoint_dict = load_distributed_checkpoint(config.load_checkpoint)
        else:
            raise ValueError(f"{config.load_checkpoint} is not a valid path to load checkpoint "
                             f"when auto_trans_ckpt is False.")
    return checkpoint_dict if checkpoint_dict else checkpoint_future


def load_ckpt(config, network, optimizer=None, model=None, future=None):
    """
    load checkpoint

    Args:
        config: MindFormerConfig object.
        network: The network for task.
        optimizer: The optimizer for task.
        model: The model for task.
        future: Future object of asynchronous task that loading ckpt to dict. The actual return value
                of the asynchronous task can be obtained through future.result()
    """
    # load checkpoint params into dict
    config.load_checkpoint = format_path(config.load_checkpoint)

    if future is not None:
        # get ckpt dict when load checkpoint async
        logger.info("............Get the checkpoint-loading asynchronous result............")
        checkpoint_dict = future.result()
    else:
        # get ckpt dict when load checkpoint sync
        logger.info(f"............Start load checkpoint from {config.load_ckpt_format}............")
        checkpoint_dict = get_load_checkpoint_result(config)
    checkpoint_dict = load_slora_ckpt(checkpoint_dict, config, network)

    if isinstance(network, (BaseModel, PreTrainedModel)):
        checkpoint_dict = network.fuse_weight_from_ckpt(checkpoint_dict)

    # replace tk in checkpoint_dict.keys()
    checkpoint_dict = replace_tk_to_mindpet(checkpoint_dict)
    if hasattr(network, 'llm_boost'):
        network.llm_boost.set_weights(checkpoint_dict)
        return

    if "global_step" in checkpoint_dict and config.runner_config.step_scale is not None:
        resume_global_step = int(checkpoint_dict["global_step"].data * config.runner_config.step_scale)
        logger.info("Set global_step from %d to: %d", \
                    int(checkpoint_dict["global_step"]), resume_global_step)
        checkpoint_dict["global_step"] = Parameter([resume_global_step])

    # pylint: disable=W0212
    if config.remove_redundancy:
        not_load_network_params = load_param_into_net(model._train_network, checkpoint_dict, remove_redundancy=True)
        logger.info("Network parameters are not loaded: %s", not_load_network_params)
    else:
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        logger.info("Network parameters are not loaded: %s", not_load_network_params)
        if optimizer:
            not_load_optim_params = load_param_into_net(optimizer, checkpoint_dict)
            logger.info("Optimizer parameters are not loaded: %s", not_load_optim_params)


def get_last_checkpoint(checkpoint_dir, ckpt_format='ckpt'):
    """get last checkpoint for resuming or finetune."""
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"{checkpoint_dir} is not a real directory,"
            f"When distributed loads are sliced weights,"
            f"load_checkpoint should be a checkpoint directory containing the directory of rank_{{0-*}},"
            f"The directory structure is as follows: **checkpoint_root_dir/rank_{{0-*}}/**.{ckpt_format}")
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith(f'.{ckpt_format}')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])
