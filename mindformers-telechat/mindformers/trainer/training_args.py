# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
# ============================================================================
"""Default Training Arguments for Trainer."""

import os
import math
import json
from enum import Enum
from collections import OrderedDict
from typing import Optional, Union, List
from dataclasses import asdict, dataclass, field, fields

from mindformers.modules.transformer import TransformerRecomputeConfig, TransformerOpParallelConfig, MoEConfig
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank, get_real_group_size
from .utils import (
    LrSchedulerType,
    OptimizerType,
    IntervalStrategy,
    SaveIntervalStrategy,
    LoggingIntervalStrategy,
    HubStrategy,
)


def _check_task_config(check_config):
    """check task config for adapting hugging-face."""
    if check_config is not None and isinstance(check_config, MindFormerConfig):
        return True
    return False


def _check_training_args(ori_value, new_value):
    """check training arguments for adapt MindFormers."""
    if new_value is not None:
        return new_value
    return ori_value


def _adapt_dict_args(dic, key, value):
    """check arguments for adapt dict."""
    dic[key] = _check_training_args(dic[key], value) if key in dic else value


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use
    in our default config which is related to the training in MindSpore.

    Args:
        output_dir (str, optional):
            The output directory where checkpoints and log will be written. Default: ``'./output'`` .
        overwrite_output_dir (bool, optional):
            Overwrite the content of the output directory. Use this to continue training if output_dir points to a
            checkpoint directory. Default: ``False`` .
        seed (int, optional): Random seed that will be set at the beginning of training. Default: ``42`` .
        data_seed (int, optional):
            Random seed to be used with data samplers. Default: ``None`` .
        only_save_strategy (bool, optional):
            If `True`, the task will only save the strategy file in `output_dir/strategy`.
            Only takes effect when the use_parallel is True. Default: ``False`` .
        auto_trans_ckpt (bool, optional):
            Whether to transform checkpoint according to parallel config. Default: ``False`` .
        src_strategy (str, optional):
            The strategy file used for transforming checkpoint when auto_trans_ckpt is True. Default: ``None`` .
        transform_process_num (int, optional):
            The number of processes responsible for checkpoint transform. Default: ``1`` .
        resume_from_checkpoint (Union[str, bool], optional):
            The path to a folder with a valid checkpoint for your model. Default: ``None`` .
        resume_training (Union[bool, str], optional):
            Decide whether to resume training or specify the name of the checkpoint from which to resume training.
            Default: ``None`` .
        ignore_data_skip (bool, optional):
            When resuming training, whether or not to skip the first epochs and batches to get to the same
            training data. Default: ``False`` .
        data_skip_steps (int, optional):
            Specify the skip steps of train dataset when resume training.
            It only takes effect when `ignore_data_skip` is set to False. Default: ``None`` .
        do_train (bool, optional):
            Whether to run training. Default: ``False`` .
        do_eval (bool, optional):
            Whether to run eval on the dev set. Default: ``False`` .
        do_predict (bool, optional):
            Whether to run predictions on the test set. Default: ``False`` .
        check_for_nan_in_loss_and_grad (bool, optional):
            Whether to check for nan in loss and grad during training. Default: ``False`` .
        calculate_per_token_loss (bool, optional):
            Whether to use megatron-style loss for training. Default: ``False`` .
        remote_save_url (str, optional):
            The OBS output dir when training on ModeArts. Default: ``None`` .
        batch_size (int, optional):
            Batch size per GPU/NPU core/CPU for training.
            If set, it will override `per_device_train_batch_size`. Default: ``None`` .
        num_train_epochs (float, optional):
            Total number of training epochs to perform. Default: ``3.0`` .
        sink_mode (bool, optional):
            Whether to directly sink data to the Device through a channel. Default: ``True`` .
        sink_size (int, optional):
            The data sink number per step for training or evaluation. Default: ``2`` .
        gradient_accumulation_steps (int, optional):
            Number of steps to accumulate before performing a backward pass. Default: ``1`` .
        mode (int, optional):
            Indicates running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: ``0`` .
        use_cpu (bool, optional):
            Whether or not to use cpu. Default: ``False`` .
        device_id (int, optional):
            The default device id for execution. Default: ``0`` .
        device_target (str, optional):
            The target device for execution, supporting 'Ascend', 'GPU', and 'CPU'. Default: ``'Ascend'`` .
        enable_graph_kernel (bool, optional):
            Whether to enable graph fusion. Default: ``False`` .
        max_call_depth (int, optional):
            Maximum depth of function calls. Default: ``10000`` .
        max_device_memory (str, optional):
            Maximum available memory of the device. The actual memory size used
            is the minimum of the device's available memory and `max_device_memory`.. Default: ``'1024GB'`` .
        save_graphs (bool, optional):
            Whether to save intermediate compilation graphs. Default: ``False`` .
        save_graphs_path (str, optional):
            Path to save intermediate compilation graphs. Default: ``'./graph'`` .
        use_parallel (bool, optional):
            Whether enable distribute parallel of the network. Default: ``False`` .
        parallel_mode (int, optional):
            Indicates running with Data Parallel(0) or Semi-Auto Parallel(1) or
            Auto Parallel(2) or Hybrid Parallel(3). Default: ``1`` .
        gradients_mean (bool, optional):
            Whether to perform the averaging operator after gradient AllReduce
            Usually, it's set to False in semi-automatic parallel mode and True in data parallel mode.
            Default: ``False`` .
        loss_repeated_mean (bool, optional):
            Calculation is repeated. Default: ``False`` .
        enable_alltoall (bool, optional):
            Whether allow generation of AllToAll communication operators during communication.
            Typically only turned on in MOE scenarios. Default: ``False`` .
        full_batch (bool, optional):
            If the entire batch dataset is loaded in auto_parallel mode, then full_batch should be set to True.
            It is currently not recommended to use this interface, please replace it with dataset_strategy.
            Default: ``True`` .
        dataset_strategy (Union[str, tuple], optional):
            Dataset sharding strategy. Semi-auto parallel mode is usually set to 'full_batch'
            while data parallel mode must be set to 'data_parallel'.
            Default: ``'full_batch'`` .
        search_mode (str, optional):
            Strategy search mode, Only effective in Auto Parallel mode, experimental interface, use with caution.
            Default: ``'sharding_propagation'`` .
        enable_parallel_optimizer (bool, optional):
            Whether enable optimizer parallel. Default: ``False`` .
        gradient_accumulation_shard (bool, optional):
            Whether the accumulated gradient variable is split along the data parallel dimension.
            It will further reduce the memory usage of model, but will introduce additional communication
            operators (ReduceScatter) during the backward gradient calculation. It is only effective in
            pipeline parallel training and gradient accumulation mode.
            Default: ``False`` .
        parallel_optimizer_threshold (int, optional):
            Set the threshold for parameter splitting. Default: ``64`` .
        optimizer_weight_shard_size (int, optional):
            Set the size of the communication domain for the specified optimizer weight splitting.
            Effective only when optimizer parallelism is enabled. The numerical range can be (0, device_num],
            and if pipeline parallelism is also enabled, the range becomes (0, device_num/stage].
            If the data parallel communication domain size ofa parameter is not divisible by
            optimizer_weight_shard_size, then the specified optimizer weight splittingcommunication domain
            size will not be effective. Default: -1, which means the optimizer weight slicecommunication
            domain size is the data parallel communication domain size of each parameter.
            Default: ``-1`` .
        strategy_ckpt_save_file (str, optional):
            Path for saving distributed strategy file. Default: ``'./ckpt_strategy.ckpt'`` .
        data_parallel (int, optional):
            The split number of data parallel. Default: ``1`` .
        model_parallel (int, optional):
            The split number of model parallel. Default: ``1`` .
        expert_parallel (int, optional):
            The split number of expert parallel. Default: ``1`` .
        pipeline_stage (int, optional):
            The number of pipeline stage. Default: ``1`` .
        micro_batch_num (int, optional):
            The number of micro batch num. Only takes effect when `pipeline_stage` > 1. Default: ``1`` .
        gradient_aggregation_group (int, optional):
            The size of the gradient communication operator fusion group. Default: ``4`` .
        micro_batch_interleave_num (int, optional):
            Enable multi-replica parallel when `micro_batch_interleave_num` > 1, it is recommended set to 2
            in model parallel. It is used for optimizing communication overhead incurred during
            model_parallel execution. However, it will incur additional memory overhead.
            It is not recommended for use in pure pipeline parallel.
            Default: ``1`` .
        use_seq_parallel (bool, optional):
            Whether enable seq parallel. Default: ``False`` .
        vocab_emb_dp (bool, optional):
            Whether to split the vocabulary only along the dp dimension. Default: ``True`` .
        expert_num (int, optional):
            The number of expert. Default: ``1`` .
        capacity_factor (float, optional):
            Expertise factor. Default: ``1.05`` .
        aux_loss_factor (float, optional):
            Loss contribution factor. Default: ``0.05`` .
        num_experts_chosen (int, optional):
            Number of experts selected for each token. Default: ``1`` .
        recompute (bool, optional):
            Whether enable recompute mode. Default: ``False`` .
        select_recompute (bool, optional):
            select recompute. Default: ``False`` .
        parallel_optimizer_comm_recompute (bool, optional):
            Whether to recompute the AllGather communication introduced by optimizer parallel. Default: ``False`` .
        mp_comm_recompute (bool, optional):
            Whether to recompute the communication operations introduced by model parallel. Default: ``True`` .
        recompute_slice_activation (bool, optional):
            Whether to slice the Cell outputs retained in memory. Default: ``False`` .
        optim (Union[OptimizerType, str], optional):
            The optimizer type to use. Default: ``'fp32_adamw'`` .
        adam_beta1 (float, optional):
             Beta1 for AdamW optimizer. Default: ``0.9`` .
        adam_beta2 (float, optional):
            Beta2 for AdamW optimizer. Default: ``0.999`` .
        adam_epsilon (float, optional):
            Epsilon for AdamW optimizer. Default: ``1.e-8`` .
        weight_decay (float, optional):
            Weight decay for AdamW if we apply some. Default: ``0.0`` .
        layer_scale (bool, optional):
            Whether to enable layer decay. Default: ``False`` .
        layer_decay (float, optional):
            Layer decay coefficient. Default: ``0.65`` .
        lr_scheduler_type (Union[LrSchedulerType, str], optional):
            The scheduler type to use. Default: ``'cosine'`` .
        learning_rate (float, optional):
            The initial learning rate. Default: ``5.e-5`` .
        lr_end (float, optional):
            The end learning rate. Default: ``1.e-6`` .
        warmup_lr_init (float, optional):
            The initial learning rate of warm up. Default: ``0.0`` .
        warmup_epochs (int, optional):
            Linear warmup over warmup_epochs fraction of total steps. Default: ``None`` .
        warmup_ratio (float, optional):
            Linear warmup over warmup_ratio fraction of total steps. Default: ``None`` .
        warmup_steps (int, optional):
            Linear warmup over warmup_steps. Default: ``0`` .
        total_steps (int, optional):
            Total number of steps used for calculating the learning rate,
            -1 means it will load the total steps of the dataset.
            Default: ``-1`` .
        lr_scale (bool, optional):
            Whether to enable learning rate scaling. Default: ``False`` .
        lr_scale_factor (int, optional):
            Learning rate scaling factor. Default: ``256`` .
        dataset_task (str, optional):
            Dataset task name. Default: ``None`` .
        dataset_type (str, optional):
            Train dataset type. Default: ``None`` .
        train_dataset (str, optional):
            Train dataset path. Default: ``None`` .
        train_dataset_in_columns (List[str], optional):
            Train dataset input column names. Default: ``None`` .
        train_dataset_out_columns (List[str], optional):
            Train dataset output column names. Default: ``None`` .
        eval_dataset (str, optional):
            Eval dataset dir. Default: ``None`` .
        eval_dataset_in_columns (List[str], optional):
            Eval data column names. Default: ``None`` .
        eval_dataset_out_columns (List[str], optional):
            Eval dataset output column names. Default: ``None`` .
        shuffle (bool, optional):
            Whether shuffle train dataset. Default: ``True`` .
        dataloader_drop_last (bool, optional):
            Drop the last incomplete batch if it is not divisible by the batch size. Default: ``True`` .
        repeat (int, optional):
            Repeat train dataset count times. If count is None or -1, iterate infinitely. Default: ``1`` .
        per_device_train_batch_size (int, optional):
            Batch size per GPU/NPU core/CPU for training. Default: ``8`` .
        per_device_eval_batch_size (int, optional):
            Batch size per GPU/NPU core/CPU for evaluation. Default: ``8`` .
        dataloader_num_workers (int, optional):
            Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            Default: ``8`` .
        python_multiprocessing (bool, optional):
            Whether to start Python multiprocessing mode to execute per_batch_map in parallel,
            where 'True' indicates Python multiprocessing mode, and 'False' indicates Python multithreading mode.
            Default: ``False`` .
        numa_enable (bool, optional):
            Set the default state of NUMA to the enabled state. Default: ``False`` .
        prefetch_size (int, optional):
            Set the queue capacity of threads in the pipeline. A larger prefetch_size can reduce the overall
            processing latency when there is an imbalance in the throughput rate of adjacent operations,
            but it also consumes more system memory.
            Default: ``1`` .
        wrapper_type (str, optional):
            Class name of wrapper. Default: ``'MFTrainOneStepCell'`` .
        scale_sense (Union[str, float], optional):
            Value or Class name of scale sense. Default: ``'DynamicLossScaleUpdateCell'`` .
        loss_scale_value (int, optional):
            Initial loss scaling factor. Default: ``65536`` .
        loss_scale_factor (int, optional):
            Increment and decrement factor for loss scaling coefficient. Default: ``2`` .
        loss_scale_window (int, optional):
            Maximum consecutive training steps to increase the loss scaling coefficient when there is no overflow.
            Default: ``1000`` .
        use_clip_grad (bool, optional):
            Whether enable gradient clipping. Default: ``True`` .
        max_grad_norm (float, optional):
            Max gradient norm. Default: ``1.0`` .
        max_scale_window (int, optional):
            Maximum scale_window of the automatic scale window list. The default value is 1000. Default: ``1000`` .
        min_scale_window (int, optional):
            Minimum scale_window of the automatic scale window list. The default value is 20. Default: ``20`` .
        metric_type (Union[List[str], str], optional):
            Whether enable gradient clipping. Default: ``None`` .
        logging_strategy (Union[LoggingIntervalStrategy, str], optional):
            The logging strategy to use. Default: ``'steps'`` .
        logging_steps (int, optional):
            Log every X updates steps. Should be an integer or a float in range `[0,1)`.
            If smaller than 1, will be interpreted as ratio of total training steps. Default: ``1`` .
        save_prefix (str, optional):
            The prefix name of checkpoint files. Default: ``'CKP'`` .
        save_directory (str, optional):
            The path of the folder which will be saved in the checkpoint file. Default: ``None`` .
        save_strategy (Union[SaveIntervalStrategy, str], optional):
            The checkpoint save strategy to use. Default: ``'steps'`` .
        save_steps (int, optional):
            Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`.
            If smaller than 1, will be interpreted as ratio of total training steps. Default: ``500`` .
        save_seconds (int, optional):
            Save checkpoint every X updates seconds. Default: ``None`` .
        save_total_limit (int, optional):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to
            `metric_for_best_model` will always be retained in addition to the most recent ones. For example,
            for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be
            retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,
            it is possible that two checkpoints are saved: the last one and the best one (if they are different).
            Default is unlimited checkpoints. Default: ``5`` .
        keep_checkpoint_per_n_minutes (int, optional):
            Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes.
            Can't be used with keep_checkpoint_max at the same time. Default: ``0`` .
        save_on_each_node (bool, optional):
            When doing multi-node distributed training, whether to save models and checkpoints on each node,
            or only on the main one. Default: ``True`` .
        integrated_save (bool, optional):
            Whether to merge and save the split Tensor in the automatic parallel scenario. Integrated save function
            is only supported in automatic parallel scene, not supported in manual parallel. If set,
            `save_on_each_node` will become invalid. Default: ``None`` .
        save_network_params (bool, optional):
            Whether to only save network weights additionally. Default: ``True`` .
        save_trainable_params (bool, optional):
            Whether to save fine-tuned weights additionally. Default: ``False`` .
        async_save (bool, optional):
            Whether asynchronous execution saves the checkpoint to a file. Default: ``False`` .
        evaluation_strategy (Union[IntervalStrategy, str], optional):
            The evaluation strategy to use. Default: ``'no'`` .
        eval_steps (float, optional):
            Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`.
            If smaller than 1, will be interpreted as ratio of total training steps. Default: ``None`` .
        eval_epochs (int, optional):
            Num of epoch intervals between each eval, 1 means eval on every epoch end. Default: ``None`` .
        profile (bool, optional):
            Whether to enable the profile performance analysis tool. Default: ``False`` .
        profile_start_step (int, optional):
            Start step for performance analysis. Default: ``1`` .
        profile_end_step (int, optional):
            End step for performance analysis. Default: ``10`` .
        init_start_profile (bool, optional):
            Whether to enable data collection at the time of Profiler initialization. Once enabled, profile_start_step
            will not be effective. It must be enabled if multi-device communication data needs to be collected.
            Default: ``False`` .
        profile_communication (bool, optional):
            Whether to collect communication performance data in multi-device training. Default: ``False`` .
        profile_memory (bool, optional):
            Whether to collect Tensor memory data. Default: ``True`` .
        auto_tune (bool, optional):
            Whether to enable automatic data acceleration. Default: ``False`` .
        filepath_prefix (str, optional):
            The save path and file prefix for the optimized global configuration. Default: ``'./autotune'`` .
        autotune_per_step (int, optional):
            Set the step interval for adjusting the configuration of automatic data acceleration. Default: ``10`` .
        push_to_hub (bool, optional):
            Whether or not to upload the trained model to the model hub after training. Default: ``False`` .
        hub_model_id (str, optional):
            The name of the repository to keep in sync with the local `output_dir`. Default: ``None`` .
        hub_strategy (Union[HubStrategy, str], optional):
            The hub strategy to use when `--push_to_hub` is activated. Default: ``'every_save'`` .
        hub_token (str, optional):
            The token to use to push to the Model Hub. Default: ``None`` .
        hub_private_repo (bool, optional):
            Whether the model repository is private or not. Default: ``False`` .
        hub_always_push (bool, optional):
            Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet. Default: ``False`` .

    Returns:
        An instance of TrainingArguments.

    Examples:
        >>> from mindformers import TrainingArguments
        >>> args = TrainingArguments(output_dir="output", seed=10)
        >>> args.output_dir
        'output'
        >>> args.seed
        10
    """
    # common config
    framework = "ms"
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where checkpoints and log will be written. Default: `./output`."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory. Default: False."
            )
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training. Default: 42."}
    )
    data_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed to be used with data samplers."}
    )
    only_save_strategy: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, the task will only save the strategy file in `output_dir/strategy`."
                "Only takes effect when the use_parallel is True. Default: False."
            )
        }
    )
    auto_trans_ckpt: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to transform checkpoint according to parallel config. See the [Transform_Ckpt documentation]"
                "(https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)"
                " for more details. Default: False."
            )
        }
    )
    src_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "The strategy file used for transforming checkpoint when auto_trans_ckpt is True"}
    )
    transform_process_num: int = field(
        default=1,
        metadata={"help": "The number of processes responsible for checkpoint transform."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    resume_training: Optional[Union[bool, str]] = field(
        default=None,
        metadata={
            "help": (
                "Decide whether to resume training or specify the name of the checkpoint "
                "from which to resume training."
            )
        },
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data. Default: False."
            )
        },
    )
    data_skip_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Specify the skip steps of train dataset when resume training."
                "It only takes effect when `ignore_data_skip` is set to False. Default: None."
            )
        }
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training. Default: False."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set. Default: False."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set. Default: False."}
    )
    check_for_nan_in_loss_and_grad: bool = field(
        default=False,
        metadata={"help": "Whether to check for nan in loss and grad during training. Default: False."}
    )
    calculate_per_token_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use megatron-style loss during training. Default: False."}
    )
    # AICC
    remote_save_url: Optional[str] = field(
        default=None,
        metadata={"help": "The OBS output dir when training on ModeArts."}
    )

    # runner config
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Batch size per GPU/NPU core/CPU for training."
                "If set, it will override `per_device_train_batch_size`. Default: None."
            )
        }
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform. Default: 3.0."}
    )
    sink_mode: bool = field(
        default=True,
        metadata={"help": "Whether to directly sink data to the Device through a channel. Default: True."}
    )
    sink_size: int = field(
        default=2,
        metadata={"help": "The data sink number per step for training or evaluation. Default: 2."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate before performing a backward pass. Default: 1."},
    )

    # context config
    mode: int = field(
        default=0,
        metadata={"help": "Indicates running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: 0."}
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cpu. Default: False."
        },
    )
    device_id: int = field(
        default=0,
        metadata={"help": "The default device id for execution. Default: 0."}
    )
    device_target: str = field(
        default="Ascend",
        metadata={"help": "The target device for execution, supporting 'Ascend', 'GPU', and 'CPU'. Default: 'Ascend'."}
    )
    enable_graph_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to enable graph fusion. Default: False."}
    )
    max_call_depth: int = field(
        default=10000,
        metadata={"help": "Maximum depth of function calls. Default: 10000."}
    )
    max_device_memory: str = field(
        default="1024GB",
        metadata={
            "help": (
                "Maximum available memory of the device. The actual memory size used"
                " is the minimum of the device's available memory and `max_device_memory`."
                "Default: '1024GB'."
            )
        }
    )
    save_graphs: bool = field(
        default=False,
        metadata={"help": "Whether to save intermediate compilation graphs. Default: False."}
    )
    save_graphs_path: str = field(
        default="./graph",
        metadata={"help": "Path to save intermediate compilation graphs. Default: './graph'."}
    )

    device_num = get_real_group_size()
    device_id = int(os.getenv("DEVICE_ID", "0"))
    rank_id = get_real_rank()

    # parallel config
    use_parallel: bool = field(
        default=False,
        metadata={"help": "Whether enable distribute parallel of the network. Default: False."}
    )
    parallel_mode: int = field(
        default=1,
        metadata={
            "help": (
                "Indicates running with Data Parallel(0) or Semi-Auto Parallel(1) or"
                "Auto Parallel(2) or Hybrid Parallel(3). Default: 1."
            )
        }
    )
    gradients_mean: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to perform the averaging operator after gradient AllReduce. "
                "Usually, it's set to False in semi-automatic parallel mode and True "
                "in data parallel mode. Default: False."
            )
        }
    )
    loss_repeated_mean: bool = field(
        default=False,
        metadata={
            "help": (
                "Calculation is repeated. Default: False."
            )
        }
    )
    enable_alltoall: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether allow generation of AllToAll communication operators during communication."
                "Typically only turned on in MOE scenarios, default is False."
            )
        }
    )
    full_batch: bool = field(
        default=True,
        metadata={
            "help": (
                "If the entire batch dataset is loaded in auto_parallel mode, then full_batch should be set to True."
                "It is currently not recommended to use this interface, please replace it with dataset_strategy."
                "Default: True. "
            )
        }
    )
    dataset_strategy: Union[str, tuple] = field(
        default='full_batch',
        metadata={
            "help": (
                "Dataset sharding strategy. Semi-auto parallel mode is usually set to 'full_batch',"
                "while data parallel mode must be set to 'data_parallel'. Default: 'full_batch'."
            ),
            "choices": ['data_parallel', 'full_batch']
        }
    )
    search_mode: str = field(
        default='sharding_propagation',
        metadata={
            "help": (
                "Strategy search mode, Only effective in Auto Parallel mode, experimental interface, use with caution."
                "Default: sharding_propagation."
            ),
            "choices": ['recursive_programming', 'dynamic_programming', 'sharding_propagation']
        }
    )
    enable_parallel_optimizer: bool = field(
        default=False,
        metadata={"help": "Whether enable optimizer parallel. Default: True."}
    )
    gradient_accumulation_shard: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether the accumulated gradient variable is split along the data parallel dimension."
                "It will further reduce the memory usage of model, but will introduce additional communication"
                " operators (ReduceScatter) during the backward gradient calculation. It is only effective in"
                " pipeline parallel training and gradient accumulation mode. Default: False."
            )
        }
    )
    parallel_optimizer_threshold: int = field(
        default=64,
        metadata={"help": "Set the threshold for parameter splitting. Default: 64"}
    )
    optimizer_weight_shard_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Set the size of the communication domain for the specified optimizer weight splitting."
                "Effective only when optimizer parallelism is enabled. The numerical range can be (0, device_num],"
                " and if pipeline parallelism is also enabled, the range becomes (0, device_num/stage]."
                " If the data parallel communication domain size ofa parameter is not divisible by"
                " optimizer_weight_shard_size, then the specified optimizer weight splittingcommunication domain"
                " size will not be effective. Default: -1, which means the optimizer weight slicecommunication"
                " domain size is the data parallel communication domain size of each parameter. Default: -1."
            )
        }
    )
    strategy_ckpt_save_file: str = field(
        default="./ckpt_strategy.ckpt",
        metadata={"help": "Path for saving distributed strategy file. Default: './ckpt_strategy.ckpt'."}
    )
    data_parallel: int = field(
        default=1,
        metadata={"help": "The split number of data parallel. Default: 1."}
    )
    model_parallel: int = field(
        default=1,
        metadata={"help": "The split number of model parallel. Default: 1."}
    )
    expert_parallel: int = field(
        default=1,
        metadata={"help": "The split number of expert parallel. Default: 1."}
    )
    pipeline_stage: int = field(
        default=1,
        metadata={"help": "The number of pipeline stage. Default: 1."}
    )
    micro_batch_num: int = field(
        default=1,
        metadata={"help": "The number of micro batch num. Only takes effect when `pipeline_stage` > 1. Default: 1."}
    )
    gradient_aggregation_group: int = field(
        default=4,
        metadata={"help": "The size of the gradient communication operator fusion group. Default: 4."}
    )
    micro_batch_interleave_num: int = field(
        default=1,
        metadata={
            "help": (
                "Enable multi-replica parallel when `micro_batch_interleave_num` > 1, it is recommended set to 2"
                " in model parallel. It is used for optimizing communication overhead incurred during"
                " model_parallel execution. However, it will incur additional memory overhead."
                "It is not recommended for use in pure pipeline parallel. Default: 1."
            )
        }
    )
    use_seq_parallel: bool = field(
        default=False,
        metadata={"help": "Whether enable seq parallel. Default: False."}
    )
    vocab_emb_dp: bool = field(
        default=True,
        metadata={"help": "Whether to split the vocabulary only along the dp dimension. Default: True."}
    )

    # moe config
    expert_num: int = field(
        default=1,
        metadata={"help": "The number of expert. Default: 1."}
    )
    capacity_factor: float = field(
        default=1.05,
        metadata={"help": "Expertise factor. Default: 1.05."}
    )
    aux_loss_factor: float = field(
        default=0.05,
        metadata={"help": "Loss contribution factor. Default: 0.05."}
    )
    num_experts_chosen: int = field(
        default=1,
        metadata={"help": "Number of experts selected for each token. Default: 1."}
    )

    # recompute config
    recompute: bool = field(
        default=False,
        metadata={"help": "Whether enable recompute mode. Default: False."}
    )
    select_recompute: bool = field(
        default=False,
        metadata={"help": "select recompute. Default: False."}
    )
    parallel_optimizer_comm_recompute: bool = field(
        default=False,
        metadata={"help": "Whether to recompute the AllGather communication introduced by optimizer parallel. \
                           Default: False."}
    )
    mp_comm_recompute: bool = field(
        default=True,
        metadata={"help": "Whether to recompute the communication operations introduced by model parallel. \
                          Default: True."}
    )
    recompute_slice_activation: bool = field(
        default=False,
        metadata={"help": "Whether to slice the Cell outputs retained in memory. Default: False."}
    )

    # optimizer config
    default_optim = "fp32_adamw"
    optim: Union[OptimizerType, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer type to use."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1.e-8,
        metadata={"help": "Epsilon for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    layer_scale: bool = field(
        default=False,
        metadata={"help": "Whether to enable layer decay. Default False."}
    )
    layer_decay: float = field(
        default=0.65,
        metadata={"help": "Layer decay coefficient."}
    )

    # scheduler config
    lr_scheduler_type: Union[LrSchedulerType, str] = field(
        default='cosine',
        metadata={"help": "The scheduler type to use. Default: cosine."},
    )
    learning_rate: float = field(
        default=5.e-5,
        metadata={"help": "The initial learning rate. Default: 5e-5."}
    )
    lr_end: float = field(
        default=1.e-6,
        metadata={"help": "The end learning rate. Default: 1e-6."}
    )
    warmup_lr_init: float = field(
        default=0.0,
        metadata={"help": "The initial learning rate of warm up. Default: 0.0."}
    )
    warmup_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Linear warmup over warmup_epochs fraction of total steps."}
    )
    warmup_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    total_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Total number of steps used for calculating the learning rate, "
                "-1 means it will load the total steps of the dataset."
            )
        },
    )
    lr_scale: bool = field(
        default=False,
        metadata={"help": "Whether to enable learning rate scaling."}
    )
    lr_scale_factor: int = field(
        default=256,
        metadata={"help": "Learning rate scaling factor."}
    )

    # dataset config
    dataset_task: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset task name. Default: CausalLanguageModelDataset."}
    )
    dataset_type: Optional[str] = field(
        default=None,
        metadata={"help": "Train dataset type. Default: MindDataset."}
    )
    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Train dataset path."}
    )
    train_dataset_in_columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Train dataset input column names. Default: ['input_ids']."}
    )
    train_dataset_out_columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Train dataset output column names. Default: None."}
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Eval dataset dir."}
    )
    eval_dataset_in_columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Eval data column names. Default: ['input_ids']."}
    )
    eval_dataset_out_columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Eval dataset output column names. Default: None."}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether shuffle train dataset. Default: True"}
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size. Default: False"}
    )
    repeat: int = field(
        default=1,
        metadata={"help": "Repeat train dataset count times. If count is None or -1, iterate infinitely. Default: 1."}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/NPU core/CPU for training. Default: 8."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/NPU core/CPU for evaluation. Default: 8."}
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded"
                " in the main process. Default: 8."
            )
        },
    )
    python_multiprocessing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to start Python multiprocessing mode to execute per_batch_map in parallel,"
                " where 'True' indicates Python multiprocessing mode, and 'False' indicates Python multithreading mode."
                "Default: False."
            )
        }
    )
    numa_enable: bool = field(
        default=False,
        metadata={"help": "Set the default state of NUMA to the enabled state. Default: Fasle."}
    )
    prefetch_size: int = field(
        default=1,
        metadata={
            "help": (
                "Set the queue capacity of threads in the pipeline. A larger prefetch_size can reduce the overall"
                " processing latency when there is an imbalance in the throughput rate of adjacent operations,"
                " but it also consumes more system memory. Default: 1."
            )
        }
    )

    # wrapper config
    wrapper_type: str = field(
        default='MFTrainOneStepCell',
        metadata={"help": "Class name of wrapper. Default: MFTrainOneStepCell."}
    )
    scale_sense: Union[str, float] = field(
        default='DynamicLossScaleUpdateCell',
        metadata={"help": "Value or Class name of scale sense. Default: DynamicLossScaleUpdateCell."}
    )
    loss_scale_value: int = field(
        default=65536,
        metadata={"help": "Initial loss scaling factor. Default: 65536."}
    )
    loss_scale_factor: int = field(
        default=2,
        metadata={"help": "Increment and decrement factor for loss scaling coefficient. Default: 2."}
    )
    loss_scale_window: int = field(
        default=1000,
        metadata={
            "help": (
                "Maximum consecutive training steps to increase the loss scaling coefficient "
                "when there is no overflow. Default: 1000."
            )
        }
    )
    use_clip_grad: bool = field(
        default=True,
        metadata={"help": "Whether enable gradient clipping. Default: False."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm."}
    )
    max_scale_window: int = field(
        default=1000,
        metadata={"help": "Maximum scale_window of the automatic scale window list. The default value is 1000."}
    )
    min_scale_window: int = field(
        default=20,
        metadata={"help": "Minimum scale_window of the automatic scale window list. The default value is 20."}
    )

    # metric config
    metric_type: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Whether enable gradient clipping. Default: False."}
    )

    # callback config
    # logging
    logging_strategy: Union[LoggingIntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    # checkpoint
    save_prefix: str = field(
        default='CKP',
        metadata={"help": "The prefix name of checkpoint files. Default: 'CKP'."}
    )
    save_directory: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the folder which will be saved in the checkpoint file. Default: None."}
    )
    save_strategy: Union[SaveIntervalStrategy, str] = field(
        default='steps',
        metadata={"help": "The checkpoint save strategy to use. Default: 'steps'."},
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps. Default: 500."
            )
        },
    )
    save_seconds: Optional[int] = field(
        default=None,
        metadata={"help": "Save checkpoint every X updates seconds."}
    )
    save_total_limit: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints. Default: 5."
            )
        },
    )
    keep_checkpoint_per_n_minutes: int = field(
        default=0,
        metadata={
            "help": (
                "Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes."
                "Can't be used with keep_checkpoint_max at the same time. Default: 0."
            )
        },
    )
    save_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one. Default: True."
            )
        },
    )
    integrated_save: bool = field(
        default=None, metadata={
            "help": (
                "Whether to merge and save the split Tensor in the automatic parallel scenario. "
                "Integrated save function is only supported in automatic parallel scene, not supported"
                "in manual parallel. If set, `save_on_each_node` will become invalid. Default: None."
            )
        }
    )
    save_network_params: bool = field(
        default=True,
        metadata={"help": "Whether to only save network weights additionally. Default: True."}
    )
    save_trainable_params: bool = field(
        default=False,
        metadata={"help": "Whether to save fine-tuned weights additionally. Default: False."}
    )
    async_save: bool = field(
        default=False,
        metadata={"help": "Whether asynchronous execution saves the checkpoint to a file. Default: False."}
    )
    # evaluate
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use. Default: 'no'."},
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    eval_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Num of epoch intervals between each eval, 1 means eval on every epoch end. "}
    )

    # profile config
    profile: bool = field(
        default=False,
        metadata={"help": "Whether to enable the profile performance analysis tool. Default: False."}
    )
    profile_start_step: int = field(
        default=1,
        metadata={"help": "Start step for performance analysis. Default: 1."}
    )
    profile_end_step: int = field(
        default=10,
        metadata={"help": "End step for performance analysis. Default: 10."}
    )
    init_start_profile: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable data collection at the time of Profiler initialization."
                "Once enabled, profile_start_step will not be effective. It must be enabled "
                "if multi-device communication data needs to be collected. Default: False."
            )
        }
    )
    profile_communication: bool = field(
        default=False,
        metadata={"help": "Whether to collect communication performance data in multi-device training. \
                          Default: False."}
    )
    profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether to collect Tensor memory data. Default: True."}
    )

    # auto tune config
    auto_tune: bool = field(
        default=False,
        metadata={"help": "Whether to enable automatic data acceleration. Default: False"}
    )
    filepath_prefix: str = field(
        default='./autotune',
        metadata={"help": "The save path and file prefix for the optimized global configuration. \
                           Default: './autotune'."}
    )
    autotune_per_step: int = field(
        default=10,
        metadata={"help": "Set the step interval for adjusting the configuration of automatic data acceleration. \
                           Default: 10."}
    )

    # hub config
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether the model repository is private or not."})
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )


    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.lr_scheduler_type is not None:
            self.lr_scheduler_type = LrSchedulerType(self.lr_scheduler_type).value
        if self.optim is not None:
            self.optim = OptimizerType(self.optim).value
        if self.logging_strategy is not None:
            self.logging_strategy = IntervalStrategy(self.logging_strategy).value
        if self.save_strategy is not None:
            self.save_strategy = SaveIntervalStrategy(self.save_strategy).value
        if self.evaluation_strategy is not None:
            self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy).value
        if self.hub_strategy is not None:
            self.hub_strategy = HubStrategy(self.hub_strategy).value

        self._check_rules()

    def _check_rules(self):
        """check rules"""
        self._check_strategy_rules()
        self._check_dataset_rules()
        self._check_metric_rules()

        if self.warmup_ratio is not None:
            if self.warmup_ratio < 0 or self.warmup_ratio > 1:
                raise ValueError(f"warmup_ratio must lie in range [0,1], but got {self.warmup_ratio}.")
            if self.warmup_ratio > 0 and self.warmup_steps > 0:
                logger.info(
                    "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                    " during training"
                )

    # pylint: disable=W0104
    def _check_strategy_rules(self):
        """check strategy rules"""
        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.EPOCH:
            logger.warning("--logging_strategy temporarily does not support epoch-level output, changing to `steps`.")

        if self.logging_strategy == IntervalStrategy.STEPS:
            self.logging_steps = self.check_step_rules(self.logging_steps, info="--logging_steps")

        if self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True
        if self.evaluation_strategy == IntervalStrategy.STEPS:
            if self.eval_steps is None or self.eval_steps == 0:
                if self.logging_steps <= 0:
                    raise ValueError(f"evaluation strategy {self.evaluation_strategy} requires"
                                     "either non-zero --eval_steps or --logging_steps.")
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}.")
                self.eval_steps = self.logging_steps
            self.eval_steps = self.check_step_rules(self.eval_steps, info="--eval_steps")
        if self.save_strategy == SaveIntervalStrategy.STEPS:
            self.save_steps = self.check_step_rules(self.save_steps, info="--save_steps")

    def _check_dataset_rules(self):
        """check dataset rules"""
        if self.batch_size is not None:
            self.per_device_train_batch_size = self.batch_size

        if self.train_dataset is not None or self.eval_dataset is not None:
            if self.dataset_task is None:
                raise ValueError(
                    "When `train_dataset` or `eval_dataset` is not None, `dataset_task` must not be None."
                    f"but found {self.dataset_task}."
                )
            if self.dataset_type is None:
                raise ValueError(
                    "When `train_dataset` or `eval_dataset` is not None, `dataset_type` must not be None."
                    f"but found {self.dataset_type}."
                )
            if self.train_dataset is not None and self.train_dataset_in_columns is None:
                raise ValueError(
                    "When `train_dataset` is not None, `train_dataset_in_columns` must not be None."
                    f"but found {self.train_dataset_in_columns}."
                )
            if self.eval_dataset is not None and self.eval_dataset_in_columns is None:
                raise ValueError(
                    "When `train_dataset` is not None, `eval_dataset_in_columns` must not be None."
                    f"but found {self.eval_dataset_in_columns}."
                )

    def _check_metric_rules(self):
        """check metric rules"""
        if self.metric_type is not None:
            self.metric_type = [self.metric_type] if isinstance(self.metric_type, str) else self.metric_type

    @staticmethod
    def check_step_rules(steps, info="steps"):
        if steps <= 0:
            raise ValueError(f"{info} must bigger than 0, but got {steps}.")
        if steps > 1:
            if steps != int(steps):
                raise ValueError(f"{info} must be an integer if bigger than 1, but got {steps}.")
            steps = int(steps)
        return steps

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        train_batch_size = self.per_device_train_batch_size * max(1, self.device_num)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        eval_batch_size = self.per_device_eval_batch_size * max(1, self.device_num)
        return eval_batch_size

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        return self.device_num

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        return self.rank_id

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        return self.device_id

    @property
    def get_device_num(self):
        """get device num for training."""
        return self.device_num

    @property
    def get_device_id(self):
        """get device id for training."""
        return self.device_id

    @property
    def get_rank_id(self):
        """get rank id for training."""
        return self.rank_id

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.

        Args:
            num_training_steps (int):
                The number of training steps.

        Returns:
            Number of training steps.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def get_recompute_config(self):
        """
        Get recompute config.

        Returns:
            An instance of TransformerRecomputeConfig.
        """
        recompute_config = TransformerRecomputeConfig(
            recompute=self.recompute,
            select_recompute=self.select_recompute,
            parallel_optimizer_comm_recompute=self.parallel_optimizer_comm_recompute,
            mp_comm_recompute=self.mp_comm_recompute,
            recompute_slice_activation=self.recompute_slice_activation
        )
        return recompute_config

    def get_parallel_config(self):
        """
        Get parallel config.

        Returns:
            An instance of TransformerOpParallelConfig.
        """
        parallel_config = TransformerOpParallelConfig(
            data_parallel=self.data_parallel,
            model_parallel=self.model_parallel,
            expert_parallel=self.expert_parallel,
            pipeline_stage=self.pipeline_stage,
            micro_batch_num=self.micro_batch_num,
            recompute=self.get_recompute_config(),
            use_seq_parallel=self.use_seq_parallel,
            gradient_aggregation_group=self.gradient_aggregation_group,
            vocab_emb_dp=self.vocab_emb_dp,
        )
        return parallel_config

    def get_moe_config(self):
        """
        Get moe config.

        Returns:
            An instance of MoEConfig.
        """
        moe_config = MoEConfig(
            expert_num=self.expert_num,
            capacity_factor=self.capacity_factor,
            aux_loss_factor=self.aux_loss_factor,
            num_experts_chosen=self.num_experts_chosen
        )
        return moe_config

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if v and isinstance(v, list) and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    # The following methods are there to simplify the instantiation of `TrainingArguments`
    def set_training(
            self,
            learning_rate: float = 5e-5,
            batch_size: int = 8,
            weight_decay: float = 0,
            num_epochs: float = 3.0,
            gradient_accumulation_steps: int = 1,
            seed: int = 42,
            **kwargs
    ):
        """
        A method that regroups all basic arguments linked to the training.

        <Tip>

        Calling this method will automatically set `self.do_train` to `True`.

        </Tip>

        Args:
            learning_rate (float, optional):
                The initial learning rate for the optimizer. Default: ``5e-5``.
            batch_size (int, optional):
                The batch size per device (GPU/NPU core/CPU...) used for training. Default: ``8``.
            weight_decay (float, optional):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
                optimizer. Default: ``0``.
            num_epochs(float, optional):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training). Default: ``3.0``.
            gradient_accumulation_steps (int, optional):
                Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
                examples. Default: ``1``.
            seed (int, optional):
                Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
                the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized
                parameters. Default: ``42``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
            >>> args.learning_rate
            1e-4
        """
        self.do_train = True
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_train_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.print_kwargs_unused(**kwargs)
        return self

    def set_evaluate(
            self,
            strategy: Union[str, IntervalStrategy] = "no",
            steps: int = 500,
            batch_size: int = 8,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the evaluation.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/NPU core/CPU...) used for evaluation.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_evaluate(strategy="steps", steps=100)
            >>> args.eval_steps
            100
        """
        self.evaluation_strategy = IntervalStrategy(strategy).value
        if self.evaluation_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.do_eval = self.evaluation_strategy != IntervalStrategy.NO
        self.eval_steps = steps
        self.per_device_eval_batch_size = batch_size
        self.print_kwargs_unused(**kwargs)
        return self

    def set_testing(
            self,
            batch_size: int = 8,
            loss_only: bool = False,
            **kwargs
    ):
        """
        A method that regroups all basic arguments linked to testing on a held-out dataset.

        <Tip>

        Calling this method will automatically set `self.do_predict` to `True`.

        </Tip>

        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_testing(batch_size=32)
            >>> args.per_device_eval_batch_size
            32
        """
        self.do_predict = True
        self.per_device_eval_batch_size = batch_size
        self.prediction_loss_only = loss_only
        self.print_kwargs_unused(**kwargs)
        return self

    def set_save(
            self,
            strategy: Union[str, IntervalStrategy] = "steps",
            steps: int = 500,
            total_limit: Optional[int] = None,
            on_each_node: bool = True,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the checkpoint saving.

        Args:
            strategy (Union[str, IntervalStrategy], optional):
                The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.

                Default: ``steps``.
            steps (int, optional):
                Number of updates steps before two checkpoint saves if `strategy="steps"`. Default: ``500``.
            total_limit (int, optional):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`. Default: ``None``.
            on_each_node (bool, optional):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.
                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node. Default: ``True``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_save(strategy="steps", steps=100)
            >>> args.save_steps
            100
        """
        self.save_strategy = IntervalStrategy(strategy).value
        if self.save_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.save_steps = steps
        self.save_total_limit = total_limit
        self.save_on_each_node = on_each_node
        self.print_kwargs_unused(**kwargs)
        return self

    def set_logging(
            self,
            strategy: Union[str, IntervalStrategy] = "steps",
            steps: int = 500,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the logging.

        Args:
            strategy (Union[str, IntervalStrategy], optional):
                The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: logging is done at the end of each epoch.
                - `"steps"`: logging is done every `save_steps`.

                Default: ``steps``.
            steps (int, optional):
                Number of update steps between two logs if `strategy="steps"`. Default: ``500``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_logging(strategy="steps", steps=100)
            >>> args.logging_steps
            100
        """
        self.logging_strategy = IntervalStrategy(strategy).value
        if self.logging_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.logging_steps = steps
        self.print_kwargs_unused(**kwargs)
        return self

    def set_push_to_hub(
            self,
            model_id: str,
            strategy: Union[str, HubStrategy] = "every_save",
            token: Optional[str] = None,
            private_repo: bool = False,
            always_push: bool = False,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

        <Tip>

        Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git
        directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is
        triggered (depending on `self.save_strategy`). Calling [`~Trainer.save_model`] will also trigger a push.

        </Tip>

        Args:
            model_id (`str`):
                The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
                which case the model will be pushed in your namespace. Otherwise it should be the whole repository
                name, for instance `"user_name/model"`, which allows you to push to an organization you are a member of
                with `"organization_name/model"`.
            strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
                Defines the scope of what is pushed to the Hub and when. Possible values are:

                - `"end"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and a
                  draft of a model card when the [`~Trainer.save_model`] method is called.
                - `"every_save"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`])
                  and
                  a draft of a model card each time there is a model save. The pushes are asynchronous to not block
                  training, and in case the save are very frequent, a new push is only attempted if the previous one is
                  finished. A last push is made with the final model at the end of training.
                - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
                  last-checkpoint, allowing you to resume training easily with
                  `trainer.train(resume_from_checkpoint="last-checkpoint")`.
                - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the
                  output
                  folder (so you will get one checkpoint folder per folder in your final repository)

            token (`str`, *optional*):
                The token to use to push the model to the Hub.
            private_repo (`bool`, *optional*, defaults to `False`):
                If True, the Hub repo will be set to private.
            always_push (`bool`, *optional*, defaults to `False`):
                Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not
                finished.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_push_to_hub("me/awesome-model")
            >>> args.hub_model_id
            'me/awesome-model'
        """
        self.push_to_hub = True
        self.hub_model_id = model_id
        self.hub_strategy = HubStrategy(strategy)
        self.hub_token = token
        self.hub_private_repo = private_repo
        self.hub_always_push = always_push
        self.print_kwargs_unused(**kwargs)
        return self

    def set_optimizer(
            self,
            name: Union[str, OptimizerType] = "adamw",
            learning_rate: float = 5e-5,
            lr_end: float = 1e-6,
            weight_decay: float = 0,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the optimizer and its hyperparameters.

        Args:
            name (Union[str, OptimizerType], optional):
                The optimizer to use: `"AdamWeightDecay"`, `"adamw"`, `"adam"`, `"sgd"`,
                `"adagrad"` or `"adafactor"`. Default: ``adamw``.
            learning_rate (float, optional):
                The initial learning rate. Default: ``5e-5``.
            lr_end (float, optional):
                The end learning rate for the optimizer. Default: ``1e-6``.
            weight_decay (float, optional):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
                Default: ``0``.
            beta1 (float, optional):
                The beta1 hyperparameter for the adam optimizer or its variants. Default: ``0.9``.
            beta2 (float, optional):
                The beta2 hyperparameter for the adam optimizer or its variants. Default: ``0.999``.
            epsilon (float, optional):
                The epsilon hyperparameter for the adam optimizer or its variants. Default: ``1e-8``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_optimizer(name="adamw", beta1=0.8)
            >>> args.optim
            'adamw'
        """
        self.optim = OptimizerType(name).value
        self.learning_rate = learning_rate
        self.lr_end = lr_end
        self.weight_decay = weight_decay
        self.adam_beta1 = beta1
        self.adam_beta2 = beta2
        self.adam_epsilon = epsilon
        self.print_kwargs_unused(**kwargs)
        return self

    def set_lr_scheduler(
            self,
            name: Union[str, LrSchedulerType] = "linear",
            num_epochs: float = 3.0,
            warmup_lr_init: float = 0.0,
            warmup_epochs: Optional[int] = None,
            warmup_ratio: Optional[float] = None,
            warmup_steps: int = 0,
            total_steps: int = -1,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

        Args:
            name (Union[str, LrSchedulerType], optional):
                The scheduler type to use. See the documentation of [`LrSchedulerType`] for all possible values.
                Default: ``linear``.
            num_epochs(float, optional):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training). Default: ``3.0``.
            warmup_lr_init (float, optional):
                The initial learning rate of warm up. Default: ``0.0``.
            warmup_epochs (int, optional):
                The number of epochs. Default: ``None``.
            warmup_ratio (float, optional):
                Ratio of total training steps used for a linear warmup from 0 to `learning_rate`. Default: ``None``.
            warmup_steps (int, optional):
                Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of
                `warmup_ratio`. Default: ``0``.
            total_steps (int, optional):
                Total number of steps used for calculating the learning rate, -1 means it will load
                the total steps of the dataset. Default: ``-1``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
            >>> args.warmup_ratio
            0.05
        """
        self.lr_scheduler_type = LrSchedulerType(name).value
        self.num_train_epochs = num_epochs
        self.warmup_lr_init = warmup_lr_init
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.print_kwargs_unused(**kwargs)
        return self

    def set_dataloader(
            self,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,
            drop_last: bool = False,
            num_workers: int = 0,
            ignore_data_skip: bool = False,
            data_skip_steps: Optional[int] = None,
            sampler_seed: Optional[int] = None,
            **kwargs
    ):
        """
        A method that regroups all arguments linked to the dataloaders creation.

        Args:
            train_batch_size (int, optional):
                Batch size per GPU/NPU core/CPU for training. Default: ``8``.
            eval_batch_size (int, optional):
                Batch size per GPU/NPU core/CPU for evaluation. Default: ``8``.
            drop_last (bool, optional):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not. Default: ``False``.
            num_workers (int, optional):
                Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                the main process. Default: ``0``.
            ignore_data_skip (bool, optional):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have. Default: ``False``.
            data_skip_steps (int, optional):
                Specify the skip steps of train dataset when resume training.
                It only takes effect when `ignore_data_skip` is set to False. Default: ``None``.
            sampler_seed (int, optional):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed. Default: ``None``.
            kwargs (Any):
                Additional parameters.

        Examples:
            >>> from mindformers import TrainingArguments
            >>> args = TrainingArguments(output_dir="output")
            >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
            >>> args.per_device_train_batch_size
            16
        """
        self.per_device_train_batch_size = train_batch_size
        self.per_device_eval_batch_size = eval_batch_size
        self.dataloader_drop_last = drop_last
        self.dataloader_num_workers = num_workers
        self.ignore_data_skip = ignore_data_skip
        self.data_skip_steps = data_skip_steps
        self.data_seed = sampler_seed
        self.print_kwargs_unused(**kwargs)
        return self

    @staticmethod
    def print_kwargs_unused(**kwargs):
        if kwargs:
            for k in kwargs:
                logger.warning("the `%s` parameter is not supported.", k)

    def convert_args_to_mindformers_config(self, task_config: MindFormerConfig = None):
        """
        Convert training arguments to mindformer config type for adapting hugging-face.

        Args:
            task_config (MindFormerConfig, optional):
                Task configuration. Default: ``None``.

        Returns:
            An instance of MindFormerConfig, processed task configuration.
        """
        if task_config is None:
            task_config = MindFormerConfig()

        self._adapt_common_config(task_config)
        self._adapt_runner_config(task_config)
        self._adapt_context_config(task_config)
        self._adapt_parallel_config(task_config)
        self._adapt_moe_config(task_config)
        self._adapt_recompute_config(task_config)
        self._adapt_optimizer_config(task_config)
        self._adapt_lr_schedule_config(task_config)
        self._adapt_dataset_config(task_config)
        self._adapt_wrapper_config(task_config)
        self._adapt_metric_config(task_config)
        self._adapt_callback_config(task_config)
        self._adapt_eval_config(task_config)
        self._adapt_profile_config(task_config)
        self._adapt_auto_tune_config(task_config)
        self._adapt_hub_config(task_config)

        return task_config

    def _adapt_common_config(self, task_config):
        """adapt common config."""
        task_config.output_dir = _check_training_args(task_config.output_dir, self.output_dir)
        task_config.overwrite_output_dir = _check_training_args(
            task_config.overwrite_output_dir, self.overwrite_output_dir)
        task_config.seed = _check_training_args(task_config.seed, self.seed)
        task_config.data_seed = _check_training_args(task_config.data_seed, self.data_seed)
        task_config.only_save_strategy = _check_training_args(task_config.only_save_strategy, self.only_save_strategy)
        task_config.auto_trans_ckpt = _check_training_args(task_config.auto_trans_ckpt, self.auto_trans_ckpt)
        task_config.src_strategy_path_or_dir = _check_training_args(task_config.src_strategy, self.src_strategy)
        task_config.transform_process_num = _check_training_args(
            task_config.transform_process_num, self.transform_process_num)
        task_config.load_checkpoint = _check_training_args(task_config.load_checkpoint, self.resume_from_checkpoint)
        task_config.resume_training = _check_training_args(task_config.resume_training, self.resume_training)
        task_config.ignore_data_skip = _check_training_args(task_config.ignore_data_skip, self.ignore_data_skip)
        task_config.data_skip_steps = _check_training_args(task_config.data_skip_steps, self.data_skip_steps)
        task_config.do_train = _check_training_args(task_config.do_train, self.do_train)
        task_config.do_eval = _check_training_args(task_config.do_eval, self.do_eval)
        task_config.do_predict = _check_training_args(task_config.do_predict, self.do_predict)
        task_config.remote_save_url = _check_training_args(task_config.remote_save_url, self.remote_save_url)
        task_config.check_for_nan_in_loss_and_grad = _check_training_args(
            task_config.check_for_nan_in_loss_and_grad, self.check_for_nan_in_loss_and_grad)
        task_config.calculate_per_token_loss = _check_training_args(
            task_config.calculate_per_token_loss, self.calculate_per_token_loss)


    def _adapt_runner_config(self, task_config):
        """adapt runner config."""
        if not _check_task_config(task_config.runner_config):
            task_config.runner_config = MindFormerConfig()

        task_config.runner_config.epochs = _check_training_args(
            task_config.runner_config.epochs, self.num_train_epochs)
        task_config.runner_config.batch_size = _check_training_args(
            task_config.runner_config.batch_size, self.per_device_train_batch_size)
        task_config.runner_config.gradient_accumulation_steps = _check_training_args(
            task_config.runner_config.gradient_accumulation_steps, self.gradient_accumulation_steps)
        task_config.runner_config.sink_size = _check_training_args(
            task_config.runner_config.sink_size, self.sink_size)
        task_config.runner_config.sink_mode = _check_training_args(
            task_config.runner_config.sink_mode, self.sink_mode)

    def _adapt_context_config(self, task_config):
        """adapt context config."""
        if not _check_task_config(task_config.context):
            task_config.context = MindFormerConfig()

        if self.use_cpu:
            self.device_target = 'CPU'
            self.use_parallel = False

        task_config.context.mode = _check_training_args(
            task_config.context.mode, self.mode)
        task_config.context.device_id = _check_training_args(
            task_config.context.device_id, self.device_id)
        task_config.context.device_target = _check_training_args(
            task_config.context.device_target, self.device_target)
        task_config.context.enable_graph_kernel = _check_training_args(
            task_config.context.enable_graph_kernel, self.enable_graph_kernel)
        task_config.context.max_call_depth = _check_training_args(
            task_config.context.max_call_depth, self.max_call_depth)
        task_config.context.max_device_memory = _check_training_args(
            task_config.context.max_device_memory, self.max_device_memory)
        task_config.context.save_graphs = _check_training_args(
            task_config.context.save_graphs, self.save_graphs)
        task_config.context.save_graphs_path = _check_training_args(
            task_config.context.save_graphs_path, self.save_graphs_path)

    def _adapt_parallel_config(self, task_config):
        """adapt parallel config."""
        task_config.use_parallel = _check_training_args(task_config.use_parallel, self.use_parallel)
        task_config.micro_batch_interleave_num = _check_training_args(
            task_config.micro_batch_interleave_num, self.micro_batch_interleave_num)

        if not _check_task_config(task_config.parallel):
            task_config.parallel = MindFormerConfig()
        if not _check_task_config(task_config.parallel.parallel_optimizer_config):
            task_config.parallel.parallel_optimizer_config = MindFormerConfig()
        if not _check_task_config(task_config.parallel_config):
            task_config.parallel_config = MindFormerConfig()

        task_config.parallel.parallel_mode = _check_training_args(
            task_config.parallel.parallel_mode, self.parallel_mode)
        task_config.parallel.gradients_mean = _check_training_args(
            task_config.parallel.gradients_mean, self.gradients_mean)
        task_config.parallel.loss_repeated_mean = _check_training_args(
            task_config.parallel.loss_repeated_mean, self.loss_repeated_mean)
        task_config.parallel.enable_alltoall = _check_training_args(
            task_config.parallel.enable_alltoall, self.enable_alltoall)
        task_config.parallel.full_batch = _check_training_args(
            task_config.parallel.full_batch, self.full_batch)
        task_config.parallel.dataset_strategy = _check_training_args(
            task_config.parallel.dataset_strategy, self.dataset_strategy)
        task_config.parallel.search_mode = _check_training_args(
            task_config.parallel.search_mode, self.search_mode)
        task_config.parallel.enable_parallel_optimizer = _check_training_args(
            task_config.parallel.enable_parallel_optimizer, self.enable_parallel_optimizer)
        task_config.parallel.strategy_ckpt_save_file = _check_training_args(
            task_config.parallel.strategy_ckpt_save_file, self.strategy_ckpt_save_file)

        task_config.parallel.parallel_optimizer_config.gradient_accumulation_shard = _check_training_args(
            task_config.parallel.parallel_optimizer_config.gradient_accumulation_shard,
            self.gradient_accumulation_shard
        )
        task_config.parallel.parallel_optimizer_config.parallel_optimizer_threshold = _check_training_args(
            task_config.parallel.parallel_optimizer_config.parallel_optimizer_threshold,
            self.parallel_optimizer_threshold
        )
        if self.optimizer_weight_shard_size > 0:
            task_config.parallel.parallel_optimizer_config.optimizer_weight_shard_size = \
                self.optimizer_weight_shard_size

        task_config.parallel_config.data_parallel = _check_training_args(
            task_config.parallel_config.data_parallel, self.data_parallel)
        task_config.parallel_config.model_parallel = _check_training_args(
            task_config.parallel_config.model_parallel, self.model_parallel)
        task_config.parallel_config.expert_parallel = _check_training_args(
            task_config.parallel_config.expert_parallel, self.expert_parallel)
        task_config.parallel_config.pipeline_stage = _check_training_args(
            task_config.parallel_config.pipeline_stage, self.pipeline_stage)
        task_config.parallel_config.micro_batch_num = _check_training_args(
            task_config.parallel_config.micro_batch_num, self.micro_batch_num)
        task_config.parallel_config.gradient_aggregation_group = _check_training_args(
            task_config.parallel_config.gradient_aggregation_group, self.gradient_aggregation_group)
        task_config.parallel_config.use_seq_parallel = _check_training_args(
            task_config.parallel_config.use_seq_parallel, self.use_seq_parallel)
        task_config.parallel_config.vocab_emb_dp = _check_training_args(
            task_config.parallel_config.vocab_emb_dp, self.vocab_emb_dp)

    def _adapt_moe_config(self, task_config):
        """adapt moe config."""
        if not _check_task_config(task_config.moe_config):
            task_config.moe_config = MindFormerConfig()

        task_config.moe_config.expert_num = _check_training_args(
            task_config.moe_config.expert_num, self.expert_num)
        task_config.moe_config.capacity_factor = _check_training_args(
            task_config.moe_config.capacity_factor, self.capacity_factor)
        task_config.moe_config.aux_loss_factor = _check_training_args(
            task_config.moe_config.aux_loss_factor, self.aux_loss_factor)
        task_config.moe_config.num_experts_chosen = _check_training_args(
            task_config.moe_config.num_experts_chosen, self.num_experts_chosen)

    def _adapt_recompute_config(self, task_config):
        """adapt recompute config."""
        if not _check_task_config(task_config.recompute_config):
            task_config.recompute_config = MindFormerConfig()

        task_config.recompute_config.recompute = _check_training_args(
            task_config.recompute_config.recompute, self.recompute)
        task_config.recompute_config.select_recompute = _check_training_args(
            task_config.recompute_config.select_recompute, self.select_recompute)
        task_config.recompute_config.parallel_optimizer_comm_recompute = _check_training_args(
            task_config.recompute_config.parallel_optimizer_comm_recompute, self.parallel_optimizer_comm_recompute)
        task_config.recompute_config.mp_comm_recompute = _check_training_args(
            task_config.recompute_config.mp_comm_recompute, self.mp_comm_recompute)
        task_config.recompute_config.recompute_slice_activation = _check_training_args(
            task_config.recompute_config.recompute_slice_activation, self.recompute_slice_activation)

    def _adapt_optimizer_config(self, task_config):
        """adapt optimizer config."""
        if not _check_task_config(task_config.optimizer):
            task_config.optimizer = MindFormerConfig()

        task_config.optimizer.type = _check_training_args(task_config.optimizer.type, self.optim)
        task_config.optimizer.beta1 = _check_training_args(task_config.optimizer.beta1, self.adam_beta1)
        task_config.optimizer.beta2 = _check_training_args(task_config.optimizer.beta2, self.adam_beta2)
        task_config.optimizer.eps = _check_training_args(task_config.optimizer.eps, self.adam_epsilon)
        task_config.optimizer.weight_decay = _check_training_args(task_config.optimizer.weight_decay, self.weight_decay)
        task_config.layer_scale = _check_training_args(task_config.layer_scale, self.layer_scale)
        task_config.layer_decay = _check_training_args(task_config.layer_decay, self.layer_decay)

    def _adapt_lr_schedule_config(self, task_config):
        """adapt lr schedule config."""
        if not _check_task_config(task_config.lr_schedule):
            task_config.lr_schedule = MindFormerConfig()

        task_config.lr_schedule.type = _check_training_args(
            task_config.lr_schedule.type, self.lr_scheduler_type)
        task_config.lr_schedule.learning_rate = _check_training_args(
            task_config.lr_schedule.learning_rate, self.learning_rate)
        task_config.lr_schedule.lr_end = _check_training_args(
            task_config.lr_schedule.lr_end, self.lr_end)
        task_config.lr_schedule.warmup_lr_init = _check_training_args(
            task_config.lr_schedule.warmup_lr_init, self.warmup_lr_init)
        task_config.lr_schedule.warmup_epochs = _check_training_args(
            task_config.lr_schedule.warmup_epochs, self.warmup_epochs)
        task_config.lr_schedule.warmup_ratio = _check_training_args(
            task_config.lr_schedule.warmup_ratio, self.warmup_ratio)
        task_config.lr_schedule.warmup_steps = _check_training_args(
            task_config.lr_schedule.warmup_steps, self.warmup_steps)
        task_config.lr_schedule.total_steps = _check_training_args(
            task_config.lr_schedule.total_steps, self.total_steps)

        task_config.lr_scale = _check_training_args(task_config.lr_scale, self.lr_scale)
        task_config.lr_scale_factor = _check_training_args(task_config.lr_scale_factor, self.lr_scale_factor)

    def _adapt_dataset_config(self, task_config):
        """adapt dataset config."""
        if not _check_task_config(task_config.train_dataset):
            task_config.train_dataset = MindFormerConfig()
        if not _check_task_config(task_config.train_dataset.data_loader):
            task_config.train_dataset.data_loader = MindFormerConfig()
        if not _check_task_config(task_config.train_dataset_task):
            task_config.train_dataset_task = MindFormerConfig()

        if not _check_task_config(task_config.eval_dataset):
            task_config.eval_dataset = MindFormerConfig()
        if not _check_task_config(task_config.eval_dataset.data_loader):
            task_config.eval_dataset.data_loader = MindFormerConfig()
        if not _check_task_config(task_config.eval_dataset_task):
            task_config.eval_dataset_task = MindFormerConfig()

        task_config.train_dataset.data_loader.type = _check_training_args(
            task_config.train_dataset.data_loader.type, self.dataset_type)
        task_config.eval_dataset.data_loader.type = _check_training_args(
            task_config.eval_dataset.data_loader.type, self.dataset_type)
        task_config.train_dataset.data_loader.dataset_dir = _check_training_args(
            task_config.train_dataset.data_loader.dataset_dir, self.train_dataset)
        task_config.train_dataset.data_loader.shuffle = _check_training_args(
            task_config.train_dataset.data_loader.shuffle, self.shuffle)
        task_config.train_dataset.input_columns = _check_training_args(
            task_config.train_dataset.input_columns, self.train_dataset_in_columns)
        task_config.train_dataset.output_columns = _check_training_args(
            task_config.train_dataset.output_columns, self.train_dataset_out_columns)
        task_config.train_dataset.batch_size = _check_training_args(
            task_config.train_dataset.batch_size, self.per_device_train_batch_size)
        task_config.train_dataset.num_parallel_workers = _check_training_args(
            task_config.train_dataset.num_parallel_workers, self.dataloader_num_workers)
        task_config.train_dataset.python_multiprocessing = _check_training_args(
            task_config.train_dataset.python_multiprocessing, self.python_multiprocessing)
        task_config.train_dataset.drop_remainder = _check_training_args(
            task_config.train_dataset.drop_remainder, self.dataloader_drop_last)
        task_config.train_dataset.repeat = _check_training_args(
            task_config.train_dataset.repeat, self.repeat)
        task_config.train_dataset.numa_enable = _check_training_args(
            task_config.train_dataset.numa_enable, self.numa_enable)
        task_config.train_dataset.prefetch_size = _check_training_args(
            task_config.train_dataset.prefetch_size, self.prefetch_size)

        task_config.eval_dataset.data_loader.dataset_dir = _check_training_args(
            task_config.eval_dataset.data_loader.dataset_dir, self.eval_dataset)
        task_config.eval_dataset.data_loader.shuffle = False
        task_config.eval_dataset.input_columns = _check_training_args(
            task_config.eval_dataset.input_columns, self.eval_dataset_in_columns)
        task_config.eval_dataset.output_columns = _check_training_args(
            task_config.eval_dataset.output_columns, self.eval_dataset_out_columns)
        task_config.eval_dataset.batch_size = _check_training_args(
            task_config.eval_dataset.batch_size, self.per_device_eval_batch_size)
        task_config.eval_dataset.num_parallel_workers = _check_training_args(
            task_config.eval_dataset.num_parallel_workers, self.dataloader_num_workers)
        task_config.eval_dataset.python_multiprocessing = _check_training_args(
            task_config.eval_dataset.python_multiprocessing, self.python_multiprocessing)
        task_config.eval_dataset.drop_remainder = False
        task_config.eval_dataset.repeat = 1
        task_config.eval_dataset.numa_enable = _check_training_args(
            task_config.eval_dataset.numa_enable, self.numa_enable)
        task_config.eval_dataset.prefetch_size = _check_training_args(
            task_config.eval_dataset.prefetch_size, self.prefetch_size)

        task_config.train_dataset_task.type = _check_training_args(
            task_config.train_dataset_task.type, self.dataset_task)
        task_config.eval_dataset_task.type = _check_training_args(
            task_config.eval_dataset_task.type, self.dataset_task)
        task_config.train_dataset_task.dataset_config = task_config.train_dataset
        task_config.eval_dataset_task.dataset_config = task_config.eval_dataset

    def _adapt_wrapper_config(self, task_config):
        """adapt wrapper config."""
        if not _check_task_config(task_config.runner_wrapper):
            task_config.runner_wrapper = MindFormerConfig()
        if isinstance(self.scale_sense, str) and \
            not _check_task_config(task_config.runner_wrapper.scale_sense):
            task_config.runner_wrapper.scale_sense = MindFormerConfig()

        task_config.runner_wrapper.type = _check_training_args(
            task_config.runner_wrapper.type, self.wrapper_type)
        task_config.runner_wrapper.use_clip_grad = _check_training_args(
            task_config.runner_wrapper.use_clip_grad, self.use_clip_grad)
        task_config.runner_wrapper.max_grad_norm = _check_training_args(
            task_config.runner_wrapper.max_grad_norm, self.max_grad_norm)
        if isinstance(self.scale_sense, str):
            task_config.runner_wrapper.scale_sense.type = _check_training_args(
                task_config.runner_wrapper.scale_sense.type, self.scale_sense)
            task_config.runner_wrapper.scale_sense.loss_scale_value = _check_training_args(
                task_config.runner_wrapper.scale_sense.loss_scale_value, self.loss_scale_value)
            task_config.runner_wrapper.scale_sense.scale_factor = _check_training_args(
                task_config.runner_wrapper.scale_sense.scale_factor, self.loss_scale_factor)
            task_config.runner_wrapper.scale_sense.scale_window = _check_training_args(
                task_config.runner_wrapper.scale_sense.scale_window, self.loss_scale_window)
            if self.scale_sense == "AdaptiveLossScaleUpdateCell":
                task_config.runner_wrapper.scale_sense.max_scale_window = _check_training_args(
                    task_config.runner_wrapper.scale_sense.max_scale_window, self.max_scale_window)
                task_config.runner_wrapper.scale_sense.min_scale_window = _check_training_args(
                    task_config.runner_wrapper.scale_sense.min_scale_window, self.min_scale_window)
        elif isinstance(self.scale_sense, (float, int)):
            task_config.runner_wrapper.scale_sense = int(self.scale_sense)

    def _adapt_metric_config(self, task_config):
        """adapt metric config."""
        if not _check_task_config(task_config.metric):
            task_config.metric = []
        if isinstance(task_config.metric, dict):
            kv_list = []
            for k, v in task_config.metric.items():
                kv_list.append((k, v))
            task_config.metric = [OrderedDict(kv_list)]
        if self.metric_type:
            type_dict = {metric['type']: i for i, metric in enumerate(task_config.metric)}
            for metric_type in self.metric_type:
                if metric_type in type_dict:
                    continue
                task_config.metric.append(
                    OrderedDict([("type", metric_type)]))

    def _adapt_callback_config(self, task_config):
        """adapt callback config."""
        if not _check_task_config(task_config.callbacks):
            task_config.callbacks = [OrderedDict([("type", "MFLossMonitor")]),
                                     OrderedDict([("type", "CheckpointMonitor")]),
                                     OrderedDict([("type", "ObsMonitor")])]

        if not isinstance(task_config.callbacks, list):
            raise ValueError(f"The type of config.callbacks should be List, but got {type(task_config.callbacks)}.")

        new_callbacks = []
        for callback in task_config.callbacks:
            if not isinstance(callback, dict):
                raise ValueError(f"The type of callback should be dict, but got {type(callback)}.")
            if callback['type'] == "CheckpointMonitor" and self.save_strategy == 'no':
                continue
            new_callbacks.append(callback)
        task_config.callbacks = new_callbacks

        def _adapt_logging_callback(callback):
            """adapt logging callback"""
            if self.logging_strategy == LoggingIntervalStrategy.STEPS:
                _adapt_dict_args(callback, 'per_print_times', self.logging_steps)
            return callback

        def _adapt_save_checkpoint_callback(callback):
            """adapt save checkpoint callback"""
            _adapt_dict_args(callback, 'prefix', self.save_prefix)
            _adapt_dict_args(callback, 'directory', self.save_directory)
            if self.save_strategy == SaveIntervalStrategy.STEPS:
                _adapt_dict_args(callback, 'save_checkpoint_steps', self.save_steps)
            elif self.save_strategy == SaveIntervalStrategy.SECONDS:
                _adapt_dict_args(callback, 'save_checkpoint_seconds', self.save_seconds)
            _adapt_dict_args(callback, 'keep_checkpoint_max', self.save_total_limit)
            _adapt_dict_args(callback, 'keep_checkpoint_per_n_minutes', self.keep_checkpoint_per_n_minutes)
            integrated_save = self.integrated_save if self.integrated_save is not None \
                else not self.save_on_each_node
            _adapt_dict_args(callback, 'integrated_save', integrated_save)
            _adapt_dict_args(callback, 'save_network_params', self.save_network_params)
            _adapt_dict_args(callback, 'save_trainable_params', self.save_trainable_params)
            _adapt_dict_args(callback, 'async_save', self.async_save)
            return callback

        for i, callback in enumerate(task_config.callbacks):
            if callback['type'] == "MFLossMonitor":
                task_config.callbacks[i] = _adapt_logging_callback(task_config.callbacks[i])
            if callback['type'] == "CheckpointMonitor":
                task_config.callbacks[i] = _adapt_save_checkpoint_callback(task_config.callbacks[i])

    def _adapt_eval_config(self, task_config):
        """adapt eval config"""
        task_config.eval_step_interval = _check_training_args(task_config.eval_step_interval, self.eval_steps)
        task_config.eval_epoch_interval = _check_training_args(task_config.eval_epoch_interval, self.eval_epochs)

    def _adapt_profile_config(self, task_config):
        """adapt profile config."""
        task_config.profile = _check_training_args(task_config.profile, self.profile)
        task_config.profile_start_step = _check_training_args(task_config.profile_start_step, self.profile_start_step)
        task_config.profile_end_step = _check_training_args(task_config.profile_end_step, self.profile_end_step)
        task_config.init_start_profile = _check_training_args(task_config.init_start_profile, self.init_start_profile)
        task_config.profile_communication = _check_training_args(task_config.profile_communication,
                                                                 self.profile_communication)
        task_config.profile_memory = _check_training_args(task_config.profile_memory, self.profile_memory)

    def _adapt_auto_tune_config(self, task_config):
        """adapt auto tune config."""
        task_config.auto_tune = _check_training_args(task_config.auto_tune, self.auto_tune)
        task_config.filepath_prefix = _check_training_args(task_config.filepath_prefix, self.filepath_prefix)
        task_config.autotune_per_step = _check_training_args(task_config.autotune_per_step, self.autotune_per_step)

    def _adapt_hub_config(self, task_config):
        """adapt hub config."""
        task_config.push_to_hub = _check_training_args(task_config.push_to_hub, self.push_to_hub)
        task_config.hub_model_id = _check_training_args(task_config.hub_model_id, self.hub_model_id)
        task_config.hub_strategy = _check_training_args(task_config.hub_strategy, self.hub_strategy)
        task_config.hub_token = _check_training_args(task_config.hub_token, self.hub_token)
        task_config.hub_private_repo = _check_training_args(task_config.hub_private_repo, self.hub_private_repo)
        task_config.hub_always_push = _check_training_args(task_config.hub_always_push, self.hub_always_push)
