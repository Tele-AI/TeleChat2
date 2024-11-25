# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Base Trainer."""
import os
import subprocess
from pprint import pprint
from functools import partial
from typing import Optional, Union, List
from collections import OrderedDict
import numpy as np
from PIL.Image import Image
import mindspore as ms
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import Dataset
from mindspore.train.metrics import get_metrics
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn import Optimizer, Cell, PipelineCell, MicroBatchInterleaved
try:
    # new interface in ms2.1.1
    from mindspore.nn.wrap.cell_wrapper import GradAccumulationCell
    GRAD_ACCUMULATION_VALID = True
except ImportError:
    GRAD_ACCUMULATION_VALID = False

from mindformers.mindformer_book import MindFormerBook
from mindformers.core import build_lr, build_optim, build_callback, build_metric
from mindformers.core.callback.callback import EvalCallBack
from mindformers.core.parallel_config import build_parallel_config
from mindformers.dataset import build_dataset, check_dataset_config, \
    check_dataset_iterable, BaseDataset
from mindformers.models import build_network, build_processor, build_tokenizer, \
    PreTrainedModel, PreTrainedTokenizerBase, BaseImageProcessor
from mindformers.pipeline import pipeline
from mindformers.wrapper import build_wrapper, PipelineCellWithTwoOutput, GradAccumulationCellWithTwoOutput
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.check_rules import check_rules
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.core.callback.callback import ColdHotExpertMointor
from .config_args import ConfigArguments
from .training_args import TrainingArguments
from .utils import (
    check_runner_config,
    transform_and_load_checkpoint,
    load_resume_context_from_checkpoint,
)
from .optimizer_grouped_parameters import get_optimizer_grouped_parameters
from .utils import set_seed, check_train_data_loader_type, \
    check_eval_data_loader_type, check_optimizer_and_lr_type, check_wrapper_config

SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_PIPELINES = MindFormerBook().get_pipeline_support_task_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CONFIG_DIR = 'configs'


class BaseTrainer:
    r"""Base Task Trainer.
    Args:
        task (str): The task name supported.
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, task: str = None, model_name: str = None):

        host_name_output = subprocess.run(['hostname'], shell=False, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, encoding='utf-8')
        host_ip_output = subprocess.run(['hostname', '-I'], shell=False, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, encoding='utf-8')
        host_name = host_name_output.stdout.strip()
        host_ip = host_ip_output.stdout.strip().split(' ')[0]
        logger.info("host_name: {}, host_ip: {}".format(host_name, host_ip))

        if model_name is None:
            model_name = "model name unspecified."
        if task is None:
            task = "task name unspecified."
        logger.info("Now Running Task is: %s, Model is: %s", task, model_name)

        self.model_name = model_name
        self.task = task
        self.config = None
        self.default_task_config = None

        self.train_dataset = None
        self.eval_dataset = None
        self.network = None
        self.optimizer = None
        self.image_processor = None
        self.audio_processor = None
        self.tokenizer = None
        self.callbacks = None
        self.eval_callbacks = None
        self.compute_metrics = None
        self.kwargs = None
        self.pipeline_task = None

        if task not in SUPPORT_TASKS.keys():
            logger.warning("Input task name is not in the supported list or unspecified.")

        if task in SUPPORT_TASKS.keys() and model_name not in SUPPORT_TASKS.get(task).keys():
            model_name_support_list = list(MindFormerBook().get_model_name_support_list_for_task(task))
            model_name_support_list.sort()
            logger.warning("Input model name is not in the supported list or unspecified.")
            logger.warning("See the list of supported task and model name: %s", model_name_support_list)
            logger.warning("The default model config: %s will now be used for the %s task ",
                           SUPPORT_TASKS.get(self.task).get("common"), task)
            self.model_name = "common"

    def set_config(self,
                   config: Optional[Union[dict, str, ConfigArguments, TrainingArguments]] = None,
                   is_full_config: bool = False):
        """Set the task config for task trainer."""
        if config is not None:
            if is_full_config:
                self.config = config
            else:
                self.setup_task_config()
                self.config = self._merge_config(config)
        else:
            self.setup_task_config()
            self.config = self.default_task_config
        build_parallel_config(self.config)
        self._check_grad_accumulation_steps()
        self._check_global_batch_size_for_auto_parallel()
        self._reset_wrapper()

        return self.config

    def setup_task_config(self):
        """Setup the default task config."""
        task_config = None
        if self.task in SUPPORT_TASKS.keys() and self.model_name in SUPPORT_TASKS.get(self.task).keys():
            task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task).get(self.model_name))

        if isinstance(task_config, MindFormerConfig):
            self.default_task_config = task_config
        else:
            logger.warning("If the default config arguments is not specified,"
                           "you must define the required model, optimizer, and so on"
                           "in the train or evaluate or predict attribute function.")

    def _check_global_batch_size_for_auto_parallel(self):
        """Check global batch size in auto parallel mode."""
        batch_size = self.config.runner_config.batch_size
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        dp = self.config.parallel_config.data_parallel
        micro_batch_num = self.config.parallel_config.micro_batch_num
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        pp = self.get_pipeline_stages()

        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            if full_batch:
                if pp > 1:
                    self.global_batch_size = batch_size * dp * micro_batch_num * micro_batch_interleave_num
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is True, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "global batch size will be changed: "
                                "global_batch_size = "
                                "batch_size * data_parallel * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s * %s).",
                                pp, self.global_batch_size, batch_size, dp, micro_batch_num,
                                micro_batch_interleave_num)
                    self.config.runner_config.batch_size = self.global_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    self.global_batch_size = batch_size * dp * micro_batch_interleave_num * gradient_accumulation_steps
                    logger.info("The current parallel mode is %s, full batch is True,"
                                "so global batch size will be changed: "
                                "global_batch_size = batch_size * data_parallel * micro_batch_interleave_num "
                                "* gradient_accumulation_steps = %s = %s * %s * %s * %s",
                                parallel_mode, self.global_batch_size, batch_size, dp, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    self.config.runner_config.batch_size = self.global_batch_size
            else:
                if pp > 1:
                    per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num
                    self.global_batch_size = per_batch_size * get_real_group_size()
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is False, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s).",
                                pp, per_batch_size, batch_size, micro_batch_num,
                                micro_batch_interleave_num)
                    logger.info("global_batch_size = per_batch_size * device_num = %s * %s = %s",
                                per_batch_size, get_real_group_size(), self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    per_batch_size = batch_size * micro_batch_interleave_num * gradient_accumulation_steps
                    self.global_batch_size = per_batch_size * get_real_group_size()
                    logger.info("The current parallel mode is %s, full batch is False, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_interleave_num * "
                                "gradient_accumulation_steps = %s = %s * %s * %s).",
                                parallel_mode, per_batch_size, batch_size, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    logger.info("global_batch_size = per_batch_size * device_num = %s * %s = %s",
                                per_batch_size, get_real_group_size(), self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
        else:
            logger.info("The current parallel mode is %s, batch size per card will not be changed: "
                        "batch_size_per_card = %s",
                        parallel_mode, batch_size)
            self.global_batch_size = batch_size * get_real_group_size() * gradient_accumulation_steps
            logger.info(
                "global_batch_size = batch_size_per_card * device_num * gradient_accumulation_steps "
                "= %s = %s * %s * %s",
                self.global_batch_size, batch_size, get_real_group_size(), gradient_accumulation_steps)
            self.config.runner_config.batch_size = batch_size * gradient_accumulation_steps
            self.config.parallel_config.data_parallel = 1
            self.config.parallel_config.model_parallel = 1
            self.config.parallel_config.pipeline_stage = 1
            self.config.parallel_config.micro_batch_num = 1
            logger.info("parallel_config will be change to default config: %s.",
                        self.config.parallel_config)
        self.config.runner_config.global_batch_size = self.global_batch_size


    def _check_grad_accumulation_steps(self):
        """check the gradient accumulation steps."""
        if self.config.runner_config.gradient_accumulation_steps is None:
            self.config.runner_config.gradient_accumulation_steps = 1
        if not isinstance(self.config.runner_config.gradient_accumulation_steps, int) or \
            isinstance(self.config.runner_config.gradient_accumulation_steps, bool):
            raise ValueError("gradient_accumulation should be integer but got "
                             f"{type(self.config.runner_config.gradient_accumulation_steps)}")
        if not self.config.runner_config.gradient_accumulation_steps >= 1:
            raise ValueError("gradient_accumulation should be greater or equal than 1, "
                             f"but got {self.config.runner_config.gradient_accumulation_steps}")
        if not GRAD_ACCUMULATION_VALID and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps only surpport mindspore version later than 2.1.1, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pp > 1 \
            and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps will not take effect when using pipeline parallel, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        # grad accumulation not supported in data parallel/standalone mode for now
        if self.config.runner_config.gradient_accumulation_steps > 1 and \
            parallel_mode not in ["semi_auto_parallel", "auto_parallel"]:
            logger.warning("gradient_accumulation_steps currently need to be used in semi_auto_parallel mode, "
                           "but got %s mode, please check your runner config and parallel config. "
                           "Reset the gradient_accumulation_steps from %s to 1. ",
                           parallel_mode, self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1

    def _check_training_network_no_use_past(self, network):
        if network is not None and hasattr(network.config, "use_past") and network.config.use_past:
            raise ValueError("In training process, network should be configured to use_past=False, "
                             f"but got use_past={network.config.use_past}")

    def _reset_wrapper(self):
        """reset wrapper in some special cases."""
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        # use gradient_accumulation, reset wrapper to MFPipelineWrapper
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and gradient_accumulation_steps > 1:
            self._reset_wrapper_for_grad_accu(gradient_accumulation_steps)

    def _reset_wrapper_for_pipeline_parallel(self):
        """Reset wrapper when pipeline parallel."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            self.config.runner_wrapper.micro_batch_num = self.config.parallel_config.micro_batch_num
            self.config.runner_wrapper.calculate_per_token_loss = self.config.calculate_per_token_loss
            logger.warning(
                "When using the pipeline parallel mode, "
                "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.info(
                "When using the pipeline parallel mode, "
                "because the wrapper class is not specified, "
                "MindSpore's built-in PipelineCell is used by default")
        logger.info("PipelineWrapper under evaluate or predict mode will not take effect.")

    def _reset_wrapper_for_grad_accu(self, gradient_accumulation_steps):
        """Reset wrapper when using grad accumulation."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            # set micro_batch_num as gradient_accumulation_steps in MFPipelineWithLossScaleCell
            self.config.runner_wrapper.micro_batch_num = gradient_accumulation_steps
            self.config.runner_wrapper.calculate_per_token_loss = self.config.calculate_per_token_loss
            logger.warning(
                "When using the gradient_accumulation_steps in semi/auto parallel mode, "
                "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.info(
                "When using the gradient_accumulation_steps in semi/auto parallel mode, "
                "because the wrapper class is not specified, "
                "MindSpore's built-in GradAccumulationCell is used by default")

    def _reset_dataset_batch_size(self):
        """Reset dataset batch size according to the global batch size of runner config."""
        check_dataset_config(self.config)

    def _merge_config(self, args):
        """Merge config from default task config."""
        if self.default_task_config is None:
            logger.warning("default task config is None, you will not be able to merge config parameters.")
            return args

        if isinstance(args, dict):
            self.default_task_config.merge_from_dict(args)
        elif isinstance(args, str):
            if not (os.path.realpath(args) and os.path.exists(args)):
                raise FileNotFoundError(f"config path must be exist, but get {args}.")
            if not args.endswith(('.yaml', '.yml')):
                raise ValueError(f"config file must be end with .yaml or .yml, but get {args}")
            self.default_task_config = MindFormerConfig(args)
        elif isinstance(args, ConfigArguments):
            if hasattr(args, 'train_dataset'):
                check_train_data_loader_type(args, self.default_task_config)
            if hasattr(args, 'eval_dataset'):
                check_eval_data_loader_type(args, self.default_task_config)
            if hasattr(args, 'optimizer'):
                check_optimizer_and_lr_type(args, self.default_task_config)
            if hasattr(args, 'runner_wrapper'):
                check_wrapper_config(args, self.default_task_config)
            self.default_task_config.merge_from_dict(args.__dict__)
        elif isinstance(args, TrainingArguments):
            args.convert_args_to_mindformers_config(self.default_task_config)
        else:
            logger.warning(
                "The type of config parameter to merge is not supported, "
                "currently supported types are [dict, config_path(str), ConfigArguments, TrainingArguments], "
                "but get %s", type(args))
        return self.default_task_config

    def create_dataset(self, is_train: bool = True, default_args: dict = None):
        """Create the dataset for training or evaluate."""
        dataset = self.train_dataset if is_train else self.eval_dataset
        dataset_task = self.config.train_dataset_task if is_train else self.config.eval_dataset_task

        if isinstance(dataset, (BaseDataset, Dataset)):
            return dataset

        if dataset is None:
            dataset = build_dataset(dataset_task, default_args=default_args)
        elif check_dataset_iterable(dataset):
            dataset_task.type = 'GeneralDataset'
            default_args = {} if default_args is None else default_args
            default_args["dataset"] = dataset
            dataset = build_dataset(dataset_task, default_args=default_args)
        else:
            raise ValueError("Dataset should be Dataset, iterator, iterable class which has `__iter__`, "
                             f"iterable class which has  `__get_item__` and `__len__`, but get {dataset}.")
        return dataset

    def create_train_dataset(self, default_args: dict = None):
        """Create the train dataset for training."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        train_dataset = self.create_dataset(is_train=True, default_args=default_args)
        return train_dataset

    def create_eval_dataset(self, default_args: dict = None):
        """Create the eval dataset for evaluate."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pipeline_stages > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.parallel_config.micro_batch_num
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.micro_batch_interleave_num
        # reduce batch size for that gradient_accumulation_steps will not take effect on eval
        if self.config.runner_config.gradient_accumulation_steps > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.runner_config.gradient_accumulation_steps
        if self.config.eval_dataset.batch_size < 1:
            logger.warning("eval_dataset batch_size is less than 1 after bs calculate, reset batch_size to 1, "
                           "please check your configs about batch_size, micro_batch_num micro_batch_interleave_num "
                           "and gradient_accumulation_steps.")
            self.config.eval_dataset.batch_size = 1
        logger.info("For evaluate phase, batch size for eval dataset is %s, different from training, "
                    "not multiplied by micro_batch_num, micro_batch_interleave_num and gradient_accumulation_steps",
                    self.config.eval_dataset.batch_size)
        eval_dataset = self.create_dataset(is_train=False, default_args=default_args)
        return eval_dataset

    def create_network(self, default_args: dict = None):
        """Create the network for task trainer."""
        logger.info(".........Build Network From Config..........")
        return build_network(self.config.model, default_args=default_args)

    def wrap_network_with_tool_cells(self, network):
        """For training process, warp the network with some tool cells."""
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if micro_batch_interleave_num > 1:
            logger.info("micro_batch_interleave_num > 1, the double copy parallel feature is turned on.")
            network = MicroBatchInterleaved(network, micro_batch_interleave_num)
        if gradient_accumulation_steps > 1 and not pp > 1:
            logger.info("gradient_accumulation_steps > 1, GradAccumulationCell is wrapped on network. "
                        "It is suggested to use `Lazy Inline` feature to save compiling time.")
            if self.config.runner_wrapper.calculate_per_token_loss:
                network = GradAccumulationCellWithTwoOutput(network, gradient_accumulation_steps)
            else:
                network = GradAccumulationCell(network, gradient_accumulation_steps)
        if pp > 1:
            micro_batch_num = self.config.parallel_config.micro_batch_num
            if self.config.runner_wrapper.calculate_per_token_loss:
                network = PipelineCellWithTwoOutput(network, micro_size=micro_batch_num)
            else:
                network = PipelineCell(network, micro_size=micro_batch_num)
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            network = _VirtualDatasetCell(network)
        return network

    def wrap_eval_network_with_tool_cells(self, network):
        """For evaluate in training process, warp the network with some tool cells."""
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            network = _VirtualDatasetCell(network)
        return network

    def create_image_processor(self, default_args: dict = None):
        """Create the image processor for predict."""
        logger.info(".........Build Image Processor From Config..........")
        self.image_processor = build_processor(
            self.config.processor.image_processor, default_args=default_args)
        return self.image_processor

    def create_optimizer_scheduler(self, network, layer_scale=False):
        """Create the optimizer for training."""
        logger.info(".........Build Optimizer From Config..........")
        # learning rate scale for multi-nodes training
        learning_scale = self.config.lr_scale
        scale_factor = self.config.lr_scale_factor

        # build learning rate schedule
        lr_schedule = self.create_lr_scheduler(learning_scale, scale_factor)

        weight_decay = self.config.optimizer.weight_decay if self.config.optimizer.weight_decay else 0.
        layer_decay = self.config.layer_decay if self.config.layer_decay else 1.0
        group_params = get_optimizer_grouped_parameters(network,
                                                        weight_decay,
                                                        lr_schedule,
                                                        layer_scale=layer_scale,
                                                        layer_decay=layer_decay)
        if lr_schedule is not None:
            self.optimizer = build_optim(
                self.config.optimizer,
                default_args={"params": group_params,
                              "learning_rate": lr_schedule})
        else:
            if self.config.optimizer.learning_rate is None:
                raise ValueError("learning_rate must be input")
            self.config.optimizer.learning_rate = self.learning_rate_scale(
                self.config.optimizer.learning_rate, scale_factor) \
                if learning_scale and scale_factor is not None else self.config.optimizer.learning_rate
            self.optimizer = build_optim(
                self.config.optimizer,
                default_args={"params": group_params})
        return self.optimizer

    def create_lr_scheduler(self, learning_scale: bool = False, scale_factor: int = 256):
        """Create the learning rate scheduler."""
        logger.info(".........Build LR Schedule From Config..........")
        train_data_size = self.get_train_data_size()

        if self.config.lr_schedule:
            warmup_epochs = self.config.lr_schedule.pop("warmup_epochs", None)
            warmup_ratio = self.config.lr_schedule.pop("warmup_ratio", None)

            if not self.config.runner_config.sink_mode:
                total_steps = int(self.config.runner_config.epochs * train_data_size)
            else:
                total_steps = int(self.config.runner_config.epochs * self.config.runner_config.sink_size)

            if warmup_epochs is not None and warmup_ratio is not None:
                logger.warning("warmup_epochs and warmup_ratio are set simultaneously,"
                               "warmup_ratio takes precedence.")
                warmup_epochs = None

            if warmup_epochs is not None:
                logger.warning("warmup_epochs was set in lr_schedule,"
                               "it will multiply the data size to represent the warmup steps")
                self.config.lr_schedule.warmup_steps = int(warmup_epochs * train_data_size)

            self.config.lr_schedule.total_steps = total_steps \
                if self.config.lr_schedule.total_steps is None or self.config.lr_schedule.total_steps == -1 \
                else int(self.config.lr_schedule.total_steps)

            if warmup_ratio is not None:
                self.config.lr_schedule.warmup_steps = int(self.config.lr_schedule.total_steps * warmup_ratio)

            self.config.lr_schedule.learning_rate = self.learning_rate_scale(
                self.config.lr_schedule.learning_rate, scale_factor) \
                if learning_scale and scale_factor is not None else self.config.lr_schedule.learning_rate
        lr_schedule = build_lr(self.config.lr_schedule)
        return lr_schedule

    def create_model_wrapper(self, network, optimizer):
        """Create the model wrapper for training."""
        logger.info(".........Build Model Wrapper for Train From Config..........")
        model_wrapper = build_wrapper(self.config.runner_wrapper,
                                      default_args={"network": network,
                                                    "optimizer": optimizer,
                                                    "parallel_config": self.config.parallel_config})
        return model_wrapper

    def create_callbacks(self, default_args: dict = None):
        """Create the callback list for training."""
        logger.info(".........Build Callbacks for Train From Config..........")
        self.callbacks = []
        if self.config.profile:
            self.callbacks.append(self.config.profile_cb)
        self.callbacks.extend(build_callback(self.config.callbacks, default_args=default_args))
        return self.callbacks

    def create_eval_callbacks(self, default_args: dict = None):
        """Create the eval callback list for training."""
        logger.info(".........Build Callbacks for Evaluate From Config..........")
        self.eval_callbacks = []
        self.eval_callbacks.extend(build_callback(self.config.eval_callbacks, default_args=default_args))
        return self.eval_callbacks

    def create_metrics(self, metric_name: str = None):
        """Create Metrics For Evaluate or Fit."""
        if self.compute_metrics is not None:
            self.compute_metrics = get_metrics(self.compute_metrics)
            return self.compute_metrics
        if self.config.metric is None:
            raise ValueError("When `do_eval` is True and `compute_metrics` is None, \
                              the config of metric must not be None.")
        if isinstance(self.config.metric, dict):
            self.config.metric = [self.config.metric]

        self.compute_metrics = {}
        for metric_config in self.config.metric:
            if "type" not in metric_config:
                raise ValueError("The type of metric is not found!")
            metric = build_metric(metric_config)
            if metric_name is None:
                metric_name = metric.__class__.__name__
            self.compute_metrics[metric_name] = metric

        return self.compute_metrics

    def count_parameters(self):
        """Count network parameters number."""
        if self.network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(self.network)))
        else:
            logger.warning("Network is None, parameters incalculable.")

    def set_seed(self, seed: int = None):
        """Set seed for training."""
        if seed is None:
            if self.config.seed is None:
                raise ValueError("seed is not set in config, it is None.")
            set_seed(self.config.seed)
        else:
            set_seed(seed)

    def set_train_dataset(self, dataset):
        """Set the attribute of train dataset."""
        if dataset is None:
            raise ValueError("Train dataset is None")
        self.train_dataset = dataset

    def set_eval_dataset(self, dataset):
        """Set the attribute of eval dataset ."""
        if dataset is None:
            raise ValueError("Eval dataset is None")
        self.eval_dataset = dataset

    def set_network(self, network, is_train: bool = True):
        """Set the attribute of network."""
        if network is None:
            raise ValueError("network is None")
        if isinstance(network, (Cell, PreTrainedModel)):
            network.set_train(is_train)
        self.network = network

    def get_train_data_size(self):
        """Get train dataset size."""
        if self.train_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.train_dataset.get_dataset_size()

    def get_eval_data_size(self):
        """Get eval dataset size."""
        if self.eval_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.eval_dataset.get_dataset_size()

    def get_pipeline_stages(self):
        """Get pipeline stages for task trainer."""
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    def learning_rate_scale(self, base_learning_rate: float = 0., scale_factor: Optional[Union[float, int]] = 256.):
        """Scale learning rate for training."""
        if not isinstance(base_learning_rate, float):
            raise ValueError(f"learning rate must be float type, but get {type(base_learning_rate)}")
        if not isinstance(scale_factor, (float, int)):
            raise ValueError(f"scale_factor must be float or int type, but get {type(scale_factor)}")

        device_num = get_real_group_size()
        per_device_batch_size = self.config.train_dataset.batch_size
        learning_rate = (base_learning_rate * device_num * per_device_batch_size) / scale_factor
        return learning_rate

    def training_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            optimizer: Optional[Optimizer] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Train or Fine-tune for BaseTrainer in MindFormers."""
        self.kwargs = kwargs
        self.train_dataset = dataset if dataset else self.train_dataset
        self.eval_dataset = kwargs.get('eval_dataset', None)
        self.compute_metrics = compute_metrics if compute_metrics else self.compute_metrics

        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Train..........")
        dataset = self.create_train_dataset()
        logger.info("Create train dataset finish, dataset size:%d", dataset.get_dataset_size())

        append_info = None
        if config.resume_training and config.load_checkpoint:
            logger.info(".............Start load resume context from checkpoint..................")
            load_resume_context_from_checkpoint(config, dataset)
            resume_dict = {
                "step_num": config.runner_config.initial_step,
                "epoch_num": config.runner_config.initial_epoch,
            }
            if config.runner_wrapper.scale_sense is not None:
                if hasattr(config.runner_wrapper.scale_sense, 'loss_scale_value'):
                    resume_dict["loss_scale"] = config.runner_wrapper.scale_sense.loss_scale_value
                else:
                    resume_dict["loss_scale"] = config.runner_wrapper.scale_sense
            append_info = [resume_dict]
        else:
            config.runner_config.initial_epoch = 0
            config.runner_config.initial_step = 0

        # check if skip datasets
        if config.data_skip_steps or config.resume_training:
            if not config.ignore_data_skip:
                data_skip_steps = config.data_skip_steps if config.data_skip_steps \
                    else config.runner_config.initial_step
                if data_skip_steps > 0:
                    dataset.set_init_step(data_skip_steps)
                    logger.info("dataset skip %d steps.", data_skip_steps)
            else:
                logger.info("ignore dataset skip.")

        self.set_train_dataset(dataset)
        check_runner_config(config, dataset)

        # check rules
        check_rules(config, mode='train', network=network, dataset=dataset)

        # build network
        logger.info(".........Build Net For Train..........")
        if network is None and self.network is None:
            check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
            calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
            network = self.create_network(
                default_args={"parallel_config": config.parallel_config,
                              "moe_config": config.moe_config,
                              "dataset_config": config.train_dataset,
                              "calculate_per_token_loss": calculate_per_token_loss,
                              "check_for_nan_in_loss_and_grad": check_for_nan_in_loss_and_grad})
        elif network is None and self.network is not None:
            logger.info(".........Using The Existing Network For Train:: %s", self.network.__class__.__name__)
            network = self.network

        self._check_training_network_no_use_past(network)

        eval_network = None
        if network is not None:
            eval_network = network
            # warp network for training
            network = self.wrap_network_with_tool_cells(eval_network)
            eval_network = self.wrap_eval_network_with_tool_cells(eval_network)
            self.set_network(network, is_train=True)

        self.count_parameters()

        # build optimizer
        logger.info(".........Build Optimizer For Train..........")
        if optimizer is None:
            optimizer = self.create_optimizer_scheduler(network, layer_scale=config.layer_scale)

        # build model wrapper
        logger.info(".........Build Running Wrapper From Config For Train..........")
        wrapper = self.create_model_wrapper(network, optimizer)

        # build callback
        logger.info(".........Build Callbacks For Train..........")
        default_callbacks = []
        if self.config.profile:
            default_callbacks.append(self.config.profile_cb)
        if isinstance(self.config.runner_config.stop_step, int) and self.config.runner_config.stop_step > 0:
            stop_step_dict = OrderedDict()
            stop_step_dict['type'] = "TrainCallBack"
            stop_step_dict['stop_step'] = self.config.runner_config.stop_step
        for callback in self.config.callbacks:
            default_args = None
            if "type" in callback and callback["type"] == "MFLossMonitor":
                default_args = {
                    "learning_rate": optimizer.learning_rate if optimizer else wrapper.optimizer.learning_rate,
                    "origin_epochs": config.runner_config.origin_epochs,
                    "dataset_size": config.data_size,
                    "micro_batch_interleave_num": config.micro_batch_interleave_num,
                    "micro_batch_num": config.parallel_config.micro_batch_num,
                    "initial_epoch": config.runner_config.initial_epoch,
                    "initial_step": config.runner_config.initial_step,
                    "global_batch_size": self.global_batch_size,
                    "gradient_accumulation_steps": self.config.runner_config.gradient_accumulation_steps,
                    "calculate_per_token_loss": getattr(config, "calculate_per_token_loss", False),
                    "check_for_nan_in_loss_and_grad": getattr(config, "check_for_nan_in_loss_and_grad", False)
                }
            elif "type" in callback and callback["type"] == "CheckpointMonitor":
                default_args = {"append_info": append_info, "global_batch_size": self.global_batch_size}

            default_callbacks.append(build_callback(callback, default_args=default_args))
        if callbacks is not None:
            if isinstance(callbacks, list):
                default_callbacks.extend(callbacks)
            if isinstance(callbacks, Callback):
                default_callbacks.append(callbacks)
        callbacks = default_callbacks

        # define compute metrics for evaluate in training
        if config.do_eval:
            compute_metrics = self.create_metrics()

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        if wrapper is not None:
            if config.train_dataset.dynamic_batch:
                from mindspore import Symbol
                divisor = config.train_dataset.divisor if config.train_dataset.divisor else 2
                remainder = config.train_dataset.remainder if config.train_dataset.remainder else 1
                s = Symbol(divisor=divisor, remainder=remainder)
                dyn_inputs = []
                for _ in config.train_dataset.input_columns:
                    dyn_inputs.append(Tensor(shape=(config.train_dataset.batch_size, s), dtype=ms.int32))
                wrapper.set_inputs(*dyn_inputs)
            model = Model(wrapper, metrics=compute_metrics, eval_network=eval_network)
        else:
            model = Model(network, optimizer=optimizer, metrics=compute_metrics, eval_network=eval_network)

        # resume checkpoint
        if config.load_checkpoint or config.only_save_strategy:
            if config.resume_training:
                logger.info(".............Start resume training from checkpoint..................")
                transform_and_load_checkpoint(config, model, network, dataset, optimizer=optimizer)
            else:
                transform_and_load_checkpoint(config, model, network, dataset)

        # build evaluate in training
        if config.do_eval:
            logger.info(".........Build Evaluate in Training Callback..........")
            eval_dataset = self.create_eval_dataset()
            logger.info("Create evaluate dataset finish, dataset size:%d", eval_dataset.get_dataset_size())

            eval_callback = EvalCallBack(
                partial(
                    self._evaluate_in_training,
                    model=model,
                    eval_dataset=eval_dataset,
                ),
                step_interval=config.eval_step_interval if config.eval_step_interval else 100,
                epoch_interval=config.eval_epoch_interval if config.eval_epoch_interval else -1,
            )
            callbacks.append(eval_callback)

        if config.moe_config.enable_cold_hot_expert:
            save_checkpoint_steps = -1
            for callback in config.callbacks:
                if callback['type'] == 'CheckpointMonitor':
                    save_checkpoint_steps = callback['save_checkpoint_steps']
            cold_hot_mointor = ColdHotExpertMointor(
                moe_config=config.moe_config,
                hidden_size=config.model.model_config.hidden_size,
                ffn_hidden_size=config.model.model_config.ffn_hidden_size,
                expert_parallel=config.parallel_config.expert_parallel,
                model_parallel=config.parallel_config.model_parallel,
                save_checkpoint_steps=save_checkpoint_steps)
            # ColdHotExpertMointor needs to be placed before CheckpointMonitor
            callbacks.insert(1, cold_hot_mointor)

        logger.info(".........Starting Training Model..........")
        if get_real_rank() % 8 == 0:
            pprint(config)
        logger.info(".........Model Compiling, Please Wait a Moment...........")
        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.sink_size,
                    initial_epoch=config.runner_config.initial_epoch)
        logger.info(".........Training Over!.............")

    def evaluate_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Evaluate for BaseTrainer in MindFormers."""
        self.eval_dataset = dataset if dataset else self.eval_dataset
        metric_name = kwargs.get("metric_name")
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)

        # build dataset
        logger.info(".........Build Dataset For Evaluate..........")
        dataset = self.create_eval_dataset()
        self.set_eval_dataset(dataset)
        logger.info("Create evaluate dataset finish, dataset size:%d", dataset.get_dataset_size())

        # check rules
        check_rules(config, mode='eval', network=network, dataset=dataset, task=self.task)

        # build network
        if network is None and self.network is None:
            network = self.create_network(
                default_args={"parallel_config": config.parallel_config,
                              "moe_config": config.moe_config})
        elif network is None and self.network is not None:
            logger.info(".........Using The Existing Network For Evaluate: %s", self.network.__class__.__name__)
            network = self.network

        self.set_network(network, is_train=False)
        self.count_parameters()

        # build metric
        logger.info(".........Build Compute Metrics For Evaluate..........")
        if compute_metrics is None:
            compute_metrics = self.create_metrics(metric_name=metric_name)

        # build callback
        logger.info(".........Build Callbacks For Evaluate..........")
        default_callbacks = self.create_eval_callbacks()
        if callbacks is not None:
            if isinstance(callbacks, list):
                default_callbacks.extend(callbacks)
            if isinstance(callbacks, Callback):
                default_callbacks.append(callbacks)
        callbacks = default_callbacks

        logger.info(".........Starting Init Evaluate Model..........")
        model = Model(network, metrics=compute_metrics, eval_network=network)

        if config.load_checkpoint or config.only_save_strategy:
            logger.info(".............Start load checkpoint for eval..................")
            transform_and_load_checkpoint(config, model, network, next(dataset.create_tuple_iterator()), do_eval=True)

        logger.info(".........Starting Evaluate Model..........")
        if get_real_rank() % 8 == 0:
            pprint(config)
        output = model.eval(dataset,
                            callbacks=callbacks,
                            dataset_sink_mode=config.runner_config.sink_mode)
        logger.info('%s = %s', metric_name, str(output))
        logger.info(".........Evaluate Over!.............")

    def predict_process(self,
                        config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                        input_data: Optional[Union[GeneratorDataset,
                                                   Tensor, np.ndarray, Image, str, list]] = None,
                        task: str = None,
                        network: Optional[Union[Cell, PreTrainedModel]] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None,
                        image_processor: Optional[BaseImageProcessor] = None,
                        audio_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """Predict for BaseTrainer in MindFormers."""
        if not self.pipeline_task:
            is_full_config = kwargs.get("is_full_config", False)
            config = self.set_config(config, is_full_config)

            # check rules
            check_rules(config, mode='predict', network=network, task=self.task)

            if ms.context.get_auto_parallel_context('parallel_mode') in \
                    ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
                if task not in ["translation", "text_generation", "multi_modal_to_text_generation"]:
                    raise ValueError("Currently distributed predict only support translation and text_generation. "
                                     "Process exit!")
                if config.parallel_config.pipeline_stage > 1:
                    raise ValueError("Currently distributed predict dose not support pipeline parallel. "
                                     "Process exit!")

            # build network
            if network is None:
                network = self.create_network(
                    default_args={"parallel_config": config.parallel_config,
                                  "moe_config": config.moe_config})
            self.set_network(network, is_train=False)
            self.count_parameters()

            if tokenizer is None and config.processor.tokenizer:
                tokenizer = build_tokenizer(config.processor.tokenizer, tokenizer_name=config.trainer.model_name)

            if image_processor is None and config.processor.image_processor:
                image_processor = build_processor(config.processor.image_processor)

            if audio_processor is None and config.processor.audio_processor:
                audio_processor = build_processor(config.processor.audio_processor)

            model = Model(network)

            if config.load_checkpoint or config.only_save_strategy:
                if ms.context.get_auto_parallel_context('parallel_mode') in \
                        ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
                    if network.config:
                        batch_size = network.config.batch_size
                        seq_length = network.config.seq_length
                    else:
                        batch_size = config.model.model_config.batch_size
                        seq_length = config.model.model_config.seq_length
                    input_ids = np.ones(shape=tuple([batch_size, seq_length]))
                    infer_data = network.prepare_inputs_for_predict_layout(input_ids)
                    transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)
                else:
                    transform_and_load_checkpoint(config, model, network, None, do_predict=True)

            self.pipeline_task = pipeline(
                task=task,
                model=network,
                tokenizer=tokenizer,
                image_processor=image_processor,
                audio_processor=audio_processor,
                **kwargs
            )

        top_k = kwargs.pop("top_k", None)
        if top_k is None:
            if config.top_k is not None:
                top_k = config.top_k
            elif config.model and config.model.model_config.top_k:
                top_k = config.model.model_config.top_k
            else:
                top_k = 1

        save_file = kwargs.pop("save_file", None)
        if save_file is None:
            if config.save_file is not None:
                save_file = config.save_file
            else:
                save_file = f"{task}_result.txt"

        if get_real_rank() % 8 == 0:
            pprint(config)
        output_results = self.pipeline_task(input_data, top_k=top_k)

        output_info = []

        for one_output in output_results:
            if isinstance(one_output, dict) and "info" in one_output:
                output_info.append(one_output["info"])
            else:
                output_info.append(one_output)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(save_file, flags_, 0o750), 'w') as file:
            file.write(str(output_info))
        file.close()

        logger.info("output result is: %s", str(output_info))
        logger.info("output result is saved at: %s", save_file)
        logger.info(".........Predict Over!.............")
        return output_results

    def _evaluate_in_training(self, model: Model, eval_dataset: BaseDataset):
        origin_phase = model.eval_network.phase
        model.eval_network.set_train(False)
        output = model.eval(
            eval_dataset, dataset_sink_mode=self.config.runner_config.sink_mode
        )
        model.eval_network.set_train(origin_phase)
        return output
