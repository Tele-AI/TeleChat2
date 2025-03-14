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
"""Trainer API For Import."""
import os
import copy
from pathlib import Path
from collections import OrderedDict
from collections.abc import Iterable
from typing import List, Optional, Union, Callable

import numpy as np
from PIL.Image import Image

import mindspore as ms
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore._checkparam import args_type_check
from mindspore import load_checkpoint, load_param_into_net
from mindspore.nn import Optimizer, Cell
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import BatchDataset, RepeatDataset, Dataset

from mindformers.core.parallel_config import build_parallel_config, \
    reset_parallel_config
from mindformers.core.callback.callback import ProfileMonitor
from mindformers.dataset import build_dataset, build_dataset_loader, \
    check_dataset_config, BaseDataset
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor, \
    PreTrainedTokenizerBase, BaseAudioProcessor
from mindformers.models.utils import WEIGHTS_NAME
from mindformers.tools.utils import (
    set_output_path,
    set_strategy_save_path,
    check_in_modelarts,
    get_real_rank,
    get_real_group_size,
    set_remote_save_url,
    get_output_root_path,
    get_device_num_per_node,
    is_publicly_accessible_path,
    clear_auto_trans_output,
    try_sync_file
)
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.register.config import ordered_yaml_dump
from mindformers.tools.resume_ckpt import get_resume_checkpoint
from mindformers.tools.download_tools import download_with_progress_bar
from mindformers.version_control import check_tft_valid
from .build_trainer import build_trainer
from .training_args import TrainingArguments
from .utils import config2dict, get_last_checkpoint

__all__ = ['Trainer']

PREFIX_CHECKPOINT_DIR = "checkpoint"
SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
SUPPORT_CHECKPOINT_NAMES = MindFormerBook().get_downloadable_model_name_list()
SUPPORT_PIPELINE_INPUT_DATA = MindFormerBook().get_pipeline_support_input_data_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CHECKPOINT_DIR = 'checkpoint'
DEFAULT_CONFIG_DIR = 'configs'


class Trainer:
    r"""
    Executor of general task trainers. It can initialize a trainer instance of the specific task through the task name
    and configuration file. It provides users with the ability to implement different tasks by encapsulating
    the train, finetune, evaluate and predict of trainer instance. It also allows users to customize
    the model, optimizer, dataset, tokenizer, processor, train_one_step, callback, and metric.

    You can initialize the Trainer using the following method:

    1. Define the `task` and `model_name` , for example, task='text_generation', model_name='gpt2'.
       By specifying the correct `task` and `model_name` , the corresponding YAML file will be found from
       MindFormerBook, and it will be read as the task configuration.
    2. Define the `task` and `model` , for example, task='text_generation', model='gpt2'.
       The `model` can be either a model instance or a model name.
       If the `model` is a model name, it will override the `model_name` .
    3. Define the `task` , `model_name` and `model` , note that the `model` is a model instance now.
    4. Define the `args` as an instance of MindFormerConfig or yaml path.
       You can also pass a model instance through the `model` parameter.
       Otherwise, the model will be initialized through the `args` configuration.
    5. Define the `args` as an instance of TrainingArguments and the `model` as a model instance.
    6. Define the `args` as an instance of TrainingArguments and just define the `task` and `model_name` .
       In this case, you needn't pass in a model instance, the model will be initialized through the
       YAML configuration obtained from `task` and `model_name` .

    Note:
        1. If you simultaneously pass in `args` , `task` , and `model_name` ,
           the task configuration will take precedence over `args` .
           The YAML configuration obtained from `task` and `model_name` will be overridden by `args` .
        2. If you use the Trainer.predict for inference, the `task` is needed.

    Args:
        args (Union[str, MindFormerConfig, TrainingArguments], optional):
            The task config which is used to configure the dataset, the hyperparameter, optimizer, etc.
            It supports yaml path, MindFormerConfig or TrainingArguments class.
            Default: ``None``.
        task (str, optional): Supported task type. Default: ``general``.
        model (Union[str, PreTrainedModel], optional):
            The network for trainer. It supports model name supported or PreTrainedModel. Default: ``None``.
        model_name (str, optional):
            Supported model name. When the incoming model is a custom instance, it is recommended to specify
            the supported model_name to get the base configuration of the model type.
            Default: ``None``.
        tokenizer (PreTrainedTokenizerBase, optional):
            The tokenizer for text preprocessing. It supports PreTrainedTokenizerBase class.
            Default: ``None``.
        train_dataset (Union[str, BaseDataset, Dataset, Iterable], optional):
            The training dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
            Default: ``None``.
        eval_dataset (Union[str, BaseDataset, Dataset, Iterable], optional):
            The evaluate dataset. It supports real dataset path or BaseDateset class or MindSpore Dataset class.
            Default: ``None``.
        data_collator (Callable, optional):
            Batch data processing function.
            Default: ``None``.
        optimizers (Optimizer, optional):
            The training network's optimizer. It supports Optimizer class of MindSpore.
            Default: ``None``.
        compute_metrics (Union[dict, set], optional):
            The metric of evaluating. It supports dict or set in MindSpore's Metric class.
            Default: ``None``.
        callbacks (Union[Callback, List[Callback]], optional):
            The training callback function. It supports CallBack or CallBack List of MindSpore.
            Default: ``None``.
        eval_callbacks (Union[Callback, List[Callback]], optional):
            The evaluate callback function. It supports CallBack or CallBack List of MindSpore.
            Default: ``None``.
        pet_method (str, optional):
            Supported pet method name.
            Default: ``''``.
        image_processor (BaseImageProcessor, optional):
            The processor for image preprocessing. It supports BaseImageProcessor class.
            Default: ``None``.
        audio_processor (BaseAudioProcessor, optional):
            The processor for audio preprocessing. It supports BaseAudioProcessor class.
            Default: ``None``.
        save_config (bool, optional):
            Save current the config of task.
            Default: ``False``.
        reset_model (bool, optional):
            Reset model instance
            Default: ``False``.

    Returns:
        An instance of Trainer.

    Raises:
        KeyError: If 'task' or 'model' not in supported trainer.

    Examples:
        >>> from mindformers import Trainer
        >>> trainer = Trainer(task="text_generation", model_name='llama2_7b')
        >>> trainer.task
        'text_generation'
        >>> trainer.model_name
        'llama2_7b'
    """

    @args_type_check(
        args=(str, MindFormerConfig, TrainingArguments), task=str, model=(str, PreTrainedModel),
        model_name=str, tokenizer=PreTrainedTokenizerBase, pet_method=str,
        image_processor=BaseImageProcessor, audio_processor=BaseAudioProcessor, optimizers=Optimizer,
        callbacks=(Callback, list), eval_callbacks=(Callback, list), compute_metrics=(dict, set), save_config=bool)
    def __init__(self,
                 args: Optional[Union[str, MindFormerConfig, TrainingArguments]] = None,
                 task: Optional[str] = 'general',
                 model: Optional[Union[str, PreTrainedModel]] = None,
                 model_name: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 train_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None,
                 eval_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None,
                 data_collator: Optional[Callable] = None,
                 optimizers: Optional[Optimizer] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 eval_callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 pet_method: Optional[str] = '',
                 image_processor: Optional[BaseImageProcessor] = None,
                 audio_processor: Optional[BaseAudioProcessor] = None,
                 save_config: bool = False,
                 reset_model: bool = False):
        self.args = args
        self.task = task
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizers = optimizers
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.eval_callbacks = eval_callbacks
        self.pet_method = pet_method
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.reset_model = reset_model
        self.default_checkpoint_name_or_path = None
        self.configs_directory = os.path.join('.', DEFAULT_CONFIG_DIR)

        # check_task_and_model
        if self.task not in SUPPORT_TASKS.keys():
            raise ValueError(
                "The value of task must be in {}, but get {}".format(SUPPORT_TASKS.keys(), self.task))

        if isinstance(self.model, (Cell, PreTrainedModel)):
            logger.info("The model instance has been entered, "
                        "and the model will not be created from model_config")
            if self.pet_method:
                logger.warning("pet_method is not valid when a model instance is passed in."
                               "Currently, only part of the model in MindFormers is supported."
                               "Please pass in the model keyword")
            self.is_model_instance = True
        else:
            self.is_model_instance = False
            if isinstance(self.model, str):
                self.model = self.model + '_{}'.format(self.pet_method) if self.pet_method else self.model
                if self.model not in SUPPORT_MODEL_NAMES:
                    raise ValueError(f"model must be in {SUPPORT_MODEL_NAMES} "
                                     f"when model's type is string, but got {self.model}.")
                if isinstance(self.model_name, str):
                    logger.warning("Detected both the `model` and the `model_name` are set simultaneously, "
                                   "`model_name` will be overridden by `model`.")
                self.model_name = self.model
                self.model = None

        if self.model_name is None:
            if self.is_model_instance:
                logger.warning(
                    "Recognizing that a model instance is sent and model_name is None,")
                logger.warning(
                    "it is recommended to select a model configuration that corresponds "
                    "to the support of MindFormers based on the instance model and set model_name.")
                logger.warning(
                    "Otherwise, they will default to a general configuration."
                    "You are advised to pass instances such as optimizers, metric, tokenizer, and processor")
            self.model_name = 'common'

        if (isinstance(self.args, (str, MindFormerConfig)) or \
            (isinstance(self.args, TrainingArguments) and self.is_model_instance)) and \
                self.task == 'general' and self.model_name != 'common':
            logger.warning("When (`args` is MindFormerConfig) "
                           "or (`args` is TrainingArguments and a model instance is passed), "
                           "The `model_name` is invalid and set to 'common'.")
            self.model_name = 'common'

        self._check_args_task_and_model()

        # config init
        task_config = self.get_task_config(self.task, self.model_name)

        self.config = self._config_init(args, task_config)
        self._reassign_monitor_config()
        # build parallel config
        build_parallel_config(self.config)

        if isinstance(args, TrainingArguments):
            ms.set_auto_parallel_context(pipeline_stages=args.pipeline_stage)

        self.rank_id = get_real_rank()
        self.device_num = get_real_group_size()
        self.config.rank_id = self.rank_id
        self.config.device_num = self.device_num

        # set checkpoint options
        self.config.remove_redundancy = self.config.get('remove_redundancy', False)
        self.config.load_ckpt_format = self.config.get('load_ckpt_format', 'ckpt')

        # set seed
        if self.config.seed and \
                ms.context.get_auto_parallel_context("parallel_mode") \
                not in ["semi_auto_parallel", "auto_parallel"]:
            set_seed(self.config.seed)
            np.random.seed(self.config.seed)

        # set output directory
        set_output_path(self.config.output_dir)
        set_strategy_save_path(self.config.parallel)
        if check_in_modelarts() and self.config.remote_save_url:
            set_remote_save_url(self.config.remote_save_url)
            logger.info(f"Set remote_save_url: %s, the output file will be uploaded to here.",
                        self.config.remote_save_url)

        # build trainer
        self.trainer = build_trainer(self.config.trainer)

        # define profile callback
        self._build_profile_cb()

        # model init
        self._init_model()

        # dataset init
        self._init_dataset()

        # callbacks init
        self._init_callbacks()

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.config.push_to_hub:
            self.init_openmind_repo()

        if save_config:
            self._save_config_to_yaml(self.config)
            logger.info("save running config success of %s_new.", task_config.trainer.model_name.lower())

        logger.info("==========Trainer Init Success!==========")

    def _reassign_monitor_config(self):
        """parse config.monitor_config and supplement settings"""
        if hasattr(self.config, 'monitor_config') and self.config.monitor_config is not None:
            monitor_config = self.config.monitor_config
            self.config.check_for_nan_in_loss_and_grad = monitor_config.get('local_loss_format') is not None
            dump_local_norm = monitor_config.get('local_norm_format') is not None
            dump_device_local_norm = monitor_config.get('device_local_norm_format') is not None
            dump_path = monitor_config.pop('dump_path') or './dump'
            ms.set_auto_parallel_context(
                dump_local_norm_path=dump_path,
                dump_local_norm=dump_local_norm,
                dump_device_local_norm=dump_device_local_norm
            )
            for callback in self.config.callbacks:
                if "type" in callback and callback["type"] == "TrainingStateMonitor":
                    callback['config'] = monitor_config
                    return
            self.config.callbacks.append({"type": "TrainingStateMonitor", "config": monitor_config})

    @args_type_check(train_checkpoint=(str, bool), resume_from_checkpoint=(str, bool),
                     resume_training=(bool, str), auto_trans_ckpt=bool, src_strategy=str,
                     transform_process_num=int, do_eval=bool)
    def train(self,
              train_checkpoint: Optional[Union[str, bool]] = False,
              resume_from_checkpoint: Optional[Union[str, bool]] = None,
              resume_training: Optional[Union[bool, str]] = None,
              ignore_data_skip: Optional[bool] = None,
              data_skip_steps: Optional[int] = None,
              auto_trans_ckpt: Optional[bool] = None,
              src_strategy: Optional[str] = None,
              transform_process_num: Optional[int] = None,
              do_eval: Optional[bool] = False):
        """
        The training API of Trainer. After setting custom settings, implement training by calling the
        training method of task-trainer instance.

        Args:
            train_checkpoint (Union[str, bool], optional):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: ``False``.
            resume_from_checkpoint (Union[str, bool], optional):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                if `train_checkpoint` is passed in, `resume_from_checkpoint` will be overridden.
                Default: ``None``.
            resume_training (Union[bool, str], optional):
                Decide whether to resume training or specify the name of the checkpoint from which to resume training.
                If set to True, the checkpoint recorded in meta.json will be loaded to resume training.
                If a checkpoint name is provided, that specific checkpoint will be loaded for resume training.
                Default: ``None``.
            ignore_data_skip (bool, optional):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have. Default: ``None``.
            data_skip_steps (int, optional):
                Specify the skip steps of train dataset when resume training.
                It only takes effect when `ignore_data_skip` is set to False. Default: ``None``.
            auto_trans_ckpt (bool, optional):
                auto transform checkpoint to load in distributed model. Default: ``None``.
            src_strategy (str, optional):
                The strategy of `load_checkpoint` . Effective only when auto_trans_ckpt is set to True,
                used for automatic checkpoint transform. Default: ``None``.
            transform_process_num (int, optional):
                The number of processes responsible for checkpoint transform. Default: ``None``.
            do_eval (bool, optional):
                Whether evaluations are performed during training. Default: ``False``.

        Raises:
            TypeError: if resume_from_checkpoint is not bool or str type.
        """
        if train_checkpoint is not None and \
                not isinstance(train_checkpoint, (bool, str)):
            raise TypeError(f"train_checkpoint must be one of [None, string, bool], "
                            f"but get {train_checkpoint}")
        if train_checkpoint is False:
            train_checkpoint = None
        else:
            logger.warning("The `train_checkpoint` will be deprecated. "
                           "Please use `resume_from_checkpoint` instead.")
            resume_from_checkpoint = train_checkpoint

        do_eval = do_eval or self.config.do_eval
        if do_eval:
            if self.eval_dataset is None:
                logger.info("do_eval is enabled, building eval_dataset from config.")
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)
            if self.eval_dataset is None:
                raise ValueError(f"if do_eval is true, eval_dataset must be input, "
                                 f"the task {self.task} is not support eval now.")
            # open do_eval for trainer config
            self.config.do_eval = True

        if resume_from_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(resume_from_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = resume_from_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            self.config.model.model_config.checkpoint_name_or_path = None

        if resume_training is not None:
            self.config.resume_training = resume_training
        if ignore_data_skip is not None:
            self.config.ignore_data_skip = ignore_data_skip
        if data_skip_steps is not None:
            self.config.data_skip_steps = data_skip_steps
        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt
        if src_strategy is not None:
            self.config.src_strategy_path_or_dir = src_strategy
        if transform_process_num is not None:
            self.config.transform_process_num = transform_process_num

        self._check_config_type()
        self._check_config_rules()
        self._init_model(is_train=True)
        # if enable_tft is False or remove_redundancy is True, can use record last_ckpt to json
        if self.config.resume_training and (not check_tft_valid() or self.config.remove_redundancy):
            if os.path.isfile(self.config.load_checkpoint) and \
                    isinstance(self.config.resume_training, str):
                logger.warning(f"`resume_training={self.config.resume_training}` is not valid "
                               "when `load_checkpoint` is a file path.")
                self.config.resume_training = True
            elif os.path.isdir(self.config.load_checkpoint):
                self.config.resume_training = get_resume_checkpoint(
                    checkpoint_dir=self.config.load_checkpoint,
                    resume_training=self.config.resume_training,
                    resume_by_meta=not self.config.resume_by_last_timestamp_ckpt,
                    ckpt_format=self.config.load_ckpt_format
                )

        self.config.load_checkpoint = self.get_load_checkpoint(self.config.load_checkpoint)
        self.trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            callbacks=self.callbacks, compute_metrics=self.compute_metrics,
            is_full_config=True)

    @args_type_check(finetune_checkpoint=(str, bool), resume_from_checkpoint=(str, bool),
                     resume_training=(bool, str), auto_trans_ckpt=bool, src_strategy=str,
                     transform_process_num=int, do_eval=bool)
    def finetune(self,
                 finetune_checkpoint: Optional[Union[str, bool]] = False,
                 resume_from_checkpoint: Optional[Union[str, bool]] = None,
                 resume_training: Optional[Union[bool, str]] = None,
                 ignore_data_skip: Optional[bool] = None,
                 data_skip_steps: Optional[int] = None,
                 auto_trans_ckpt: Optional[bool] = None,
                 src_strategy: Optional[str] = None,
                 transform_process_num: Optional[int] = None,
                 do_eval: bool = False):
        """
        The fine-tuning API of Trainer. After setting custom settings, implement fine-tuning by calling the
        training method of task-trainer instance.

        Args:
            finetune_checkpoint (Union[str, bool], optional):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                if resume_training is true, this checkpoint will be used to restore training of the network.
                Default: ``False``.
            resume_from_checkpoint (Union[str, bool], optional):
                Used to restore training or fine-tune the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                if resume_training is true, this checkpoint will be used to restore training of the network.
                if `finetune_checkpoint` is passed in, `resume_from_checkpoint` will be overridden.
                Default: ``None``.
            resume_training (Union[bool, str], optional):
                Decide whether to resume training or specify the name of the checkpoint from which to resume training.
                If set to True, the checkpoint recorded in meta.json will be loaded to resume training.
                If a checkpoint name is provided, that specific checkpoint will be loaded for resume training.
                Default: ``None``.
            ignore_data_skip (bool, optional):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True` , the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have. Default: ``None``.
            data_skip_steps (int, optional):
                Specify the skip steps of train dataset when resume training.
                It only takes effect when `ignore_data_skip` is set to False. Default: ``None``.
            auto_trans_ckpt (bool, optional):
                Auto transform checkpoint to load in distributed model. Default: ``None``.
            src_strategy (str, optional):
                The strategy of `resume_from_checkpoint` . Effective only when auto_trans_ckpt is set to True,
                used for automatic checkpoint transform. Default: ``None``.
            transform_process_num (int, optional):
                The number of processes responsible for checkpoint transform. Default: ``None``.
            do_eval (bool, optional):
                Whether evaluations are performed during training. Default: ``False``.

        Raises:
            TypeError: if load_checkpoint is not bool or str type.
        """
        if finetune_checkpoint is not None and \
                not isinstance(finetune_checkpoint, (bool, str)):
            raise TypeError(f"train_checkpoint must be one of [None, string, bool], "
                            f"but get {finetune_checkpoint}")
        if finetune_checkpoint is False:
            finetune_checkpoint = None
        else:
            logger.warning("The `finetune_checkpoint` will be deprecated. "
                           "Please use `resume_from_checkpoint` instead.")
            resume_from_checkpoint = finetune_checkpoint

        do_eval = do_eval or self.config.do_eval
        if do_eval:
            if self.eval_dataset is None:
                logger.info("do_eval is enabled, building eval_dataset from config.")
                self.eval_dataset = build_dataset(self.config.eval_dataset_task)
            if self.eval_dataset is None:
                raise ValueError(f"if do_eval is true, eval_dataset must be input, "
                                 f"the task {self.task} is not support eval now.")
            # open do_eval for trainer config
            self.config.do_eval = True

        if resume_from_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(resume_from_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = resume_from_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be fine-tune are specified"
                        "by the finetune_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None
            self.config.model.model_config.checkpoint_name_or_path = None

        if resume_training is not None:
            self.config.resume_training = resume_training
        if ignore_data_skip is not None:
            self.config.ignore_data_skip = ignore_data_skip
        if data_skip_steps is not None:
            self.config.data_skip_steps = data_skip_steps
        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt
        if src_strategy is not None:
            self.config.src_strategy_path_or_dir = src_strategy
        if transform_process_num is not None:
            self.config.transform_process_num = transform_process_num

        self._check_config_type()
        self._check_config_rules()
        self._init_model(is_train=True)

        if self.config.resume_training:
            if os.path.isfile(self.config.load_checkpoint) and \
                    isinstance(self.config.resume_training, str):
                logger.warning(f"`resume_training={self.config.resume_training}` is not valid "
                               "when `load_checkpoint` is a file path.")
                self.config.resume_training = True
            elif os.path.isdir(self.config.load_checkpoint):
                self.config.resume_training = get_resume_checkpoint(
                    checkpoint_dir=self.config.load_checkpoint,
                    resume_training=self.config.resume_training,
                    resume_by_meta=not self.config.resume_by_last_timestamp_ckpt,
                    ckpt_format=self.config.load_ckpt_format
                )

        self.config.load_checkpoint = self.get_load_checkpoint(self.config.load_checkpoint)

        self.trainer.train(
            config=self.config, network=self.model,
            dataset=self.train_dataset, optimizer=self.optimizers,
            eval_dataset=self.eval_dataset if do_eval else None,
            callbacks=self.callbacks, compute_metrics=self.compute_metrics,
            is_full_config=True)

    @args_type_check(eval_checkpoint=(str, bool), auto_trans_ckpt=bool, src_strategy=str,
                     transform_process_num=int)
    def evaluate(self,
                 eval_dataset: Optional[Union[str, BaseDataset, Dataset, Iterable]] = None,
                 eval_checkpoint: Optional[Union[str, bool]] = False,
                 auto_trans_ckpt: Optional[bool] = None,
                 src_strategy: Optional[str] = None,
                 transform_process_num: Optional[int] = None,
                 **kwargs):
        """
        The evaluation API of Trainer. After setting custom settings, implement evaluation by calling the
        evaluation method of task-trainer instance.

        Args:
            eval_dataset (Union[str, BaseDataset, Dataset, Iterable], optional):
                Evaluate dataset. Default: ``None``.
            eval_checkpoint (Union[str, bool], optional):
                Used to evaluate the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: ``False``.
            auto_trans_ckpt (bool, optional):
                Auto transform checkpoint to load in distributed model. Default: ``None``.
            src_strategy (str, optional):
                The strategy of `resume_from_checkpoint` . Effective only when auto_trans_ckpt is set to True,
                used for automatic checkpoint transform. Default: ``None``.
            transform_process_num (int, optional):
                The number of processes responsible for checkpoint transform. Default: ``None``.
            kwargs (Any):
                Additional parameters.

        Raises:
            TypeError: if eval_checkpoint is not bool or str type.
        """
        if eval_checkpoint is not None and not isinstance(eval_checkpoint, (bool, str)):
            raise TypeError(f"eval_checkpoint must be one of [None, string, bool], "
                            f"but get {eval_checkpoint}")

        if eval_checkpoint is False:
            eval_checkpoint = None

        if eval_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(eval_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = eval_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be evaluated are specified"
                        "by the eval_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None

        if eval_dataset is not None:
            self.eval_dataset = eval_dataset
            self._init_dataset()
        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt
        if src_strategy is not None:
            self.config.src_strategy_path_or_dir = src_strategy
        if transform_process_num is not None:
            self.config.transform_process_num = transform_process_num

        self._check_config_type()
        self._check_config_rules()
        self._init_model()

        self.config.load_checkpoint = self.get_load_checkpoint(self.config.load_checkpoint)

        self.trainer.evaluate(
            config=self.config, network=self.model,
            dataset=self.eval_dataset, callbacks=self.eval_callbacks,
            compute_metrics=self.compute_metrics, is_full_config=True, **kwargs)

    @args_type_check(predict_checkpoint=(str, bool), auto_trans_ckpt=bool, src_strategy=str,
                     transform_process_num=int, input_data=(GeneratorDataset, Tensor, np.ndarray, Image, str, list),
                     batch_size=int)
    def predict(self,
                predict_checkpoint: Optional[Union[str, bool]] = None,
                auto_trans_ckpt: Optional[bool] = None,
                src_strategy: Optional[str] = None,
                transform_process_num: Optional[int] = None,
                input_data: Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]] = None,
                batch_size: int = None,
                **kwargs):
        """
        The prediction API of Trainer. After setting custom settings, implement prediction by calling the
        prediction method of task-trainer instance.

        Args:
            predict_checkpoint (Union[str, bool], optional):
                Used to predict the weight of the network.
                It supports real checkpoint path or valid model name of mindformers or bool value.
                if it's true, the last checkpoint file saved from the previous training round is automatically used.
                Default: ``None``.
            auto_trans_ckpt (bool, optional):
                Auto transform checkpoint to load in distributed model. Default: ``None``.
            src_strategy (str, optional):
                The strategy of `resume_from_checkpoint` . Effective only when auto_trans_ckpt is set to True,
                used for automatic checkpoint transform. Default: ``None``.
            transform_process_num (int, optional):
                The number of processes responsible for checkpoint transform. Default: ``None``.
            input_data (Union[Tensor, np.ndarray, Image, str, list], optional):
                The predict data. Default: ``None``.
            batch_size (int, optional):
                Batch size of predict data. Default: ``None``.
            kwargs (Any):
                Additional parameters.

        Return:
            predict result (dict).

        Raises:
            TypeError: if predict_checkpoint is not bool or str type.
            TypeError: if input_data is not Tensor, np.ndarray, Image, str or list.
        """
        if predict_checkpoint is not None and not isinstance(predict_checkpoint, (bool, str)):
            raise TypeError(f"predict_checkpoint must be one of [None, string, bool], "
                            f"but get {predict_checkpoint}")

        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if predict_checkpoint is False:
            predict_checkpoint = None

        if predict_checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = self.get_last_checkpoint()
        elif isinstance(predict_checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = None
            self.config.load_checkpoint = predict_checkpoint
        else:
            self.default_checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
            if auto_trans_ckpt:
                if self.is_model_instance:
                    logger.warning(
                        "When a model instance is identified,"
                        "the weights that are currently proposed to be evaluated are specified"
                        "by the eval_checkpoint argument or a model instance "
                        "with the model weights already loaded is imported")
                else:
                    self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                self.config.load_checkpoint = None

        if auto_trans_ckpt is not None:
            self.config.auto_trans_ckpt = auto_trans_ckpt
        if src_strategy is not None:
            self.config.src_strategy_path_or_dir = src_strategy
        if transform_process_num is not None:
            self.config.transform_process_num = transform_process_num

        self._check_config_type()
        self._check_config_rules()
        self._init_model()

        self.config.load_checkpoint = self.get_load_checkpoint(self.config.load_checkpoint)

        if input_data is None:
            input_data = build_dataset_loader(self.config.eval_dataset.data_loader)
            logger.info("dataset by config is used as input_data.")

        if not isinstance(input_data, (GeneratorDataset, BaseDataset, RepeatDataset, BatchDataset, Tensor,
                                       np.ndarray, Image, str, list)):
            raise ValueError("Input data's type must be one of [GeneratorDataset, "
                             "str, ms.Tensor, np.ndarray, PIL.Image.Image]")

        output_result = self.trainer.predict(
            config=self.config, input_data=input_data,
            network=self.model, image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            tokenizer=self.tokenizer,
            is_full_config=True,
            **kwargs)
        return output_result

    def add_callback(self, callback):
        """add callback."""
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class.__name__} to the callbacks of this Trainer, but there is already one.\n"
                f"The current list of callbacks is:\n{self.callback_list}"
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        """pop callback."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb
        return callback

    def remove_callback(self, callback):
        """remove callback."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    @args_type_check(data_parallel=int, model_parallel=int, expert_parallel=int, pipeline_stage=int,
                     micro_batch_interleave_num=int, micro_batch_num=int, use_seq_parallel=bool, optimizer_shard=bool,
                     gradient_aggregation_group=int, vocab_emb_dp=bool)
    def set_parallel_config(
            self, data_parallel=1, model_parallel=1, context_parallel=1, expert_parallel=1, pipeline_stage=1,
            micro_batch_interleave_num=1, micro_batch_num=1, use_seq_parallel=False, optimizer_shard=False,
            gradient_aggregation_group=4, vocab_emb_dp=True):
        """
        set_parallel_config for the setting global data parallel, model parallel and fusion group.
        The parallel configure setting for Trainer.

        Args:
            data_parallel (int):
                The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int):
                The model parallel way. The parameters of dense layers in Multi-head Attention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            context_parallel (int):
                The context parallel way. The sequence length of the input data in Multi-head Attention and
                FeedForward layer will be sliced according to the context parallel way. Default: 1.
            expert_parallel (int):
                The expert parallel way. This is effective only when MoE (Mixture of Experts)
                is applied. This value specifies the number of partitions to split the experts into.
            pipeline_stage (int):
                The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num (int):
                The micro size of the batches for the pipeline training. Default: 1.
            use_seq_parallel (bool):
                Whether to enable sequence parallel. Default False.
            optimizer_shard (bool):
                Whether to enable optimizer shard. Default False.
            gradient_aggregation_group (int):
                The fusion group size of the optimizer state sharding. Default: 4.
            vocab_emb_dp (bool):
                Shard embedding in model parallel or data parallel. Default: True.
            micro_batch_interleave_num (int):
                split num of batch size. Default: 1.

        Returns:
            None
        """
        self.config.parallel_config.data_parallel = data_parallel
        self.config.parallel_config.model_parallel = model_parallel
        self.config.parallel_config.context_parallel = context_parallel
        self.config.parallel_config.expert_parallel = expert_parallel
        self.config.parallel_config.pipeline_stage = pipeline_stage
        self.config.parallel_config.use_seq_parallel = use_seq_parallel
        self.config.parallel_config.optimizer_shard = optimizer_shard
        self.config.parallel_config.micro_batch_num = micro_batch_num
        self.config.parallel_config.vocab_emb_dp = vocab_emb_dp
        self.config.parallel_config.gradient_aggregation_group = gradient_aggregation_group
        self.config.micro_batch_interleave_num = micro_batch_interleave_num

        self.reset_model = True
        logger.info("The incoming model will be reinit when parallel config is reconfigured.")

    @args_type_check(recompute=bool, parallel_optimizer_comm_recompute=bool,
                     select_recompute=bool, mp_comm_recompute=bool,
                     recompute_slice_activation=bool)
    def set_recompute_config(self, recompute=False, parallel_optimizer_comm_recompute=False, select_recompute=False,
                             mp_comm_recompute=True, recompute_slice_activation=False):
        r"""
        Set recompute config.

        Args:
            recompute (bool):
                Enable recomputation of the transformer block or not. Default: False.
            select_recompute (bool):
                Only Enable recomputation of the attention layer or not. Default: False.
            parallel_optimizer_comm_recompute (bool):
                Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
            mp_comm_recompute (bool):
                Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            recompute_slice_activation (bool):
                Slice the cell output which would remains in memory. Default: False.

        Returns:
            None
        """
        self.config.recompute_config.recompute = recompute
        self.config.recompute_config.select_recompute = select_recompute
        self.config.recompute_config.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self.config.recompute_config.mp_comm_recompute = mp_comm_recompute
        self.config.recompute_config.recompute_slice_activation = recompute_slice_activation

        self.reset_model = True
        logger.info("The incoming model will be reinit when recompute config is reconfigured.")

    @args_type_check(expert_num=int, capacity_factor=float, aux_loss_factor=float, num_experts_chosen=int,
                     expert_group_size=int, group_wise_a2a=bool, comp_comm_parallel=bool, comp_comm_parallel_degree=int)
    def _set_moe_config(self,
                        expert_num=1,
                        capacity_factor=1.1,
                        aux_loss_factor=0.05,
                        num_experts_chosen=1,
                        expert_group_size=None,
                        group_wise_a2a=False,
                        comp_comm_parallel=False,
                        comp_comm_parallel_degree=2):
        """
        Sef the configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int):
                The number of experts employed. Default: 1
            capacity_factor (float):
                The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float):
                The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int):
                The number of experts is chosen by each token, it should not be larger
                than expert_num. Default: 1.
            expert_group_size (int):
                The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            group_wise_a2a (bool):
                Whether to enable group-wise alltoall communication, which can reduce communication
                time by converting part of intercommunication into intra communication. Default: False. This parameter
                is effective only when model parallel > 1 and data_parallel equal to expert parallel.
            comp_comm_parallel (bool):
                Whether to enable ffn compute and communication parallel, which can reduce pure
                communication time by splitting and overlapping compute and communication. Default: False.
            comp_comm_parallel_degree (int):
                The split number of compute and communication. The larger the numbers,
                the more overlap there will be but will consume more memory. Default: 2. This parameter is effective
                only when comp_comm_parallel enable.
        Returns:
            None
        """
        self.config.moe_config.expert_num = expert_num
        self.config.moe_config.capacity_factor = capacity_factor
        self.config.moe_config.aux_loss_factor = aux_loss_factor
        self.config.moe_config.num_experts_chosen = num_experts_chosen
        self.config.moe_config.expert_group_size = expert_group_size
        self.config.moe_config.group_wise_a2a = group_wise_a2a
        self.config.moe_config.comp_comm_parallel = comp_comm_parallel
        self.config.moe_config.comp_comm_parallel_degree = comp_comm_parallel_degree

        self.reset_model = True
        logger.info("The incoming model will be reinit when moe config is reconfigured.")

    def _reset_model_instance(self, is_train=False):
        """Reset model instance for new model config."""
        if not isinstance(self.model, PreTrainedModel):
            raise NotImplementedError("Currently only the integrated model structure in MindFormers is supported.")

        build_parallel_config(self.config)
        model_config = self.model.config
        if self.reset_model or (is_train and hasattr(model_config, 'use_past') and model_config.use_past):
            logger.info("..........Reinit Model..........")
            if is_train and hasattr(model_config, 'use_past') and model_config.use_past:
                model_config.use_past = False
                logger.warning("The `use_past` is set to False and reinit the incoming model.")
            model_config.parallel_config = self.config.parallel_config
            model_config.moe_config = self.config.moe_config
            self.model.__init__(model_config)
            self.reset_model = False

    @staticmethod
    def get_task_config(task, model_name):
        """get task config based on task and model_name."""
        default_config_path = SUPPORT_TASKS.get(task).get(model_name)
        relative_config_path = default_config_path[default_config_path.rfind("configs/"):]
        current_config_path = os.path.join(os.getcwd(), relative_config_path)
        if os.path.exists(current_config_path):
            default_config_path = current_config_path
        config_path = default_config_path
        task_config = MindFormerConfig(config_path)
        logger.info(f"Load configs in {config_path} to build trainer.")
        return task_config

    def get_train_dataloader(self):
        """get train dataloader of mindspore."""
        return build_dataset_loader(self.config.train_dataset.data_loader)

    def get_eval_dataloader(self):
        """get eval dataloader of mindspore."""
        return build_dataset_loader(self.config.eval_dataset.data_loader)

    def get_last_checkpoint(self):
        """get last checkpoint for resuming or finetune."""
        output_folder = self.config.output_dir
        checkpoint_dir = os.path.join(
            output_folder, DEFAULT_CHECKPOINT_DIR, 'rank_{}'.format(self.rank_id))
        return get_last_checkpoint(checkpoint_dir, self.config.load_ckpt_format)

    def get_load_checkpoint(self, checkpoint):
        """get checkpoint path which will be loaded."""
        if not checkpoint:
            return None

        if not isinstance(checkpoint, str):
            raise TypeError(f"checkpoint should be a str, but got {type(checkpoint)}")

        if os.path.exists(checkpoint):
            return checkpoint

        if not checkpoint.startswith("mindspore/") and "/" in checkpoint:
            raise FileNotFoundError("The load_checkpoint must be correct, "
                                    f"but get {checkpoint}")

        if checkpoint not in SUPPORT_CHECKPOINT_NAMES:
            raise ValueError(f"{checkpoint} is not a supported default model"
                             f" or a valid path to checkpoint,"
                             f" please select from {SUPPORT_CHECKPOINT_NAMES}.")

        checkpoint_name = checkpoint
        if checkpoint.startswith('mindspore'):
            # Adaptation the name of checkpoint at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/vit_base_p16"
            checkpoint_name = checkpoint.split('/')[1]
            default_checkpoint_download_folder = os.path.join(
                MindFormerBook.get_xihe_checkpoint_download_folder(),
                checkpoint_name.split('_')[0])
        else:
            # Default the name of checkpoint,
            # the relevant file will be downloaded from the Obs platform.
            # such as "vit_base_p16"
            default_checkpoint_download_folder = os.path.join(
                MindFormerBook.get_default_checkpoint_download_folder(),
                checkpoint.split("_")[0])

        if not os.path.exists(default_checkpoint_download_folder) and not get_real_rank():
            os.makedirs(default_checkpoint_download_folder, exist_ok=True)
        while True:
            if os.path.exists(default_checkpoint_download_folder):
                break

        ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name + ".ckpt")
        if not os.path.exists(ckpt_file):
            url = MindFormerBook.get_model_ckpt_url_list()[checkpoint][0]
            succeed = download_with_progress_bar(url, ckpt_file)
            if not succeed:
                logger.info("checkpoint download failed, and pretrained weights are unloaded.")
                return None
        try_sync_file(ckpt_file)
        return ckpt_file

    def _config_init(self,
                     args: Optional[Union[str, MindFormerConfig, TrainingArguments]] = None,
                     task_config: dict = None):
        """init config from args"""
        logger.info("..........Init Config..........")
        if task_config is None:
            logger.warning(
                "The `task_config` is detected as `None`, "
                "`task_config` will be initialized to a generic configuration.")
            config_path = SUPPORT_TASKS.get('general').get('common')
            task_config = MindFormerConfig(config_path)
        if args is None:
            return task_config

        if isinstance(args, MindFormerConfig):
            config_path = None
            task_config = args
        elif isinstance(args, str):
            if not (os.path.realpath(args) and os.path.exists(args)):
                raise ValueError(f"config path must be exist, but get {args}.")
            if not args.endswith(('.yaml', '.yml')):
                raise ValueError(f"config file must be end with .yaml or .yml, but get {args}.")
            config_path = args
            logger.info(f"Load configs in {config_path} to build trainer.")
            task_config = MindFormerConfig(config_path)
        elif isinstance(args, TrainingArguments):
            logger.warning(
                "When using the TrainingArguments class, "
                "its arguments will override the default config configuration.")
            args.convert_args_to_mindformers_config(task_config)

        return task_config

    def _build_profile_cb(self):
        """build profile callback from config."""
        if self.config.profile:
            sink_size = self.config.runner_config.sink_size
            sink_mode = self.config.runner_config.sink_mode
            if sink_mode:
                if self.config.profile_start_step % sink_size != 0:
                    self.config.profile_start_step -= self.config.profile_start_step % sink_size
                    self.config.profile_start_step = max(self.config.profile_start_step, sink_size)
                    logger.warning("profile_start_step should divided by sink_size, \
                        set profile_start_step to %s", self.config.profile_start_step)
                if self.config.profile_stop_step % sink_size != 0:
                    self.config.profile_stop_step += self.config.profile_stop_step % sink_size
                    self.config.profile_stop_step = max(self.config.profile_stop_step, \
                                                        self.config.profile_start_step + sink_size)
                    logger.warning("profile_stop_step should divided by sink_size, \
                        set profile_stop_step to %s", self.config.profile_stop_step)

            start_profile = self.config.init_start_profile
            profile_communication = self.config.profile_communication
            profile_cb = ProfileMonitor(
                start_step=self.config.profile_start_step,
                stop_step=self.config.profile_stop_step,
                start_profile=start_profile,
                profile_rank_ids=self.config.profile_rank_ids,
                profile_pipeline=self.config.profile_pipeline,
                profile_communication=profile_communication,
                profile_memory=self.config.profile_memory,
                output_path=self.config.profile_output,
                profiler_level=self.config.profiler_level,
                with_stack=self.config.with_stack,
                data_simplification=self.config.data_simplification,
                config=self.config)
            logger.warning(
                "Please reduce the data sample size with 'num_samples' in MindSpore data format according to "
                "https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html.")
            logger.warning("In profiler mode, auto-tune will be turned off.")
            self.config.auto_tune = False
            self.config.profile_cb = profile_cb

    def _init_model(self, is_train=False):
        """init network"""
        logger.info("..........Init Model..........")
        self.is_model_instance = False
        if isinstance(self.model, (Cell, PreTrainedModel)):
            self.is_model_instance = True
        else:
            if self.config.model is None:
                raise ValueError("When `model` is not instance, `self.config.model` must not be None.")

        if self.is_model_instance and (self.reset_model or is_train):
            self._reset_model_instance(is_train)

    def _init_tokenizer(self):
        """init tokenizer"""
        if self.tokenizer is not None:
            logger.info("..........Init Tokenizer..........")
            if self.config.train_dataset is not None:
                self.config.train_dataset.tokenizer = self.tokenizer
            if self.config.eval_dataset is not None:
                self.config.eval_dataset.tokenizer = self.tokenizer

    def _init_dataset(self):
        """init dataset"""
        if isinstance(self.train_dataset, str):
            logger.info("..........Init Train Dataset..........")
            if not os.path.exists(self.train_dataset):
                raise ValueError(f"train dataset path must be exist, but got {self.train_dataset}.")
            self.config.train_dataset.data_loader.dataset_dir = self.train_dataset
            self.train_dataset = None

        if isinstance(self.eval_dataset, str):
            logger.info("..........Init Eval Dataset..........")
            if not os.path.exists(self.eval_dataset):
                raise ValueError(f"eval dataset path must be exist, but got {self.eval_dataset}.")
            self.config.eval_dataset.data_loader.dataset_dir = self.eval_dataset
            self.eval_dataset = None
        check_dataset_config(self.config)

    def _init_callbacks(self):
        """Init callbacks."""
        callbacks = []
        if self.callbacks is not None:
            logger.info("..........Init Callbacks..........")
            if isinstance(self.callbacks, list):
                for callback in self.callbacks:
                    callback = callback() if isinstance(callback, type) else callback
                    if not isinstance(callback, Callback):
                        raise ValueError(f"The callback must be an instance of the Callback class, but got {callback}.")
                    callbacks.append(callback)
            elif isinstance(self.callbacks, Callback):
                callbacks.append(self.callbacks)
            else:
                raise ValueError("The callback must be an instance of the Callback class, "
                                 f"but get {self.callbacks}")

        self.callbacks = callbacks

        if self.eval_callbacks is not None:
            logger.info("..........Init Eval Callbacks..........")
            logger.warning("`eval_callbacks` might be deprecated in the future, "
                           "prefer using `callbacks` uniformly.")
            if isinstance(self.eval_callbacks, list):
                for callback in self.eval_callbacks:
                    self.add_callback(callback)
            elif isinstance(self.eval_callbacks, Callback):
                self.add_callback(self.eval_callbacks)
            else:
                raise ValueError("The callback must be an instance of the Callback class, "
                                 f"but get {self.eval_callbacks}")

    def init_openmind_repo(self):
        """
        Initializes a git repo in `self.config.hub_model_id`.
        """
        from modelfoundry_hub import create_repo
        if self.config.rank_id:
            return

        if self.config.hub_model_id is None:
            repo_name = Path(self.config.output_dir).absolute().name
        else:
            repo_name = self.config.hub_model_id

        repo_url = create_repo(repo_name, token=self.config.hub_token,
                               private=self.config.hub_private_repo, exist_ok=True)

        self.hub_model_id = repo_url.repo_id

    def save_model(self, output_dir: Optional[str] = None, internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = get_output_root_path()

        self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save checkpoint, tokenizer and config."""
        output_dir = output_dir if output_dir is not None else get_output_root_path()
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.trainer.network is not None:
            network = self.trainer.network
        else:
            network = self.model

        supported_classes = (PreTrainedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(network, supported_classes):
            if state_dict is None:
                state_dict = {}
                for item in network.get_parameters():
                    state_dict[item.name] = item.data
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            ms.save_checkpoint(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            network.save_pretrained(output_dir, state_dict=state_dict, save_json=True)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir, save_json=True)

        self._save_config_to_yaml(self.config, config_dir=output_dir)

    def _save_config_to_yaml(self, config: dict = None, config_dir: Optional[str] = None):
        """Save now config file to yaml file."""
        if config is None:
            config = self.config
        model_name = self.config.trainer.model_name
        config_dict = _reset_config_for_save(config)
        if config_dir is None:
            config_dir = os.path.join(
                self.configs_directory, model_name.lower() + '_new')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        run_yaml_path = os.path.join(config_dir, 'run_{}.yaml'.format(model_name.lower()))

        _save_config_to_yaml(run_yaml_path, config_dict)

    def _load_model_checkpoint(self):
        """Load model checkpoint to network."""
        checkpoint_name_or_path = self.config.model.model_config.checkpoint_name_or_path
        if checkpoint_name_or_path is None:
            logger.warning("checkpoint_name_or_path is None, not load input checkpoint.")
        elif isinstance(checkpoint_name_or_path, str):
            is_exist_path = os.path.exists(checkpoint_name_or_path)
            is_checkpoint_name = checkpoint_name_or_path in SUPPORT_MODEL_NAMES
            if is_exist_path:
                logger.info("now input valid checkpoint path, it will load to network.")
                checkpoint_dict = load_checkpoint(checkpoint_name_or_path)
                not_load_params = load_param_into_net(self.model, checkpoint_dict)
                logger.info("not load parameters is: %s", str(not_load_params))
            elif is_checkpoint_name:
                logger.info("now input valid checkpoint name, it will load to network.")
                if isinstance(self.model, (Cell, PreTrainedModel)):
                    self.model.load_checkpoint(self.config.model.model_config)
                else:
                    logger.warning("model must be PreTrainedModel or Cell type, but get %s", type(self.model))
            else:
                logger.warning("input checkpoint args is invalid, "
                               "it must be valid and real checkpoint path or a valid checkpoint name,"
                               "but get %s", checkpoint_name_or_path)
        else:
            raise TypeError(f"checkpoint_name_or_path type error, "
                            f"it should be one of [None, str], "
                            f"but get {type(checkpoint_name_or_path)}")

    def _check_checkpoint_config(self, checkpoint: Optional[Union[str, bool]] = None):
        """Check checkpoint config."""
        if checkpoint is True:
            self.config.model.model_config.checkpoint_name_or_path = self.get_last_checkpoint()
        elif isinstance(checkpoint, str):
            self.config.model.model_config.checkpoint_name_or_path = checkpoint
        else:
            if self.default_checkpoint_name_or_path is not None:
                self.config.model.model_config.checkpoint_name_or_path = self.default_checkpoint_name_or_path

    def _check_config_type(self):
        """Check config type."""
        if self.config.resume_training is not None and not isinstance(self.config.resume_training, (bool, str)):
            raise TypeError(f"resume_training must be bool or str, "
                            f"but get {self.config.resume_training}")
        if self.config.auto_trans_ckpt is not None and not isinstance(self.config.auto_trans_ckpt, bool):
            raise TypeError(f"auto_trans_ckpt must be bool, "
                            f"but get {self.config.auto_trans_ckpt}")
        if isinstance(self.config.metric, dict):
            self.config.metric = [self.config.metric]

    def _check_config_rules(self):
        """Check config rules."""
        if self.config.auto_trans_ckpt and self.config.load_ckpt_format == 'ckpt':
            if not is_publicly_accessible_path(get_output_root_path()):
                raise ValueError(f"When device num > {get_device_num_per_node()} and auto_trans_ckpt is set to True,"
                                 "the output_dir should be a shared directory that can be accessed by all nodes."
                                 f"but {os.path.abspath(self.config.output_dir)} is not a shared directory.")
            clear_auto_trans_output()

        if (self.config.auto_trans_ckpt or self.config.resume_training) and not self.config.load_checkpoint:
            if self.config.model and self.config.model.model_config.checkpoint_name_or_path:
                self.config.load_checkpoint = self.config.model.model_config.checkpoint_name_or_path
                self.config.model.model_config.checkpoint_name_or_path = None
            else:
                raise ValueError("when `auto_trans_ckpt` or `resume_training` is True, "
                                 "the `load_checkpoint` should not be empty string or None."
                                 "If you are using TrainingArguments, `resume_from_checkpoint` should be set.")
        if self.config.load_checkpoint and self.config.model \
                and self.config.model.model_config.checkpoint_name_or_path:
            self.config.model.model_config.checkpoint_name_or_path = None
            logger.info("The `load_checkpoint` is set, the `checkpoint_name_or_path` will be set to None.")

        if isinstance(self.config.resume_training, str) and \
                self.config.load_checkpoint and os.path.isfile(self.config.load_checkpoint):
            logger.warning(f"`resume_training={self.config.resume_training}` is not valid "
                           "when `load_checkpoint` is a file path")
            self.config.resume_training = True

    def _check_args_task_and_model(self):
        """Check args, task and model."""
        # get support model names of task
        model_name_support_list = list(MindFormerBook().get_model_name_support_list_for_task(self.task))
        model_name_support_list.sort()
        # if task is not general, model_name should be supported by task
        if self.task != 'general' and self.model_name not in model_name_support_list:
            raise ValueError(f"The `model_name`={self.model_name} is not support in task: {self.task},\n"
                             f"Support model name of {self.task}: {model_name_support_list}.")

        if isinstance(self.args, (str, MindFormerConfig)) or \
                (isinstance(self.args, TrainingArguments) and self.is_model_instance):
            return

        if self.task == 'general':
            if self.model_name != 'common':
                # task is not defined but model name is defined, raise error
                task_name_support_list = []
                for task in list(SUPPORT_TASKS.keys()):
                    if self.model_name in list(MindFormerBook().get_model_name_support_list_for_task(task)):
                        task_name_support_list.append(task)
                task_name_support_list.sort()
                raise ValueError(f"The `task` is needed, \
                    please select an appropriate task from {task_name_support_list}.")
            if self.args is None:
                if self.is_model_instance:
                    # only model instance is defined, need train args.
                    raise ValueError("The `args` is needed, it could be an instance of "
                                     "`TrainingArguments` or `MindFormerConfig` or `yaml path`.")
                raise ValueError("Neither `task`, `model`, `model_name`, nor `args` are configured.\n")
            if isinstance(self.args, TrainingArguments) and not self.is_model_instance:
                # only train args is defined, need model instance.
                raise ValueError("A model instance is needed, which is passed through the `model` parameter.")
        elif self.model_name == 'common':
            if not self.is_model_instance:
                # only task is defined, need model_name by raise error.
                raise ValueError("A model name is needed, which is passed through the `model` or `model_name`.\n"
                                 f"Support model name of {self.task}: {model_name_support_list}.")
            if self.args is None:
                # only task and model instance is defined, need model_name by warning.
                logger.warning("\n===================================================================\n"
                               "Note that the `model_name` is not passed and it is defined as 'common'. "
                               "You'd better choose a suitable model name, otherwise you may end up using "
                               "an inappropriate YAML file as the task configuration.\n"
                               f"Support model name of {self.task}: {model_name_support_list}.\n"
                               "===================================================================\n")
                return
        return

    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True) -> str:
        """
        Upload `self.model` and `self.tokenizer` to the model hub on the repo `self.args.hub_model_id`.

        Args:
            commit_message (Optional[str]):
                Message to commit while pushing, defaults to "End of training".
            blocking (Optional[bool]):
                Whether the function should return only when the `git push` has finished, default is True

        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """
        from modelfoundry_hub import upload_folder
        if self.hub_model_id is None:
            self.init_openmind_repo()

        self.save_model(internal_call=True)

        if self.config.rank_id:
            return None

        # Wait for the current upload to be finished.
        return upload_folder(
            repo_id=self.hub_model_id,
            folder_path=get_output_root_path(),
            commit_message=commit_message,
            token=self.config.hub_token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*", "transformed_checkpoint"],
        )


def _save_config_to_yaml(save_file_path: str = None, save_config: dict = None):
    """
    Save config to yaml file.

    Args:
        save_file_path (str): The real path to save yaml file. Default: None.
        save_config (dict): The task config. Default: None.

    Returns:
        None
    """
    if save_config is None:
        save_config = {}
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(save_file_path, flags_, 0o750), 'w', encoding='utf-8') as file_pointer:
        file_pointer.write(
            ordered_yaml_dump(
                save_config,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False))


def _reset_config_for_save(config: dict = None):
    """
    Reset config according to the yaml file number.

    Args:
        config (dict): The task config. Default: None.

    Returns:
        the reset config
    """
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    reset_parallel_config(config)

    config_dict = OrderedDict()

    if config.get('model') is not None:
        model_config = config2dict(config.pop('model'))
        config_dict.setdefault('model', model_config)

    if config.get('processor') is not None:
        processor_config = config2dict(config.pop('processor'))
        config_dict.setdefault('processor', processor_config)

    if config.get('train_dataset_task') is not None and config.get('train_dataset') is not None:
        train_dataset_config = config2dict(config.pop('train_dataset'))
        train_dataset_task_config = config2dict(config.pop('train_dataset_task'))
        config_dict.setdefault('train_dataset', train_dataset_config)
        config_dict.setdefault('train_dataset_task', train_dataset_task_config)

    if config.get('eval_dataset_task') is not None and config.get('eval_dataset') is not None:
        eval_dataset_config = config2dict(config.pop('eval_dataset'))
        eval_dataset_task_config = config2dict(config.pop('eval_dataset_task'))
        config_dict.setdefault('train_dataset', eval_dataset_config)
        config_dict.setdefault('train_dataset_task', eval_dataset_task_config)

    if config.get('context') is not None:
        context_config = config2dict(config.pop('context'))
        parallel_context_config = config2dict(config.pop('parallel'))
        moe_config = config2dict(config.pop('moe_config'))
        recompute_config = config2dict(config.pop('recompute_config'))
        parallel_config = config2dict(config.pop('parallel_config'))
        config_dict.setdefault('context', context_config)
        config_dict.setdefault('parallel', parallel_context_config)
        config_dict.setdefault('moe_config', moe_config)
        config_dict.setdefault('recompute_config', recompute_config)
        config_dict.setdefault('parallel_config', parallel_config)

    if config.get('runner_config') is not None:
        runner_config = config2dict(config.pop('runner_config'))
        config_dict.setdefault('runner_config', runner_config)

    if config.get('runner_wrapper') is not None:
        wrapper_config = config2dict(config.pop('runner_wrapper'))
        config_dict.setdefault('runner_wrapper', wrapper_config)

    if config.get('optimizer') is not None:
        optim_config = config2dict(config.pop('optimizer'))
        config_dict.setdefault('optimizer', optim_config)

    if config.get('lr_schedule') is not None:
        lr_config = config2dict(config.pop('lr_schedule'))
        config_dict.setdefault('lr_schedule', lr_config)

    if config.get('callbacks') is not None:
        cb_config = config2dict(config.pop('callbacks'))
        config_dict.setdefault('callbacks', cb_config)

    run_config = config2dict(config)
    for key, value in run_config.items():
        config_dict.setdefault(key, value)

    return config_dict
