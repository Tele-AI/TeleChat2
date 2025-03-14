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
"""MindFormer Self-Define Callback."""
import json
import os
import re
import glob
import sys
import time
import tempfile
import datetime
import hashlib

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Callable, Optional, Union
import numpy as np

import mindspore as ms
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import (
    Callback,
    Profiler,
    ModelCheckpoint,
    CheckpointConfig,
    context,
    save_checkpoint,
    Tensor,
    get_auto_parallel_context,
    set_auto_parallel_context
)
from mindspore.train.callback import SummaryCollector
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.train.serialization import _get_merged_param_data
from mindspore.nn.cell import Cell
from mindspore.ops.operations.comm_ops import Broadcast
from mindspore.common import jit
from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy
from mindspore.common.api import flops_collection
from mindspore.communication.management import create_group, get_group_size, get_rank
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.profiler import ProfilerLevel

from mindformers.tools.cloud_adapter.cloud_adapter import Local2ObsMonitor
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import (
    get_output_root_path,
    get_output_subpath,
    get_remote_save_url,
    check_in_modelarts,
    get_real_rank,
    get_real_group_size,
    get_pipeline_rank_ids
)
from mindformers.utils.tensorboard import get_tensorboard_writer, get_tensorboard_args
from mindformers.version_control import check_stress_detect_valid, is_version_ge, check_arf_status

__all__ = ['ObsMonitor', 'MFLossMonitor', 'CheckpointMonitor', 'SummaryMonitor', 'ProfileMonitor', 'EvalCallBack']

_cur_dir = os.getcwd()
SAVE_DIR = _cur_dir

VOLTAGE_ERROR_CODE = 574007


class AllReduceNet(Cell):
    """
    Used to accumulate flops in pipeline parallel.
    """

    def __init__(self, group_name):
        super(AllReduceNet, self).__init__()
        self.allreduce_sum = P.AllReduce(op=P.ReduceOp.SUM, group=group_name)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self, x):
        return self.allreduce_sum(x)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ObsMonitor:
    """
    Obs Monitor For Local and AICC.

    Args:
        src_dir (str): The output path in Local/AICC. Default: None.
        target_dir (str): The remote url to save files. Default: None.
        step_upload_frequence (int): The step interval of uploading. Default: 100.
        epoch_upload_frequence (int): The epoch interval of uploading. Default: -1, means epoch upload is disabled.
        keep_last (bool): Check the consistency of obs files and AICC. Default: True.

    Examples:
        >>> from mindformers.core.callback import ObsMonitor
        >>> monitor = ObsMonitor(src_dir='./root_path', target_dir='./remote_url')
    """

    def __new__(cls,
                src_dir: str = None,
                target_dir: str = None,
                step_upload_frequence: int = 100,
                epoch_upload_frequence: int = -1,
                keep_last: bool = True):
        if src_dir is None:
            src_dir = get_output_root_path()
        if target_dir is None:
            target_dir = get_remote_save_url()
        return Local2ObsMonitor(src_dir, target_dir, step_upload_frequence, epoch_upload_frequence, keep_last)


def _check_nan(loss, local_norm, global_norm):
    """Check if Nan in loss, local/global norm of grad then terminate training"""
    if isinstance(loss, ms.Tensor):
        loss = loss.asnumpy()
        if np.any(np.isnan(loss)):
            raise ValueError(f"loss is {loss}, terminate training.")

    if isinstance(local_norm, ms.Tensor):
        local_norm = local_norm.asnumpy()
        if np.any(np.isnan(local_norm)):
            raise ValueError(f"local_norm is {local_norm}, terminate training.")

    if isinstance(global_norm, ms.Tensor):
        global_norm = global_norm.asnumpy()
        if np.any(np.isnan(global_norm)):
            raise ValueError(f"global_norm is {global_norm}, terminate training.")


def _get_loss_output(output, check_for_nan_in_loss_and_grad=False):
    """Get output of task for MFLossMonitor."""
    overflow = False
    scaling_sens = False
    loss = output
    learning_rate = None
    global_norm = None
    local_norm = None
    if isinstance(output, (tuple, list)):
        if len(output) in [3, 4, 5, 7]:
            loss, overflow, scaling_sens, *res = output
            if len(res) == 4:
                learning_rate, global_norm, local_norm, norm_size = res[0], res[1], res[2], res[3]
                logger.info(f" norm_size: {norm_size}\nlocal_norm:\n{local_norm}")
            if len(res) == 2:
                learning_rate, global_norm = res[0], res[1]
            if len(res) == 1:
                learning_rate = res[0]
            if isinstance(scaling_sens, ms.Tensor):
                scaling_sens = scaling_sens.asnumpy()
        else:
            if isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                loss = output[0]

    # Boundary check.
    if check_for_nan_in_loss_and_grad:
        _check_nan(loss, local_norm, global_norm)

    if isinstance(global_norm, ms.Tensor):
        global_norm = global_norm.asnumpy()

    if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())

    if isinstance(overflow, ms.Tensor):
        overflow = overflow.asnumpy()

    if isinstance(learning_rate, ms.Tensor):
        learning_rate = learning_rate.asnumpy()

    return loss, overflow, scaling_sens, learning_rate, global_norm


def _get_weight_norm(network):
    """Get the L2 norm of network trainable parameters. Return 0 if there's no trainable parameter"""
    norms = []
    for param in network.trainable_params():
        norms.append(param.to(ms.float32).norm())
    if not norms:
        return 0.0
    norm = float(F.stack(norms).norm().item())
    return norm


def _get_optimizer_state(optim_params, filter_fn: Callable = None):
    """Get the respective L2 norms of specified optimizer parameters. Return a dict"""
    norms = {}
    for param in optim_params:
        if filter_fn is None or filter_fn(param.name):
            norms[param.name] = float(param.to(ms.float32).norm().item())
    return norms


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFLossMonitor(Callback):
    """
    Monitor loss and other parameters in training process.

    Args:
        learning_rate (Union[float, LearningRateSchedule], optional): The learning rate schedule. Default: ``None``.
        per_print_times (int, optional): Every how many steps to print the log information. Default: ``1``.
        micro_batch_num (int, optional): MicroBatch size for Pipeline Parallel. Default: ``1``.
        micro_batch_interleave_num (int, optional): split num of batch size. Default: ``1``.
        origin_epochs (int, optional): Training epoches. Default: ``None``.
        dataset_size (int, optional): Training dataset size. Default: ``None``.
        initial_epoch (int, optional): The beginning epoch. Default: ``0``.
        initial_step (int, optional): The beginning step. Default: ``0``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
        gradient_accumulation_steps (int, optional): The gradient accumulation steps. Default: ``1``.
        check_for_nan_in_loss_and_grad (bool, optional): Whether to check loss and norm of grad is Nan.
            Default: ``False``.
        calculate_per_token_loss (bool, optional): Whether to calculate the loss of each token. Default: ``False``.

    Examples:
        >>> from mindformers.core import MFLossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = MFLossMonitor(learning_rate=lr, per_print_times=10)
    """

    def __init__(self,
                 learning_rate: Optional[Union[float, LearningRateSchedule]] = None,
                 per_print_times: int = 1,
                 micro_batch_num: int = 1,
                 micro_batch_interleave_num: int = 1,
                 origin_epochs: int = None,
                 dataset_size: int = None,
                 initial_epoch: int = 0,
                 initial_step: int = 0,
                 global_batch_size: int = 0,
                 gradient_accumulation_steps: int = 1,
                 check_for_nan_in_loss_and_grad: bool = False,
                 calculate_per_token_loss: bool = False):
        super(MFLossMonitor, self).__init__()
        self.per_print_times = per_print_times
        self.learning_rate = deepcopy(learning_rate)
        self.last_print_time = 0
        self.mirco_size = micro_batch_num
        self.print_warning_flag = True
        self.loss_list = []
        self.step_time = time.time()
        self.epoch_time = time.time()
        self.run_context = None
        self.steps_per_epoch = dataset_size
        self.micro_batch_interleave_num = micro_batch_interleave_num
        self.origin_epochs = origin_epochs
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.global_batch_size = global_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_num = get_real_group_size()
        self.mf_support = None
        self.mf_calculated = False
        self.current_phase = None
        self.full_model_flops = 0.0
        self.tensor_writer = get_tensorboard_writer()
        self.tensorboard = get_tensorboard_args()
        self.check_for_nan_in_loss_and_grad = check_for_nan_in_loss_and_grad
        self.calculate_per_token_loss = calculate_per_token_loss

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.loss_list = []
        self.epoch_time = time.time()
        self.run_context = run_context

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context

    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        parallel_mode = get_auto_parallel_context("parallel_mode")
        full_batch = get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        step_seconds = (time.time() - self.step_time) * 1000
        net_outputs = cb_params.net_outputs
        loss, overflow, scaling_sens, learning_rate, global_norm = \
            _get_loss_output(net_outputs, self.check_for_nan_in_loss_and_grad)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        loss = self._fix_loss_for_parallel(loss)
        self.loss_list.append(loss)

        if not overflow:
            overflow = "False"
        if not scaling_sens:
            scaling_sens = "unavailable"

        if self.mf_support is None:
            self.mf_support = self._can_calculate_model_flops(cb_params)
        if (not self.mf_calculated or check_arf_status(cb_params)) and self.mf_support:
            self._calculate_model_flops()

        origin_epochs = self.origin_epochs
        if cb_params.dataset_sink_mode:
            per_step_seconds = step_seconds / cb_params.batch_num
            steps_per_epoch = self.steps_per_epoch
            cur_epoch_num = (cb_params.cur_step_num + self.initial_step - 1) // steps_per_epoch + 1
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1
        else:
            per_step_seconds = step_seconds
            steps_per_epoch = cb_params.batch_num
            cur_epoch_num = cb_params.cur_epoch_num
            cur_step_num = (cb_params.cur_step_num + self.initial_step - 1) % steps_per_epoch + 1

        # compute time remaining
        step_remain = (origin_epochs - cur_epoch_num + 1) * steps_per_epoch - cur_step_num
        time_remain = step_remain * per_step_seconds / 1000

        # compute throughput
        throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)

        # compute percent
        percent = ((cur_epoch_num - 1) * steps_per_epoch + cur_step_num) / origin_epochs / steps_per_epoch * 100

        if (cb_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = cb_params.cur_step_num
            self.print_output_info(cb_params, cur_epoch_num, origin_epochs, throughput,
                                   cur_step_num, steps_per_epoch, loss, per_step_seconds,
                                   overflow, scaling_sens, time_remain, percent, global_norm)

        if check_in_modelarts() and get_real_rank() == get_real_group_size() - 1:
            self.dump_info_to_modelarts(ma_step_num=cur_step_num, ma_loss=loss)

        if auto_parallel:
            set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

    def _fix_loss_for_parallel(self, loss):
        """Fix loss value in pipeline or double parallel mode."""
        pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
        if pipeline_stages > 1 and self.print_warning_flag:
            logger.warning("pipeline stages: %s > 1, the loss on the last card is valid.",
                           pipeline_stages)

        if self.micro_batch_interleave_num > 1 and self.print_warning_flag:
            logger.warning("micro_batch_interleave_num: %s > 1, multiple copies in parallel is open.",
                           self.micro_batch_interleave_num)

        if pipeline_stages > 1 and not self.calculate_per_token_loss:
            loss = loss / self.mirco_size
        if self.micro_batch_interleave_num > 1:
            loss = loss / self.micro_batch_interleave_num
        if self.gradient_accumulation_steps > 1 and not self.calculate_per_token_loss:
            loss = loss / self.gradient_accumulation_steps

        return loss

    def _get_pipeline_group(self):
        """
        Calculate the communication group between all pipeline stages
        """
        rank = get_rank()
        stage_nums = auto_parallel_context().get_pipeline_stages()
        device_nums = get_group_size()
        per_stage_device_nums = device_nums // stage_nums
        local_stage_rank_id = rank % per_stage_device_nums
        group = range(0, stage_nums)
        rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
        rank_str_list = [str(r) for r in rank_list]

        rank_list_str = "-".join(rank_str_list)
        return rank_list, rank_list_str

    def _can_calculate_model_flops(self, cb_params):
        """
        Check whether the model flops can be collected
        """
        if cb_params.mode == 'train':
            network = cb_params.train_network
        elif cb_params.mode == 'eval':
            network = cb_params.eval_network
        else:
            logger.warning('Model Flops computation only support train and eval mode!')
            return False
        if ms.get_context('mode') != ms.GRAPH_MODE:
            logger.warning('Model Flops computation only support graph mode!')
            return False
        if not hasattr(network, 'current_phase'):
            logger.warning('This model dose not support collecting model flops now!')
            return False
        self.current_phase = network.current_phase
        return True

    def _calculate_model_flops(self):
        """
        Calculate the full model flops
        """
        try:
            full_model_flops, _, shard_model_flops, \
                _, is_dynamic_shape = flops_collection(self.current_phase)
        except RuntimeError as e:
            logger.warning("%s", e)
            self.mf_support = False
            return
        if is_dynamic_shape:
            logger.warning("Model Flops computation now do not support dynamic shape.")
            self.mf_support = False
            return
        self.full_model_flops = full_model_flops / 1.0
        self.mf_calculated = True
        if auto_parallel_context().get_pipeline_stages() > 1:
            pipeline_group_list, pipeline_group_name = self._get_pipeline_group()
            hashed = hashlib.md5(
                pipeline_group_name.encode()).hexdigest()[:48]
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)
            self.full_model_flops = AllReduceNet(pipeline_group_name)(
                Tensor([self.full_model_flops])).asnumpy()[0]

        if auto_parallel_context().get_parallel_mode() != "stand_alone":
            self.full_model_flops = self.full_model_flops / get_group_size()

        logger.info("Full model flops is %d, Shard model flops is %d.",
                    full_model_flops, shard_model_flops)

    def print_output_info(self, cb_params, cur_epoch_num, origin_epochs, throughput,
                          cur_step_num, steps_per_epoch, loss, per_step_seconds,
                          overflow, scaling_sens, time_remain, percent, global_norm):
        """print output information."""
        if self.learning_rate is not None:
            if isinstance(self.learning_rate, (float, Tensor, np.ndarray)):
                current_lr = str(self.learning_rate)
            elif isinstance(self.learning_rate, LearningRateSchedule):
                if ms.context.get_context('device_target') == 'CPU':
                    if self.print_warning_flag:
                        logger.warning(
                            "device target not support CPU when generating the learning rate value, "
                            "please use: mindspore.context.set_context(device_target='Ascend')")
                        self.print_warning_flag = False
                    current_lr = None
                else:
                    if cb_params.optimizer is not None:
                        global_step = cb_params.optimizer.global_step
                    else:
                        global_step = cb_params.network.optimizer.global_step

                    # temporary set_train to avoid error on Atlas 800T A2
                    origin_phase = cb_params.train_network.phase
                    cb_params.train_network.set_train(False)
                    current_lr = self.learning_rate(global_step)
                    cb_params.train_network.set_train(origin_phase)

                    current_lr = np.array2string(current_lr.asnumpy())
            else:
                if self.print_warning_flag:
                    logger.warning(
                        "The current learning rate cannot be calculated in real time."
                        "Only the type of LearningRateSchedule is supported in the callback of MFLossMonitor,"
                        "but the input learning rate function type is %s", type(self.learning_rate)
                    )
                    self.print_warning_flag = False
                current_lr = None
        else:
            if self.print_warning_flag:
                logger.warning(
                    "MFLossMonitor callback is not set learning rate arguments."
                    "To display the learning rate, you must input the arguments, "
                    "which can be LearningRateSchedule or a fixed float"
                )
                self.print_warning_flag = False
            current_lr = None

        global_step = cur_step_num + (cur_epoch_num - 1) * steps_per_epoch
        if self.mf_calculated:
            throughput_per_npu = self.full_model_flops / per_step_seconds / 1e9
            throughput_info = ', train_throughput_per_npu: %.3fT' % (throughput_per_npu)
            if self.tensor_writer is not None:
                self.tensor_writer.add_scalar('model-flops-throughput-per-npu',
                                              float(throughput_per_npu),
                                              global_step=global_step)
        else:
            throughput_info = ''

        if current_lr is not None:
            if cb_params.dataset_sink_mode:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss: %5.3f, "
                            "per_step_time: %dms, lr: %s, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss,
                            int(per_step_seconds), current_lr, overflow, scaling_sens, global_norm, throughput_info)
            else:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss:[%5.3f/%5.3f], "
                            "per_step_time: %dms, lr: %s, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                            int(per_step_seconds), current_lr, overflow, scaling_sens, global_norm, throughput_info)
            if self.tensor_writer is not None:
                self.tensor_writer.add_scalar('learning-rate', float(current_lr), global_step=global_step)
                self.tensor_writer.add_scalar('learning-rate vs samples', float(current_lr),
                                              global_step=global_step * self.global_batch_size)
        else:
            if cb_params.dataset_sink_mode:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss: %5.3f, "
                            "per_step_time: %dms, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss,
                            int(per_step_seconds), overflow, scaling_sens, global_norm, throughput_info)
            else:
                logger.info("{ Epoch:[%3d/%3d], step:[%5d/%5d], loss:[%5.3f/%5.3f], "
                            "per_step_time: %dms, overflow cond: %s, loss_scale: %s, global_norm: %s%s",
                            cur_epoch_num, origin_epochs, cur_step_num, steps_per_epoch, loss, np.mean(self.loss_list),
                            int(per_step_seconds), overflow, scaling_sens, global_norm, throughput_info)
        show_str = ('|%%-%ds|' % 50) % (int(50 * percent / 100) * "█")
        logger.info("  %4.1f%% %s %.5f samples/s/p  %s }", percent, show_str, throughput,
                    datetime.timedelta(seconds=int(time_remain)))
        if self.tensor_writer is not None:
            self.tensor_writer.add_scalar('batch-size', self.global_batch_size, global_step=global_step)
            self.tensor_writer.add_scalar('batch-size vs samples', self.global_batch_size,
                                          global_step=global_step * self.global_batch_size)
            self.tensor_writer.add_scalar('loss', loss, global_step=global_step)
            self.tensor_writer.add_scalar('loss vs samples', loss,
                                          global_step=global_step * self.global_batch_size)
            if self.tensorboard.get('log_loss_scale_to_tensorboard', False):
                self.tensor_writer.add_scalar('loss-scale', scaling_sens, global_step=global_step)
                self.tensor_writer.add_scalar('loss-scale vs samples', scaling_sens,
                                              global_step=global_step * self.global_batch_size)
            self.tensor_writer.add_scalar('grad-norm', global_norm, global_step=global_step)
            self.tensor_writer.add_scalar('grad-norm vs samples', global_norm,
                                          global_step=global_step * self.global_batch_size)
            if self.tensorboard.get('log_timers_to_tensorboard', False):
                self.tensor_writer.add_scalar('iteration-time', int(per_step_seconds),
                                              global_step=global_step)
                self.tensor_writer.add_scalar('iteration-time vs samples', int(per_step_seconds),
                                              global_step=global_step * self.global_batch_size)
                self.tensor_writer.add_scalar('throughput', throughput, global_step=global_step)
                seconds_per_day = 86400
                billion_samples_per_day = throughput * get_group_size() * seconds_per_day / 1e9
                self.tensor_writer.add_scalar('B-samples-per-day', billion_samples_per_day, global_step=global_step)
                self.tensor_writer.add_scalar('throughput vs samples', throughput,
                                              global_step=global_step * self.global_batch_size)


    def dump_info_to_modelarts(self, ma_step_num, ma_loss):
        """dump modelarts info to display evaluation result page"""
        ma_loss = float(ma_loss)
        obj = None
        modelarts_dir = os.path.join(get_output_root_path(), "modelarts")
        if not os.path.exists(modelarts_dir):
            os.mkdir(modelarts_dir)
        if not os.path.exists(os.path.join(modelarts_dir, "model_analysis_results.json")):
            obj = {
                "en-us": {
                    "common": {},
                    "precision_performance": {
                        "pr": {
                            "title": "loss", "description": "loss of model", "value": {"current_loss": 0},
                            "line_chart": {
                                "pr_line_chart": {
                                    "name": "loss line chart of model",
                                    "x_axis_name": "step",
                                    "y_axis_name": "loss",
                                    "curve": {"loss": []}}}}},
                    "feature_sensitivity": {},
                    "computational_performance": {},
                    "abstract_feature": {},
                    "adversary": {}
                },
                "zh-cn": {
                    "common": {},
                    "precision_performance": {
                        "pr": {
                            "title": "loss", "description": "模型损失", "value": {"当前loss": 0},
                            "line_chart": {
                                "pr_line_chart": {
                                    "name": "loss line chart of model",
                                    "x_axis_name": "step",
                                    "y_axis_name": "loss",
                                    "curve": {"loss": []}}}}},
                    "feature_sensitivity": {},
                    "computational_performance": {},
                    "abstract_feature": {},
                    "adversary": {}
                }
            }
        else:
            with open(os.path.join(modelarts_dir, "model_analysis_results.json"), "r") as fp:
                obj = json.load(fp)

        if obj is not None:
            en_precision_performance = obj["en-us"]["precision_performance"]
            en_precision_performance["pr"]["value"]["loss_value"] = ma_loss
            en_loss_list = en_precision_performance["pr"]["line_chart"]["pr_line_chart"]["curve"]["loss"]
            en_loss_list.append([ma_step_num, ma_loss])

            zh_precision_performance = obj["zh-cn"]["precision_performance"]
            zh_precision_performance["pr"]["value"]["当前loss"] = ma_loss
            zh_loss_list = zh_precision_performance["pr"]["line_chart"]["pr_line_chart"]["curve"]["loss"]
            zh_loss_list.append([ma_step_num, ma_loss])

            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            file_path = os.path.join(modelarts_dir, "model_analysis_results.json")
            with os.fdopen(os.open(file_path, flags_, 0o750), 'w', encoding="utf8") as fp:
                json.dump(obj, fp)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TrainingStateMonitor(Callback):
    """
    Monitor metrics such as local norm and local loss in training process.

    Args:
        origin_epochs (int): Required. Training epoches.
        config (dict, optional): The config specified how to display metrics. Keys are shown below. Default: ``None``,
        mean that keys will be set as the default values as below.

            - target: Specify the name or regular expression of params to monitor.
              Must be list of str, e.g. ["layers.[01]", "attention"]. Default: ['*'] , all params are selected.

            - invert: Whether to invert `target`, i.e. params in `target` won't be monitored.
              Must be `bool`.  Default: `False`

            - local_norm_format: Determine where to display the local norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Only params specified will be monitored.
              may cause a large amount of print info if 'log' is selected.
              Set to ``None`` to ignore this metric. Default: ``None``.

            - device_local_norm_format: Determine where to display the device local norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric. Default: ``None``.

            - local_loss_format: Determine where to display the local loss.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric.
              Default: ``None``.

            - optimizer_state_format: Determine where to display the optimizer state.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Only the optimizer state of params specified
              will be monitored, may cause a large amount of print info if 'log' is selected.
              Set to ``None`` to ignore this metric. Default: ``None``.

            - weight_state_format: Determine where to display the weight L2-norm.
              Should be a `str` in ['tensorboard', 'log'] (mean that write data to tensorboard or log file),
              or a `list` containing them,  or ``None``. Set to ``None`` to ignore this metric.
              Default: ``None``.

            - throughput_baseline: The model throughput baseline to calculate linearity. Must be a positive number.
              Will be displayed both to tensorboard and log. Set to ``None`` to ignore this metric. Default: ``None``.

            - print_struct: Whether to print the structure of model. If ``True``, callback will print the names of all
              trainable params at the first step and then quit training process. Default: ``False``.

        per_print_times (int, optional): Every how many steps to print the log information. Default: ``1``.
        dataset_size (int, optional): Required in sink mode. Training dataset size. Default: ``None``.
        initial_epoch (int, optional): The beginning epoch. Default: ``0``.
        initial_step (int, optional): The beginning step. Default: ``0``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
    """
    def __init__(self,
                 origin_epochs: int,
                 config: dict = None,
                 per_print_times: int = 1,
                 dataset_size: int = None,
                 initial_epoch: int = 0,
                 initial_step: int = 0,
                 global_batch_size: int = 0):
        super(TrainingStateMonitor, self).__init__()
        self.per_print_times = per_print_times
        self.last_print_time = 0
        self.step_time = time.time()
        self.epoch_time = time.time()
        self.run_context = None
        self.steps_per_epoch = dataset_size
        self.origin_epochs = origin_epochs
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.global_batch_size = global_batch_size
        self.device_num = get_real_group_size()
        self.tensor_writer = get_tensorboard_writer()
        self.outputer = {'tensorboard': self._to_tensorboard, 'log': self._to_log}
        self._init_config(config)
        if get_auto_parallel_context("dump_local_norm_path"):
            self.dump_path = os.path.join(get_auto_parallel_context("dump_local_norm_path"), f'rank_{get_real_rank()}')
            self.dump_key = {0: -1}
            self.dump_step = 1
            if is_version_ge(ms.__version__, '2.5.0'):
                self.dump_name_mode = 0
                self.finish_pattern = 'finish_step_*_*'
                self.local_loss_pattern = re.compile('(local_loss)_[a-z]+[0-9]+_([0-9]+)')
                self.local_norm_pattern = re.compile('(local_norm)__(.+)_[a-z]+[0-9]+_([0-9]+)')
                self.device_local_norm_pattern = re.compile('(device_local_norm)_[a-z]+[0-9]+_([0-9]+)')
            else:
                self.dump_name_mode = 1
                self.finish_pattern = '*_finish_step'
                self.local_loss_pattern = re.compile('([0-9]+)_(local_loss)')
                self.local_norm_pattern = re.compile('([0-9]+)_(local_norm)__(.+)')
                self.device_local_norm_pattern = re.compile('([0-9]+)_(device_local_norm)')

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.epoch_time = time.time()
        self.run_context = run_context

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()
        self.run_context = run_context
        if self.print_struct:
            network = run_context.original_args().network
            if isinstance(network, ms.nn.TrainOneStepCell):
                network = network.network
            for param in network.trainable_params():
                logger.info(param.name)
            self.run_context.request_stop()

    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        if self.print_struct:
            return
        step_seconds = (time.time() - self.step_time) * 1000
        parallel_mode = get_auto_parallel_context("parallel_mode")
        full_batch = get_auto_parallel_context("full_batch")
        auto_parallel = parallel_mode in ['semi_auto_parallel', 'auto_parallel']
        if auto_parallel:
            set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
        cb_params = run_context.original_args()
        if cb_params.dataset_sink_mode:
            per_step_seconds = step_seconds / cb_params.batch_num
        else:
            self.steps_per_epoch = cb_params.batch_num
            per_step_seconds = step_seconds
        if (cb_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = cb_params.cur_step_num
            if get_auto_parallel_context("dump_local_norm_path"):
                self._dump_data_in_step(cb_params.cur_step_num)
            if self.optimizer_state_format:
                self._dump_optimizer_state(cb_params)
            if self.weight_state_format:
                network = cb_params.network
                if isinstance(network, ms.nn.TrainOneStepCell):
                    network = network.network
                weight_norm = _get_weight_norm(network)
                self._output('weight_norm', weight_norm, cb_params.cur_step_num, self.weight_state_format)
            if self.throughput_baseline is not None:
                # compute throughput
                throughput = self.global_batch_size / self.device_num / (per_step_seconds / 1000)
                linearity = throughput / self.throughput_baseline
                self._output('throughput_linearity', linearity, cb_params.cur_step_num, ['log', 'tensorboard'])

        if auto_parallel:
            set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)

    def _init_config(self, config):
        """Initialize members from config"""
        if config is None:
            logger.warning("The param `config` of TrainingStateMonitor is not set. Will use the default config.")
            config = {}
        if not isinstance(config, dict):
            raise TypeError("The param `config` of TrainingStateMonitor should be a dict.")
        self.target = config.get('target') or ['.*']
        self.invert = config.get('invert', False)
        self.target_cache = {}
        self.local_norm_format = config.get('local_norm_format', None)
        self.local_loss_format = config.get('local_loss_format', None)
        self.device_local_norm_format = config.get('device_local_norm_format', None)
        self.optimizer_state_format = config.get('optimizer_state_format', None)
        self.weight_state_format = config.get('weight_state_format', None)
        self.throughput_baseline = config.get('throughput_baseline', None)
        self.print_struct = config.get('print_struct', False)
        if not (isinstance(self.target, list) and self.target and all([isinstance(i, str) for i in self.target])):
            raise TypeError(f"The value of 'target' should be a list of str.")
        if not isinstance(self.invert, bool):
            raise TypeError(f"The value of 'invert' should be bool.")
        if (self.throughput_baseline is not None and
                not (isinstance(self.throughput_baseline, (int, float)) and self.throughput_baseline > 0)):
            raise ValueError(f"The value of 'throughput_baseline' should be None or positive number.")
        if not isinstance(self.print_struct, bool):
            raise TypeError(f"The value of 'print_struct' should be bool.")
        attrs = ['local_norm_format', 'local_loss_format', 'device_local_norm_format',
                 'optimizer_state_format', 'weight_state_format']
        for attr in attrs:
            self._check_attr_formats(attr)

    def _check_attr_formats(self, attr):
        """Check the validation of formats in config"""
        if getattr(self, attr):
            if not isinstance(getattr(self, attr), (str, list)):
                raise TypeError(f"The value of {attr} should be a `str` in 'tensorboard' or 'log', "
                                f"or a list containing them, or None, but get type {type(getattr(self, attr))}")
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, set([getattr(self, attr)]))
            else:
                setattr(self, attr, set(getattr(self, attr)))
            diff = getattr(self, attr) - {'tensorboard', 'log'}
            if diff:
                raise ValueError(f"The value of {attr} should be a `str` in 'tensorboard' or 'log', "
                                 f"or a list containing them, or None, but get unexpected value {diff}")
        else:
            setattr(self, attr, None)
        if self.tensor_writer is None and getattr(self, attr):
            logger.warning("Tensorboard config is unset. '%s' will use 'log' only.", attr)
            getattr(self, attr).discard('tensorboard')
            getattr(self, attr).add('log')

    def _parse_step(self):
        """record the finish dump id of each step"""

        def check_step(pattern, id_pos):
            search_path = glob.glob(os.path.join(self.dump_path, f'{pattern}.npy'))
            if not search_path:
                return None
            step_ids = []
            for f in search_path:
                tag, _ = os.path.splitext(os.path.basename(f))
                step_ids.append(int(tag.split('_')[id_pos]))
                os.remove(f)
            step_ids.sort()
            return step_ids

        step_ids = check_step(self.finish_pattern, self.dump_name_mode - 1)
        if step_ids is None:
            return
        cur_steps = len(self.dump_key)
        for i, step_id in enumerate(step_ids):
            self.dump_key[cur_steps + i] = step_id

    def _dump_data_in_step(self, global_step):
        """write the dumped data each step to tensorboard"""

        def match_pattern(pattern, filename):
            name, _ = os.path.splitext(os.path.basename(filename))
            parsed = re.fullmatch(pattern, name)
            if parsed is None:
                return None, None, None
            groups = parsed.groups()
            dump_id = int(groups[self.dump_name_mode - 1])
            prefix = groups[self.dump_name_mode]
            param_name = None if len(groups) < 3 else groups[self.dump_name_mode + 1]
            return dump_id, prefix, param_name

        self._parse_step()
        while self.dump_step <= global_step and self.dump_key.get(self.dump_step) is not None:
            begin_id = self.dump_key[self.dump_step - 1]
            end_id = self.dump_key[self.dump_step]
            file_list = os.listdir(self.dump_path)
            local_losses = []
            for f in file_list:
                parsed_name = None, None, None
                if self.local_norm_format:
                    parsed_name = match_pattern(self.local_norm_pattern, f)
                if not any(parsed_name) and self.device_local_norm_format:
                    parsed_name = match_pattern(self.device_local_norm_pattern, f)
                if not any(parsed_name) and self.local_loss_format:
                    parsed_name = match_pattern(self.local_loss_pattern, f)
                if not any(parsed_name):
                    continue
                dump_id, prefix, param_name = parsed_name
                if not begin_id < dump_id < end_id:
                    continue
                data = np.load(os.path.join(self.dump_path, f), allow_pickle=True)
                if prefix == 'device_local_norm':
                    self._output(f'device_local_norm', data, self.dump_step, self.device_local_norm_format)
                elif prefix == 'local_loss':
                    # collect all local loss if there are more than one local loss within one step
                    local_losses.append(data)
                elif prefix == 'local_norm' and self._check_param_name(param_name):
                    self._output(f'local_norm/{param_name}', data, self.dump_step, self.local_norm_format)
            # write the mean of local loss to tensorboard
            if local_losses:
                self._output(f'local_loss', np.stack(local_losses).mean(), self.dump_step, self.local_loss_format)
            for f in file_list:
                os.remove(os.path.join(self.dump_path, f))
            self.dump_step += self.per_print_times

    def _dump_optimizer_state(self, cb_params):
        """write the optimizer state to tensorboard"""
        optimizer = cb_params.optimizer
        if optimizer is None:
            optimizer = getattr(cb_params.network, "optimizer", None)

        if hasattr(optimizer, "moment1") and hasattr(optimizer, "moment2"):
            adam_m, adam_v = optimizer.moment1, optimizer.moment2
        elif hasattr(optimizer, "moments1") and hasattr(optimizer, "moments2"):
            adam_m, adam_v = optimizer.moments1, optimizer.moments2
        elif hasattr(optimizer, "exp_avg") and hasattr(optimizer, "exp_avg_sq"):
            adam_m, adam_v = optimizer.exp_avg, optimizer.exp_avg_sq
        else:
            return
        global_step = cb_params.cur_step_num
        adam_m_norms = _get_optimizer_state(adam_m, self._check_param_name)
        adam_v_norms = _get_optimizer_state(adam_v, self._check_param_name)
        for param_name, adam_m_norm in adam_m_norms.items():
            param_name = param_name.split('.', maxsplit=1)[1]
            self._output(f'adam_m_norm/{param_name}', adam_m_norm, global_step, self.optimizer_state_format)
        for param_name, adam_v_norm in adam_v_norms.items():
            param_name = param_name.split('.', maxsplit=1)[1]
            self._output(f'adam_v_norm/{param_name}', adam_v_norm, global_step, self.optimizer_state_format)

    def _check_param_name(self, param_name):
        if self.target_cache.get(param_name) is None:
            for pattern in self.target:
                if re.search(pattern, param_name) is not None:
                    self.target_cache[param_name] = not self.invert
                    return not self.invert
            self.target_cache[param_name] = self.invert
            return self.invert
        return self.target_cache[param_name]

    def _to_tensorboard(self, tag, data, global_step):
        """Write data to tensorboard if possible"""
        if self.tensor_writer is not None:
            self.tensor_writer.add_scalar(tag, data, global_step=global_step)

    def _to_log(self, tag, data, global_step):
        """Write data to log file"""
        cur_epoch_num = (global_step + self.initial_step - 1) // self.steps_per_epoch + 1
        cur_step_num = (global_step + self.initial_step - 1) % self.steps_per_epoch + 1
        logger.info("Epoch:[%3d/%3d], step:[%5d/%5d] %s: %.4f",
                    cur_epoch_num, self.origin_epochs, cur_step_num, self.steps_per_epoch, tag, data)

    def _output(self, tag, data, global_step, formats):
        """Write data in specified formats"""
        if formats is None:
            return
        for fmt in formats:
            self.outputer[fmt](tag, data, global_step)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class SummaryMonitor:
    """
    Summary Monitor can help you to collect some common information, such as loss,
    learning late, computational graph and so on.

    Note:
        referring to
        `note <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryCollector.html>`_ .

    Args:
        summary_dir (str, optional):
            The collected data will be persisted to this directory. If the directory does not exist,
            it will be created automatically. Default: ``None``.
        collect_freq (int, optional):
            Set the frequency of data collection, it should be greater than zero, and the unit is `step`.
            Default: ``10``.
        collect_specified_data (Union[None, dict], optional):
            Perform custom operations on the collected data. Default: ``None``.
        keep_default_action (bool, optional):
            This field affects the collection behavior of the 'collect_specified_data' field. Default: ``True``.
        custom_lineage_data (Union[dict, None], optional):
            Allows you to customize the data and present it on the MingInsight `lineage page <https://
            www.mindspore.cn/mindinsight/docs/en/master/lineage_and_scalars_comparison.html>`_ . Default: ``None``.
        collect_tensor_freq (Optional[int], optional):
            The same semantics as the `collect_freq`, but controls TensorSummary only. Default: ``None``.
        max_file_size (Optional[int], optional):
            The maximum size in bytes of each file that can be written to the disk. For example,
            to write not larger than 4GB, specify max_file_size=4*1024**3. Default: ``None``, which means no limit.
        export_options (Union[None, dict], optional):
            Perform custom operations on the export data. Default: ``None``, it means that the data is not exported.

    Examples:
        >>> from mindformers.core import SummaryMonitor
        >>> monitor = SummaryMonitor(summary_dir='./summary_dir')
    """

    def __new__(cls,
                summary_dir=None,
                collect_freq=10,
                collect_specified_data=None,
                keep_default_action=True,
                custom_lineage_data=None,
                collect_tensor_freq=None,
                max_file_size=None,
                export_options=None):
        if summary_dir is None:
            rank_id = get_real_rank()
            summary_dir = get_output_subpath('summary', rank_id)
        kwargs = {
            "summary_dir": summary_dir,
            "collect_freq": collect_freq,
            "collect_specified_data": collect_specified_data,
            "keep_default_action": keep_default_action,
            "custom_lineage_data": custom_lineage_data,
            "collect_tensor_freq": collect_tensor_freq,
            "max_file_size": max_file_size,
            "export_options": export_options
        }
        return SummaryCollector(**kwargs)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class CheckpointMonitor(ModelCheckpoint):
    """
    Checkpoint Monitor For Save LossScale.

    Args:
        prefix (str, optional): The prefix name of checkpoint files. Default: ``'CKP'``.
        directory (str, optional): The path of the folder which will be saved in the checkpoint file. Default: ``None``.
        config (CheckpointConfig, optional): Checkpoint strategy configuration. Default: ``None``.
        save_checkpoint_steps (int, optional): Steps to save checkpoint. Default: ``1``.
        save_checkpoint_seconds (int, optional): Seconds to save checkpoint.
            Can't be used with save_checkpoint_steps at the same time. Default: ``0``.
        keep_checkpoint_max (int, optional): Maximum number of checkpoint files can be saved. Default: ``5``.
        keep_checkpoint_per_n_minutes (int, optional): Save the checkpoint file every "keep_checkpoint_per_n_minutes"
            minutes. Can't be used with keep_checkpoint_max at the same time. Default: ``0``.
        integrated_save (bool, optional): Whether to merge and save the split Tensor in the automatic parallel scenario.
            Integrated save function is only supported in automatic parallel scene. Default: ``True``.
        save_network_params (bool, optional): Whether to only save network weights additionally. Default: ``True``.
        save_trainable_params (bool, optional): Whether to save fine-tuned weights additionally. Default: ``False``.
        async_save (bool, optional): Whether asynchronous execution saves the checkpoint to a file. Default: ``False``.
        saved_network (Cell, optional): Network to be saved in checkpoint file. Default: ``None``.
        append_info (list, optional): The information save to checkpoint file.
            Support "epoch_num", "step_num" and dict. Default: ``None``.
        enc_key (Union[None, bytes], optional): Byte type key used for encryption. Default: ``None``.
        enc_mode (str, optional): This parameter is valid only when "enc_key" is not set to None. Specifies the
            encryption mode, currently supports 'AES-GCM', 'AES-CBC' and 'SM4-CBC'. Default: ``'AES-GCM'``.
        exception_save (bool, optional): Whether to save the current checkpoint when an exception occurs.
            Default: ``False``.
        global_batch_size (int, optional): The total batch size. Default: ``0``.
        checkpoint_format (str, optional): The format of checkpoint to save. Support 'ckpt' or 'safetensors'.
            Default: ``'ckpt'``.
        remove_redundancy (bool, optional): Whether to remove redundancy when saving checkpoint. Default: ``False``.

    Raises:
        ValueError: If `prefix` is not str or contains the '/' character.
        ValueError: If `directory` is not str.
        TypeError: If the config is not CheckpointConfig type.

    Examples:
        >>> from mindformers.core import CheckpointMonitor
        >>> monitor = CheckpointMonitor(directory='./checkpoint_dir')
    """

    def __init__(self, prefix='CKP',
                 directory=None,
                 config=None,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 save_network_params=True,
                 save_trainable_params=False,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM',
                 exception_save=False,
                 global_batch_size=None,
                 checkpoint_format='ckpt',
                 remove_redundancy=False):

        self.config = config
        self.save_network_params = save_network_params
        self.save_trainable_params = save_trainable_params
        self.rank_id = get_real_rank()
        prefix = prefix + "_rank_{}".format(self.rank_id)

        self.global_batch_size = global_batch_size
        # this list records parametrs which will be ignored when saving ckpt.
        self.filter_list = ['accu_grads', 'fi_parameter', 'zeros_k_pe', 'zeros_k_nope', 'zeros_value_states', '_cache']

        self.save_info_list = defaultdict(
            lambda: {
                'ckpt': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
                'network': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
                'trainable_params': {'ckpt_file_path': None, 'save_start_time': None, 'save_end_time': None},
            }
        )

        if append_info is None:
            append_info = [{
                "epoch_num": 0,
                "step_num": 0,
                "global_step": 0,
                "loss_scale": 1
            }]
        ckpt_directory = os.path.join(directory, f"checkpoint/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint', self.rank_id)
        self.network_directory = os.path.join(directory, f"checkpoint_network/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint_network', self.rank_id)
        self.trainable_directory = os.path.join(directory, f"checkpoint_trainable/rank_{self.rank_id}") \
            if directory else get_output_subpath('checkpoint_trainable', self.rank_id)
        if context.get_auto_parallel_context('parallel_mode') in \
                ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
            logger.info("Integrated_save is changed to False when using auto_parallel.")
            integrated_save = False
        config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                     save_checkpoint_seconds=save_checkpoint_seconds,
                                     keep_checkpoint_max=keep_checkpoint_max,
                                     keep_checkpoint_per_n_minutes=keep_checkpoint_per_n_minutes,
                                     integrated_save=integrated_save,
                                     async_save=async_save,
                                     saved_network=saved_network,
                                     append_info=append_info,
                                     enc_key=enc_key,
                                     enc_mode=enc_mode,
                                     format=checkpoint_format,
                                     exception_save=exception_save,
                                     remove_redundancy=remove_redundancy)
        super(CheckpointMonitor, self).__init__(prefix, ckpt_directory, config=config_ck)
        self.meta_json = os.path.join(self._directory, "meta.json")
        if self._config.async_save:
            self.last_epoch_num = None
            self.last_step_num_in_epoch = None
            self.last_ckpoint_file = None
            self.meta_updated = True

    def print_savetime(self, record_step, batch_num):
        """print the time cost of saving checkpoint files."""
        epoch = int((record_step - 1) // batch_num + 1)
        step = int((record_step - 1) % batch_num + 1)

        def output_if_exists(key):
            save_info = self.save_info_list[record_step][key]
            file = save_info['ckpt_file_path']
            if file is not None and os.path.exists(file):
                save_info['save_end_time'] = os.path.getmtime(file)
                cost_time = save_info['save_end_time'] - save_info['save_start_time']
                logger.info(f'Finish saving {key} of epoch {epoch} step {step}'
                            f' using {cost_time:.3f} seconds')
                save_info['ckpt_file_path'] = None

        output_if_exists('ckpt')
        output_if_exists('network')
        output_if_exists('trainable_params')

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        # pylint: disable=E0203
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)

        # if async_save is True, check whether saving processes are completed each step
        if self._config.async_save:
            keys = list(self.save_info_list.keys())
            for record_step in keys:
                self.print_savetime(record_step, cb_params.batch_num)
                if not any([self.save_info_list[record_step][key]['ckpt_file_path']
                            for key in ['ckpt', 'network', 'trainable_params']]):
                    self.save_info_list.pop(record_step)

        if self._config.async_save and not ms.async_ckpt_thread_status() and \
            self.last_epoch_num and self.last_step_num_in_epoch and self.last_ckpoint_file and \
                not self.meta_updated:
            self.record_last_ckpt_to_json(self.last_epoch_num, self.last_step_num_in_epoch, self.last_ckpoint_file)
            self.meta_updated = True

        if save_ckpt:
            self.save_checkpoint(cb_params)
            self.save_checkpoint_network(cb_params)
            # if async_save is False, output the time cost directly
            if not self._config.async_save:
                self.print_savetime(cb_params.cur_step_num, cb_params.batch_num)

    def save_checkpoint(self, cb_params):
        """save checkpoint suitable for resume training."""
        logger.info('......Saving ckpt......')
        self.save_info_list[cb_params.cur_step_num]['ckpt']['save_start_time'] = time.time()
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        cur_ckpoint_file = (f"{self._prefix}-{str(cb_params.cur_epoch_num)}"
                            f"_{str(step_num_in_epoch)}.{self._config.format}")
        # update checkpoint file list.
        self._manager.update_ckpoint_filelist(self._directory, self._prefix)
        # keep checkpoint files number equal max number.
        if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
            self._manager.remove_oldest_ckpoint_file()
        elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
            # pylint: disable=E0203
            self._cur_time_for_keep = time.time()
            if (self._cur_time_for_keep - self._last_time_for_keep) \
                    < self._config.keep_checkpoint_per_n_minutes * 60:
                self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                           self._cur_time_for_keep)

        # generate the new checkpoint file and rename it.
        global SAVE_DIR
        SAVE_DIR = self._directory
        cur_file = os.path.join(self._directory, cur_ckpoint_file)
        self._last_time_for_keep = time.time()
        self._last_triggered_step = cb_params.cur_step_num

        if "epoch_num" in self._append_dict:
            self._append_dict["epoch_num"] = cb_params.cur_epoch_num
        if "step_num" in self._append_dict:
            self._append_dict["step_num"] = self._append_step_num + cb_params.cur_step_num
        if cb_params.optimizer is not None:
            self._append_dict["global_step"] = cb_params.optimizer.global_step
        else:
            self._append_dict["global_step"] = cb_params.network.optimizer.global_step
        if "loss_scale" in self._append_dict:
            outputs = cb_params.net_outputs
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                self._append_dict["loss_scale"] = outputs[2]
        if self.global_batch_size is not None:
            self._append_dict["global_batch_size"] = self.global_batch_size
            logger.info("global_batch_size: %d", self._append_dict["global_batch_size"])
        logger.info("epoch_num: %d", self._append_dict["epoch_num"])
        logger.info("step_num: %d", self._append_dict["step_num"])
        logger.info("global_step: %d", self._append_dict["global_step"])
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network

        self.remove_redundancy(network, cur_file, self._append_dict, None)

        self._latest_ckpt_file_name = cur_file
        self.save_info_list[cb_params.cur_step_num]['ckpt']['ckpt_file_path'] = cur_file

        if self._config.async_save:
            self.last_epoch_num = cb_params.cur_epoch_num
            self.last_step_num_in_epoch = step_num_in_epoch
            self.last_ckpoint_file = cur_ckpoint_file
            self.meta_updated = False
        else:
            if "__exception_save__" not in self._append_dict:
                self.record_last_ckpt_to_json(cb_params.cur_epoch_num, step_num_in_epoch, cur_ckpoint_file)

    def _get_cur_dp(self, cur_rank, parameter_redundancy_dict):
        """get the current dp"""
        value_len = sys.maxsize
        min_value = ()
        min_value_set = set()
        for key, value in parameter_redundancy_dict.items():
            if key.startswith("accu_grads") or key.startswith("inputs"):
                continue
            for item in value:
                if cur_rank not in item:
                    continue
                # if item is subset of min_value_set, update min_value_set and min_value
                if len(item) < value_len:
                    if min_value_set and not set(item).issubset(min_value_set):
                        return (cur_rank,)
                    value_len = len(item)
                    min_value_set = set(item)
                    min_value = item
                # if value is not smaller than len of min_value len,
                # check if min_value_set is subset of current item
                elif not min_value_set.issubset(set(item)):
                    return (cur_rank,)

        return min_value

    def _filter_ckpt_not_save(self, x, filter_list):
        return all(not x.startswith(item) and item not in x for item in filter_list)

    def _tft_save_ckpt(self, param_layout_set, save_param_names, cur_file, append_dict, network):
        """save checkpoint with remove redundancy for TFT training."""
        def choice_func(x):
            return (x not in param_layout_set or (save_param_names is not None and x in save_param_names)) \
                and self._filter_ckpt_not_save(x, self.filter_list)

        save_checkpoint(network, cur_file, False, False,
                        append_dict, self._config.enc_key, self._config.enc_mode,
                        format=self._config.format, choice_func=choice_func)

    # pylint: disable=W0640
    def _do_remove_redundancy_for_tft(self, redundancy_info, cur_file, network, append_dict):
        """save checkpoint with remove redundancy for TFT training."""
        rank_id, param_redundancy_dict, single_params, param_layout = redundancy_info

        pattern = rf'_(\d+)\.{self._config.format}$'
        match = re.search(pattern, cur_file)
        cur_step_in_epoch = int(match.group(1))

        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        cur_dp = self._get_cur_dp(rank_id, param_redundancy_dict)
        # loop through all ranks in the current dp
        for rank in cur_dp:
            save_param_names = single_params.get(rank)
            if save_param_names == param_layout.keys():
                logger.warning(
                    f"For remove_redundancy save checkpoint, the saved parameters are non-redundant.")
            param_layout_set = set(param_layout.keys()) if parallel_mode else set()
            cur_file = re.sub(r'rank_\d+', f'rank_{rank}', cur_file)
            self._tft_save_ckpt(param_layout_set, save_param_names, cur_file, append_dict, network)
            append_dict["__exception_save__"] = True
            self.meta_json = re.sub(r'rank_\d+', f'rank_{rank}', self.meta_json)
            self.record_last_ckpt_to_json(append_dict["epoch_num"], cur_step_in_epoch, os.path.basename(cur_file))

    def remove_redundancy(self, network, cur_file, append_dict, train_network):
        """remove redundancy when saving checkpoint files."""
        if self._config.remove_redundancy:
            logger.info('......Removing redundancy......')
            parallel_mode = context.get_auto_parallel_context("parallel_mode")
            if parallel_mode == "stand_alone":
                raise TypeError(f"The deduplication feature for saving checkpoint can only be used "
                                f"in parallel scenarios, but got {parallel_mode}.")

            if train_network:
                param_layout = train_network.parameter_layout_dict
            else:
                param_layout = network.parameter_layout_dict
            rank_id = get_real_rank()
            if param_layout:
                device_num = get_real_group_size()
                stage_num = get_auto_parallel_context("pipeline_stages")
                chunk_size = device_num // stage_num
                initial_rank = (rank_id // chunk_size) * chunk_size
                param_redundancy_dict = get_parameter_redundancy(param_layout, initial_rank)
                single_params = remove_param_redundancy(param_redundancy_dict)
                save_param_names = single_params.get(rank_id)
                param_layout_set = set(param_layout.keys())
                if save_param_names == param_layout.keys():
                    logger.warning(
                        f"For remove_redundancy save checkpoint, the saved parameters are non-redundant.")

                def choice_func(x):
                    return (x not in param_layout_set or (save_param_names is not None and x in save_param_names)) \
                        and self._filter_ckpt_not_save(x, self.filter_list)
            else:
                param_redundancy_dict = get_parameter_redundancy(network)
                single_params = remove_param_redundancy(param_redundancy_dict)
                save_param_names = single_params.get(rank_id)

                def choice_func(x):
                    return save_param_names is not None and x in save_param_names \
                        and self._filter_ckpt_not_save(x, self.filter_list)

            # __exception_save__ is used to indicate that the checkpoint is saved by the TFT process
            if "__exception_save__" in append_dict:
                redundancy_info = (rank_id, param_redundancy_dict, single_params, param_layout)
                self._do_remove_redundancy_for_tft(redundancy_info, cur_file, network, append_dict)
                return

            save_checkpoint(network, cur_file, False, self._config.async_save,
                            append_dict, self._config.enc_key, self._config.enc_mode,
                            format=self._config.format, choice_func=choice_func)
        else:
            save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                            append_dict, self._config.enc_key, self._config.enc_mode,
                            format=self._config.format,
                            choice_func=lambda x: self._filter_ckpt_not_save(x, self.filter_list))

    def save_checkpoint_network(self, cb_params):
        """save checkpoint only network params, which is suitable for train, evaluate and predict."""
        save_obj = cb_params.network
        network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network

        if hasattr(save_obj, 'optimizer') and save_obj.optimizer is not None:
            save_obj = save_obj.network
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

        if self.save_trainable_params:
            self.save_info_list[cb_params.cur_step_num]['trainable_params']['save_start_time'] = time.time()
            save_obj.init_parameters_data()
            param_dict = OrderedDict()
            for param in save_obj.trainable_params():
                param_dict[param.name] = param
            param_list = []
            for (key, value) in param_dict.items():
                cur_param_info = value.param_info
                if hasattr(cur_param_info, 'is_pipeline_shared_param') and cur_param_info.is_pipeline_shared_param:
                    continue
                each_param = {"name": key}
                param_data = Tensor(value.data.asnumpy())

                # in automatic model parallel scenario, some parameters were split to all the devices,
                # which should be combined before saving
                if key in save_obj.parameter_layout_dict:
                    param_data = _get_merged_param_data(save_obj, key, param_data, self._config.integrated_save)

                each_param["data"] = param_data
                param_list.append(each_param)
            save_obj = param_list
            cb_cur_ckpoint_file = (f"{self._prefix}-trainable_params-{str(cb_params.cur_epoch_num)}"
                                   f"_{str(step_num_in_epoch)}.{self._config.format}")
            cb_cur_file = os.path.join(self.trainable_directory, cb_cur_ckpoint_file)
            os.makedirs(self.trainable_directory, exist_ok=True)
            self.remove_redundancy(save_obj, cb_cur_file, {}, network)
            self.save_info_list[cb_params.cur_step_num]['trainable_params']['ckpt_file_path'] = cb_cur_file
            return

        if self.save_network_params:
            self.save_info_list[cb_params.cur_step_num]['network']['save_start_time'] = time.time()
            cb_cur_ckpoint_file = (f"{self._prefix}-network-{str(cb_params.cur_epoch_num)}"
                                   f"_{str(step_num_in_epoch)}.{self._config.format}")
            cb_cur_file = os.path.join(self.network_directory, cb_cur_ckpoint_file)
            os.makedirs(self.network_directory, exist_ok=True)
            self.remove_redundancy(save_obj, cb_cur_file, {}, network)
            self.save_info_list[cb_params.cur_step_num]['network']['ckpt_file_path'] = cb_cur_file



    def record_last_ckpt_to_json(self, epoch, step, ckpt_file):
        """record last ckpt info to json"""
        meta_data = {
            "last_epoch": epoch,
            "last_step": step,
            "last_ckpt_file": ckpt_file
        }
        with tempfile.NamedTemporaryFile('w', delete=False, dir=self._directory) as temp_file:
            json.dump(meta_data, temp_file)
            temp_file_path = temp_file.name
        os.replace(temp_file_path, self.meta_json)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ProfileMonitor(Callback):
    """
    Profile analysis in training.

    Args:
        start_step (int, optional): The step to start profiling. Default: ``1``.
        stop_step (int, optional): The step to stop profiling. Default: ``10``.
        output_path (str, optional): The result of profiling will be saved in this path. Default: ``None``.
        start_profile (str, optional): Whether to enable profiling. Default: ``True``.
        profile_rank_ids (list, optional): Specify rank ids to enable profiling. Default: ``None`` (All rank ids
            are enabled).
        profile_pipeline (str, optional): Whether to enable profiling on one card of each parallel stage.
            Default: ``False``.
        profile_communication (str, optional): Whether to collect communication performance data
            during multi-device training. Default: ``False``.
        profile_memory (str, optional): Whether to collect Tensor memory data. Default: ``False``.
        config (dict, optional): Configuration items, used to profile relevant configuration information,
            such as parallel configuration. Default: ``None``.
        profiler_level (int, optional): Collection level of profiling data(0, 1, 2). Default: ``0``.

            - 0: The most streamlined level of performance data collection,
              only collecting execution time data for computational operators and
              basic data for large communication operators.
            - 1: In addition to level 0, extra data is collected for CANN layer AscendCL,
              AICORE performance data, and small communication operators.
            - 2: In addition to level 1, extra data is collected for graph compile level O2
              and Runtime in the CANN layer.

        with_stack (str, optional): Whether to collect Python-side stack trace data. Default: ``False``.
        data_simplification (str, optional): Whether to enable data simplification, which will delete the FRAMEWORK
            directory and other extraneous data after exporting profiling data. Default: ``True``.

    Examples:
        >>> from mindformers.core import ProfileMonitor
        >>> monitor = ProfileMonitor(output_path='./profile_dir')
    """

    def __init__(self, start_step=1, stop_step=10, output_path=None,
                 start_profile=True, profile_rank_ids=None, profile_pipeline=False,
                 profile_communication=False, profile_memory=False, config=None,
                 profiler_level=0, with_stack=False, data_simplification=True, **kwargs):
        super(ProfileMonitor, self).__init__()
        self.mstx_range_id = None
        self.mstx_avail = is_version_ge(ms.__version__, '2.5.0')
        self.start_step = start_step
        self.stop_step = stop_step
        self.start_profile = start_profile
        self.profile_rank_ids = profile_rank_ids
        self.profile_pipeline = profile_pipeline
        self.profile_communication = profile_communication
        self.profiler_level = self._get_profiler_level(profiler_level)
        self.profiler = None

        if profile_communication and not start_profile:
            raise ValueError("When profile_communication is True, start_profile must also be True")

        rank_id = get_real_rank()
        self.pipeline_rank_ids = get_pipeline_rank_ids() if self.profile_pipeline else None
        if self.pipeline_rank_ids == [-1]:
            raise ValueError(f"Device num should be divided by pipeline stage num.")

        if self._is_profile_required(rank_id):
            if not output_path:
                output_path = get_output_subpath('profile', rank_id)
            else:
                output_path = os.path.join(output_path, 'profile', 'rank_{}'.format(rank_id))
            logger.info("Profile save path: %s", output_path)

            if ms.get_context("device_target") == "GPU" and profile_memory:
                logger.warning("The parameter profile_memory is not supported on GPU currently, "
                               "so is changed to False. ")
                profile_memory = False

            self.profiler = Profiler(
                start_profile=start_profile, output_path=output_path,
                profile_communication=profile_communication, profile_memory=profile_memory,
                profiler_level=self.profiler_level, with_stack=with_stack,
                data_simplification=data_simplification, **kwargs
                )
            self._record_metadata(config)
            self.run_context = None
            self.output_path = output_path

    def step_begin(self, run_context):
        """
        Start profile at the beginning of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step and not self.start_profile and self.profiler:
            self.profiler.start()
        if self.mstx_avail:
            from mindspore.profiler import mstx
            self.mstx_range_id = mstx.range_start(f'Training step {step_num}')

    def step_end(self, run_context):
        """
        Stop profile at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.mstx_avail:
            from mindspore.profiler import mstx
            mstx.range_end(self.mstx_range_id)
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step and self.profiler:
            self.profiler.stop()
            self.profiler.analyse()
            logger.info("End of Profiling, please view the profile data under %s and analyze it using mindinsight."
                        "MindInsight order as follow: "
                        "mindinsight start --summary-base-dir %s", self.output_path, self.output_path)

    def _record_metadata(self, config):
        """
        Record metadata from config.

        Args:
            config (dict): config of the train running.
        """
        if config is None:
            return

        parallel = config.parallel
        parallel_config = config.parallel_config.to_dict()

        try:
            self.profiler.add_metadata_json('distributed_args', json.dumps({
                'tensor_model_parallel_size': parallel_config.get('model_parallel', 1),
                'pipeline_model_parallel_size': parallel_config.get('pipeline_stage', 1),
                'data_parallel_size': parallel_config.get('data_parallel', 1),
                'expert_model_parallel_size': parallel_config.get('expert_parallel', 1),
                'sequence_parallel': parallel_config.get('use_seq_parallel', False),
                'parallel_mode': parallel.get('parallel_mode', None),
                'world_size': parallel.get('device_num', None)
            }))
        except AttributeError as e:
            logger.warning("Profiler failed to record distributed args,  %s", e)

    def _is_profile_required(self, rank_id):
        """
        Determine whether current rank id needs to enable profiling.

        Args:
            rank_id (int): current rank id.
        """
        if not self.profile_rank_ids and not self.pipeline_rank_ids:
            return True

        profile_ids = self.profile_rank_ids if isinstance(self.profile_rank_ids, list) else []
        pipeline_ids = self.pipeline_rank_ids if isinstance(self.pipeline_rank_ids, list) else []

        if rank_id in profile_ids or rank_id in pipeline_ids:
            return True

        return False

    @staticmethod
    def _get_profiler_level(level):
        """
        Obtain profiler level based on the level value with integer type.

        Args:
            level (int): the value of profiler_level in MF config.
        """
        if level is None:
            return ProfilerLevel.Level0

        max_level = len(ProfilerLevel.__members__) - 1
        if level < 0 or level > max_level:
            logger.warning("Invalid profiler_level: %s, return None.", level)
            return None
        profiler_level = getattr(ProfilerLevel, f"Level{level}")
        return profiler_level


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class EvalCallBack(Callback):
    """
    Evaluate Callback used in training progress.

    Args:
        eval_func (Callable): The function used to evaluate the model results
            and can be customized according to specific task.
        step_interval (int, optional): Determine the num of step intervals between each eval.
            Default ``100``. Note that it will not take effects when running in data sink mode.
        epoch_interval (int, optional): Determine the num of epoch intervals between each eval.
            Default ``-1``, means eval on every epoch end.

    Examples:
        >>> from mindformers.core.callback import EvalCallBack
        >>> def eval_func():
        ...     print("output result")
        >>> eval_callback = EvalCallBack(eval_func=eval_func)
        >>> type(eval_callback)
    """

    def __init__(self, eval_func: Callable, step_interval: int = 100, epoch_interval: int = -1):
        self.eval_func = eval_func
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval

    def epoch_end(self, run_context):
        # if not use epoch end
        if self.epoch_interval <= 0:
            return
        callback_params = run_context.original_args()
        cur_epoch_num = callback_params.cur_epoch_num
        if cur_epoch_num % self.epoch_interval == 0:
            self._execute_eval()

    def step_end(self, run_context):
        # if not use step end
        if self.step_interval <= 0:
            return
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        if cur_step_num % self.step_interval == 0:
            self._execute_eval()

    def _execute_eval(self):
        start_time = time.time()
        output = self.eval_func()
        eval_time = time.time() - start_time
        logger.info("Eval result: %s, eval time is %f s.", output, eval_time)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class ColdHotExpertMointor(Callback):
    """
        ColdHotExpertMointor Callback used in MoE model training progress.

        Args:
            config : Read config from configuration file.

        Examples:
            >>> from mindformers.core.callback import ColdHotExpertMointor
            >>> callback = ColdHotExpertMointor(config)
            >>> type(callback)
            <class 'mindformers.core.callback.callback.ColdHotExpertMointor'>
    """

    def __init__(self, moe_config=None, hidden_size=None, ffn_hidden_size=None, expert_parallel=None,
                 model_parallel=None, save_checkpoint_steps=None):
        self.update_step = moe_config.update_step if hasattr(moe_config, "update_step") else 10000
        self.expert_num = moe_config.expert_num
        self.hot_expert_num = moe_config.hot_expert_num
        self.moe_module_name = moe_config.moe_module_name
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ep = expert_parallel
        self.mp = model_parallel
        self.save_checkpoint_steps = save_checkpoint_steps
        self.rank_id = int(os.getenv("RANK_ID"))
        self.local_expert_num = self.expert_num // self.ep
        start_index = (self.rank_id // self.mp) * self.local_expert_num
        end_index = start_index + self.local_expert_num
        self.local_expert_index = [i for i in range(start_index, end_index)]
        self.rank_size = int(os.getenv("RANK_SIZE"))

    def on_train_step_end(self, run_context):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.update_step <= 0:
            return
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        if ((cur_step_num < self.update_step and cur_step_num & (cur_step_num - 1) == 0) or
                (cur_step_num == self.save_checkpoint_steps) or (cur_step_num % self.update_step == 0)):
            total_start = time.time()
            train_network = callback_params.train_network
            if train_network is None:
                return
            blocks = self.get_attribute_by_path(train_network, self.moe_module_name)
            for block in blocks:
                if cur_step_num > 1:
                    self.return_back_hot_expert(block)
                self.switch_hot_expert(block, cur_step_num)
            total_end = time.time()
            logger.info("switch hot experts spent time is %f s.", total_end - total_start)

    def on_train_end(self, run_context):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        train_network = callback_params.train_network
        if train_network is None:
            return
        blocks = self.get_attribute_by_path(train_network, self.moe_module_name)
        for block in blocks:
            if cur_step_num > 1:
                self.return_back_hot_expert(block)

    def get_attribute_by_path(self, obj, path):
        """
        Obtains MoE blocks modules in obj by path..

        Args:
            obj : Model.
            path(str) : Path of the MoE layer in the model
        """
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj

    def return_back_hot_expert(self, block):
        """
        When the popular experts change, return the replica parameters to the old popular experts.

        Args:
            block : MoE layer.
        """
        old_hot_expert_index = block.output.hot_expert_index.value()[0]
        if self.hot_expert_num == 1:
            if old_hot_expert_index[0] in self.local_expert_index:
                ffn_index = old_hot_expert_index[0] - (self.rank_id // self.mp) * self.local_expert_num
                block.output.ffn.mapping.weight[ffn_index] = block.output.mlp.mapping.weight
                block.output.ffn.mapping.bias[0][ffn_index][0] = block.output.mlp.mapping.bias
                block.output.ffn.projection.weight[ffn_index] = block.output.mlp.projection.weight
                block.output.ffn.projection.bias[0][ffn_index][0] = block.output.mlp.projection.bias
        elif self.hot_expert_num > 1:
            for i in range(self.hot_expert_num):
                if old_hot_expert_index[i] in self.local_expert_index:
                    ffn_index = old_hot_expert_index[i] - (self.rank_id // self.mp) * self.local_expert_num
                    block.output.ffn.mapping.weight[ffn_index] = block.output.mlp.mapping.weight[i]
                    block.output.ffn.mapping.bias[0][ffn_index][0] = block.output.mlp.mapping.bias[0][i][0]
                    block.output.ffn.projection.weight[ffn_index] = block.output.mlp.projection.weight[i]
                    block.output.ffn.projection.bias[0][ffn_index][0] = block.output.mlp.projection.bias[0][i][0]

    def switch_hot_expert(self, block, cur_step_num):
        """
        Switch popular expert copies when there is a change in popular experts at the step.

        Args:
            block : MoE layer.
            cur_step_num : Current training step
        """
        old_hot_expert_index = block.output.hot_expert_index.value()[0]
        cumsum_tensor = block.output.router.router.cumsum_value.value()
        _, new_expert_index = cumsum_tensor.topk(self.expert_num, largest=True)
        new_hot_expert_index = new_expert_index[0:self.hot_expert_num]
        new_cold_expert_index = new_expert_index[self.hot_expert_num:self.expert_num]
        broadcasts = [self.BroadcastCell(i) for i in range(self.rank_size)]
        if self.hot_expert_num == 1:
            if cur_step_num > 1 and old_hot_expert_index[0] == new_hot_expert_index[0]:
                return
            # Broadcast new hot expert and copy the weights of new hot experts to mlp
            for i in range(self.mp):
                ffn_index = new_hot_expert_index[0] % self.local_expert_num
                rank_id = new_hot_expert_index[0] // self.local_expert_num * self.mp + i
                expert_part = broadcasts[rank_id]((block.output.ffn.mapping.weight[ffn_index],
                                                   block.output.ffn.mapping.bias[0][ffn_index][0],
                                                   block.output.ffn.projection.weight[ffn_index],
                                                   block.output.ffn.projection.bias[0][ffn_index][0]))
                if self.rank_id % self.mp == i:
                    block.output.mlp.mapping.weight = expert_part[0]
                    block.output.mlp.mapping.bias = expert_part[1]
                    block.output.mlp.projection.weight = expert_part[2]
                    block.output.mlp.projection.bias = expert_part[3]
        elif self.hot_expert_num > 1:
            new_hot_expert_index, _ = new_hot_expert_index.topk(self.hot_expert_num, largest=False)
            if cur_step_num > 1 and old_hot_expert_index.equal(new_hot_expert_index).all():
                return
            # Broadcast new hot expert and copy the weights of new hot experts to mlp
            for index in range(self.hot_expert_num):
                for i in range(self.mp):
                    ffn_index = new_hot_expert_index[index] % self.local_expert_num
                    rank_id = new_hot_expert_index[index] // self.local_expert_num * self.mp + i
                    expert_part = broadcasts[rank_id]((block.output.ffn.mapping.weight[ffn_index],
                                                       block.output.ffn.mapping.bias[0][ffn_index][0],
                                                       block.output.ffn.projection.weight[ffn_index],
                                                       block.output.ffn.projection.bias[0][ffn_index][0]))
                    if self.rank_id % self.mp == i:
                        block.output.mlp.mapping.weight[index] = expert_part[0]
                        block.output.mlp.mapping.bias[0][index][0] = expert_part[1]
                        block.output.mlp.projection.weight[index] = expert_part[2]
                        block.output.mlp.projection.bias[0][index][0] = expert_part[3]
        block.output.hot_expert_index = new_hot_expert_index.reshape((1, -1))
        block.output.cold_expert_index = new_cold_expert_index.reshape((1, -1))
        del broadcasts

    class BroadcastCell(Cell):
        def __init__(self, rank_id):
            super().__init__(auto_prefix=False)
            self.broadcast = Broadcast(rank_id)
            self.add_flags(skip_auto_parallel_compile=True)

        @jit()
        def construct(self, x):
            x = self.broadcast(x)
            return x


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TrainCallBack(Callback):
    """
    Train Callback used in training progress.

    Args:
        stop_step (int): The function stop train process at the step.
                             Default None, set in yaml.
    Examples:
        >>> from mindformers.core.callback import TrainCallBack
        >>> stop_step = TrainCallBack(stop_step=10)
        <class 'mindformers.core.callback.callback.TrainCallBack'>
    """

    def __init__(self, stop_step: int = None):
        self.stop_step = stop_step

    def step_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        cb_params = run_context.original_args()
        if self.stop_step is not None and cb_params.cur_step_num >= self.stop_step:
            run_context.request_stop()
            logger.info("set train process early stop at %s steps in yaml", self.stop_step)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class StressDetectCallBack(Callback):
    """
    Stress Detect Callback used in training progress.

    Args:
        detection_interval (int): (int, optional): The number of steps between each hardware precision stress detection.
            Default: ``None``.
        num_detections (int, optional): The number of consecutive hardware precision stress detections for each round.
            Default: ``None``.
        dataset_size (int, optional): Training dataset size. Default: ``None``.

    Examples:
        >>> from mindformers.core.callback import StressDetectCallBack
        >>> stress_detect_callback = StressDetectCallBack(detection_interval=10, num_detections=3, dataset_size=1024)
        >>> type(stress_detect_callback)
    """

    def __init__(self, detection_interval: int = None, num_detections: int = None, dataset_size: int = None):
        logger.warning('StressDetectCallBack serves as an experimental interface and its functionality is '
                       'not yet stable.')
        self.detection_interval = detection_interval
        self.num_detections = num_detections
        self.steps_per_epoch = dataset_size
        self.ms_version_valid = check_stress_detect_valid()

        if self.detection_interval > self.steps_per_epoch:
            logger.warning(f"detection_interval = {self.detection_interval} is bigger than "
                           f"steps_per_epoch = {self.steps_per_epoch}")


    def step_end(self, run_context):
        """
        Stress detect at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        callback_params = run_context.original_args()
        cur_step_num = callback_params.cur_step_num
        # stress detect
        detect_ret_list = []
        if self.ms_version_valid:
            from mindspore.utils import stress_detect

            if cur_step_num % self.detection_interval == 0:
                logger.info("Start to stress detect")
                for _ in range(self.num_detections):
                    ret = stress_detect()
                    detect_ret_list.append(ret)

            self.log_stress_detect_result(detect_ret_list)


    def log_stress_detect_result(self, detect_ret_list):
        """print output information."""
        for ret in detect_ret_list:
            if ret == 0:
                logger.info("Stress detection passed")
            elif ret == VOLTAGE_ERROR_CODE:
                raise RuntimeError(f"Voltage recovery failed with error code: {ret}")
            else:
                logger.warning(f"Stress detection failed with error code: {ret}")


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TopkBiasBalanceCallback(Callback):
    """Callback for topk bias balance feature in moe module.

    Args:
        balance_via_topk_bias (bool): Whether to use topk bias update, should be consistent with moe config.
        topk_bias_update_rate (float): How fast is the bias updated.
        num_layers (int): How many layers in the model.
        mtp_depth (int): How many layers in the mtp module.
        expert_num (int): How many experts in the moe module.
        micro_batch_num (int): Micro batch number in pipeline parallel. Default to 1.
        gradient_accumulation_steps (int): Gradient accumulation steps for training. Default to 1.
    """
    def __init__(self,
                 balance_via_topk_bias: bool,
                 topk_bias_update_rate: float,
                 num_layers: int,
                 mtp_depth: int,
                 expert_num: int,
                 micro_batch_num: int = 1,
                 gradient_accumulation_steps: int = 1):
        # for aux loss free
        # this process is to update the expert load
        self.update_topk_bias_flag = balance_via_topk_bias
        if self.update_topk_bias_flag:
            self.assign = P.Assign()
            self.assign.recompute(False)
            self.afb_sub = P.Sub()
            self.afb_add = P.Add()
            self.sign = P.Sign()
            self.afb_mul = P.Mul()
            self.afb_div = P.Div()
            self.pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
            self.micro_batch_num = micro_batch_num if self.pipeline_stages > 1 else 1
            self.acc_step_over_expert_num = \
                Tensor([micro_batch_num * gradient_accumulation_steps / expert_num], ms.float32)
            self.num_layers = num_layers + mtp_depth
            self.topk_bias_update_rate = topk_bias_update_rate
            self.zeros_tensor = ms.Tensor(np.zeros([expert_num]), ms.float32)

    def _update_topk_bias(self, network):
        """update topk bias tensor during training."""
        for i in range(self.num_layers):
            while hasattr(network, "network"):
                network = network.network
            if hasattr(network.model.layers[i].feed_forward, "routed_experts"):
                if hasattr(network.model.layers[i].feed_forward.routed_experts, "router"):
                    expert_load_data = \
                        network.model.layers[i].feed_forward.routed_experts.router.router.expert_load.value()
                    if expert_load_data.sum() > 0:
                        err = self.afb_sub(self.acc_step_over_expert_num, expert_load_data)
                        topk_bias_new = self.afb_add(
                            network.model.layers[i].feed_forward.routed_experts.router.router.topk_bias.value(),
                            self.afb_mul(self.sign(err), self.topk_bias_update_rate)
                        )
                        self.assign(network.model.layers[i].feed_forward.routed_experts.router.router.topk_bias,
                                    topk_bias_new)
                        self.assign(network.model.layers[i].feed_forward.routed_experts.router.router.expert_load,
                                    self.zeros_tensor)
                else:
                    expert_load_data = network.model.layers[i].feed_forward.routed_experts.expert_load.value()
                    if expert_load_data.sum() > 0:
                        err = self.afb_sub(self.acc_step_over_expert_num, expert_load_data)
                        topk_bias_new = self.afb_add(
                            network.model.layers[i].feed_forward.routed_experts.topk_bias.value(),
                            self.afb_mul(self.sign(err), self.topk_bias_update_rate)
                        )
                        self.assign(network.model.layers[i].feed_forward.routed_experts.topk_bias, topk_bias_new)
                        self.assign(network.model.layers[i].feed_forward.routed_experts.expert_load, self.zeros_tensor)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if self.update_topk_bias_flag:
            # pylint: disable=W0212
            self._update_topk_bias(cb_params.train_network.network.network._backbone)


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MoEDropRateCallback(Callback):
    """Callback drop rate in moe module.

    Args:
        expert_num (int): How many experts in the moe module.
        capacity_factor (float): Capcity factor in the moe module.
        num_layers (int): How many layers in the model.
        mtp_depth (int): How many layers in the mtp module.

    Examples:
        >>> from mindformers.core.callback import MoEDropRateCallback
        >>> stop_step = MoEDropRateCallback(expert_num=8, capacity_factor=1.5, num_layers=4, mtp_depth=1)
        <class 'mindformers.core.callback.callback.MoEDropRateCallback'>
    """
    def __init__(self,
                 expert_num: int,
                 capacity_factor: float,
                 num_layers: int,
                 mtp_depth: int):
        self.capacity_factor_over_expert_num = capacity_factor / expert_num
        self.num_layers = num_layers + mtp_depth

    def _callback_droprate(self, network):
        """callback drop rate."""
        for i in range(self.num_layers):
            while hasattr(network, "network"):
                network = network.network
            if hasattr(network.model.layers[i].feed_forward, "routed_experts"):
                if hasattr(network.model.layers[i].feed_forward.routed_experts, "router"):
                    fi = network.model.layers[i].feed_forward.routed_experts.router.router.fi_parameter.value()
                    if fi.sum() > 0:
                        delta = fi - self.capacity_factor_over_expert_num
                        droprate = ms.ops.sum(delta * (delta > 0))
                        logger.info("layer: %d, drop_rate: %.5f" % (i, droprate))
            else:
                if hasattr(network.model.layers[i].feed_forward, "router"):
                    fi = network.model.layers[i].feed_forward.router.router.fi_parameter.value()
                    if fi.sum() > 0:
                        delta = fi - self.capacity_factor_over_expert_num
                        droprate = ms.ops.sum(delta * (delta > 0))
                        logger.info("layer: %d, drop_rate: %.5f" % (i, droprate))

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        # pylint: disable=W0212
        self._callback_droprate(cb_params.train_network.network.network._backbone)
