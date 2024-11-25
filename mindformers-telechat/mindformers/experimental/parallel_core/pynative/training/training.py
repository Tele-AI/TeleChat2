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
"""For Training"""

import time
import contextlib

import mindspore.common.dtype as mstype
from mindspore import nn, Tensor, Parameter, mint, value_and_grad
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
import mindspore.communication.comm_func as comm_func
from mindspore.experimental.optim.optimizer import Optimizer as mintOptimizer

from mindformers.tools import logger
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
    set_virtual_pipeline_model_parallel_rank,
    is_pipeline_first_stage,
    get_data_modulo_expert_parallel_group,
    get_expert_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.distributed import DistributedDataParallelConfig, \
    DistributedDataParallel
from mindformers.experimental.parallel_core.pynative.optimizer import MixedPrecisionOptimizer, DistributedOptimizer
from mindformers.experimental.parallel_core.pynative.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
    forward_backward_pipelining_with_interleaving
)
from mindformers.experimental.parallel_core.pynative.config import GeneralConfig
from mindformers.experimental.parallel_core.pynative.dist_checkpointing import save_checkpoint
from mindformers.experimental.parallel_core.pynative.transformer.moe.utils import MoEAuxLossAutoScaler

from .grad_handler import inplace_apply_to_tensor_list, get_grad_process_func, GradAccumulator


def get_sp_params(training_config):
    """get reduce parameters for sequence parallel"""
    use_lora = training_config.lora_config.use_lora
    if use_lora:
        sp_params = [
            'qkv_proj.lora_a',
            'out_proj.lora_b',
            'mapping.lora_a',
            'projection.lora_b',
            'q_proj.lora_a',
            'kv_proj.lora_a',
            'gating.lora_a'
        ]
    else:
        sp_params = ["norm", "mlp.projection.bias", "attention.out_proj.bias"]
    return sp_params


def rename_set_hidden_states_parameter(model, model_chunk_id=None):
    """ rename set_hidden_states parameter """
    weight_untrainable = model.untrainable_params()
    for param in weight_untrainable:
        if "set_hidden_states" in param.name:
            param.name = param.name + f"_{model_chunk_id}_chunk"


def model_zero_grad_buffer(model, wrap_with_ddp):
    """ zero grad buffer if wrap_with_ddp=True """
    if wrap_with_ddp:
        if isinstance(model, nn.CellList):
            for model_chunk_id, _ in enumerate(model):
                model[model_chunk_id].zero_grad_buffer()
        else:
            model.zero_grad_buffer()


class ParallelTrainingReducer:
    """The reducer for parallel training"""

    def __init__(self, params, training_config):
        super(ParallelTrainingReducer, self).__init__()

        self.enable_grad_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,  # only valid in the case of sequence parallel
            "ep-dp": False
        }

        self.enable_loss_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.enable_grad_flag_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.sp_reduce_params = get_sp_params(training_config)
        self.expert_params = ["mlp.experts.local_experts"]

        self.batch_reduction = training_config.loss_reduction
        # dp
        if get_data_parallel_world_size() > 1:
            self.enable_loss_reduce["dp"] = True
            if training_config.parallel_config.zero_level is None \
                and not training_config.wrap_with_ddp:
                self.enable_grad_reduce["dp"] = True
            else:
                self.enable_grad_flag_reduce["dp"] = True

        # tp / sp
        if get_tensor_model_parallel_world_size() > 1:
            self.enable_grad_flag_reduce["tp"] = True
            if training_config.parallel_config.sequence_parallel:
                self.enable_grad_reduce["tp"] = True
                self.sp_reduce_filter = [
                    any([sp_param in param.name for sp_param in self.sp_reduce_params]) for param in params
                ]

        # pp
        if get_pipeline_model_parallel_world_size() > 1:
            self.enable_grad_flag_reduce["pp"] = True

        # ep
        if get_expert_model_parallel_world_size() > 1:
            self.enable_grad_reduce["ep-dp"] = True
            self.expert_filter = [
                any([ep_param in param.name for ep_param in self.expert_params]) for param in params
            ]

    def get_reduce_group(self, idx):
        if self.enable_grad_reduce["ep-dp"] and self.expert_filter[idx]:
            group = get_data_modulo_expert_parallel_group()
        else:
            group = get_data_parallel_group()
        return group

    def inplace_reduce_dp_grad(self, grads, params=None):
        """Reduce the gradients in data parallel mode."""
        if self.enable_grad_reduce["dp"]:
            if params is not None:
                for idx, param in enumerate(params):
                    if param.grad is None:
                        continue
                    group = self.get_reduce_group(idx)
                    param.grad = comm_func.all_reduce(param.grad, "sum", group)[0]
                    if self.batch_reduction == "mean":
                        param.grad = mint.div(param.grad, get_data_parallel_world_size())
            else:
                if self.batch_reduction == "mean":
                    for idx, grad in enumerate(grads):
                        group = self.get_reduce_group(idx)
                        grads[idx] = mint.div(
                            comm_func.all_reduce(grad, "sum", group)[0], get_data_parallel_world_size())
                elif self.batch_reduction == "sum":
                    for idx, grad in enumerate(grads):
                        group = self.get_reduce_group(idx)
                        grads[idx] = comm_func.all_reduce(grad, "sum", group)[0]

    def inplace_reduce_sp_grad(self, grads, params=None):
        """Reduce the gradients in sequence parallel mode over tp group."""
        if self.enable_grad_reduce["tp"]:
            if params is not None:
                for idx, param in enumerate(params):
                    if param.grad is None or not self.sp_reduce_filter[idx]:
                        continue
                    param.grad.copy_(comm_func.all_reduce(param.grad, "sum", get_tensor_model_parallel_group())[0])
            else:
                for idx, reduce_flag in enumerate(self.sp_reduce_filter):
                    if reduce_flag:
                        grads[idx] = comm_func.all_reduce(grads[idx], "sum", get_tensor_model_parallel_group())[0]

    def inplace_reduce_grad(self, grads, params=None):
        """Reduce the gradients in all parallel modes."""
        self.inplace_reduce_dp_grad(grads, params)
        self.inplace_reduce_sp_grad(grads, params)

    def reduce_dp_loss(self, loss):
        """Reduce the loss in data parallel mode."""
        if self.enable_loss_reduce["dp"]:
            if self.batch_reduction == "mean":
                loss = mint.div(
                    comm_func.all_reduce(loss, "sum", get_data_parallel_group())[0], get_data_parallel_world_size())
            else:
                loss = comm_func.all_reduce(loss, "sum", get_data_parallel_group())[0]
        return loss

    def reduce_overflow(self, overflow):
        """Reduce the overflow status in all parallel modes."""
        # logical or
        overflow = Tensor(overflow, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_tensor_model_parallel_group())[0]

    def reduce_is_finite(self, is_finite):
        """Reduce the is_finite status in all parallel modes."""
        # logical and
        is_finite = Tensor(is_finite, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_tensor_model_parallel_group())[0]
        return is_finite.astype(mstype.bool_)


def get_model(model_provider_func, training_config):
    """ get model """
    model = nn.CellList(auto_prefix=False)
    parallel_config = training_config.parallel_config
    if training_config.bf16 and training_config.wrap_with_ddp and \
            not training_config.accumulate_allreduce_grads_in_fp32:
        logger.warning("Using bf16 with ddp, automatically set 'accumulate_allreduce_grads_in_fp32=True'.")
        training_config.accumulate_allreduce_grads_in_fp32 = True
    if get_pipeline_model_parallel_world_size() > 1:
        if parallel_config.virtual_pipeline_model_parallel_size is not None and \
           parallel_config.virtual_pipeline_model_parallel_size > 1:
            for i in range(parallel_config.virtual_pipeline_model_parallel_size):
                set_virtual_pipeline_model_parallel_rank(i)
                pre_process = is_pipeline_first_stage()
                post_process = is_pipeline_last_stage()
                this_model = model_provider_func(pre_process=pre_process,
                                                 post_process=post_process)
                rename_set_hidden_states_parameter(this_model, i)
                model.append(this_model)
        else:
            pre_process = is_pipeline_first_stage()
            post_process = is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process,
                                             post_process=post_process)
            # wrap with PP cell if pipeline parallelism is used
            model.append(this_model)
    else:
        this_model = model_provider_func(pre_process=True, post_process=True)
        model.append(this_model)

    if training_config.wrap_with_ddp:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=training_config.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=training_config.overlap_grad_reduce,
            use_distributed_optimizer=training_config.use_distributed_optimizer,
            bucket_size=training_config.bucket_size,
            average_in_collective=(training_config.loss_reduction == 'mean'),
            check_for_nan_in_grad=training_config.check_for_nan_in_grad,
            enable_mem_align=training_config.enable_mem_align,
        )
        training_config.ddp_config = ddp_config
        logger.warning("Wrap model with DistributedDataParallel, ddp config:\n{}".format(ddp_config))
        model = nn.CellList([DistributedDataParallel(config=training_config,
                                                     ddp_config=ddp_config,
                                                     module=model_chunck) for model_chunck in model],
                            auto_prefix=False)

    return model


def get_forward_backward_func(network_with_loss, params, training_config, model_config):
    """
    Returns a forward-backward function for training a network with or without pipeline parallelism.

    Args:
        network_with_loss (callable): A function that takes inputs and returns the network output and loss.
        params (list): List of parameters to compute gradients for.
        training_config (TrainingConfig): Training configuration.
        model_config (TransformerConfig): Model configuration.

    Returns:
        callable: A forward-backward function that can be used for training the network.

    Raises:
        NotImplementedError: If pipeline parallelism is not implemented yet.
    """
    forward_backward_func = None
    seq_length = model_config.seq_length
    micro_batch_num = training_config.dataset_config.micro_batch_num
    micro_batch_size = training_config.dataset_config.batch_size
    data_layout = model_config.dataset_config.data_layout

    # no pipeline parallel
    if get_pipeline_model_parallel_world_size() == 1:

        def forward_with_loss_scale(*inputs_tuple, loss_scale=None, **inputs_dict):
            logits = None
            output = network_with_loss(*inputs_tuple, **inputs_dict)
            if isinstance(output, tuple):
                loss, logits = output[0], output[1]
            else:
                loss = output
            if loss_scale is not None:
                loss = mint.mul(loss, loss_scale.astype(loss.dtype))
            return loss, logits

        grad_position = None
        # If parallel_config.zero_level == z3, gradient with respect to inputs and weights
        if model_config.parallel_config.zero_level == "z3":
            grad_position = 0
        forward_backward_once_func = value_and_grad(
            forward_with_loss_scale, grad_position=grad_position, weights=params, has_aux=True
        )

        # if overlap_grad_reduce, grad will be accumulate in grad buffer
        if micro_batch_num > 1 and not training_config.wrap_with_ddp:
            grad_accumulator = GradAccumulator(micro_batch_num, op="sum")

        def forward_backward_func_with_grad_acc(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            loss = None
            logits = None
            grads = None

            # reset grad buffer
            model_zero_grad_buffer(network_with_loss, training_config.wrap_with_ddp)

            # fuse loss scale and grad accumulation if do grad acc
            if training_config.loss_reduction == "mean" and micro_batch_num > 1:
                if loss_scale is None:
                    loss_scale = Tensor(1, mstype.float32)
                actual_loss_scale = mint.div(loss_scale, micro_batch_num)
            else:
                actual_loss_scale = loss_scale

            if training_config.wrap_with_ddp:
                no_sync_func = network_with_loss.no_sync
            else:
                no_sync_func = contextlib.nullcontext

            def forward_backward_on_microbatch(idx):
                nonlocal loss
                nonlocal logits
                nonlocal grads

                # slice inputs over batch size dimension
                inputs_tuple_micro = [
                    input_data[idx * micro_batch_size : (idx + 1) * micro_batch_size] for input_data in inputs_tuple
                ]
                inputs_dict_micro = {
                    key: value[idx * micro_batch_size : (idx + 1) * micro_batch_size]
                    for key, value in inputs_dict.items()
                }

                # step on micro batch
                if forward_only:
                    loss_micro, logits_micro = forward_with_loss_scale(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                else:
                    (loss_micro, logits_micro), grads_micro = forward_backward_once_func(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                    if grad_position == 0:
                        grads_micro = grads_micro[1]
                    # accumulate grads
                    if micro_batch_num > 1 and not training_config.wrap_with_ddp:
                        grads = grad_accumulator(grads_micro)
                    else:
                        grads = grads_micro

                # process output, loss will be averaged in loss unscaling
                loss = loss_micro if loss is None else loss + loss_micro

                if logits is None:
                    logits = logits_micro
                else:
                    cat_dim = 0 if data_layout == "BSH" else 1
                    logits = mint.cat((logits, logits_micro), dim=cat_dim)

            # trigger dp reduce only on last step
            with no_sync_func():
                for idx in range(micro_batch_num - 1):
                    forward_backward_on_microbatch(idx)
            forward_backward_on_microbatch(micro_batch_num - 1)

            # unscale loss
            if loss_scale is not None:
                loss = mint.div(loss, loss_scale)

            if forward_only:
                return loss, logits

            # finalize ddp grad reduce
            if training_config.wrap_with_ddp:
                network_with_loss.final_grad_reduce()
            return (loss, logits), grads

        forward_backward_func = forward_backward_func_with_grad_acc

    else:
        all_config = GeneralConfig(
            model_config=model_config,
            training_config=training_config,
        )

        def forward_backward_with_pipelining(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            # reset grad buffer
            model_zero_grad_buffer(network_with_loss, training_config.wrap_with_ddp)

            if loss_scale is None:
                loss_scale = Tensor(1, mstype.float32)
            if model_config.parallel_config.virtual_pipeline_model_parallel_size is not None and \
               model_config.parallel_config.virtual_pipeline_model_parallel_size > 1:
                loss, logits, grads = forward_backward_pipelining_with_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=all_config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            else:
                loss, logits, grads = forward_backward_pipelining_without_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=all_config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            if forward_only:
                return loss, logits
            return (loss, logits), grads

        forward_backward_func = forward_backward_with_pipelining

    return forward_backward_func


class TrainOneStepCell(nn.Cell):
    r"""TrainOneStepCell with loss scaling, grad clipping, and grad accumulation.

    Args:
        network_with_loss (nn.Cell): The network with loss, output of the network should be loss,
            which is a scalar Tensor.
        optimizer (Optimizer): The optimizer used for training.
        opt_param_scheduler(OptimizerParamScheduler): Learning rate scheduler
        training_config (TrainingConfig): Training Configuration.
        model_config (TransformerConfig): Transformer Configuration.
        **kwargs: Additional keyword arguments.

    Raises:
        NotImplementedError: If gradient accumulation is not supported in pipeline parallel.

    Inputs:
        - **inputs_tuple** (Tuple[Tensor]) - Tuple of input tensors.
        - **inputs_dict** (Dict[str, Tensor]) - Dict of input tensors.

    Outputs:
        Tuple of 4 Tensor, the loss, overflow flag, current loss scale value, and learning rate.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **is_finite** (Tensor) -  A bool, whether grads is finite.
        - **loss scale** (Union[Tensor, None]) -  The loss scale value, None if not using loss scaling.
        - **learning rate** (Union[Tensor, List[Tensor])) -  The model learning rate.
    """

    # pylint: disable=W0613
    def __init__(self, network_with_loss, optimizer, opt_param_scheduler, training_config, model_config, **kwargs):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        if isinstance(network_with_loss, nn.CellList) and len(network_with_loss) == 1:
            network_with_loss = network_with_loss[0]
        self.network_with_loss = network_with_loss
        self.optimizer = optimizer
        self.model_config = model_config
        self.opt_param_scheduler = opt_param_scheduler
        self.increment = training_config.dataset_config.micro_batch_num * \
                    training_config.dataset_config.batch_size * \
                    get_data_parallel_world_size()
        self.wrap_with_ddp = training_config.wrap_with_ddp
        self.use_mixed_precision_optimizer = isinstance(optimizer, MixedPrecisionOptimizer)
        if isinstance(optimizer, DistributedOptimizer) and optimizer.config.overlap_param_gather:
            optimizer.enable_pre_hook(network_with_loss)

        self.params_with_grad = None if not self.use_mixed_precision_optimizer \
            else network_with_loss.trainable_params()

        # init loss scaler
        if training_config.loss_scale is not None:
            self.loss_scaler = StaticLossScaler(scale_value=training_config.loss_scale)
        else:
            # dynamic loss scaler is used only if the model is computed in float16
            if model_config.compute_dtype == mstype.float16:
                self.loss_scaler = DynamicLossScaler(
                    scale_value=training_config.loss_scale_value,
                    scale_factor=training_config.loss_scale_factor,
                    scale_window=training_config.loss_scale_window,
                )
            else:
                logger.warning(
                    "Dynamic loss scale is only supported for float16 computation. Not using loss scaling."
                )
                self.loss_scaler = StaticLossScaler(scale_value=1)

        # init grad clip func
        self.use_grad_clip = training_config.grad_clip_kwargs is not None
        if self.use_grad_clip:
            self.grad_clip_func = get_grad_process_func(
                training_config, not model_config.untie_embeddings_and_output_weights,
                params=network_with_loss.trainable_params()
            )
        # init grad scale func
        self.grad_scale_func = inplace_apply_to_tensor_list(mint.mul)

        # init parallel reducer
        self.parallel_reducer = ParallelTrainingReducer(network_with_loss.trainable_params(), training_config)

        self.micro_batch_num = training_config.dataset_config.micro_batch_num
        # init forward_backward_func
        self.forward_backward_func = get_forward_backward_func(
            network_with_loss, network_with_loss.trainable_params(), training_config, model_config
        )
        self.accumulate_allreduce_grads_in_fp32 = training_config.accumulate_allreduce_grads_in_fp32


    def unscale_and_clip_grads(self, grads, loss_scale=None):
        """Handle grads with scaling and clipping.

        Args:
            grads (tuple): The gradients.
            loss_scale (Tensor, optional): The scaling factor of loss. Defaults: None.
        """
        if loss_scale is not None:
            inv_scale = mint.reciprocal(loss_scale).astype(grads[0].dtype)
            self.grad_scale_func(grads, inv_scale)
        global_norm = None
        if self.use_grad_clip:
            global_norm = self.grad_clip_func(grads)
        return global_norm

    def construct(self, *inputs_tuple, **inputs_dict):
        """Forward, backward, grad process, and optimizer step."""
        # forward and backward
        if self.use_mixed_precision_optimizer:
            self.optimizer.zero_grad()

        if self.loss_scaler is not None:
            current_step_loss_scale = self.loss_scaler.scale_value
        else:
            current_step_loss_scale = None
        if self.model_config.moe_config.num_experts > 1:
            MoEAuxLossAutoScaler.set_loss_scale(mint.div(current_step_loss_scale, self.micro_batch_num))

        # loss is scale and unscale in forward_backward_func
        (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)

        # apply grad reducer
        grads = list(grads)
        if self.accumulate_allreduce_grads_in_fp32:
            grads = [grad.to(mstype.float32) for grad in grads]
        self.parallel_reducer.inplace_reduce_grad(grads, self.params_with_grad)

        # check overflow. When using mixed precision optimizer,
        # this process will be done in optimizer
        is_finite = True
        if not self.use_mixed_precision_optimizer and not self.wrap_with_ddp:
            is_finite = all_finite(grads)
            # sync over tp and pp group
            is_finite = self.parallel_reducer.reduce_is_finite(is_finite)

            if self.loss_scaler is not None:
                self.loss_scaler.adjust(is_finite)

        global_norm = None
        if is_finite:
            # scale grads and clip grads if enabled
            if not self.use_mixed_precision_optimizer:
                global_norm = self.unscale_and_clip_grads(grads, current_step_loss_scale)
                grads_tuple = tuple(grads)
                self.optimizer(grads_tuple)
            else:
                self.optimizer()

        # Update learning rate.
        if self.opt_param_scheduler:
            self.opt_param_scheduler.step(increment=self.increment)
        if isinstance(self.optimizer, mintOptimizer):
            learning_rate = self.optimizer.lrs
        else:
            learning_rate = self.optimizer.get_lr()
        if isinstance(learning_rate, (Parameter, Tensor)):
            learning_rate = float(learning_rate.value())
        if isinstance(learning_rate, (tuple, list)):
            learning_rate = tuple(
                float(individual_learning_rate.value())
                if isinstance(individual_learning_rate, (Parameter, Tensor))
                else individual_learning_rate
                for individual_learning_rate in learning_rate
            )

        # reduce loss if dp
        loss = self.parallel_reducer.reduce_dp_loss(loss)

        return loss, is_finite, current_step_loss_scale, learning_rate, global_norm


def train(
        train_one_step_cell,
        train_dataset_iterator,
        training_config,
        val_dataset_iterator=None,
        metrics=None,
        evaluation_func=None,
        resume_dict=None,
        **kwargs,
):
    """
    Train the model using the provided training configuration.

    Args:
        train_one_step_cell (TrainOneStepCell): The training cell object.
        train_dataset_iterator (Dataset): The iterator for the training dataset.
        training_config (TrainingConfig): The configuration object for training.
        val_dataset_iterator (iterable, optional): The iterator for the validation dataset. Defaults: None
        metrics (dict[str, Metric], optional): A dictionary of metrics to track during training.
            Defaults: None
        evaluation_func (callable, optional): The evaluation function to use for validation. Defaults: None
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    if training_config.resume_training and resume_dict is not None:
        initial_epoch = resume_dict.get("epoch_num")
        initial_step = resume_dict.get("step_num")
    else:
        initial_epoch = 0
        initial_step = 1
    train_one_step_cell.set_train()
    model_config = train_one_step_cell.model_config
    # set input to use dynamic shape
    model_input_data = next(train_dataset_iterator.create_tuple_iterator())
    set_input_data = [
        Tensor(shape=(None,) * len(input_data.shape), dtype=input_data.dtype) for input_data in model_input_data
    ]
    train_one_step_cell.set_inputs(*set_input_data)

    dataset_size = train_dataset_iterator.get_dataset_size()
    global_step = 1
    epoch_step = 1
    current_epoch = 0
    if training_config.resume_training:
        global_step = initial_step + initial_epoch * dataset_size + 1
        epoch_step = global_step % dataset_size
        current_epoch = global_step // dataset_size
    evaluation_flag = (
        val_dataset_iterator is not None
        and evaluation_func is not None
        and metrics is not None
        and training_config.eval_interval is not None
        and training_config.best_metric_comparison is not None
        and training_config.eval_metric is not None
    )
    save_ckpt_flag = training_config.save_interval is not None and training_config.training_iters != 0
    correct_metric_flag = is_pipeline_last_stage() # not use pp or pp last_stage

    if evaluation_flag:
        if training_config.best_metric_comparison == "less_equal":
            best_metric_compare_func = mint.less_equal
            best_metric = Tensor(float("inf"))
        elif training_config.best_metric_comparison == "greater_equal":
            best_metric_compare_func = mint.greater_equal
            best_metric = Tensor(float("-inf"))
        elif training_config.best_metric_comparison == "less":
            best_metric_compare_func = mint.less
            best_metric = Tensor(float("inf"))
        elif training_config.best_metric_comparison == "greater":
            best_metric_compare_func = mint.greater
            best_metric = Tensor(float("-inf"))

    # check if the training should be stopped
    while not (
            training_config.epochs is not None
            and current_epoch >= training_config.epochs
            or global_step > training_config.training_iters
    ):
        if epoch_step > 1:
            logger.debug(f"skip {epoch_step} step data")
            dataset_iterator = train_dataset_iterator.skip(epoch_step - 1).create_dict_iterator(num_epochs=1)
        else:
            dataset_iterator = train_dataset_iterator.create_dict_iterator(num_epochs=1)
        for data in dataset_iterator:
            # check if the training should be stopped
            if global_step > training_config.training_iters:
                break
            start_time = time.time()
            loss, is_finite, loss_scale, learning_rate, _ = train_one_step_cell(**data)
            end_time = time.time()
            if training_config.log_interval is not None and global_step % training_config.log_interval == 0:
                if not correct_metric_flag:
                    logger.warning("Metrics is only calculated on the last stage.")
                if isinstance(learning_rate, (tuple, list)):
                    report_learning_rate = '('
                    for lr in learning_rate:
                        report_learning_rate += "{:e},".format(lr)
                    report_learning_rate += ')'
                else:
                    report_learning_rate = "{:e}".format(learning_rate)
                logger.warning(
                    f"Epoch: {current_epoch}, Step: {epoch_step}, Loss: {loss}, "
                    + f"Finite_grads: {is_finite}, "
                    + f"Loss_scale: {loss_scale.value() if loss_scale is not None else None}, "
                    + f"Learning_rate: {report_learning_rate}, Time: {(end_time - start_time) * 1000:.2f} ms"
                )

            if evaluation_flag and global_step % training_config.eval_interval == 0:
                is_best = Tensor(False, dtype=mstype.int8)
                results = evaluation_func(train_one_step_cell, val_dataset_iterator, metrics, **kwargs)

                # update best_metrics only on last stage
                if correct_metric_flag and best_metric_compare_func(results[training_config.eval_metric], best_metric):
                    best_metric = results[training_config.eval_metric]
                    is_best = Tensor(True, dtype=mstype.int8)

                if get_pipeline_model_parallel_world_size() > 1:
                    is_best = comm_func.all_reduce(is_best, "max", get_pipeline_model_parallel_group())[0]

                # save ckpt
                if is_best and save_ckpt_flag:
                    logger.warning("saving best checkpoint")
                    if save_ckpt_flag:
                        save_checkpoint(model_config,
                                        train_one_step_cell.network_with_loss,
                                        train_one_step_cell.optimizer,
                                        train_one_step_cell.opt_param_scheduler,
                                        training_config.output_dir,
                                        format=training_config.ckpt_format,
                                        prefix=training_config.prefix + "_best",
                                        epoch_num=current_epoch,
                                        step_num=epoch_step,
                                        crc_check=training_config.crc_check,
                                        keep_checkpoint_max=training_config.keep_checkpoint_max + 1)

            if save_ckpt_flag and global_step % training_config.save_interval == 0:
                save_checkpoint(model_config,
                                train_one_step_cell.network_with_loss,
                                train_one_step_cell.optimizer,
                                train_one_step_cell.opt_param_scheduler,
                                training_config.output_dir,
                                format=training_config.ckpt_format,
                                prefix=training_config.prefix,
                                epoch_num=current_epoch,
                                step_num=epoch_step,
                                crc_check=training_config.crc_check,
                                keep_checkpoint_max=training_config.keep_checkpoint_max)
            epoch_step += 1
            global_step += 1
        epoch_step = 1
        current_epoch += 1

    if isinstance(train_one_step_cell.optimizer, DistributedOptimizer) \
            and train_one_step_cell.optimizer.config.overlap_param_gather:
        train_one_step_cell.optimizer.sync_gather_all_model_params(True)

    if save_ckpt_flag:
        logger.info("Saving last step checkpoint.")
        # at the end of training loop, we use `global_step += 1`,
        # so the right global step should be 'global_step - 1',
        epoch_step = (global_step - 1) % dataset_size
        current_epoch = (global_step - 1) // dataset_size
        # to avoid situation like 'epoch 1, step 0'
        if epoch_step == 0:
            epoch_step = dataset_size
            current_epoch -= 1
        save_checkpoint(model_config,
                        train_one_step_cell.network_with_loss,
                        train_one_step_cell.optimizer,
                        train_one_step_cell.opt_param_scheduler,
                        training_config.output_dir,
                        format=training_config.ckpt_format,
                        prefix=training_config.prefix,
                        epoch_num=current_epoch,
                        step_num=epoch_step,
                        crc_check=training_config.crc_check,
                        keep_checkpoint_max=training_config.keep_checkpoint_max)
