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
"""optimizer registration and factory method"""
from collections import OrderedDict

from mindspore import mint, ops, Tensor, Parameter, ParameterTuple
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.communication.comm_func as comm_func
from mindformers.experimental.parallel_core.pynative.training.optimizer_param_scheduler import OptimizerParamScheduler

from mindformers.tools import logger
from mindformers.experimental.parallel_core.pynative.training.grad_handler import inplace_apply_to_tensor_list, \
    get_grad_norm_fp32, clip_grad_by_total_norm_fp32, param_is_not_shared
from mindformers.experimental.parallel_core.pynative.parallel_state import get_tensor_model_parallel_rank, \
    get_data_parallel_world_size, get_model_parallel_group


def get_optimizer_param_scheduler(optimizer, optimizer_config, dataset_config, training_config):
    """ Build the learning rate scheduler."""
    # Iteration-based training.
    dp = get_data_parallel_world_size()
    global_batch_size = dataset_config.batch_size * dp * dataset_config.micro_batch_num
    if training_config.training_iters > 0:
        if optimizer_config.lr_decay_iters is None:
            optimizer_config.lr_decay_iters = training_config.training_iters

        lr_decay_steps = optimizer_config.lr_decay_iters * global_batch_size
        wd_incr_steps = training_config.training_iters * global_batch_size
        wsd_decay_steps = None
        if optimizer_config.lr_wsd_decay_iters is not None:
            wsd_decay_steps = optimizer_config.lr_wsd_decay_iters * global_batch_size
        if optimizer_config.lr_warmup_fraction is not None:
            lr_warmup_steps = optimizer_config.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = optimizer_config.lr_warmup_iters * global_batch_size
    # Sample-based training.
    elif dataset_config.train_samples:
        training_config.training_iters = dataset_config.train_samples // global_batch_size
        if optimizer_config.lr_decay_samples is None:
            optimizer_config.lr_decay_samples = dataset_config.train_samples
        lr_decay_steps = optimizer_config.lr_decay_samples
        wd_incr_steps = dataset_config.train_samples
        wsd_decay_steps = optimizer_config.lr_wsd_decay_samples
        if optimizer_config.lr_warmup_fraction is not None:
            lr_warmup_steps = optimizer_config.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = optimizer_config.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be positive number.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=optimizer_config.lr_warmup_init,
        max_lr=optimizer_config.learning_rate,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=optimizer_config.lr_decay_style,
        start_wd=optimizer_config.start_weight_decay,
        end_wd=optimizer_config.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=optimizer_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=optimizer_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=optimizer_config.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=optimizer_config.lr_wsd_decay_style
        )

    return opt_param_scheduler


def _zero_param_group_grad_helper(group):
    """ zero grad data of parameters. """
    for param in group:
        if hasattr(param, 'grad') and param.grad is not None:
            param.grad = None


class MixedPrecisionOptimizer(nn.Cell):
    """
    MixedPrecision Optimizer base class.

    Args:
        optimizer (mindspore.experimental.optim.optimizer): Base optimizer.
        config (OptimizerConfig): Configuration object for optimizer.
        grad_scaler (GradScaler): Gradient scaling. When `grad_scaler=None`, no scaler will be used for
            gradients.
        init_state_fn: Function to initialize state parameters of optimizer.
    """
    def __init__(
            self,
            optimizer,
            config,
            grad_scaler,
            init_state_fn
        ):
        super(MixedPrecisionOptimizer, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.config = config
        self.grad_scaler = grad_scaler
        if init_state_fn is not None:
            logger.warning("Float16OptimizerWithFloat16Params only support AdamW optimizer for now. "
                           "The 'init_state_fn' will not be used.")
        self.init_state_fn = init_state_fn

        if self.grad_scaler is None:
            self._scale_one = Tensor([1.0], dtype=mstype.float32)

        self.grad_scale_func = inplace_apply_to_tensor_list(mint.mul)

        self.grads = []
        self.found_inf = Tensor(False, dtype=mstype.bool_)
        self._scale_zero = Tensor([0.0], dtype=mstype.float32)

    def _get_lrs(self):
        """ get lrs. """
        return self.optimizer.lrs

    def _set_lrs(self, value):
        """ set lrs. """
        self.optimizer.lrs = value

    lrs = property(_get_lrs, _set_lrs)

    def zero_grad(self):
        """ zero grad data. """
        return

    # pylint: disable=R1705
    def get_loss_scale(self):
        """ get loss scale. """
        if self.grad_scaler is None:
            return self._scale_one
        elif isinstance(self.grad_scaler, Tensor):
            return self.grad_scaler
        return self.grad_scaler.scale

    def reload_model_params(self):
        """ copy model params to its fp32 copy. """
        self._copy_model_params_to_main_params()

    def get_parameters_(self):
        """ get parameters registered to optimizer in order. """
        return self.optimizer.parameters

    def get_lr(self):
        """ get learning rate. """
        return tuple(self.optimizer.lrs)

    def get_model_parallel_group(self):
        """ return model_parallel_group for global norm allreduce. """
        return get_model_parallel_group()

    def get_main_grads_for_grad_norm(self):
        """ collect main gradients for grad norm compute. """
        params = self.get_parameters_()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = not (
                ("norm" in param.name)
                or ("mlp.projection.bias" in param.name)
                or ("attention.out_proj.bias" in param.name)
            ) or (get_tensor_model_parallel_rank() == 0)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        return grads_for_norm

    def clip_grad_norm(self, clip_grad):
        """ clip gridients by global norm. """
        params = self.get_parameters_()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, model_parallel_group=self.get_model_parallel_group())
        clip_grad_by_total_norm_fp32(params, clip_grad, grad_norm)
        return grad_norm

    def _unscale_main_grads_and_check_for_nan(self):
        """ check nan in main grads and unscale when using grad_scaler. """
        self._collect_main_grad_data()
        self.found_inf = Tensor(False, mstype.bool_)
        inv_scale = mint.reciprocal(self.grad_scaler).astype(mstype.float32)
        self.grad_scale_func(self.grads, inv_scale)
        for grad in self.grads:
            self.found_inf = mint.logical_and(self.found_inf, mint.logical_not(mint.isfinite(grad)).all())
        self.found_inf = comm_func.all_reduce(
            self.found_inf.astype(mstype.float32), 'max', get_model_parallel_group())[0]
        return mint.greater(self.found_inf, self._scale_zero)

    # pylint: disable=R1705
    def prepare_grads(self):
        """ grads overflow check and unscaling. """
        self._copy_model_grads_to_main_grads()
        if self.grad_scaler:
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            return found_inf_flag
        else:
            self._collect_main_grad_data()
        return False

    def step_with_ready_grads(self):
        """ optimizer update and copy from fp32 copy to model params. """
        success = self.optimizer(self.grads)
        self._copy_main_params_to_model_params()
        return success

    def construct(self):
        """ construct function. """
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        num_zeros_in_grad = None
        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros_in_grad

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """
    Mixed precision optimizer with float16/bfloat16 parameters and gradients.

    Args:
        optimizer (mindspore.experimental.optim.optimizer): Base optimizer.
        config (OptimizerConfig): Configuration object for optimizer.
        grad_scaler (GradScaler): Gradient scaling. When `grad_scaler=None`, no scaler will be used for
            gradients.
        init_state_fn: Function to initialize state parameters of optimizer.
        wrap_with_ddp: Indicate whether the model has been wrapped with DistributedDataParallel. Default: False.
    """
    def __init__(
            self,
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            wrap_with_ddp=False,
        ):
        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        )
        self.wrap_with_ddp = wrap_with_ddp

        self.optimizer.ori_parameters = self.optimizer.parameters
        self.optimizer.parameters = list(self.optimizer.parameters)
        self.optimizer.exp_avg = list(self.optimizer.exp_avg)
        self.optimizer.exp_avg_sq = list(self.optimizer.exp_avg_sq)

        # group parameters by their dtype
        self.fp16_groups = []
        self.fp32_from_fp16_groups = []
        self.fp32_groups = []
        self.grads = []

        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            fp16_params_this_group = []
            fp32_from_fp16_params_this_group = []
            fp32_params_this_group = []

            for param_idx, param in enumerate(param_group['params']):
                if not param.requires_grad:
                    continue

                if param.dtype in [mstype.bfloat16, mstype.float16]:
                    fp16_params_this_group.append(param)
                    # create fp32 copy
                    main_param = Tensor(param.value().astype(mstype.float32))
                    main_param.name = param.name
                    param.main_param = main_param
                    fp32_from_fp16_params_this_group.append(main_param)

                    # global index of parameter in optimizer.parameters
                    param_world_index = self.optimizer.group_start_id[group_idx] + param_idx
                    # replace registered parameter with its fp32 copy
                    param_group['params'][param_idx] = main_param
                    self.optimizer.parameters[param_world_index] = main_param

                    # update state with fp16/bf16 to fp32 copy
                    # only support AdamW optimizer for now.
                    self.optimizer.exp_avg[param_world_index] = Parameter(
                        self.optimizer.exp_avg[param_world_index].value().astype(mstype.float32),
                        name=self.optimizer.exp_avg[param_world_index].name,
                        requires_grad=self.optimizer.exp_avg[param_world_index].requires_grad
                    )
                    self.optimizer.exp_avg_sq[param_world_index] = Parameter(
                        self.optimizer.exp_avg_sq[param_world_index].value().astype(mstype.float32),
                        name=self.optimizer.exp_avg_sq[param_world_index].name,
                        requires_grad=self.optimizer.exp_avg_sq[param_world_index].requires_grad
                    )
                elif param.dtype == mstype.float32:
                    fp32_params_this_group.append(param)
                else:
                    raise TypeError("Invalid parameter type, parameter registered in optimizer should be "
                                    "one of [mstype.float32, mstype.bfloat16, mstype.float16], "
                                    "but got {}".format(param.dtype))

                if not self.wrap_with_ddp:
                    # register hook function for parameters which sets param.grad attr
                    param.register_hook(self._make_param_hook(param))

            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            self.fp32_groups.append(fp32_params_this_group)

        self.optimizer.exp_avg = ParameterTuple(self.optimizer.exp_avg)
        self.optimizer.exp_avg_sq = ParameterTuple(self.optimizer.exp_avg_sq)

        self.parameters = self.optimizer.ori_parameters
        self.defaults = self.optimizer.defaults
        self.reload_model_params()

    def zero_grad(self):
        """ reset parameter grad. """
        for group in self.fp16_groups:
            _zero_param_group_grad_helper(group)
        for group in self.fp32_from_fp16_groups:
            _zero_param_group_grad_helper(group)
        for group in self.fp32_groups:
            _zero_param_group_grad_helper(group)
        # reset self.grads
        self.grads = []

    def reload_main_params(self):
        """ reload main params to model params. """
        self._copy_main_params_to_model_params()

    def load_state_dict(self, state_dict):
        """ load state dict into optimizer. """
        param_dict = list(self.optimizer.parameters) + \
                     list(self.optimizer.exp_avg) + \
                     list(self.optimizer.exp_avg_sq)
        for param in param_dict:
            if not param.name in state_dict:
                logger.warning("No state data found for {}, it will not be loaded.".format(param.name))
            param.copy_(state_dict[param.name])

        if 'state_step' in state_dict.keys():
            self.optimizer.state_step.assign_value(state_dict['state_step'].value())

        # load learning rate
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            if lr_name in state_dict.keys():
                lr = state_dict[lr_name]
                self.optimizer.param_groups[group_idx]['lr'] = lr.item()
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            if wd_name in state_dict.keys():
                self.optimizer.param_groups[group_idx]['weight_decay'] = state_dict.get(wd_name).item()

    def state_dict(self):
        """ get optimizer state dict for saving checkpoint. """
        param_dict = OrderedDict()

        for param in self.optimizer.parameters:
            if isinstance(param, Parameter):
                param_dict[param.name] = param
            elif isinstance(param, Tensor):
                param_dict[param.name] = Parameter(param, name=param.name)
            else:
                raise TypeError("Instance in optimizer.parameters should be mindspore.Parameter or "
                                "mindspore.Tensor, but got {}".format(type(param)))

        for param in self.optimizer.exp_avg:
            param_dict[param.name] = param

        for param in self.optimizer.exp_avg_sq:
            param_dict[param.name] = param

        # add state step to state_dict
        param_dict['state_step'] = self.optimizer.state_step

        # add learning rate and weight decay to state_dict
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            param_dict[lr_name] = lr
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            param_dict[wd_name] = Parameter(
                ops.Tensor(
                    self.optimizer.param_groups[group_idx]['weight_decay'],
                    dtype=mstype.float64,
                ),
                name=wd_name,
                requires_grad=False,
            )

        return param_dict

    def _collect_main_grad_data(self):
        """ collect main grad for unscaling """
        for param in self.optimizer.parameters:
            self.grads.append(param.grad)

    def _get_model_and_main_params_data(self):
        """ get model and main params. """
        model_params = []
        main_params = []
        for model_group, main_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_params.append(model_param)
                main_params.append(main_param)
        return model_params, main_params

    def _copy_model_grads_to_main_grads(self):
        """ copy model grads to main grads. """
        for model_group, main_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.astype(mstype.float32)
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.astype(mstype.float32)

        for model_group in self.fp32_groups:
            for model_param in model_group:
                if hasattr(model_param, 'main_grad'):
                    model_param.grad = model_param.main_grad

    def _copy_model_params_to_main_params(self):
        """ copy model param to main param. """
        model_params, main_params = self._get_model_and_main_params_data()
        for model_param, main_param in zip(model_params, main_params):
            main_param.copy_(model_param.value().astype(mstype.float32))

    def _copy_main_params_to_model_params(self):
        """ copy main param to model param. """
        model_params, main_params = self._get_model_and_main_params_data()
        for main_param, model_param in zip(main_params, model_params):
            model_param.copy_(main_param.value().astype(model_param.dtype))

    def _make_param_hook(self, param):
        """ make closure function as the param hook. """
        def param_hook(grad):
            # when using bf16, gradients shuold be cast to fp32 for communication and optim
            if param.grad is not None:
                # grad accumulate
                param.grad = mint.add(param.grad, grad)
            else:
                if grad.dtype == mstype.bfloat16:
                    param.grad = ops.cast(grad, mstype.float32)
                else:
                    param.grad = grad
            return param.grad

        return param_hook
