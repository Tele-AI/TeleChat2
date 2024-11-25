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

"""mindformers init"""
import os
from mindspore import mint, ops
from mindspore.common import dtype as mstype
from mindspore.nn import Adam, SGD
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.experimental.optim.adamw import SpeedAdamW

from mindformers.core.optim import Came
from mindformers.core.optim import AdamW as mf_AdamW
from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry
from mindformers.experimental.parallel_core.pynative.distributed import DistributedDataParallel
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_group
from mindformers.experimental.parallel_core.pynative.dist_checkpointing import get_checkpoint_name
from mindformers.experimental.parallel_core.pynative.optimizer.lr_scheduler import get_learning_rate_scheduler

from . import zero
from . import lr_scheduler
from .distrib_optimizer import DistributedOptimizer
from .optimizer import MixedPrecisionOptimizer, Float16OptimizerWithFloat16Params, get_optimizer_param_scheduler

__all__ = [
    "DistributedOptimizer", "MixedPrecisionOptimizer", "Float16OptimizerWithFloat16Params", \
    "get_optimizer", "get_optimizer_param_scheduler"
]
__all__.extend(zero.__all__)
__all__.extend(lr_scheduler.__all__)

ModuleRegistry.register(mf_AdamW, ModuleType.OPTIMIZER)
ModuleRegistry.register(Adam, ModuleType.OPTIMIZER)
ModuleRegistry.register(SGD, ModuleType.OPTIMIZER)
ModuleRegistry.register(Came, ModuleType.OPTIMIZER)
ModuleRegistry.register(mint.optim.AdamW, ModuleType.OPTIMIZER, item_name='mint.AdamW')
ModuleRegistry.register(SpeedAdamW, ModuleType.OPTIMIZER, item_name='SpeedAdamW')


def get_ditributed_optimizer(optimizer, optimizer_config, training_config, model_chunks):
    " warp non-parallel optimizer with distributed optimizer. "
    if model_chunks is None:
        raise ValueError("When using DistributedOptimizer based on DDP, network instance should be passed "
                         "to get_optimizer method but got None.")
    per_model_buffers = {}
    per_model_ep_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if not isinstance(model_chunk, DistributedDataParallel):
            raise TypeError("When using DistribtedOptimizer, the network passed to get_optimizer should be "
                            "wrapped with DistributedDataParallel.")
        per_model_buffers[model_idx] = model_chunk.buffers
        per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers
    grad_scaler = None if not training_config.loss_scale \
        else ops.Tensor(training_config.loss_scale, mstype.float32)
    distributed_optimizer = DistributedOptimizer(
        optimizer=optimizer,
        config=optimizer_config,
        grad_scaler=grad_scaler,
        init_state_fn=None,
        per_model_buffers=per_model_buffers,
        data_parallel_group=get_data_parallel_group(with_context_parallel=True),
    )
    ckpt_file, _ = get_checkpoint_name(os.path.join(training_config.output_dir, 'opt_shard_info'),
                                       format='json', prefix='dist_opt_shard_info', epoch_num=0, step_num=0)

    distributed_optimizer.save_opt_shard_strategy(ckpt_file)

    return distributed_optimizer


def get_non_distributed_mixed_precision_optimizer(optimizer, optimizer_config, training_config):
    " warp non-parallel optimizer with Float16OptimizerWithFloat16Params optimizer. "
    grad_scaler = None if not training_config.loss_scale \
        else ops.Tensor(training_config.loss_scale, mstype.float32)
    optimizer = Float16OptimizerWithFloat16Params(
        optimizer,
        optimizer_config,
        grad_scaler=grad_scaler,
        init_state_fn=None,
        wrap_with_ddp=training_config.wrap_with_ddp,
    )
    return optimizer


def _set_group_lr_and_weight_decay(optimizer_config, params, lr, weight_decay):
    if isinstance(params[0], dict) and not optimizer_config.optimizer_type.startswith("mint") \
        and not optimizer_config.optimizer_type.startswith("Speed"):
        using_group_lr = any("lr" in param for param in params)
        for param in params:
            if "order_params" not in param:
                if "lr" not in param and using_group_lr:
                    param["lr"] = lr
                if "weight_decay" not in param:
                    param["weight_decay"] = weight_decay


def _append_order_param_group(params, network, optimizer_cls):
    """
    Append 'order_params' parameter group to params when a user invokes
    'get_optimizer' with parameter groups and intends to create a
    subclass instance of mindspore.nn.optim.optimizer.Optimizer

    NOTE: mindspore.nn.optim.optimizer.Optimizer assumes that 'order_params' contains
    the original parameter list of network and arranges its parameter list
    following the order of 'order_params'.
    """
    if issubclass(optimizer_cls, Optimizer) and \
        isinstance(params, list) and \
        all(isinstance(t, dict) and "params" in t for t in params):
        if network is None:
            raise ValueError("Network must be provided when using built-in "
                             "mindspore.nn.optim.optimizer.Optimizer")
        params.append({"order_params": network.trainable_params()})
    return params


def get_optimizer(optimizer_config, training_config, params=None, network=None, return_instance: bool = True, **kwargs):
    """
    Get an optimizer instance or class based on the provided optimizer configuration.

    Args:
        optimizer_config (OptimizerConfig): The configuration object for the optimizer.
        params (list or dict, optional): The parameters to optimize. Default: None.
        network (nn.Cell, optional): The network model, should be provided when use ZeRO optimizer. Default: None.
        return_instance (bool): Whether to return an instance of the optimizer or just the optimizer class.
        **kwargs: Additional keyword arguments to be passed to the optimizer class.

    Returns:
        Optimizer or type: An instance of the optimizer class if `return_instance` is True,
        otherwise the optimizer class itself.

    Raises:
        ValueError: If `params` is None and `return_instance` is True.
        ValueError: If `network` is None and use ZeRO optimizer.
        NotImplementedError: If `weight_decay_kwargs` is not supported yet.

    """
    if optimizer_config.parallel_config.zero_level is not None:
        optimizer_type = optimizer_config.optimizer_type + "ZeRO"
    else:
        optimizer_type = optimizer_config.optimizer_type

    optimizer_cls = ModuleRegistry.get_item(module_type=ModuleType.OPTIMIZER, item_name=optimizer_type)
    if not return_instance:
        return optimizer_cls

    if params is None:
        raise ValueError("params must be provided when return_instance is True.")

    params = _append_order_param_group(params, network, optimizer_cls)

    if optimizer_config.weight_decay_kwargs is not None:
        raise NotImplementedError("weight_decay_kwargs is not supported yet.")

    weight_decay = optimizer_config.weight_decay

    if optimizer_config.learning_rate_scheduler_kwargs is not None:
        learning_rate = get_learning_rate_scheduler(optimizer_config)
    else:
        learning_rate = optimizer_config.learning_rate

    _set_group_lr_and_weight_decay(optimizer_config, params, learning_rate, weight_decay)

    optimizer_kwargs = optimizer_config.get_needed_params_for_class(optimizer_cls)
    if optimizer_config.optimizer_type.startswith("mint") or optimizer_config.optimizer_type.startswith("Speed"):
        optimizer_kwargs["lr"] = learning_rate
        optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
    else:
        optimizer_kwargs["learning_rate"] = learning_rate
    optimizer_kwargs["weight_decay"] = weight_decay
    optimizer_kwargs["params"] = params
    if "grad_allreduce_op" in kwargs:
        if optimizer_config.parallel_config.zero_level is not None:
            optimizer_kwargs["grad_allreduce_op"] = kwargs["grad_allreduce_op"]
        kwargs.pop("grad_allreduce_op", None)
    if optimizer_config.parallel_config.zero_level is not None:
        if network is None:
            raise ValueError("Network must be provided when get ZeRO optimizer instance.")
        optimizer_kwargs["zero_level"] = optimizer_config.parallel_config.zero_level
        optimizer_kwargs["network"] = network
        if optimizer_config.zero_config is not None:
            optimizer_kwargs.update(optimizer_config.zero_config)
    optimizer_kwargs.update(kwargs)
    return_item = optimizer_cls(**optimizer_kwargs)

    if training_config.wrap_with_ddp and training_config.use_distributed_optimizer:
        return_item = get_ditributed_optimizer(
            return_item,
            optimizer_config,
            training_config,
            network,
        )
    elif training_config.fp16 or training_config.bf16:
        return_item = get_non_distributed_mixed_precision_optimizer(
            return_item,
            optimizer_config,
            training_config,
        )
    return return_item
