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
"""clip grad and scale grad"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore._checkparam as validator
from mindspore import mint
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.communication import get_group_size, GlobalComm
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage
)
from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry


def inplace_apply_to_tensor_list(func: callable):
    """Apply a function to a list of tensors in place.

    Args:
        func (callable): The function to apply to each tensor in the list.
    Returns:
        callable: The function that applies the input function to each tensor in the list in place.
    """

    def inplace_apply_func(tensor_list, *args, **kwargs):
        for idx, _ in enumerate(tensor_list):
            tensor_list[idx].copy_(func(tensor_list[idx], *args, **kwargs))

    return inplace_apply_func


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


def get_grad_norm_fp32(grads_for_norm, norm_type=2.0, model_parallel_group=None):
    """ get fp32 global grad norm. """
    if isinstance(grads_for_norm, ms.Tensor):
        grads_for_norm = [grads_for_norm]

    if not model_parallel_group:
        model_parallel_group = GlobalComm.WORLD_COMM_GROUP

    norm_type = float(norm_type)
    total_norm = ms.Tensor(0.0, mstype.float32)

    if norm_type == 2.0:
        for grad in grads_for_norm:
            grad_norm = mint.norm(grad)
            total_norm = mint.add(total_norm, mint.square(grad_norm))
        total_norm = mint.sqrt(total_norm)
        total_norm = mint.square(total_norm)
    else:
        raise NotImplementedError("for global norm, l2 norm only support now")

    if get_group_size(model_parallel_group) > 1:
        total_norm = comm_func.all_reduce(total_norm, "sum", model_parallel_group)[0]
    total_norm = total_norm.item() ** (1.0 / norm_type)
    return total_norm


def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm):
    """ clip gradients by global norm. """
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grad_func = inplace_apply_to_tensor_list(mint.mul)
        grad_func(grads, clip_coeff)


@ModuleRegistry.register_decorator(ModuleType.GRAD_PROCESS_FUNC)
class GradClipByValue(nn.Cell):
    def __init__(self, clip_value):
        super(GradClipByValue, self).__init__()
        self.clip_value = clip_value
        self.clip_func = inplace_apply_to_tensor_list(F.clip_by_value)

    def construct(self, grads):
        return self.clip_func(grads, -self.clip_value, self.clip_value)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = ops.norm(grad) ** 2.0
    norm = F.expand_dims(norm, 0)
    return norm


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = ops.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


@ModuleRegistry.register_decorator(ModuleType.GRAD_PROCESS_FUNC)
class ClipGlobalNorm(nn.Cell):
    """
    clip grad by global norm
    """

    def __init__(self, params, reduce_comm_group, clip_value=1.0, norm_type="l2",
                 share_embeddings_and_output_weights=True):
        super(ClipGlobalNorm, self).__init__()
        self.params = params
        self.clip_value = clip_value
        self.hyper_map = C.HyperMap()
        self.norm_type = norm_type
        self.reduce_comm_group = reduce_comm_group
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.clip_func = inplace_apply_to_tensor_list(self.grad_scale_func)

    def grad_scale_func(self, grad, scale):
        """ function of scaling grads """
        return grad * scale

    def get_grads(self, grads):
        """
        get grads to norm, include weight/bias(not duplicate) and layernorm(duplicate, only pick grad on rank0)
        """
        rank_id = get_tensor_model_parallel_rank()
        norm_grads = ()
        for i, param in enumerate(self.params):
            tp_duplicate_params = (
                ("norm" in param.name)
                or ("mlp.projection.bias" in param.name)
                or ("attention.out_proj.bias" in param.name)
            )
            if tp_duplicate_params:
                if rank_id == 0:
                    norm_grads = norm_grads + (grads[i],)
            elif is_pipeline_last_stage():
                if self.share_embeddings_and_output_weights and 'language_model.output_layer.weight' in param.name:
                    continue
                else:
                    norm_grads = norm_grads + (grads[i],)
            else:
                norm_grads = norm_grads + (grads[i],)
        return norm_grads


    def construct(self, grads):
        """clip grad by global norm."""
        norm_grads = self.get_grads(grads)
        if self.norm_type == "l2":
            l2_norm = 2.0
        else:
            raise NotImplementedError("for global norm, l2 norm only support now")
        total_norm = get_grad_norm_fp32(norm_grads, norm_type=l2_norm, model_parallel_group=self.reduce_comm_group)
        clip_coeff = self.clip_value / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            self.clip_func(grads, clip_coeff)
        return total_norm

def get_grad_process_func(training_config, share_embeddings_and_output_weights=True, return_instance=True, **kwargs):
    """
    Get the gradient processing function based on the provided training configuration.

    Args:
        training_config (TrainingConfig): The training configuration object.
        return_instance (bool, optional): Whether to return an instance of the gradient processing function.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        Union[Type[GradProcessFunc], GradProcessFunc]: The gradient processing function or its instance.

    Raises:
        ValueError: If `params` is not provided for the "ClipGlobalNorm" gradient clip type.

    """
    grad_process_func_kwargs = training_config.grad_clip_kwargs.copy()
    grad_clip_type = grad_process_func_kwargs.pop("grad_clip_type")
    grad_clip_cls = ModuleRegistry.get_item(module_type=ModuleType.GRAD_PROCESS_FUNC, item_name=grad_clip_type)
    if return_instance:
        if grad_clip_type == "ClipGlobalNorm":
            if "params" not in kwargs:
                raise ValueError("params is required for ClipGlobalNorm")
            grad_process_func_kwargs["params"] = kwargs["params"]
            if training_config.use_distributed_optimizer or \
                    training_config.parallel_config.zero_level in ["z2", "z3"]:
                reduce_comm_group = GlobalComm.WORLD_COMM_GROUP
            else:
                reduce_comm_group = get_model_parallel_group()
            grad_process_func_kwargs["reduce_comm_group"] = reduce_comm_group
            grad_process_func_kwargs["share_embeddings_and_output_weights"] = share_embeddings_and_output_weights
        return grad_clip_cls(**grad_process_func_kwargs)
    return grad_clip_cls


class GradAccumulator:
    '''
    Gradient accumulator.

    Args:
        micro_batch_num (int): Gradient accumulation steps.
        op (str): Operate on the result of gradient accumulation. like sum, mean. Default: "mean".

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of parameters, the shape is the same as parameters.

    Outputs:
        - Tensor, accumulated gradients, the shape and type is the same asgradients.

    Raises:
        NotImplementedError: If `op` is not mean or sum.

    Examples:
        >>> from mindformers.experimental.distri_cores.grad_handler import GradAccumulator
        >>> micro_batch_num = 2
        >>> accumulator = GradAccumulator(micro_batch_num)
        >>> grad_func = ops.value_and_grad(network, grad_position=0, weights=optimizer.parameters)
        >>> loss, grads = grad_func(input_ids, labels)
        >>> grads = accumulator(grads)
        >>> if grads is not None:
        ...     print("do optimizer")
        ...     optimizer(grads)
    '''
    def __init__(self, micro_batch_num, op="mean"):
        self.counter = 0
        validator.check_non_negative_int(micro_batch_num, "accumulate_step")
        self.accumulate_step = micro_batch_num
        if op not in ["mean", "sum"]:
            raise NotImplementedError(f"{op} is not supported in GradAccumulator yet.")
        self.mean_op = op == "mean"
        self.map = ops.HyperMap()
        self.has_init = False
        self.need_clear = False
        self.inner_grads = None

        self.zeroslike = ops.ZerosLike()

    def _init_inner_grads(self, param):
        return self.zeroslike(param)

    def _clear_value(self, inner_grads):
        zeros = self.zeroslike(inner_grads)
        inner_grads.assign_value(zeros)

    def _mean_value(self, inner_grads):
        inner_grads.assign_value(inner_grads / self.accumulate_step)

    def __call__(self, grads):
        if not self.has_init:
            self.inner_grads = self.map(ops.partial(self._init_inner_grads), grads)
            self.has_init = True
        if self.need_clear:
            self.map(ops.partial(self._clear_value), self.inner_grads)
            self.need_clear = False
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        self.counter += 1
        if self.counter % self.accumulate_step == 0:
            if self.mean_op:
                self.map(ops.partial(self._mean_value), self.inner_grads)
            self.need_clear = True
            return self.inner_grads
        return None
