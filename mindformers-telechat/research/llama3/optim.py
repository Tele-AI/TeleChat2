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
"""AdamWeightX API"""
import numpy as np
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.optim.optim import _check_param_value

op_mul = P.Mul()
op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_maximum = P.Maximum()
addcmul = P.Addcmul()

__all__ = ['AdamWeightDecayX']

_adam_opt = C.MultitypeFuncGraph("adam_opt")


@_adam_opt.register("Float", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Bool")
def _update_run_op(weight_decay, lr, eps, state_step, beta1, beta2, param, grad,
                   exp_avg, exp_avg_sq, optim_filter):
    """Apply adamw optimizer to the weight parameter."""
    op_cast = P.Cast()
    if optim_filter:
        param_fp32 = op_cast(param, mstype.float32)
        next_param = op_mul(param_fp32, 1 - lr * weight_decay)
        gradient_fp32 = op_cast(grad, mstype.float32)

        F.depend(next_param, F.assign(exp_avg, op_mul(exp_avg, beta1) + op_mul(gradient_fp32,
                                                                               op_cast(F.tuple_to_array((1.0,)),
                                                                                       mstype.float32) - beta1)))
        F.depend(next_param, F.assign(exp_avg_sq, addcmul(op_mul(exp_avg_sq, beta2), gradient_fp32, gradient_fp32,
                                                          op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta2)))

        bias_correction1 = 1 - op_pow(op_cast(beta1, mstype.float32), state_step)
        bias_correction2 = 1 - op_pow(op_cast(beta2, mstype.float32), state_step)
        step_size = lr * op_sqrt(bias_correction2) / bias_correction1

        denom = op_sqrt(exp_avg_sq) + eps

        return_param = next_param - op_mul(exp_avg / denom, step_size)
        F.assign(param, op_cast(return_param, F.dtype(param)))
        return op_cast(return_param, F.dtype(param))
    return op_cast(grad, F.dtype(param))


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class AdamWeightDecayX(Optimizer):
    """AdamWeightDecayX Optimizer."""
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super(AdamWeightDecayX, self).__init__(learning_rate, params, weight_decay=weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.exp_avg = self.clone_state(prefix="adam_m", init='zeros')
        self.exp_avg_sq = self.clone_state(prefix="adam_v", init='zeros')
        self.weight_decay = weight_decay
        if context.get_context("device_target") == "Ascend":
            self.use_fused_opt = False
        else:
            self.use_fused_opt = True

    def clone_state(self, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        parameter_tuple = self.parameters
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)

    # pylint: disable=W0221
    def construct(self, gradients):
        """construct with gradients, scaling, step"""
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        # assume: self.is_group=True, and self.is_group_lr=False
        optim_result = self.hyper_map(F.partial(_adam_opt, weight_decay, lr, self.eps,
                                                self.global_step, self.beta1, self.beta2),
                                      self._parameters, gradients, self.exp_avg,
                                      self.exp_avg_sq, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
