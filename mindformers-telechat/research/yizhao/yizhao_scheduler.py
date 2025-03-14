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
"""YiZhao Scheduler."""
import math
from typing import Optional

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P

from mindformers.core.lr.lr_schedule import CosineWithWarmUpLR, _check_decay_method, _get_warmup_steps, \
    LearningRateSchedule
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def get_decay_steps(decay_steps: Optional[int], decay_ratio: Optional[float], total_steps: int):
    if decay_steps is not None:
        return decay_steps
    if total_steps is None:
        raise ValueError("total steps cannot be None.")
    if decay_ratio is not None:
        return int(decay_ratio * total_steps)
    return total_steps


def get_constant_steps(constant_steps: Optional[int], constant_ratio: Optional[float], total_steps: int):
    if constant_steps is not None:
        return constant_steps
    if total_steps is None:
        raise ValueError("total steps cannot be None.")
    if constant_ratio is not None:
        return int(constant_ratio * total_steps)
    return total_steps


@MindFormerRegister.register(MindFormerModuleType.LR)
class MegatronLR(CosineWithWarmUpLR):
    """ MegatronLR. """

    # pylint: disable=W0613
    def __init__(self,
                 learning_rate: float, warmup_steps: int = 0, total_steps: Optional[int] = None,
                 num_cycles: float = 0.5, lr_end: float = 0, warmup_lr_init: float = 0,
                 warmup_ratio: Optional[float] = None, decay_steps: Optional[int] = None,
                 decay_ratio: Optional[float] = None, actual_total_steps: int = 0,
                 actual_warmup_ratio: Optional[float] = None,
                 decay_style: Optional[str] = None, **kwargs):
        if actual_total_steps != 0:
            total_steps = actual_total_steps
        if actual_warmup_ratio:
            warmup_ratio = actual_warmup_ratio
        super().__init__(
            learning_rate, warmup_steps, total_steps, num_cycles, lr_end, warmup_lr_init, warmup_ratio, decay_steps,
            **kwargs
        )
        _check_decay_method(decay_steps, total_steps)
        logger.info("total_steps is %s", total_steps)
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        logger.info("warmup_steps is %s", warmup_steps)
        decay_steps = get_decay_steps(decay_steps, decay_ratio, total_steps)
        logger.info("decay_steps is %s", decay_steps - warmup_steps)
        anneal_step = total_steps - decay_steps
        logger.info("anneal step is %s", anneal_step)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.lr_end = Tensor(lr_end, mstype.float32)

        self.total_steps = total_steps
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.anneal_step = Tensor(anneal_step, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.greater_equal = P.GreaterEqual()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.cast = P.Cast()

    def __call__(self, global_step):
        """compute current step lr."""
        if self.greater_equal(global_step, self.decay_steps):
            # Include global_step in computation to circumvent mindspore control flow issues
            return global_step - global_step + self.lr_end

        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            num_steps_ = global_step - self.warmup_steps
            decay_steps_ = self.decay_steps - self.warmup_steps
            decay_ratio = num_steps_ / decay_steps_
            percent = 0.5 * (1.0 + self.cos(self.math_pi * decay_ratio))
            learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float, **_kwargs):
        super(ConstantLR, self).__init__()
        self.lr = Tensor(lr, mstype.float32)

    # pylint: disable=W0613
    def construct(self, global_step):
        return self.lr


@MindFormerRegister.register(MindFormerModuleType.LR)
class AnnealLR(LearningRateSchedule):
    """ AnnealLR """
    # pylint: disable=W0613
    def __init__(self,
                 learning_rate: float, warmup_steps: int = 0, total_steps: Optional[int] = None, lr_end: float = 0,
                 warmup_lr_init: float = 0, warmup_ratio: Optional[float] = None, constant_steps: Optional[int] = None,
                 constant_ratio: Optional[float] = None, actual_total_steps: int = 0,
                 actual_warmup_ratio: Optional[float] = None,
                 decay_style: Optional[str] = None, **kwargs):
        super(AnnealLR, self).__init__()
        if actual_total_steps != 0:
            total_steps = actual_total_steps
        if actual_warmup_ratio:
            warmup_ratio = actual_warmup_ratio
        _check_decay_method(constant_steps, total_steps)
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        constant_steps = get_constant_steps(constant_steps, constant_ratio, total_steps)
        decay_steps = total_steps - warmup_steps - constant_steps
        logger.info("total_steps is %s", total_steps)
        logger.info("warmup_steps is %s", warmup_steps)
        logger.info("constant_steps is %s", constant_steps)
        logger.info("anneal_step is %s", decay_steps)
        self.kwargs = kwargs
        self.learning_rate = Tensor(learning_rate, mstype.float32)
        self.lr_end = Tensor(lr_end, mstype.float32)
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.constant_steps = Tensor(constant_steps, mstype.float32)
        self.warmup_constant_steps = Tensor(constant_steps + warmup_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.greater = P.Greater()
        self.greater_equal = P.GreaterEqual()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.cast = P.Cast()
        self.total_steps = total_steps

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater_equal(global_step, self.warmup_constant_steps):
            percent = (global_step - self.warmup_constant_steps) / self.decay_steps
            learning_rate = self.learning_rate - (self.learning_rate - self.lr_end) * percent
            return learning_rate
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            num_steps_ = global_step - self.warmup_steps
            decay_ratio = num_steps_ / self.constant_steps
            percent = 0.5 * (1.0 + self.cos(self.math_pi * decay_ratio))
            learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
        return learning_rate
