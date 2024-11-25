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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
# ============================================================================
"""Self-Define LR Schedule."""
import math

from mindspore._checkparam import args_type_check
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.common.tensor import Tensor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger

__all__ = [
    'LearningRateWiseLayer', 'ConstantWarmUpLR',
    'LinearWithWarmUpLR', 'CosineWithWarmUpLR',
    'CosineWithRestartsAndWarmUpLR', 'PolynomialWithWarmUpLR',
    'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']


def _get_warmup_steps(warmup_steps: int, warmup_ratio: float, total_steps: int):
    """check warmup args and get warmup steps."""
    if warmup_ratio is None:
        if not isinstance(warmup_steps, int):
            raise TypeError(f"The type of warmup_steps must be int, but got {type(warmup_steps)}")
        if warmup_steps < 0:
            raise ValueError(f"Warmup_steps must be >= 0, but got {warmup_steps}")
        return warmup_steps

    if not isinstance(warmup_ratio, float):
        raise TypeError(f"The type of warmup_ratio must be float, but got {type(warmup_ratio)}")

    if warmup_ratio > 1.0 or warmup_ratio < 0.0:
        raise ValueError(f"Warmup_ratio's value range must be in [0,1], but got {warmup_ratio}")

    if total_steps is None:
        raise ValueError(f"When warmup_ratio takes effect, total_steps must be set, but got {total_steps}")
    if not isinstance(total_steps, int):
        raise TypeError(f"The type of total_steps must be int, but got {type(total_steps)}")

    warmup_steps = int(total_steps * warmup_ratio)
    logger.info("Current warmup_ratio is %s, total_steps is %s, warmup_steps will be set to %s",
                warmup_ratio, total_steps, warmup_steps)
    return warmup_steps


def _check_decay_method(decay_steps: int, total_steps: int):
    """check decay method."""
    if decay_steps is not None:
        return

    if decay_steps is None and total_steps is None:
        raise ValueError(f"When decay_steps is None, total_steps must be set, but got {total_steps} ")


@MindFormerRegister.register(MindFormerModuleType.LR)
class ConstantWarmUpLR(LearningRateSchedule):
    r"""
    Constant Warm Up Learning Rate.

    This learning rate strategy maintains a constant learning rate during the warm-up phase.
    It is particularly suitable for scenarios where a stable, lower learning rate is needed at the
    beginning of training to avoid issues such as gradient explosion, before transitioning to
    the main learning rate schedule.

    During the warm-up phase, the learning rate is kept at a fixed value, denoted as :math:`\eta_{\text{warmup}}` .
    The formula for the learning rate during the warm-up phase is:

    .. math::
        \eta_t = \eta_{\text{warmup}}

    Here, :math:`\eta_{\text{warmup}}` is the fixed learning rate applied during the warm-up steps,
    and :math:`t` represents the current step.

    After the warm-up phase concludes, the learning rate transitions to the main learning rate,
    denoted as :math:`\eta_{\text{main}}` . The formula for the learning rate after the transition is:

    .. math::
        \eta_t = \eta_{\text{main}}

    Args:
        learning_rate (float): Initial value of learning rate.
        warmup_steps (int): The number of warm up steps. Default: None.
        warmup_lr_init (float): Initial learning rate in warm up steps. Default: 0.
        warmup_ratio (float): Ratio of total training steps used for warmup. Default: None.
        total_steps (int): The number of warm up steps. Default: None.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import ConstantWarmUpLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>>
        >>> constant_warmup = ConstantWarmUpLR(learning_rate=learning_rate,
        ...                                    warmup_steps=warmup_steps,
        ...                                    total_steps=total_steps)
        >>> print(constant_warmup(1))
        0.0005
        >>> print(constant_warmup(15))
        0.005
    """

    @args_type_check(
        learning_rate=(int, float), warmup_steps=int, warmup_lr_init=(int, float), warmup_ratio=(int, float),
        total_steps=int
    )
    def __init__(self, learning_rate: float, warmup_steps: int = None, warmup_lr_init: float = 0.,
                 warmup_ratio: float = None, total_steps: int = None, **kwargs):
        super(ConstantWarmUpLR, self).__init__()
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.greater = P.Greater()
        self.kwargs = kwargs

    def construct(self, global_step):
        """compute current step lr."""
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            percent = self.one_constant
            learning_rate = self.learning_rate * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class LinearWithWarmUpLR(LearningRateSchedule):
    r"""
    Linear with Warm Up Learning Rate.

    The LinearWithWarmUpLR scheduler uses a linear warm-up strategy to gradually increase the learning rate
    for each parameter group, followed by a linear adjustment of the learning rate after the warm-up phase ends.

    During the warm-up phase, the learning rate increases linearly from a smaller initial value to the base
    learning rate, as described by the following formula:

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    where :math:`\eta_{\text{warmup}}` is the initial learning rate during the warm-up phase,
    and :math:`\eta_{\text{base}}` is the base learning rate after the warm-up phase.

    After the warm-up phase, the learning rate is adjusted according to the following linear schedule:

    .. math::
        \eta_t = \eta_{\text{base}} - t \times \frac{\eta_{\text{base}} - \eta_{\text{end}}}{\text{total_steps}
        - \text{warmup_steps}}

    where :math:`\eta_{\text{end}}` is the minimum learning rate at the end of training, :math:`\text{total_steps}`
    is the total number of training steps, and :math:`\text{warmup_steps}` is the number of steps in the warm-up phase.

    This method allows for a smooth increase in learning rate through linear warm-up, followed by a gradual
    decrease during the remainder of the training, enhancing the stability and effectiveness of the training process.

    Args:
        learning_rate (float): Initial value of learning rate.
        total_steps (int): The number of total steps.
        warmup_steps (int): The number of warm up steps. Default: None.
        warmup_lr_init (float): Initial learning rate in warm up steps. Default: 0.
        warmup_ratio (float): Ratio of total training steps used for warmup. Default: None.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import LinearWithWarmUpLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>>
        >>> linear_warmup = LinearWithWarmUpLR(learning_rate=learning_rate,
        ...                                    warmup_steps=warmup_steps,
        ...                                    total_steps=total_steps)
        >>> print(linear_warmup(1))
        0.0005
        >>> print(linear_warmup(15))
        0.0025
    """

    @args_type_check(
        learning_rate=(int, float), warmup_steps=int, warmup_lr_init=(int, float), warmup_ratio=(int, float),
        total_steps=int
    )
    def __init__(self, learning_rate: float, total_steps: int, warmup_steps: int = None,
                 warmup_lr_init: float = 0., warmup_ratio: float = None,
                 **kwargs):
        super(LinearWithWarmUpLR, self).__init__()
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        linear_steps = max(1, total_steps - warmup_steps)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            percent = self.max(self.zero_constant, (self.total_steps - global_step) / self.linear_steps)
            learning_rate = self.learning_rate * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineWithWarmUpLR(LearningRateSchedule):
    r"""
    Cosine with Warm Up Learning Rate.

    The CosineWithWarmUpLR learning rate scheduler applies a cosine annealing schedule with warm-up steps to
    set the learning rate for each parameter group. Initially, the learning rate increases linearly during
    the warm-up phase, after which it follows a cosine function to decay.

    During the warm-up phase, the learning rate increases from a small initial value to the base
    learning rate as follows:

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{base}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    where :math:`\eta_{\text{warmup}}` is the initial learning rate, and :math:`\eta_{\text{base}}` is the
    learning rate after the warm-up phase.

    once the warm-up phase is completed, the learning rate follows a cosine decay schedule:

    .. math::
        \eta_t = \eta_{\text{end}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{end}})\left(1
        + \cos\left(\frac{t_{cur}}{t_{max}}\pi\right)\right)

    where :math:`t_{cur}` is the number of epochs since the end of the warm-up phase, and :math:`t_{max}` is
    the total number of epochs until the next restart.

    Args:
        learning_rate (float): Initial value of learning rate.
        warmup_steps (int): The number of warm up steps. Default: None.
        total_steps (int): The number of total steps. Default: None.
        num_cycles (float): The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine). Default: 0.5.
        lr_end (float): Final value of learning rate. Default: 0.
        warmup_lr_init (float): Initial learning rate in warm up steps. Default: 0.
        warmup_ratio (float): Ratio of total training steps used for warmup. Default: None.
        decay_steps (int): The number of decay steps. Default: None.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import CosineWithWarmUpLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>>
        >>> cosine_warmup = CosineWithWarmUpLR(learning_rate=learning_rate,
        ...                                    warmup_steps=warmup_steps,
        ...                                    total_steps=total_steps)
        >>> print(cosine_warmup(1))
        0.0005
        >>> print(cosine_warmup(15))
        0.0024999997
    """

    @args_type_check(
        learning_rate=(int, float), warmup_steps=int, warmup_lr_init=(int, float), warmup_ratio=(int, float),
        total_steps=int, num_cycles=(int, float), lr_end=(int, float)
    )
    def __init__(self, learning_rate: float, warmup_steps: int = 0, total_steps: int = None,
                 num_cycles: float = 0.5, lr_end: float = 0., warmup_lr_init: float = 0.,
                 warmup_ratio: float = None, decay_steps: int = None, **kwargs):
        super(CosineWithWarmUpLR, self).__init__()
        _check_decay_method(decay_steps, total_steps)
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        decay_steps = max(1, decay_steps) \
            if decay_steps is not None else max(1, total_steps)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.lr_end = Tensor(lr_end, mstype.float32)
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.cosine_steps = Tensor(cosine_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.greater_equal = P.GreaterEqual()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater_equal(global_step, self.decay_steps):
            # Include global_step in computation to circumvent mindspore control flow issues
            return global_step - global_step + self.lr_end

        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            progress = (global_step - self.warmup_steps) / self.cosine_steps
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * self.num_cycles * 2.0 * progress)))
            learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineWithRestartsAndWarmUpLR(LearningRateSchedule):
    r"""
    Cosine with Restarts and Warm Up Learning Rate.

    The CosineWithRestartsAndWarmUpLR schedule sets the learning rate for each parameter group using a cosine
    annealing with restarts and warm-up, where :math:`\eta_{max}` is set to the initial learning rate,
    and :math:`T_{cur}` represents the number of steps since the last restart:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1
            + \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right), & T_{cur} \neq (2k+1)T_{i}; \ \eta_{t+1}
            & = \eta_{\text{max}}, & T_{cur} = (2k+1)T_{i}.
        \end{aligned}

    When last_epoch=-1, the initial learning rate is set to lr. During the restart phase, the learning rate begins
    anew from the maximum value and gradually decreases to the set minimum value. This strategy helps avoid getting
    trapped in local minima and accelerates convergence during training.

    This method was proposed in SGDR: Stochastic Gradient Descent with Warm Restarts, extending the concept of
    cosine annealing to allow for multiple restarts.

    Args:
        learning_rate (float): Initial value of learning rate.
        warmup_steps (int): The number of warm up steps. Default: None.
        total_steps (int): The number of total steps. Default: None.
        num_cycles (float): The number of waves in the cosine schedule (the defaults is to just decrease
            from the max value to 0 following a half-cosine). Default: 1.0.
        lr_end (float): Final value of learning rate. Default: 0.
        warmup_lr_init (float): Initial learning rate in warm up steps. Default: 0.
        warmup_ratio (float): Ratio of total training steps used for warmup. Default: None.
        decay_steps (int): The number of decay steps. Default: None.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import CosineWithRestartsAndWarmUpLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>>
        >>> cosine_warmup_restart = CosineWithRestartsAndWarmUpLR(learning_rate=learning_rate,
        ...                                                       warmup_steps=warmup_steps,
        ...                                                       total_steps=total_steps)
        >>> print(cosine_warmup_restart(1))
        0.0005
        >>> print(cosine_warmup_restart(15))
        0.0024999997
    """

    @args_type_check(
        learning_rate=(int, float), warmup_steps=int, warmup_lr_init=(int, float), warmup_ratio=(int, float),
        total_steps=int, num_cycles=(int, float), lr_end=(int, float)
    )
    def __init__(self, learning_rate: float, warmup_steps: int = None, total_steps: int = None,
                 num_cycles: float = 1., lr_end: float = 0., warmup_lr_init: float = 0.,
                 warmup_ratio: float = None, decay_steps: int = None, **kwargs):
        super(CosineWithRestartsAndWarmUpLR, self).__init__()
        _check_decay_method(decay_steps, total_steps)
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        decay_steps = max(1, decay_steps) \
            if decay_steps is not None else max(1, total_steps)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.lr_end = Tensor(lr_end, mstype.float32)
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.cosine_steps = Tensor(cosine_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.greater_equal = P.GreaterEqual()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater_equal(global_step, self.decay_steps):
            # Include global_step in computation to circumvent mindspore control flow issues
            return global_step - global_step + self.lr_end

        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        else:
            progress = (global_step - self.warmup_steps) / self.cosine_steps
            if self.greater(self.one_constant, progress):
                percent = self.max(
                    self.zero_constant,
                    0.5 * (1.0 + self.cos(self.math_pi * ((self.num_cycles * progress) % self.one_constant))))
                learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
            else:
                return self.zero_constant
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class PolynomialWithWarmUpLR(LearningRateSchedule):
    r"""
    Polynomial with Warm Up Learning Rate.

    At the beginning of training, the learning rate gradually increases from a lower initial
    value, :math:`\eta_{\text{warmup}}` , to the starting learning rate, :math:`\eta_{\text{start}}` .
    The change in learning rate during the warm-up phase, depending on the step :math:`t` ,
    is described by the following formula:

    .. math::
        \eta_t = \eta_{\text{warmup}} + t \times \frac{\eta_{\text{start}} - \eta_{\text{warmup}}}{\text{warmup_steps}}

    where :math:`\text{warmup\_steps}` represents the total number of steps in the warm-up phase.

    After the warm-up phase concludes, the learning rate gradually decays according to a polynomial function,
    reaching the final learning rate, :math:`\eta_{\text{end}}` . The change in learning rate over the total
    number of steps :math:`\text{total\_steps}` is given by the formula:

    .. math::
        \eta_t = \eta_{\text{end}} + (\eta_{\text{start}} - \eta_{\text{end}}) \times \left(1
        - \frac{t - \text{warmup_steps}}{\text{decay_steps}}\right)^{\text{power}}

    where :math:`\text{power}` is the exponent of the polynomial, controlling the decay rate.

    This learning rate strategy is well-suited for scenarios where a stable learning rate is needed during
    the early stages of training, with a gradual decrease in the later stages. By preventing gradient explosion
    initially and reducing the learning rate during the latter part of training, it helps the model achieve better
    generalization as it converges.

    Args:
        learning_rate (float): Initial value of learning rate.
        total_steps (int): The number of total steps.
        warmup_steps (int): The number of warm up steps.
        lr_end (float): Final value of learning rate. Default: 1e-7.
        power (float): The power of the polynomial. Default: 1.0.
        warmup_lr_init (float): Initial learning rate in warm up steps. Default: 0.
        warmup_ratio (float): Ratio of total training steps used for warmup. Default: None.
        decay_steps (int): The number of decay steps, which must be smaller than total_steps - warmup_steps.
            If the value is None, decay steps will be total_steps - warmup_steps. Default: None.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import PolynomialWithWarmUpLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>> lr_end = 0.0000001
        >>>
        >>> polynomial_warmup = PolynomialWithWarmUpLR(learning_rate=learning_rate,
        ...                                            warmup_steps=warmup_steps,
        ...                                            total_steps=total_steps,
        ...                                            lr_end=lr_end)
        >>> print(polynomial_warmup(1))
        0.0005
        >>> print(polynomial_warmup(15))
        0.0025000498
    """

    @args_type_check(
        learning_rate=(int, float), warmup_steps=int, warmup_lr_init=(int, float), warmup_ratio=(int, float),
        total_steps=int, lr_end=(int, float), power=(int, float)
    )
    def __init__(self, learning_rate: float, total_steps: int, warmup_steps: int = None,
                 lr_end: float = 1e-7, power: float = 1.0, warmup_lr_init: float = 0.,
                 warmup_ratio: float = None, decay_steps: int = None, **kwargs):
        super(PolynomialWithWarmUpLR, self).__init__()
        _check_decay_method(decay_steps, total_steps)
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        decay_steps = max(1, decay_steps) \
            if decay_steps is not None else max(1, total_steps - warmup_steps)
        if not learning_rate > lr_end:
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({learning_rate})")
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.lr_end = Tensor(lr_end, mstype.float32)
        self.power = power
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.greater = P.Greater()
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)

        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + (self.learning_rate - self.warmup_lr_init) * percent
        elif self.greater(global_step, self.total_steps):
            # Include global_step in computation to circumvent mindspore control flow issues
            return global_step - global_step + self.lr_end
        else:
            lr_range = self.learning_rate - self.lr_end
            pct_remaining = 1 - (global_step - self.warmup_steps) / self.decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            percent = decay / self.learning_rate
            learning_rate = self.learning_rate * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class LearningRateWiseLayer(LearningRateSchedule):
    r"""
    Learning Rate Wise Layer.

    This approach allows each layer to adapt its learning rate according to its specific needs, leading
    to more efficient and effective training. The learning rate for each layer is determined by a base
    learning rate modulated by a scaling factor specific to that layer.

    Initially, the learning rate for each layer is set based on a linear scaling strategy:

    .. math::
        \eta_{t,l} = \eta_{\text{base}} \times \alpha_l

    where :math:`\eta_{t,l}` is the learning rate for layer :math:`l` at time :math:`t` , :math:`\eta_{\text{base}}`
    is the base learning rate, and :math:`\alpha_l` is the scaling factor for layer :math:`l` .

    As training progresses, the learning rate for each layer is adjusted according to the
    following cosine annealing schedule:

    .. math::
        \eta_{t,l} = \eta_{\text{end}} + \frac{1}{2}(\eta_{\text{base}} \times \alpha_l - \eta_{\text{end}})\left(1
        + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    where :math:`T_{cur}` is the number of epochs completed since the learning rate was last reset,
    and :math:`T_{max}` is the total number of epochs before the next reset. :math:`\eta_{\text{end}}` represents
    the minimum learning rate at the end of the training.

    Args:
        base_lr (mindspore.nn.learning_rate_schedule.LearningRateSchedule): The base learning rate schedule.
        lr_scale (float): The value for learning rate scaling.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import LinearWithWarmUpLR
        >>> from mindformers.core import LearningRateWiseLayer
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> total_steps = 20
        >>> warmup_steps = 10
        >>> learning_rate = 0.005
        >>>
        >>> linear_warmup = LinearWithWarmUpLR(learning_rate=learning_rate,
        ...                                    warmup_steps=warmup_steps,
        ...                                    total_steps=total_steps)
        >>> learning_rate_wise_layer = LearningRateWiseLayer(linear_warmup, 0.5)
        >>> print(learning_rate_wise_layer(1))
        0.00025
        >>> print(learning_rate_wise_layer(15))
        0.00125
    """

    def __init__(self, base_lr, lr_scale):
        super(LearningRateWiseLayer, self).__init__()
        self.base_lr = base_lr
        self.lr_scale = lr_scale

    def construct(self, global_step):
        lr = self.base_lr(global_step)
        return self.lr_scale * lr


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineAnnealingLR(LearningRateSchedule):
    r"""
    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_ .
    Note that this only implements the cosine annealing part of SGDR, and not the restarts.

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    Args:
        base_lr (float): Initial value of learning rate.
        t_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import CosineAnnealingLR
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> base_lr = 0.005
        >>> t_max = 10
        >>> eta_min = 0.0000001
        >>>
        >>> cosine_annealing = CosineAnnealingLR(base_lr=base_lr, t_max=t_max, eta_min=eta_min)
        >>> print(cosine_annealing(1))
        0.0048776437
        >>> print(cosine_annealing(15))
        0.0025000498
    """

    @args_type_check(base_lr=(int, float), t_max=int, eta_min=(int, float))
    def __init__(self, base_lr: float, t_max: int, eta_min: float = 0., **kwargs):
        super(CosineAnnealingLR, self).__init__()
        if t_max < 1 or not isinstance(t_max, int):
            raise ValueError("Expected positive integer T_max, but got {}".format(t_max))
        self.kwargs = kwargs
        self.base_lr = base_lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.max = P.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        percent = self.max(
            self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * global_step / self.t_max)))
        learning_rate = self.eta_min + (self.base_lr - self.eta_min) * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineAnnealingWarmRestarts(LearningRateSchedule):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_ .

    Args:
        base_lr (float): Initial value of learning rate.
        t_0 (int): Number of iterations for the first restart.
        t_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.

    Inputs:
        - **global_step** (int) - The global step.

    Outputs:
        Learning rate.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.core import CosineAnnealingWarmRestarts
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> base_lr = 0.005
        >>> t_0 = 10
        >>> t_mult = 2
        >>> eta_min = 0.0000001
        >>>
        >>> cosine_annealing_restart = CosineAnnealingWarmRestarts(base_lr=base_lr,
        ...                                                        t_0=t_0,
        ...                                                        t_mult=t_mult,
        ...                                                        eta_min=eta_min)
        >>> print(cosine_annealing_restart(1))
        0.0048776437
        >>> print(cosine_annealing_restart(15))
        0.0042677815
    """

    @args_type_check(base_lr=(int, float), t_0=int, t_mult=int, eta_min=(int, float))
    def __init__(self, base_lr: float, t_0: int, t_mult: int = 1, eta_min: float = 0., **kwargs):
        super(CosineAnnealingWarmRestarts, self).__init__()
        if t_0 < 1 or not isinstance(t_0, int):
            raise ValueError("Expected positive integer t_0, but got {}".format(t_0))
        if t_mult < 1 or not isinstance(t_mult, int):
            raise ValueError("Expected positive integer t_mult, but got {}".format(t_mult))
        self.kwargs = kwargs
        self.base_lr = base_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.max = P.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.floor = P.Floor()
        self.log = P.Log()
        self.log_t_mult = math.log(t_mult)

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if global_step < self.t_0:
            t_cur = global_step
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * t_cur / self.t_0)))
        elif self.t_mult == 1:
            t_index = global_step // self.t_0
            t_cur = global_step - t_index * self.t_0
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * t_cur / self.t_0)))
        else:
            t_index = self.floor(self.log(global_step / self.t_0 * (self.t_mult - 1.0) + 1.0) / self.log_t_mult)
            q_n = self.t_mult ** t_index
            t_start = self.t_0 * (1.0 - q_n) / (1.0 - self.t_mult)
            t_i = self.t_0 * q_n
            t_cur = global_step - t_start
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * t_cur / t_i)))
        learning_rate = self.eta_min + (self.base_lr - self.eta_min) * percent
        return learning_rate
