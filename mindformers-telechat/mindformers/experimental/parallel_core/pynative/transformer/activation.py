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
"""Activations."""

__all__ = ["get_act_func", "get_act_func_gated_version"]

import inspect

import mindspore.common.dtype as mstype
from mindspore import nn, Tensor
from mindspore import ops, mint

from mindformers.experimental.parallel_core.pynative.register import ModuleType, ModuleRegistry


@ModuleRegistry.register_decorator(ModuleType.ACTIVATION_FUNC, 'gelu')
class GELU(nn.Cell):
    r"""
    Gaussian error linear unit activation function.

    Applies GELU function to each element of the input. The input is a Tensor with any valid shape.

    GELU is defined as:

    .. math::

        GELU(x_i) = x_i*P(X < x_i),

    where :math:`P` is the cumulative distribution function
    of standard Gaussian distribution and :math:`x_i` is the element of the input.

    Args:
        approximate (bool): Whether to enable approximation. Default: ``False`` .

            If `approximate` is ``True``, The gaussian error linear activation is:

            :math:`0.5 * x * (1 + tanh(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))`

            else, it is:

            :math:`x * P(X <= x) = 0.5 * x * (1 + erf(x / \sqrt(2)))`, where P(X) ~ N(0, 1).

    Inputs:
        - **x** (Tensor) - The input of GELU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindformers.experimental import get_act_func
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> gelu = get_act_func("gelu")
        >>> output = gelu(x)
        >>> print(output)
        [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
         [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]
        >>> gelu = get_act_func("gelu", approximate=False)
        >>> # CPU not support "approximate=False", using "approximate=True" instead
        >>> output = gelu(x)
        >>> print(output)
        [[-1.5865526e-01  3.9998732e+00 -0.0000000e+00]
         [ 1.9544997e+00 -1.4901161e-06  9.0000000e+00]]
    """

    def __init__(self, approximate=False):
        """Initialize GELU."""
        super(GELU, self).__init__()
        self.approximate = approximate
        if not self.approximate:
            self.const0 = Tensor(0.5, mstype.float32)
            self.const1 = Tensor(1.0, mstype.float32)
            self.const2 = Tensor(2.0, mstype.float32)

    def construct(self, x):
        """construct method"""
        if self.approximate:
            return mint.nn.functional.gelu(x, approximate='tanh')
        return (
            x
            * ops.cast(self.const0, x.dtype)
            * (
                ops.cast(self.const1, x.dtype)
                + mint.erf(x / mint.sqrt(ops.cast(self.const2, x.dtype)))
            )
        )


@ModuleRegistry.register_decorator(ModuleType.ACTIVATION_FUNC, 'fast_gelu')
class FastGelu(nn.Cell):
    r"""
    Fast Gaussian error linear unit activation function.

    Applies FastGelu function to each element of the input. The input is a Tensor with any valid shape.

    FastGelu is defined as:

    .. math::
        FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} *
                           \exp(0.851 * (x_i - \left| x_i \right|))

    where :math:`x_i` is the element of the input.

    Inputs:
        - **x** (Tensor) - The input of FastGelu with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindformers.experimental import get_act_func
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> fast_gelu = get_act_func("fast_gelu")
        >>> output = fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """

    def __init__(self):
        """Initialize FastGelu."""
        super(FastGelu, self).__init__()
        self.fast_gelu = ops.FastGeLU()

    def construct(self, x):
        """construct method"""
        return self.fast_gelu(x)


@ModuleRegistry.register_decorator(ModuleType.ACTIVATION_FUNC)
def swiglu(x):
    r"""
    Swish-Gated Linear Unit activation function.

    Inputs:
        - **x** (Tensor) - The input of Swish-Gated Linear Unit with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.
    """
    x0, x1 = mint.split(x, x.shape[-1] // 2, dim=-1)
    return mint.nn.functional.silu(x0) * x1


@ModuleRegistry.register_decorator(ModuleType.ACTIVATION_FUNC)
def fused_swiglu(x):
    r"""
    Fused kernel implementation of Swish-Gated Linear Unit activation function.

    Inputs:
        - **x** (Tensor) - The input of Swish-Gated Linear Unit with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.
    """
    return ops.swiglu(x)


@ModuleRegistry.register_decorator(ModuleType.ACTIVATION_FUNC)
def squared_relu(x):
    r"""
    Squared ReLU activation function.

    Inputs:
        - **x** (Tensor) - The input of Squared ReLU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.
    """
    return mint.pow(mint.nn.functional.relu(x), 2)


ModuleRegistry.register(mint.nn.functional.silu, ModuleType.ACTIVATION_FUNC, "silu", meta={"gated_version": "swiglu"})


def get_act_func(activation_type, **kwargs):
    r"""
    Get activation function by name and parameters.

    Args:
        activation_type (str): The name of the activation function.
        **kwargs: Arbitrary keyword arguments for the activation function.

    Returns:
        callable, the activation function.
    """
    activation_func_item = ModuleRegistry.get_item(ModuleType.ACTIVATION_FUNC, activation_type)
    if inspect.isclass(activation_func_item):
        kwargs = ModuleRegistry.get_needed_params_for_init(activation_func_item, kwargs)
        return activation_func_item(**kwargs)
    return activation_func_item


def get_act_func_gated_version(activation_type):
    r"""
    Get the gated version of the activation function. If not exist, return None.

    Args:
        activation_type (str): The name of the activation function.

    Returns:
        Union[str, None], the name of the gated version of the activation function.
    """
    return ModuleRegistry.get_item_meta_info(ModuleType.ACTIVATION_FUNC, activation_type).get("gated_version", None)
