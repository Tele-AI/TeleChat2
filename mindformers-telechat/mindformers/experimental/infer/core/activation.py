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
import inspect

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops

from mindformers.version_control import check_valid_big_kernel

__all__ = ["get_act_func"]


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
        approximate (bool): Whether to enable approximation. Default: ``True`` .

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

    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate
        if self.approximate:
            self.gelu = ops.GeLU()
        else:
            self.erf = ops.Erf()
            self.sqrt = ops.Sqrt()
            self.const0 = Tensor(0.5, mstype.float32)
            self.const1 = Tensor(1.0, mstype.float32)
            self.const2 = Tensor(2.0, mstype.float32)

    def construct(self, x):
        if self.approximate:
            return self.gelu(x)
        return (
            x
            * ops.cast(self.const0, x.dtype)
            * (
                ops.cast(self.const1, x.dtype)
                + self.erf(x / self.sqrt(ops.cast(self.const2, x.dtype)))
            )
        )


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
        super(FastGelu, self).__init__()
        self.fast_gelu = ops.FastGeLU()

    def construct(self, x):
        return self.fast_gelu(x)


def swiglu(x):
    r"""
    Swish-Gated Linear Unit activation function.

    Inputs:
        - **x** (Tensor) - The input of Swish-Gated Linear Unit with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.
    """
    x = ops.chunk(x, 2, axis=-1)
    return ops.silu(x[0]) * x[1]


def squared_relu(x):
    r"""
    Squared ReLU activation function.

    Inputs:
        - **x** (Tensor) - The input of Squared ReLU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - Tensor with the same type and shape as the `x`.
    """
    return ops.pow(ops.relu(x), 2)


class SiLU(nn.Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = silu(x).
    """

    def __init__(self):
        super().__init__()
        if check_valid_big_kernel():
            # pylint: disable=W0212
            self.silu = nn.SiLU()
            self.self_define = False
        else:
            self.sigmoid = ops.Sigmoid()
            self.mul = ops.Mul()
            self.silu = self._self_silu
            self.self_define = True

    def _self_silu(self, x):
        return self.mul(x, self.sigmoid(x))

    def construct(self, x):
        return self.silu(x)


ACTIVATION_MAP = {
    "gelu": GELU,
    "fast_gelu": FastGelu,
    "swiglu": swiglu,
    "squared_relu": squared_relu,
    "silu": SiLU
}


def get_act_func(activation_type, *args, **kwargs):
    r"""
    Get activation function by name and parameters.

    Args:
        activation_type (str): The name of the activation function.
        *args: Variable length argument list for the activation function.
        **kwargs: Arbitrary keyword arguments for the activation function.

    Returns:
        callable, the activation function.
    """
    if activation_type.lower() not in ACTIVATION_MAP:
        raise NotImplementedError(
            f"Invalid activation function: {activation_type}.\
                Supported activation functions are: {ACTIVATION_MAP.keys()}"
        )
    activation_func = ACTIVATION_MAP[activation_type.lower()]
    if inspect.isclass(activation_func):
        return activation_func(*args, **kwargs)
    return activation_func
