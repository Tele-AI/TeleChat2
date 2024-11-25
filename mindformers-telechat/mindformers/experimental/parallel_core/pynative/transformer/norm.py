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
"""Normalization"""

__all__ = ["get_norm"]

import mindspore.common.dtype as mstype

from mindspore import nn, Parameter, Tensor, mint, ops
from mindspore.common.initializer import initializer


class LayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16], "
                            "but got the type : {}.".format(type(param_init_type)))
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.eps = eps
        self.normalized_shape = normalized_shape

    def construct(self, x):
        """construct method"""
        output = mint.nn.functional.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        return output


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32):
        super(RMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)
        self.weight = Parameter(initializer("ones", (dim,), dtype=param_init_type))

    def construct(self, x):
        """Forward of RMSNorm."""
        origin_dtype = x.dtype
        x = ops.cast(x, mstype.float32)
        norm_factor = mint.square(x)
        norm_factor = mint.mean(norm_factor, dim=-1, keepdim=True)
        norm_factor = mint.add(norm_factor, self.eps)
        norm_factor = mint.rsqrt(norm_factor)
        output = mint.mul(x, norm_factor)
        output = ops.cast(output, origin_dtype)
        output = mint.mul(output, self.weight)
        return output


class FusedRMSNorm(nn.Cell):
    r"""
    A RMSNorm fused kernel implementation.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1.e-6, compute_type=mstype.float32):
        super(FusedRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer("ones", (dim,), dtype=self.compute_type), parallel_optimizer=False)

    def construct(self, x):
        """Forward of FusedRMSNorm."""
        output = ops.rms_norm(x, self.weight, self.eps)[0]
        return output


def get_norm(config):
    r"""
    Get normalization layer.

    Args:
        config: The config of the model.

    Returns:
        callable, the normalization layer.
    """
    if config.normalization == "LayerNorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon)
    if config.normalization == "RMSNorm":
        return RMSNorm(dim=config.hidden_size,
                       eps=config.norm_epsilon)
    if config.normalization == "FusedRMSNorm":
        return FusedRMSNorm(dim=config.hidden_size, eps=config.norm_epsilon, compute_type=config.compute_dtype)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
