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
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, nn
from mindspore.common.initializer import initializer

from mindformers.version_control import check_rmsnorm_big_kernel_valid

__all__ = ["get_norm"]


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

    def __init__(self, normalized_shape, eps=1e-5, compute_type=mstype.float32):
        super().__init__()
        if compute_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16], "
                            "but got the type : {}.".format(type(compute_type)))
        self.compute_type = compute_type
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps)
        self.gamma = Parameter(initializer('ones', normalized_shape, compute_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, compute_type), name="beta",
                              parallel_optimizer=False)

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        x = self.cast(x, self.compute_type)
        output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        output = self.cast(x, original_type)
        return output


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32):
        super().__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.compute_type), parallel_optimizer=False)

        if check_rmsnorm_big_kernel_valid():
            self.norm = P.RmsNorm(eps)
            self.rms_norm = self._rms_norm
            self.self_define = False
            self.cast = P.Cast()
            self.rcast = P.Cast()
        else:
            self.cast = P.Cast()
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.square = P.Square()
            self.mean = P.ReduceMean(keep_dims=True)
            self.add = P.Add()
            self.rsqrt = P.Rsqrt()
            self.rms_norm = self._self_norm
            self.self_define = True

    def _self_norm(self, x):
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, self.cast(norm_factor, original_type))
        output = self.mul2(output, self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)


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
            eps=config.layernorm_epsilon,
            compute_type=config.layernorm_compute_dtype)
    if config.normalization == "RMSNorm":
        return RMSNorm(dim=config.hidden_size,
                       eps=config.layernorm_epsilon,
                       compute_type=config.layernorm_compute_dtype)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
