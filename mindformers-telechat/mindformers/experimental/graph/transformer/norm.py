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
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

from mindspore import nn, Parameter
from mindspore.common.initializer import initializer

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

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16, bfoat16], "
                            "but got the type : {}.".format(type(param_init_type)))

        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)

        self.mean = P.ReduceMean(keep_dims=True)
        self.mean2 = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub = P.Sub()
        self.add = P.Add()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()
        self.compute_type = param_init_type
        self.cast = P.Cast()

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        mean = self.mean(self.cast(x, self.compute_type), -1)
        diff = self.sub(self.cast(x, self.compute_type), mean)
        varaince = self.mean2(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(varaince, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy):
        """shard method"""
        if not strategy:
            raise TypeError('The strategy length must bigger than 0! Strategy {} not supported'.format(strategy))

        self.mean.shard((strategy,))
        self.sub.shard((strategy, strategy[:-1] + (1,)))
        self.square.shard((strategy,))
        self.mean2.shard((strategy,))
        self.add.shard((strategy[:-1] + (1,), ()))
        self.sqrt.shard((strategy[:-1] + (1,),))
        self.real_div.shard((strategy, strategy[:-1] + (1,)))
        self.mul.shard((strategy, (strategy[-1],)))
        self.add2.shard((strategy, (strategy[-1],)))


class FusedLayerNorm(nn.Cell):
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
        super(FusedLayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16, bfoat16], "
                            "but got the type : {}.".format(type(param_init_type)))

        self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      epsilon=eps)
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.compute_type = param_init_type
        self.cast = P.Cast()

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        output, _, _ = self.layer_norm(self.cast(x, self.compute_type), self.gamma, self.beta)
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy):
        """shard method"""
        if not strategy:
            raise TypeError('The strategy length must bigger than 0! Strategy {} not supported'.format(strategy))

        if strategy[-1] != 1:
            raise TypeError(
                'The last dim in FusedlayerNorm can not equal to 1! Strategy {} not supported!'.format(strategy))

        self.layer_norm.shard((strategy, (strategy[-1],), (strategy[-1],)))


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
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16, bfoat16], "
                            "but got the type : {}.".format(type(param_init_type)))

        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim), param_init_type))
        self.compute_type = param_init_type

        self.square = P.Square()
        self.mean = P.ReduceMean(keep_dims=True)
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(self.cast(x, self.compute_type), norm_factor)
        output = self.mul2(output, self.weight)
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy):
        """shard method"""
        if not strategy:
            raise TypeError('The strategy length must bigger than 0! Strategy {} not supported'.format(strategy))

        self.square.shard((strategy,))
        self.mean.shard((strategy,))
        self.add.shard((strategy[:-1] + (1,), ()))
        self.rsqrt.shard((strategy[:-1] + (1,),))
        self.mul.shard((strategy, strategy[:-1] + (1,)))
        self.mul2.shard((strategy, (strategy[-1],)))


class FusedRMSNorm(nn.Cell):
    r"""
    FusedRMSNorm operation

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
        super(FusedRMSNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16, bfoat16], "
                            "but got the type : {}.".format(type(param_init_type)))

        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim), param_init_type))
        self.compute_type = param_init_type

        self.norm = P.RmsNorm(eps)
        self.cast = P.Cast()

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy):
        """shard method"""
        if not strategy:
            raise TypeError('The strategy length must bigger than 0! Strategy {} not supported'.format(strategy))

        if strategy[-1] != 1 and ms.get_context('mode') == ms.GRAPH_MODE:
            raise TypeError(
                'The last dim in FusedlayerNorm can not equal to 1! Strategy {} not supported!'.format(strategy))

        self.norm.shard((strategy, (strategy[-1],)))


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
            param_init_type=config.layernorm_compute_type)
    if config.normalization == "FusedLayerNorm":
        return FusedLayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type)
    if config.normalization == "RMSNorm":
        return RMSNorm(
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type)
    if config.normalization == "FusedRMSNorm":
        return FusedRMSNorm(
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
