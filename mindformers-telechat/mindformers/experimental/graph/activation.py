# Copyright 2020-2024 Huawei Technologies Co., Ltd
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

"""
Activation functions for transformer.
"""
from mindspore import nn, dtype, Tensor
from mindspore.ops import operations as P
from mindformers.experimental.graph.transformer.transformer_config import ModelParallelConfig

__all__ = [
    "GELU",
    "SwiGlu",
    "SiLU",
    "bias_gelu_impl",
    "bias_swiglu_impl",
    "get_activation"
]


class SwiGlu(nn.Cell):
    """
    SwiGlu activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
    """
    def __init__(self, config: ModelParallelConfig = None):
        super(SwiGlu, self).__init__()
        self.slice = P.StridedSlice()
        # pylint: disable=W0212
        self.silu = P._inner_ops.SiLU()
        self.mul = P.Mul()
        self.add = P.Add()
        self.shard(config)

    def construct(self, x: Tensor, bias: Tensor = None) -> Tensor:
        if bias is not None:
            x = self.add(x, bias)
        bs, seq_len, hidden_size = x.shape
        x_1 = self.slice(x, (0, 0, 0), (bs, seq_len, hidden_size // 2), (1, 1, 1))
        x_2 = self.slice(x, (0, 0, hidden_size // 2), (bs, seq_len, hidden_size), (1, 1, 1))
        return self.mul(self.silu(x_1), x_2)

    def shard(self, config: ModelParallelConfig):
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        cp = config.context_parallel if config and config.context_parallel is not None else 1
        silu_in_strategy = ((dp, cp, 1),)
        self.silu.shard(silu_in_strategy)
        slice_in_strategy = ((dp, cp, 1),)
        self.slice.shard(slice_in_strategy)
        mul_in_strategy = ((dp, cp, 1), (dp, cp, 1))
        self.mul.shard(mul_in_strategy)


class GELU(nn.Cell):
    """
    GELU activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
        approximate (bool): Whether to use the approximate version of GELU.
    """
    def __init__(self, config: ModelParallelConfig = None, approximate: bool = True):
        super(GELU, self).__init__()
        self.approximate = approximate
        self.add_bias = P.Add()
        if self.approximate:
            self.gelu = P.GeLU()
        else:
            self.erf = P.Erf()
            self.sqrt = P.Sqrt()
            self.mul_tensor = P.Mul()
            self.mul_const = P.Mul()
            self.add_erf = P.Add()
            self.div = P.Div()
            self.const0 = Tensor(0.5, dtype.float32)
            self.const1 = Tensor(1.0, dtype.float32)
            self.const2 = Tensor(2.0, dtype.float32)
            self.cast = P.Cast()

        self.shard(config)

    def construct(self, x: Tensor, bias: Tensor = None) -> Tensor:
        """
        Apply GELU activation function.

        Args:
            x (Tensor): The input tensor.
            bias (Tensor): The bias tensor.
        """
        if bias is not None:
            x = self.add_bias(x, bias)
        # [bs, seq_len, hidden_size]
        if self.approximate:
            return self.gelu(x)

        const0 = self.cast(self.const0, x.dtype)
        const1 = self.cast(self.const1, x.dtype)
        const2 = self.cast(self.const2, x.dtype)
        return self.mul_tensor(self.mul_const(x, const0),
                               self.add_erf(const1, self.erf(self.div(x, self.sqrt(const2)))))

    def shard(self, config: ModelParallelConfig):
        """
        Shard operators in GELU activation function.

        Args:
            config (ModelParallelConfig): The model parallel configuration.
        """
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        cp = config.context_parallel if config and config.context_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1
        if self.approximate:
            gelu_in_strategy = ((dp, cp, tp),)
            self.gelu.shard(gelu_in_strategy)
        else:
            div_in_strategy = ((dp, cp, tp), ())
            self.div.shard(div_in_strategy)
            erf_in_strategy = ((dp, cp, tp),)
            self.erf.shard(erf_in_strategy)
            add_erf_in_strategy = ((), (dp, cp, tp))
            self.add_erf.shard(add_erf_in_strategy)
            mul_const_in_strategy = ((dp, cp, tp), ())
            self.mul_const.shard(mul_const_in_strategy)
            mul_tensor_in_strategy = ((dp, cp, tp), (dp, cp, tp))
            self.mul_tensor.shard(mul_tensor_in_strategy)


class SiLU(nn.Cell):
    """
    SiLU activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
    """
    def __init__(self, config: ModelParallelConfig = None):
        super(SiLU, self).__init__()
        # pylint: disable=W0212
        self.silu = P._inner_ops.SiLU()
        self.add = P.Add()
        self.shard(config)

    def construct(self, x: Tensor, bias: Tensor = None) -> Tensor:
        if bias is not None:
            x = self.add(x, bias)
        return self.silu(x)

    def shard(self, config: ModelParallelConfig):
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        cp = config.context_parallel if config and config.context_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1
        silu_in_strategy = ((dp, cp, tp),)
        self.silu.shard(silu_in_strategy)


ACTIVATION_MAP = {
    'gelu': GELU,
    'swiglu': SwiGlu,
    'silu': SiLU
}


def get_activation(activation_name, *args, **kwargs):
    activation_name = activation_name.lower()
    if activation_name not in ACTIVATION_MAP:
        raise NotImplementedError(f"Activation '{activation_name}' is not supported for now. \
                                  Supported activations are: {ACTIVATION_MAP}")

    # activation should be a Cell in Static
    activation = ACTIVATION_MAP[activation_name]
    return activation(*args, **kwargs)


def bias_gelu_impl(x: Tensor, bias: Tensor = None, config: ModelParallelConfig = None) -> Tensor:
    apply_func = GELU(config)
    return apply_func(x, bias)


def bias_swiglu_impl(x: Tensor, bias: Tensor = None, fp8_input_store: bool = False,
                     config: ModelParallelConfig = None) -> Tensor:
    if fp8_input_store:
        raise NotImplementedError("For SwiGlu, fp8 input store is not supported for now.")
    apply_func = SwiGlu(config)
    return apply_func(x, bias)
