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
"""mindformers lora layer"""
from typing import List, Optional, Callable

from mindspore import nn
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
try:
    from mindspore._checkparam import Validator, Rel
    INC_LEFT = Rel.INC_LEFT
except ImportError:
    import mindspore._checkparam as Validator
    INC_LEFT = Validator.INC_LEFT
from mindspore.common import dtype
from mindspore.common.initializer import initializer
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.utils import init_method_zero

__all__ = ["LoRAColumnParallelLinear", "LoRARowParallelLinear"]


class LoRAColumnParallelLinear(nn.Cell):
    r"""
    The LoRA layer with weight sliced in column dimension.

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        init_method (Callable): The initialization method. Default: None
        bias (bool): Whether to include bias in the linear layer. Default: True.
        gather_output (bool): Whether to gather the output. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        embedding_activation_buffer (List[Tensor]): The buffer for embedding activation. Default: None.
        grad_output_buffer (List[Tensor]): The buffer for gradient output. Default: None.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        disable_grad_reduce (bool): Whether to disable gradient reduce. Default: False.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        lora_rank (int): The value of rank. Default: 8.
        lora_alpha (int):  The alpha value for lora. Default: 32.
        lora_dropout (float):  The dropout rate for lora. Default: 0.0.
        lora_a_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_a weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
        lora_b_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_b weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'zeros'.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, input\_size)`. The `input_size` in `Args` should be equal
          to :math:`input\_size` in `Inputs`.
        - **weight** (Tensor) - Tensor of shape :math:`(input\_size, output\_size)`/:math`(output\_size, input\_size)`.
          Default: None.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Tensor): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = True,
                 gather_output: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False,
                 skip_weight_param_allocation: bool = False,
                 embedding_activation_buffer: Optional[List[Tensor]] = None,
                 grad_output_buffer: Optional[List[Tensor]] = None,
                 is_expert: bool = False,
                 tp_comm_buffer_name: str = None,
                 disable_grad_reduce: bool = False,
                 transpose_b: bool = True,
                 compute_dtype: dtype = dtype.float16,
                 bias_init: Callable = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.0,
                 lora_a_init='normal',
                 lora_b_init='zeros',
                 ):
        super(LoRAColumnParallelLinear, self).__init__()
        if gather_output:
            raise NotImplementedError("For LoRAColumnParallelLinear, `gather_output` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For LoRAColumnParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For LoRAColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For LoRAColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if grad_output_buffer is not None:
            raise NotImplementedError("For LoRAColumnParallelLinear, `grad_output_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For LoRAColumnParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For LoRAColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")
        if disable_grad_reduce:
            raise NotImplementedError("For LoRAColumnParallelLinear, `disable_grad_reduce` is not supported for now")

        # Define and initialize params
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = compute_dtype
        self.cast = P.Cast()
        self.transpose_b = transpose_b
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.has_bias = bias
        self.params_dtype = config.params_dtype

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.mindpet_delta_lora_a = Parameter(
            initializer(lora_a_init, [lora_rank, input_size], self.params_dtype),
            name='mindpet_delta_lora_A')
        self.mindpet_delta_lora_b = Parameter(initializer(lora_b_init, [output_size, lora_rank], self.params_dtype),
                                              name='mindpet_delta_lora_B')
        self.scaling = self.lora_alpha / self.lora_rank

        if skip_weight_param_allocation:
            self.weight = None
        else:
            weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
            self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        # Calculation utils
        self.matmul = P.MatMul(transpose_b=transpose_b)
        if not skip_bias_add:
            self.bias_add = P.Add()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.add = P.Add()
        self.lora_a_matmul = P.MatMul(transpose_b=True)
        self.lora_b_matmul = P.MatMul(transpose_b=True)
        # init shard
        self.shard(config)

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of LoRAColumnParallelLinear.

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        # Data type operation
        ori_dtype = input_.dtype
        input_ = self.cast(input_, self.compute_dtype)
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For LoRAColumnParallelLinear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight
        weight = self.cast(weight, self.compute_dtype)
        lora_a = self.cast(self.mindpet_delta_lora_a, self.compute_dtype)
        lora_b = self.cast(self.mindpet_delta_lora_b, self.compute_dtype)
        scaling = self.cast(self.scaling, self.compute_dtype)

        # Shape operations
        x_shape = input_.shape
        input_ = self.reshape(input_, (-1, x_shape[-1]))

        # Dense result
        dense_result = self.matmul(input_, weight)
        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            dense_result = self.bias_add(dense_result, bias)
            bias = None
        else:
            bias = self.bias

        # LoRA result
        input_ = self.lora_dropout(input_)
        input_ = self.lora_a_matmul(input_, lora_a)
        input_ = self.lora_b_matmul(input_, lora_b)
        input_ = self.mul(input_, scaling)

        # Result addition
        dense_result = self.add(dense_result, input_)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            dense_result = self.reshape(dense_result, out_shape)
        dense_result = self.cast(dense_result, ori_dtype)
        return dense_result, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in LoRAColumnParallelLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.

        """
        dp = config.data_parallel if config.data_parallel is not None else 1
        tp = config.tensor_parallel if config.tensor_parallel is not None else 1
        cp = config.context_parallel if config.context_parallel is not None else 1
        # shard of matmul
        if self.transpose_b:
            weight_strategy = (1, tp)
        else:
            weight_strategy = (tp, 1)
        matmul_in_strategy = ((dp * cp, tp), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)

        # shard of bias add
        if not self.skip_bias_add:
            add_in_strategy = ((dp * cp, 1), (1,))
            self.bias_add.shard(in_strategy=add_in_strategy)

        # shard of lora operator
        self.lora_dropout.dropout.shard((matmul_in_strategy[0],))
        if self.transpose_b:
            self.lora_a_matmul.shard((matmul_in_strategy[0], (1, matmul_in_strategy[1][1])))
            self.lora_b_matmul.shard(((matmul_in_strategy[0][0], 1), (matmul_in_strategy[1][0], 1)))
            self.mul.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][0]), ()))
            self.add.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][0]),
                            (matmul_in_strategy[0][0], matmul_in_strategy[1][0])))
        else:
            self.lora_a_matmul.shard((matmul_in_strategy[0], (matmul_in_strategy[1][0], 1)))
            self.lora_b_matmul.shard(((matmul_in_strategy[0][0], 1), (1, matmul_in_strategy[1][1])))
            self.mul.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][1]), ()))
            self.add.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][1]),
                            (matmul_in_strategy[0][0], matmul_in_strategy[1][1])))


class LoRARowParallelLinear(nn.Cell):
    r"""
    The LoRA layer with weight sliced in row dimension.

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        init_method (Callable): The initialization method. Default: None
        bias (bool): Whether to include bias in the linear layer. Default: True.
        input_is_parallel (bool): Whether the input is already parallelized. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        lora_rank (int): The value of rank. Default: 8.
        lora_alpha (int):  The alpha value for lora. Default: 32.
        lora_dropout (float):  The dropout rate for lora. Default: 0.0.
        lora_a_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_a weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
        lora_b_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_b weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'zeros'.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, input\_size)`. The `input_size` in `Args` should be equal
          to :math:`input\_size` in `Inputs`.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Parameter): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 skip_bias_add: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 is_expert: bool = False,
                 tp_comm_buffer_name: str = None,
                 transpose_b: bool = True,
                 compute_dtype: dtype = dtype.float16,
                 bias_init: Callable = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.0,
                 lora_a_init='normal',
                 lora_b_init='zeros',
                 ):
        super(LoRARowParallelLinear, self).__init__()
        if input_is_parallel:
            raise NotImplementedError("For LoRARowParallelLinear, `input_is_parallel` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For LoRARowParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For LoRARowParallelLinear, `keep_master_weight_for_test` is not supported now")
        if is_expert:
            raise NotImplementedError("For LoRARowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For LoRARowParallelLinear, `tp_comm_buffer_name` is not supported for now")

        # Define and initialize params
        self.input_size = input_size
        self.output_size = output_size
        self.transpose_b = transpose_b
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.has_bias = bias

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.mindpet_delta_lora_a = Parameter(
            initializer(lora_a_init, [lora_rank, input_size], self.params_dtype),
            name='mindpet_delta_lora_A')
        self.mindpet_delta_lora_b = Parameter(initializer(lora_b_init, [output_size, lora_rank], self.params_dtype),
                                              name='mindpet_delta_lora_B')
        self.scaling = self.lora_alpha / self.lora_rank

        weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
        self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        # Calculation utils
        self.matmul = P.MatMul(transpose_b=transpose_b)
        if not skip_bias_add:
            self.bias_add = P.Add()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.add = P.Add()
        self.lora_a_matmul = P.MatMul(transpose_b=True)
        self.lora_b_matmul = P.MatMul(transpose_b=True)
        # init shard
        self.shard(config)

    def construct(self, input_: Tensor) -> tuple[Tensor, Tensor]:
        """Forward of RowParallelLinear.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        # Data type operation
        ori_dtype = input_.dtype
        input_ = self.cast(input_, self.compute_dtype)
        weight = self.cast(self.weight, self.compute_dtype)
        lora_a = self.cast(self.mindpet_delta_lora_a, self.compute_dtype)
        lora_b = self.cast(self.mindpet_delta_lora_b, self.compute_dtype)
        scaling = self.cast(self.scaling, self.compute_dtype)

        # Shape operations
        x_shape = input_.shape
        input_ = self.reshape(input_, (-1, x_shape[-1]))

        # Dense result
        dense_result = self.matmul(input_, weight)
        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            dense_result = self.bias_add(dense_result, bias)
            bias = None
        else:
            bias = self.bias

        # LoRA result
        input_ = self.lora_dropout(input_)
        input_ = self.lora_a_matmul(input_, lora_a)
        input_ = self.lora_b_matmul(input_, lora_b)
        input_ = self.mul(input_, scaling)

        # Result addition
        dense_result = self.add(dense_result, input_)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            dense_result = self.reshape(dense_result, out_shape)
        dense_result = self.cast(dense_result, ori_dtype)
        return dense_result, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in LoRARowParallelLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel if config.data_parallel is not None else 1
        cp = config.context_parallel if config.context_parallel is not None else 1
        tp = config.tensor_parallel if config.tensor_parallel is not None else 1
        # shard of matmul
        if self.transpose_b:
            weight_strategy = (1, tp)
        else:
            weight_strategy = (tp, 1)
        matmul_in_strategy = ((dp * cp, tp), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)

        # shard of bias add
        if not self.skip_bias_add:
            add_in_strategy = ((dp * cp, 1), (1,))
            self.bias_add.shard(in_strategy=add_in_strategy)

        # shard of lora operator
        self.lora_dropout.dropout.shard((matmul_in_strategy[0],))
        if self.transpose_b:
            self.lora_a_matmul.shard((matmul_in_strategy[0], (1, matmul_in_strategy[1][1])))
            self.lora_b_matmul.shard(((matmul_in_strategy[0][0], 1), (matmul_in_strategy[1][0], 1)))
            self.mul.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][0]), ()))
            self.add.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][0]),
                            (matmul_in_strategy[0][0], matmul_in_strategy[1][0])))
        else:
            self.lora_a_matmul.shard((matmul_in_strategy[0], (matmul_in_strategy[1][0], 1)))
            self.lora_b_matmul.shard(((matmul_in_strategy[0][0], 1), (1, matmul_in_strategy[1][1])))
            self.mul.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][1]), ()))
            self.add.shard(((matmul_in_strategy[0][0], matmul_in_strategy[1][1]),
                            (matmul_in_strategy[0][0], matmul_in_strategy[1][1])))
