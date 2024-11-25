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
Linear units for tensor parallelism.
"""
from typing import List, Optional, Callable

try:
    from mindformer._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import nn, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.utils import init_method_zero

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding"
]


class ColumnParallelLinear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
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
        init_method (Callable): The initialization method. Default: None
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
                 bias_init: Callable = None
                 ):
        super(ColumnParallelLinear, self).__init__()
        if gather_output:
            raise NotImplementedError("For ColumnParallelLinear, `gather_output` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if grad_output_buffer is not None:
            raise NotImplementedError("For ColumnParallelLinear, `grad_output_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ColumnParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")
        if disable_grad_reduce:
            raise NotImplementedError("For ColumnParallelLinear, `disable_grad_reduce` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = compute_dtype
        self.cast = P.Cast()
        self.transpose_b = transpose_b
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.has_bias = bias
        self.params_dtype = config.params_dtype

        if skip_weight_param_allocation:
            self.weight = None
        else:
            weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
            # we use `zeros` to generate a tensor as the `init_method` parameter.
            self.weight = Parameter(init_method(initializer('zeros', weight_shape)), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias_ = Parameter(bias_init(initializer('zeros', (output_size,))), name='bias')
        else:
            self.bias_ = None

        self.matmul = P.MatMul(transpose_b=transpose_b)
        if not skip_bias_add:
            self.add = P.Add()
        self.reshape = P.Reshape()
        self.shard(config)

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of ColumnParallelLinear.

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For ColumnParallelLinear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight
        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias_, self.compute_dtype)
            input_ = self.add(input_, bias)
            bias = None
        else:
            bias = self.bias_

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel if config.data_parallel is not None else 1
        tp = config.tensor_parallel if config.tensor_parallel is not None else 1
        cp = config.context_parallel if config.context_parallel is not None else 1

        if self.transpose_b:
            weight_strategy = (tp, 1)
        else:
            weight_strategy = (1, tp)
        matmul_in_strategy = ((dp * cp, 1), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)

        if not self.skip_bias_add:
            add_in_strategy = ((dp * cp, tp), (tp,))
            self.add.shard(in_strategy=add_in_strategy)


class RowParallelLinear(nn.Cell):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along
    its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        input_is_parallel (bool): Whether the input is already parallelized. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        init_method (Callable): The initialization method. Default: None
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
                 bias_init: Callable = None
                 ):
        super(RowParallelLinear, self).__init__()
        if input_is_parallel:
            raise NotImplementedError("For RowParallelLinear, `input_is_parallel` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For RowParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For RowParallelLinear, `tp_comm_buffer_name` is not supported for now")
        self.input_size = input_size
        self.output_size = output_size
        self.transpose_b = transpose_b
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.has_bias = bias

        weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
        # we use `zeros` to generate a tensor as the `init_method` parameter.
        self.weight = Parameter(init_method(initializer('zeros', weight_shape)), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias_ = Parameter(bias_init(initializer('zeros', (output_size,))), name='bias')
        else:
            self.bias_ = None

        self.matmul = P.MatMul(transpose_b=transpose_b)
        if not skip_bias_add:
            self.add = P.Add()
        self.reshape = P.Reshape()
        self.shard(config)

    def construct(self, input_: Tensor) -> tuple[Tensor, Tensor]:
        """Forward of RowParallelLinear.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        output_shape = input_.shape[:-1] + (self.output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype
        weight = self.cast(self.weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias_, self.compute_dtype)
            input_ = self.add(input_, bias)
            bias = None
        else:
            bias = self.bias_

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output, bias

    def shard(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinear.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel if config.data_parallel is not None else 1
        cp = config.context_parallel if config.context_parallel is not None else 1
        tp = config.tensor_parallel if config.tensor_parallel is not None else 1

        if self.transpose_b:
            weight_strategy = (1, tp)
        else:
            weight_strategy = (tp, 1)
        matmul_in_strategy = ((dp * cp, tp), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)

        if not self.skip_bias_add:
            add_in_strategy = ((dp * cp, 1), (1,))
            self.add.shard(in_strategy=add_in_strategy)


class VocabParallelEmbedding(nn.Cell):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The size of each embedding vector.
        parallel_config (ModelParallelConfig): The model parallel configuration.
        init_method (str): The initialization method. Default: 'normal'.
        init_type (dtype): The data type of the initialization. Default: dtype.float32.
    """
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            parallel_config,
            init_method: Callable,
            init_type=dtype.float32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init_method(
                initializer('zeros', [self.num_embeddings, self.embedding_dim], dtype=init_type)
            ), name="weight"
        )
        self.gather = P.Gather()
        self.parallel_config = parallel_config
        self.shard(self.parallel_config)

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        Validator.check_type_name("input_ids", F.dtype(input_ids), [dtype.int32, dtype.int64], self.cls_name)
        output = self.gather(self.weight, input_ids, 0)

        return output

    def shard(self, config: TransformerConfig):
        """sharding for embedding"""
        dp = 1 if config.data_parallel is None else config.data_parallel
        tp = 1 if config.tensor_parallel is None else config.tensor_parallel
        cp = 1 if config.context_parallel is None else config.context_parallel
        if config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp, cp)))
        else:
            if self.num_embeddings % tp != 0:
                self.gather.shard(((1, 1), (dp, cp)))
            else:
                self.gather.shard(((tp, 1), (1, 1)))
