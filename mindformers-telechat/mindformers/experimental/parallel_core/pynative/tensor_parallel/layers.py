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
"""Layers"""

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "LinearWithGradAccumulationAndAsyncCommunication"
]

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, ops, mint
from mindspore.common.initializer import initializer, Zero
from mindspore.common.api import _pynative_executor
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_stream
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    ReduceFromModelParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    ScatterToModelParallelRegion,
    GatherFromSequenceParallelRegion
)
from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import (
    get_rng_tracer,
    TENSOR_PARALLEL_GENERATOR,
    EXPERT_PARALLEL_GENERATOR,
)


class LinearWithGradAccumulationAndAsyncCommunication(nn.Cell):
    r"""
    Linear execution with asynchronous communication in backprop.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix passed to the layer,
    and :math:`\text{bias}` is a bias vector passed to the layer.

    The gradient of weight is calculated simultaneously with
    all reduce communication of input gradient under tensor parallel condition.

    For sequence parallel, the calculation of weight gradient is overlapped with
    reduce scatter communication of input gradient.

    Args:
        bias (bool): Specifies whether the layer uses a bias vector.
        gradient_accumulation_fusion (bool): Specifies whether accumulate gradient in backprop.
        sequence_parallel (bool): Specifies whether sequence parallel is enabled. If enabled, the input is all
            gathered in the forward pass and the gradients are reduce scattered in the backward pass.
        allreduce_dgrad (bool): Specifies whether calculation and communication are overlapped.
        grad_output_buffer (Tensor): Buffer used to save output gradients. Default: ``None``.
        wgrad_deferral_limit (int): Limit on the number of micro-batches for which gradient GEMM of embedding weight
            should be deferred. Default: ``0``.
        transpose_b (bool): use transposed weight shape for initialization and compute. Default: ``True``.
        data_layout (str): Input layout. Default: ``"BSH"``.
        recompute_comm(bool) : Recompute allgather before dw of matmul

    Inputs:
        - **x** (Tensor) - The input Tensor of Shape (B, S, H) or (S, B, H).
        - **weight** (Tensor) - The weight Tensor of shape (H, H).
        - **bias** (Tensor) - The bias vector of shape (H, ).

    Outputs:
        - **output_parallel** (Tensor) - The Tensor of shape (B, S, H) or (S, B, H), which is consistent with input.

    Raises:
        NotImplementedError: `grad_output_buffer` is not supported for now.
        NotImplementedError: `wgrad_deferral_limit != 0` is not supported for now.
        NotImplementedError: `allreduce_dgrad` is currently not supported when sequence_parallel is enabled.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
        ...     LinearWithGradAccumulationAndAsyncCommunication
        ... )
        >>> import numpy as np
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> class LinearWithGradNet(nn.Cell):
        ...     def __init__(self):
        ...         super(LinearWithGradNet, self).__init__()
        ...         self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
        ...             bias=False,
        ...             gradient_accumulation_fusion=False,
        ...             sequence_parallel=False,
        ...             allreduce_dgrad=False,
        ...             data_layout="BSH"
        ...         )
        ...     def construct(self, x, weight):
        ...         output = self.forward_impl_(x, weight, bias=None, weight_param=None)
        ...         return output
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=1)
        >>> input_shape = (2, 3, 3)
        >>> weight_shape = (3, 3)
        >>> x = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> weight = Tensor(np.random.standard_normal(weight_shape).astype(np.float32))
        >>> net = LinearWithGradNet()
        >>> output = net(x, weight)
        >>> print(output)
        [[[ 0.96709514  1.54926     1.0753305 ]
          [-0.60277843 -3.2891138  -1.3835094 ]
          [-1.3610287  -2.396154   -1.5938796 ]]
         [[-0.3728725  -3.8218632  -1.4241107 ]
          [-0.95458585 -3.4623666  -1.678731  ]
          [ 1.0809017   2.203352    1.3136505 ]]]
    """
    def __init__(
            self,
            bias,
            gradient_accumulation_fusion,
            sequence_parallel,
            allreduce_dgrad,
            grad_output_buffer=None,
            wgrad_deferral_limit=0,
            transpose_b=True,
            data_layout="BSH",
            recompute_comm=False
        ):
        super(LinearWithGradAccumulationAndAsyncCommunication, self).__init__()
        if grad_output_buffer:
            raise NotImplementedError("`grad_output_buffer` is not supported for now.")
        if wgrad_deferral_limit != 0:
            raise NotImplementedError("`wgrad_deferral_limit != 0` is not supported for now.")
        self.use_bias = bias
        self.allreduce_dgrad = allreduce_dgrad
        self.grad_output_buffer = grad_output_buffer
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.transpose_b = transpose_b

        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.matmul_g_in = P.BatchMatMul(transpose_a=False, transpose_b=not self.transpose_b)
        self.matmul_g_w = P.BatchMatMul(transpose_a=True, transpose_b=False)
        self.tp_group = get_tensor_model_parallel_group()
        self.stream = get_stream()
        self.input_parallel = []
        self.weight_param = None
        self.data_layout = data_layout
        self.recompute_comm = recompute_comm and self.sequence_parallel
        self.used_bprop_inputs = [0, 1, 2, 3]

    # pylint: disable=C0111
    def construct(self, x, weight, bias, weight_param=None):
        if bias is None:
            self.use_bias = False
        self.weight_param = weight_param
        if self.sequence_parallel:
            if self.data_layout == "BSH":
                x = x.swapaxes(0, 1)
            x = comm_func.all_gather_into_tensor(x, group=self.tp_group)[0]
            if self.data_layout == "BSH":
                x = x.swapaxes(0, 1)

        if _pynative_executor.grad_flag() and not self.recompute_comm:
            self.input_parallel.append(x)

        output_parallel = self.matmul(x, weight)
        if self.use_bias:
            output_parallel = mint.add(
                output_parallel, bias
            )

        return output_parallel

    def prepare_input_tensors_for_wgrad_compute(self, dout, x):
        if self.data_layout == "SBH" and len(dout.shape) == 3:
            dout = dout.view(dout.shape[0] * dout.shape[1], dout.shape[2])
            x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        return dout, x

    # pylint: disable=W0613, C0111
    def bprop(self, *args):
        dout = args[-1]
        weight = args[1]
        if self.recompute_comm:
            x = args[0]
            if self.data_layout == "BSH":
                x = x.swapaxes(0, 1)
            x, x_handler = comm_func.all_gather_into_tensor(x, group=self.tp_group, async_op=True)
            grad_input = self.matmul_g_in(dout, weight)
            x_handler.wait()
            if self.data_layout == "BSH":
                x = x.swapaxes(0, 1)
            grad_input.reshape(x.shape)
        else:
            x = self.input_parallel.pop(0)
            grad_input = self.matmul_g_in(dout, weight).reshape(x.shape)
        wgrad_compute = True

        if wgrad_compute:
            dout, x = self.prepare_input_tensors_for_wgrad_compute(dout, x)

        if self.allreduce_dgrad:
            grad_input, grad_input_handle = comm_func.all_reduce(grad_input, group=self.tp_group, async_op=True)

        if self.sequence_parallel:
            if self.allreduce_dgrad:
                raise NotImplementedError("`allreduce_dgrad` is not supported for now.")
            if self.data_layout == "BSH":
                grad_input = grad_input.swapaxes(0, 1)
            grad_input, grad_input_handle = comm_func.reduce_scatter_tensor(
                grad_input, group=self.tp_group, async_op=True)

        if self.transpose_b:
            grad_weight = self.matmul_g_w(dout, x)
        else:
            grad_weight = self.matmul_g_w(x, dout)

        if len(dout.shape) > 2:     # b,s,h / s,b,h
            grad_weight = mint.sum(grad_weight, 0)
            dout = dout.sum(axis=0)

        grad_weight = grad_weight.reshape(weight.shape)
        grad_bias = dout.sum(axis=0) if self.use_bias else None

        if self.gradient_accumulation_fusion and self.weight_param is not None and \
            isinstance(self.weight_param, Parameter):
            if hasattr(self.weight_param, 'grad_accumulated'):
                origin_dtype = None
                if grad_weight.dtype != self.weight_param.dtype:
                    grad_weight = ops.cast(grad_weight, self.weight_param.dtype)
                    origin_dtype = grad_weight.dtype
                self.weight_param.main_grad[:] = mint.add(self.weight_param.main_grad, grad_weight)
                self.weight_param.grad_accumulated = True
                if origin_dtype:
                    grad_weight = ops.cast(grad_weight, origin_dtype)

        if self.sequence_parallel or self.allreduce_dgrad:
            grad_input_handle.wait()

        if self.sequence_parallel and self.data_layout == "BSH":
            grad_input = grad_input.swapaxes(0, 1)

        return grad_input, grad_weight, grad_bias, None


class LinearWithFrozenWeight(nn.Cell):
    r"""
    Linear execution with frozen weight.

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    which :math:`inputs` is the input tensors, :math:`weight` is a weight matrix,
    and :math:`\text{bias}` is a bias vector.

    The gradient of weight is not calculated during backward propagation.

    Args:
        bias (bool): Specifies whether the layer uses a bias vector.
        allreduce_dgrad (bool): Specifies whether calculation and communication are overlapped. Default: None.
        transpose_b (bool): use transposed weight shape for initialization and compute. Default: True.

    Inputs:
        - **x** (Tensor) - The input Tensor.
        - **weight** (Tensor) - The weight matrix for multiplication.
        - **bias** (Tensor) - The bias vector for addition after multiplication.

    Outputs:
        Tensor, the output after the calculation.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import LinearWithFrozenWeight
        >>> input = Tensor(np.random.random((2, 3, 3)).astype(np.float32))
        >>> weight = Tensor(np.random.random((3, 3)).astype(np.float32))
        >>> bias = Tensor(np.random.random((1)).astype(np.float32))
        >>> model = LinearWithFrozenWeight(bias=bias)
        >>> output = model(input, weight, bias)
    """
    def __init__(self, bias, allreduce_dgrad=None, transpose_b=True):
        super(LinearWithFrozenWeight, self).__init__()
        self.bias = bias
        self.allreduce_dgrad = allreduce_dgrad
        self.transpose_b = transpose_b
        self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.matmul_g_in = P.BatchMatMul(transpose_a=False, transpose_b=not self.transpose_b)
        self.tp_group = get_tensor_model_parallel_group()

    def construct(self, input_, weight, bias):
        output = self.matmul(input_, weight)
        if self.bias and bias is not None:
            output = mint.add(output, bias)
        return output

    # pylint: disable=W0613
    def bprop(self, x, weight, bias, out, dout):
        grad_input = self.matmul_g_in(dout, weight).reshape(x.shape)
        grad_bias = F.full(bias.shape, 0, dtype=bias.dtype) if self.bias else None
        if self.allreduce_dgrad:
            grad_input = comm_func.all_reduce(grad_input, group=self.tp_group)[0]
        return grad_input, None, grad_bias


class ColumnParallelLinear(nn.Cell):
    r"""
    The dense layer with weight sliced on second dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): The config of the transformer model. For details, please refer to TransformerConfig.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`.
        bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: ``False``.
        stride (int): For the strided linear layers. Default: ``1``.
        keep_master_weight_for_test (bool): For testing and should be set to False. It returns the master weights used
            for initialization. Default: ``False``.
        skip_bias_add (bool): If True, do not add the bias term, instead return it for fusion. Default: ``False``.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: ``False``.
        embedding_activation_buffer (Tensor): This buffer holds the input activations of the final embedding linear
            layer on the last pipeline stage. Default: ``None``.
        grad_output_buffer (Tensor): This buffer holds the gradient outputs of the final embedding linear layer on
            the last pipeline stage. Default: ``None``.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: ``False``.
        tp_comm_buffer_name (str): Communication buffer name is not used in non-Transformer-Engine modules.
            Default: ``None``.
        disable_grad_reduce (bool): If True, reduction of output gradients across tensor-parallel ranks will be
            disabled. Default: ``False``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: ``Zero()``.
        param_init_dtype (dtype.Number): The parameter initialization type. Default: ``None``.
        compute_dtype (dtype.Number): The computation type. Default: ``None``.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape. Default:
            ``True``.

    Inputs:
        - **input\_** (Tensor) - Tensor of shape :math:`(*, in\_channels)` . The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.
        - **weight** (Tensor) - Tensor of shape :math:`(in\_channels, out\_channels)`. The `input_size` in `Args` should
          be equal to :math:`in\_channels` in `Inputs`. The `output_size` in `Args` should be equal to
          :math:`out\_channels` in `Outputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`. The `output_size` in `Args` should be equal to :math:`out\_channels`
        in `Outputs`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.
        NotImplementedError: `stride > 1` is not supported for now.
        NotImplementedError: `keep_master_weight_for_test=True` is not supported for now.
        NotImplementedError: `embedding_activation_buffer` is not supported for now.
        NotImplementedError: `grad_output_buffer` is not supported for now.
        NotImplementedError: `tp_comm_buffer_name` is not supported for now.
        NotImplementedError: `disable_grad_reduce=True` is not supported for now.
        NotImplementedError: `config.parallel_config.use_cpu_initialization` is not supported for now.
        RuntimeError: use zero3 optimizer parallel without initializing data parallel communication.
        RuntimeError: `allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method without any third-party
            or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.config import (
        ...     ModelParallelConfig,
        ...     TransformerConfig
        ... )
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel import ColumnParallelLinear
        >>> from mindformers.experimental.utils import init_method_normal
        >>> class TestNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(TestNet, self).__init__()
        ...         hidden_size = config.hidden_size
        ...         self.columnlinear = ColumnParallelLinear(input_size=hidden_size,
        ...                                                  output_size=hidden_size,
        ...                                                  config=config,
        ...                                                  init_method=config.init_method,
        ...                                                  bias=config.mlp_has_bias,
        ...                                                  gather_output=False,
        ...                                                  skip_bias_add=False,
        ...                                                  bias_init=config.bias_init)
        ...     def construct(self, input_):
        ...         output, _ = self.columnlinear(input_)
        ...         return output
        >>> dataset_size = 1
        >>> seq_length = 2
        >>> hidden_size = 4
        >>> tensor_parallel = 1
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
        >>> input_data = Tensor(np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32))
        >>> parallel_config = ModelParallelConfig()
        >>> model_config = TransformerConfig(vocab_size=40000,
        ...                                  num_layers=1,
        ...                                  num_attention_heads=1,
        ...                                  mlp_has_bias=True,
        ...                                  gated_linear_unit=False,
        ...                                  hidden_size=hidden_size,
        ...                                  ffn_hidden_size=4*hidden_size,
        ...                                  hidden_act='gelu',
        ...                                  parallel_config=parallel_config,
        ...                                  params_dtype='float32',
        ...                                  compute_dtype='float32')
        >>> network = TestNet(config=model_config)
        >>> output = network(input_data)
        >>> print(output)
        [[[ 0.01780816  0.00895902 -0.00554341 -0.00185049]
          [ 0.02319741 -0.00320548 -0.0062025  -0.0050142 ]]]
    """
    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method,
            bias=True,
            gather_output=False,
            stride=1,
            keep_master_weight_for_test=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            embedding_activation_buffer=None,
            grad_output_buffer=None,
            is_expert=False,
            tp_comm_buffer_name=None,
            disable_grad_reduce=False,
            bias_init=Zero(),
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
        ):
        super(ColumnParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("`stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("`keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if embedding_activation_buffer:
            raise NotImplementedError("`embedding_activation_buffer` is not supported "
                                      "for now.")
        if grad_output_buffer:
            raise NotImplementedError("`grad_output_buffer` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("`tp_comm_buffer_name` is not supported for now.")
        if disable_grad_reduce:
            raise NotImplementedError("`disable_grad_reduce=True` is not supported for now.")
        if config.parallel_config.use_cpu_initialization:
            raise NotImplementedError("`config.parallel_config.use_cpu_initialization` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        tensor_parallel_group_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, tensor_parallel_group_size)

        self.is_expert = is_expert
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.config = config
        self.param_init_dtype = param_init_dtype if param_init_dtype else self.config.params_dtype
        self.compute_dtype = compute_dtype if compute_dtype else self.config.compute_dtype
        self.transpose_b = transpose_b

        self.expert_parallel = self.config.parallel_config.expert_model_parallel_size > 1
        self.sequence_parallel = self.config.parallel_config.sequence_parallel
        self.use_zero3 = self.config.parallel_config.zero_level == 'z3'
        if self.use_zero3:
            try:
                dp_size = get_data_parallel_world_size()
            except AssertionError as e:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.") from e

        if self.transpose_b:
            if self.use_zero3 and self.output_size_per_partition % dp_size == 0:
                self.output_size_per_partition = divide(self.output_size_per_partition, dp_size)
            else:
                self.use_zero3 = False
        else:
            if self.use_zero3 and self.input_size % dp_size == 0 and self.output_size_per_partition % dp_size == 0:
                self.input_size = divide(self.input_size, dp_size)
            else:
                self.use_zero3 = False

        if self.sequence_parallel and tensor_parallel_group_size <= 1:
            self.sequence_parallel = False

        mode = EXPERT_PARALLEL_GENERATOR if self.is_expert and self.expert_parallel else TENSOR_PARALLEL_GENERATOR

        weight_shape = (self.output_size_per_partition, self.input_size) if self.transpose_b \
                        else (self.input_size, self.output_size_per_partition)

        with get_rng_tracer().rng_fork(mode):
            if not self.skip_weight_param_allocation:
                self.weight = Parameter(
                    initializer(
                        init_method,
                        weight_shape,
                        self.param_init_dtype,
                    ),
                    name="weight",
                )
            if self.has_bias:
                if self.use_zero3 and not self.transpose_b:
                    self.output_size_per_partition = divide(self.output_size_per_partition, dp_size)
                self.bias = Parameter(
                    initializer(
                        bias_init, (self.output_size_per_partition), self.param_init_dtype
                    ),
                    name="bias",
                )

        self.explicit_expert_comm = self.is_expert and (
            config.parallel_config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gather_from_mp_region = GatherFromModelParallelRegion()
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=self.config.dataset_config.data_layout == "BSH"
        )
        self.allreduce_dgrad = (
            tensor_parallel_group_size > 1 and not self.sequence_parallel and not disable_grad_reduce
        )
        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )
        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion
        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=(self.has_bias and not self.skip_bias_add),
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
            allreduce_dgrad=False if self.explicit_expert_comm else self.allreduce_dgrad,
            transpose_b=self.transpose_b,
            data_layout=self.config.dataset_config.data_layout,
            recompute_comm=self.config.select_comm_recompute
        )

        self.frozen_weight_forward_impl_ = LinearWithFrozenWeight(
            bias=(self.has_bias and not self.skip_bias_add),
            allreduce_dgrad=False if self.explicit_expert_comm else self.allreduce_dgrad
        )

    def construct(self, input_, weight=None):
        """construct method."""
        if weight is None and self.skip_weight_param_allocation:
            raise ValueError("when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")
        if weight is not None and not self.skip_weight_param_allocation:
            raise ValueError("when skip_weight_param_allocation=False,"
                             "weight should not be passed to construct(), but got {}".format(weight))

        if (
                self.sequence_parallel
                or self.explicit_expert_comm
                or self.allreduce_dgrad
        ):
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        if self.skip_weight_param_allocation:
            weight_requires_grad = not isinstance(weight, Parameter) or weight.requires_grad
            weight_param = weight
            weight = ops.cast(weight, self.compute_dtype)
        else:
            weight_requires_grad = self.weight.requires_grad
            weight_param = self.weight
            weight = ops.cast(self.weight, self.compute_dtype)
        if input_parallel.dtype != self.compute_dtype:
            input_parallel = ops.cast(input_parallel, self.compute_dtype)

        bias = ops.cast(self.bias, self.compute_dtype) if self.has_bias and not self.skip_bias_add else None

        if not weight_requires_grad:
            if self.sequence_parallel:
                input_parallel = self.gather_from_sp_region(input_parallel)
            output_parallel = self.frozen_weight_forward_impl_(input_parallel, weight, bias)
        else:
            output_parallel = self.forward_impl_(input_parallel, weight, bias, weight_param=weight_param)

        if output_parallel.dtype != origin_dtype:
            output_parallel = ops.cast(output_parallel, origin_dtype)

        if self.gather_output:
            output = self.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.has_bias and self.skip_bias_add else None

        return output, output_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tensor_model_parallel_world_size()
        w_shard = (tp_size, 1) if self.transpose_b else (1, tp_size)
        state_dict = {}
        opt_weight_shard_step = get_tensor_model_parallel_world_size() if self.use_zero3 else 0
        opt_weight_shard_size = get_data_parallel_world_size() if self.use_zero3 else 0
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {
                'shape': self.weight.shape,
                'shard': w_shard,
                'opt_weight_shard_step': opt_weight_shard_step,
                'opt_weight_shard_size': opt_weight_shard_size
            }
        if self.has_bias:
            state_dict[self.bias.name] = {
                'shape': self.bias.shape,
                'shard': (tp_size,),
                'opt_weight_shard_step': opt_weight_shard_step,
                'opt_weight_shard_size': opt_weight_shard_size
            }
        return state_dict


class RowParallelLinear(nn.Cell):
    r"""
    The dense layer with weight sliced on first dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): The config of the transformer model. For details, please refer to TransformerConfig.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`.
        bias (bool): Specifies whether the layer uses a bias vector.
        input_is_parallel (bool): If True, we assume that the input is already split across the tensor parallel group
            and we do not split again.
        skip_bias_add (bool): If True, do not add the bias term, instead return it for fusion. Default: ``True``.
        stride (int): For the strided linear layers. Default: ``1``.
        keep_master_weight_for_test (bool): For tesing and should be set to False. It returns the master weights used
            for initialization. Default: ``False``.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: ``False``.
        tp_comm_buffer_name (str): Communication buffer name is not used in non-Transformer-Engine modules.
            Default: ``None``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: ``Zero()``.
        param_init_dtype (dtype.Number): The parameter initialization type. Default: ``None``.
        compute_dtype (dtype.Number): The computation type. Default: ``None``.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape. Default:
            ``True``.

    Inputs:
        - **input\_** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`. The `output_size` in `Args` should be equal to :math:`out\_channels`
        in `Outputs`.

    Raises:
        ValueError: `sequence_parallel` should be False when `input_is_parallel` is False , but got True.
        ValueError: `skip_bias_add` should be True when `explicit_expert_comm` is True , but got False.
        NotImplementedError: `stride > 1` is not supported for now.
        NotImplementedError: `keep_master_weight_for_test=True` is not supported for now.
        NotImplementedError: `tp_comm_buffer_name` is not supported for now.
        RuntimeError: use zero3 optimizer parallel without initializing data parallel communication.
        RuntimeError: To enable `sequence_parallel`, `input_is_parallel` must be True but got False.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method without any third-party
            or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.config import (
        ...     ModelParallelConfig,
        ...     TransformerConfig
        ... )
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel import RowParallelLinear
        >>> from mindformers.experimental.utils import init_method_normal
        >>> class TestNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(TestNet, self).__init__()
        ...         hidden_size = config.hidden_size
        ...         self.rowlinear = RowParallelLinear(input_size=hidden_size,
        ...                                            output_size=hidden_size,
        ...                                            config=config,
        ...                                            init_method=config.init_method,
        ...                                            bias=config.mlp_has_bias,
        ...                                            input_is_parallel=True,
        ...                                            skip_bias_add=False,
        ...                                            bias_init=config.bias_init)
        ...     def construct(self, input_):
        ...         output, _ = self.rowlinear(input_)
        ...         return output
        >>> dataset_size = 1
        >>> seq_length = 2
        >>> hidden_size = 4
        >>> tensor_parallel = 1
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
        >>> input_data = Tensor(np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32))
        >>> parallel_config = ModelParallelConfig()
        >>> model_config = TransformerConfig(vocab_size=40000,
        ...                                  num_layers=1,
        ...                                  num_attention_heads=1,
        ...                                  mlp_has_bias=True,
        ...                                  gated_linear_unit=False,
        ...                                  hidden_size=hidden_size,
        ...                                  ffn_hidden_size=4*hidden_size,
        ...                                  hidden_act='gelu',
        ...                                  parallel_config=parallel_config,
        ...                                  params_dtype='float32',
        ...                                  compute_dtype='float32')
        >>> network = TestNet(config=model_config)
        >>> output = network(input_data)
        >>> print(output)
        [[[ 0.01780816  0.00895902 -0.00554341 -0.00185049]
          [ 0.02319741 -0.00320548 -0.0062025  -0.0050142 ]]]
    """
    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method,
            bias,
            input_is_parallel,
            skip_bias_add=True,
            stride=1,
            keep_master_weight_for_test=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            bias_init=Zero(),
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
        ):
        super(RowParallelLinear, self).__init__()
        if stride > 1:
            raise NotImplementedError("`stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("`keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("`tp_comm_buffer_name` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

        tensor_parallel_group_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, tensor_parallel_group_size)

        self.config = config
        self.param_init_dtype = param_init_dtype if param_init_dtype else self.config.params_dtype
        self.compute_dtype = compute_dtype if compute_dtype else self.config.compute_dtype
        self.is_expert = is_expert
        self.expert_parallel = self.config.parallel_config.expert_model_parallel_size > 1
        self.sequence_parallel = self.config.parallel_config.sequence_parallel
        self.use_zero3 = self.config.parallel_config.zero_level == 'z3'
        self.transpose_b = transpose_b

        if self.use_zero3:
            try:
                dp_size = get_data_parallel_world_size()
            except AssertionError as e:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.") from e

        if self.transpose_b:
            if self.use_zero3 and self.output_size % dp_size == 0:
                self.output_size = divide(self.output_size, dp_size)
        else:
            if self.use_zero3 and self.input_size_per_partition % dp_size == 0 and self.output_size % dp_size == 0:
                self.input_size_per_partition = divide(self.input_size_per_partition, dp_size)
            else:
                self.use_zero3 = False

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
            )

        mode = EXPERT_PARALLEL_GENERATOR if self.is_expert and self.expert_parallel else TENSOR_PARALLEL_GENERATOR

        weight_shape = (self.output_size, self.input_size_per_partition) if self.transpose_b \
                        else (self.input_size_per_partition, self.output_size)

        with get_rng_tracer().rng_fork(mode):
            self.weight = Parameter(
                initializer(
                    init_method,
                    weight_shape,
                    self.param_init_dtype,
                ),
                name="weight",
            )

            if self.has_bias:
                if self.use_zero3 and not self.transpose_b:
                    self.output_size = divide(self.output_size, dp_size)
                self.bias = Parameter(
                    initializer(bias_init, (self.output_size), self.param_init_dtype), name="bias"
                )

        self.explicit_expert_comm = self.is_expert and (
            config.parallel_config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        self.scatter_to_mp_region = ScatterToModelParallelRegion()
        if self.sequence_parallel:
            self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion(
                need_to_swapaxes=self.config.dataset_config.data_layout == "BSH"
            )
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion

        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=False,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=False,
            allreduce_dgrad=False,
            transpose_b=self.transpose_b,
            data_layout=self.config.dataset_config.data_layout
        )

        self.frozen_weight_forward_impl_ = LinearWithFrozenWeight(
            bias=False,
            allreduce_dgrad=False
        )

    def construct(self, input_):
        """construct method"""
        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.sequence_parallel:
                raise ValueError("sequence_parallel should be False here, but got True.")
            input_parallel = self.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = ops.cast(self.weight, self.compute_dtype)
        weight_param = self.weight if self.gradient_accumulation_fusion else None
        if input_parallel.dtype != self.compute_dtype:
            input_parallel = ops.cast(input_parallel, self.compute_dtype)
        if self.weight.requires_grad:
            output_parallel = self.forward_impl_(input_parallel, weight, bias=None, weight_param=weight_param)
        else:
            output_parallel = self.frozen_weight_forward_impl_(input_parallel, weight, bias=None)
        if self.explicit_expert_comm:
            if not self.skip_bias_add:
                raise ValueError("skip_bias_add should be True here, but got False.")
            output = output_parallel
        elif self.sequence_parallel:
            output = self.reduce_scatter_to_sp_region(output_parallel)
        else:
            output = self.reduce_from_mp_region(output_parallel)

        output_bias = None
        if not self.skip_bias_add:
            output = mint.add(output, ops.cast(self.bias, self.compute_dtype)) if self.has_bias else output
        else:
            if self.has_bias:
                output_bias = self.bias
        if output.dtype != origin_dtype:
            output = ops.cast(output, origin_dtype)

        return output, output_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tensor_model_parallel_world_size()
        w_shard = (1, tp_size) if self.transpose_b else (tp_size, 1)
        state_dict = {}
        opt_weight_shard_step = get_tensor_model_parallel_world_size() if self.use_zero3 else 0
        opt_weight_shard_size = get_data_parallel_world_size() if self.use_zero3 else 0
        state_dict[self.weight.name] = {
            'shape': self.weight.shape,
            'shard': w_shard,
            'opt_weight_shard_step': opt_weight_shard_step,
            'opt_weight_shard_size': opt_weight_shard_size
        }
        if self.has_bias:
            state_dict[self.bias.name] = {
                'shape': self.bias.shape,
                'shard': (1,),
                'opt_weight_shard_step': opt_weight_shard_step,
                'opt_weight_shard_size': opt_weight_shard_size
            }
        return state_dict


class VocabParallelEmbedding(nn.Cell):
    r"""
    Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`.
        reduce_scatter_embeddings (bool): Decides whether to perform ReduceScatter after embedding lookup. Default:
            ``False``.
        config (dict): The config of the transformer model. For details, please refer to TransformerConfig.
        param_init_dtype (dtype.Number): The parameter initialization type. Default: ``None``.

    Inputs:
        - **input\_** (Tensor) - Tensor of shape (B, S) or (S, B).

    Outputs:
        - **output** (Tensor) - Tensor of shape (B, S, H) or (S, B, H), which is consistent with the input.

    Raises:
        ValueError: The vocabulary size is not divisible by size of tensor parallel.
        NotImplementedError: `config.parallel_config.deterministic_mode` is not supported for now.
        NotImplementedError: `config.parallel_config.use_cpu_initialization` is not supported for now.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method without any third-party
            or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import VocabParallelEmbedding
        >>> from mindformers.experimental.parallel_core.pynative.config import (
        ...     ModelParallelConfig,
        ...     DatasetConfig,
        ...     TransformerConfig
        ... )
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> ms.set_context(pynative_synchronize=True)
        >>> ms.set_seed(2024)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        >>> class ParallelTransformerLayerNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(ParallelTransformerLayerNet, self).__init__()
        ...         self.config = config
        ...         self.embedding = VocabParallelEmbedding(
        ...             num_embeddings=config.vocab_size,
        ...             embedding_dim=config.hidden_size,
        ...             init_method=config.init_method,
        ...             reduce_scatter_embeddings=config.parallel_config.sequence_parallel,
        ...             config=config,
        ...         )
        ...     def construct(self, x):
        ...         x = self.embedding(x)
        ...         return x
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=2,
        ...                                       pipeline_model_parallel_size=1,
        ...                                       context_parallel_size=1,
        ...                                       expert_model_parallel_size=1,
        ...                                       sequence_parallel=True)
        >>> dataset_config = DatasetConfig(batch_size=1,
        ...                                dataset_size=2,
        ...                                seq_length=1024)
        >>> model_config = TransformerConfig(vocab_size=50304,
        ...                                  num_layers=1,
        ...                                  num_attention_heads=32,
        ...                                  hidden_size=2560,
        ...                                  ffn_hidden_size=7680,
        ...                                  parallel_config=parallel_config)
        >>> model_config.dataset_config = dataset_config
        >>> batch_size = dataset_config.batch_size
        >>> dataset_size = dataset_config.dataset_size
        >>> seq_length = dataset_config.seq_length
        >>> vocab_size = model_config.vocab_size
        >>> init()
        >>> tensor_parallel = parallel_config.tensor_model_parallel_size
        >>> initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
        >>> network = ParallelTransformerLayerNet(config=model_config)
        >>> input_shape = (batch_size, seq_length)
        >>> input_ids = Tensor(np.ones(input_shape).astype(np.float32))
        >>> output = network(input_ids)
        >>> print(output)
        [[-0.00546161]
         [ 0.00440422]
         [ 0.00252223]
         ...
         [ 0.00539334]
         [-0.00625365]
         [-0.01025379]]
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            *,
            init_method,
            reduce_scatter_embeddings=False,
            config,
            param_init_dtype=None
    ):
        super().__init__()
        if config.parallel_config.deterministic_mode:
            raise NotImplementedError("`config.parallel_config.deterministic_mode` is not supported for now.")
        if config.parallel_config.use_cpu_initialization:
            raise NotImplementedError("`config.parallel_config.use_cpu_initialization` is not supported for now.")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.param_init_dtype = param_init_dtype if param_init_dtype else config.params_dtype
        self.data_layout = config.dataset_config.data_layout

        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        with get_rng_tracer().rng_fork():
            self.weight = Parameter(
                initializer(
                    init=init_method,
                    shape=(self.num_embeddings_per_partition, self.embedding_dim),
                    dtype=self.param_init_dtype,
                ),
                name="weight",
            )
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        if self.reduce_scatter_embeddings:
            self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion(
                need_to_swapaxes=self.data_layout == "BSH"
            )

    def construct(self, input_):
        """ construct. """
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = mint.logical_or((input_ < self.vocab_start_index), (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.copy() - self.vocab_start_index
            masked_input = ops.masked_fill(
                masked_input, input_mask, Tensor(0, masked_input.dtype)
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = mint.nn.functional.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel = ops.masked_fill(
                output_parallel,
                input_mask.expand_dims(2),
                Tensor(0.0, output_parallel.dtype),
            )
        if self.reduce_scatter_embeddings:
            if self.data_layout == "SBH":
                output_parallel = output_parallel.swapaxes(0, 1)    # BSH
            output = self.reduce_scatter_to_sp_region(output_parallel)
        else:
            # Reduce across all the model parallel devices.
            output = self.reduce_from_mp_region(output_parallel)
        return output

    # pylint: disable=W0613
    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(f"The vocabulary size is {global_vocab_size},"
                             f"which is not divisible by size of tensor parallel({world_size}).")
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tensor_model_parallel_world_size()
        w_shard = (tp_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {
            'shape': self.weight.shape,
            'shard': w_shard,
            'opt_weight_shard_step': 0,
            'opt_weight_shard_size': 0
        }

        return state_dict
