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
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, mint
from mindspore.common.initializer import initializer

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    GatherFromSequenceParallelRegion,
    ReduceFromModelParallelRegion,
    ReduceScatterToSequenceParallelRegion,
    ScatterToModelParallelRegion,
)
from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import (
    get_rng_tracer,
    TENSOR_PARALLEL_GENERATOR,
    EXPERT_PARALLEL_GENERATOR,
)


class ColumnParallelLoRA(nn.Cell):
    r"""
    The LoRA layer with weight sliced on second dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias} + \text{inputs} * \text{lora_a} * \text{lora_b},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True),
    :math:`\text{lora_a}` is the lora_a matrix with shape (input_size, rank),
    and :math:`\text{lora_b}` is the lora_b matrix with shape (rank, output_size) .

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Parallel configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        stride (int): For the strided linear layers. Default: 1.
        keep_master_weight_for_test (bool): For testing and should be set to False. It returns the master weights used
            for initialization. Default: False.
        skip_bias_add (bool): If True, do not add the bias term, instead return it for fusion. Default: False.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        embedding_activation_buffer (Tensor): This buffer holds the input activations of the final embedding linear
            layer on the last pipeline stage. Default: None.
        grad_output_buffer (Tensor): This buffer holds the gradient outputs of the final embedding linear layer on
            the last pipeline stage. Default: None.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        tp_comm_buffer_name (str): Communication buffer name is not used in non-Transformer-Engine modules.
            Default: None.
        disable_grad_reduce (bool): If True, reduction of output gradients across tensor-parallel ranks will be
            disabled. Default: False.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: 'zeros'.
        param_init_dtype (dtype.Number): The parameter initialization type. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: None.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            Default: True.
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
          When `skip_weight_param_allocation=True`, this input must be provided. Default: None.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Parameter): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    Supported Platforms:
        ``Ascend``
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
            bias_init='zeros',
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.0,
            lora_a_init='normal',
            lora_b_init='zeros',
        ):
        super(ColumnParallelLoRA, self).__init__()
        if stride > 1:
            raise NotImplementedError("`stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError("`keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if skip_bias_add:
            raise NotImplementedError("`skip_bias_add=True` is not supported for now.")
        if embedding_activation_buffer:
            raise NotImplementedError("`embedding_activation_buffer` is not supported "
                                      "for now.")
        if grad_output_buffer:
            raise NotImplementedError("`grad_output_buffer` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("`tp_comm_buffer_name` is not supported for now.")
        if disable_grad_reduce:
            raise NotImplementedError("`disable_grad_reduce=True` is not supported for now.")

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        tensor_parallel_group_size = get_tensor_model_parallel_world_size()

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = mint.nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.lora_rank

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
            except AssertionError as ex:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.") from ex

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
                self.weight = Parameter(initializer(init_method, weight_shape, self.param_init_dtype), name="weight")
            self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)

            if self.has_bias:
                if self.use_zero3 and not self.transpose_b:
                    self.output_size_per_partition = divide(self.output_size_per_partition, dp_size)
                self.bias = Parameter(
                    initializer(bias_init, (self.output_size_per_partition), self.param_init_dtype), name="bias")

        self.explicit_expert_comm = self.is_expert and (self.sequence_parallel or self.expert_parallel)

        lora_a_shape = (lora_rank, weight_shape[1]) if self.transpose_b else (weight_shape[0], lora_rank)
        lora_b_shape = (weight_shape[0], lora_rank) if self.transpose_b else \
                       (lora_rank, weight_shape[1])
        self.lora_a_matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=self.transpose_b)

        self.lora_a = Parameter(initializer(lora_a_init, lora_a_shape, self.param_init_dtype))
        self.lora_b = Parameter(initializer(lora_b_init, lora_b_shape, self.param_init_dtype))

        self.explicit_expert_comm = self.is_expert and (self.sequence_parallel or self.expert_parallel)

        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gather_from_mp_region = GatherFromModelParallelRegion()
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=self.config.dataset_config.data_layout == "BSH"
        )

    def construct(self, input_, weight=None):
        """construct method."""
        if weight is None and self.skip_weight_param_allocation:
            raise ValueError("when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        lora_a = ops.cast(self.lora_a, self.compute_dtype)
        lora_input = ops.cast(input_, self.compute_dtype)
        lora_parallel = self.lora_a_matmul(self.lora_dropout(lora_input), lora_a)

        if self.sequence_parallel or self.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)
            lora_parallel = self.copy_to_mp_region(lora_parallel)

        origin_dtype = F.dtype(input_parallel)
        if self.skip_weight_param_allocation:
            weight = ops.cast(weight, self.compute_dtype)
        else:
            weight = ops.cast(self.weight, self.compute_dtype)
        input_parallel = ops.cast(input_parallel, self.compute_dtype)

        lora_b = ops.cast(self.lora_b, self.compute_dtype)
        scaling = ops.cast(self.scaling, self.compute_dtype)

        if self.sequence_parallel:
            input_parallel = self.gather_from_sp_region(input_parallel)
            lora_parallel = self.gather_from_sp_region(lora_parallel)

        output_parallel = self.matmul(input_parallel, weight)
        if self.has_bias and not self.skip_bias_add:
            output_parallel = mint.add(
                output_parallel, ops.cast(self.bias, self.compute_dtype)
            )

        lora_output = self.lora_b_matmul(lora_parallel, lora_b)
        lora_output = lora_output * scaling
        output_parallel = output_parallel + lora_output
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
        lora_a_shard = (1, 1)
        lora_b_shard = (tp_size, 1) if self.transpose_b else (1, tp_size)
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
            state_dict[self.lora_a.name] = {
                'shape': self.lora_a.shape,
                'shard': lora_a_shard,
                'opt_weight_shard_step': opt_weight_shard_step,
                'opt_weight_shard_size': opt_weight_shard_size
            }
            state_dict[self.lora_b.name] = {
                'shape': self.lora_b.shape,
                'shard': lora_b_shard,
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


class RowParallelLoRA(nn.Cell):
    r"""
    The LoRA layer with weight sliced on first dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias} + \text{inputs} * \text{lora_a} * \text{lora_b},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True),
    :math:`\text{lora_a}` is the lora_a matrix with shape (input_size, rank),
    and :math:`\text{lora_b}` is the lora_b matrix with shape (rank, output_size) .

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Parallel configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias (bool): Specifies whether the layer uses a bias vector.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
        skip_bias_add (bool): If True, do not add the biad term, instead return it for fusion.
        stride (int): For the strided linear layers. Default: 1.
        keep_master_weight_for_test (bool): For testing and should be set to False. It returns the master weights used
            for initialization. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        tp_comm_buffer_name (str): Communication buffer name is not used in non-Transformer-Engine modules.
            Default: None.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: 'zeros'.
        param_init_dtype (dtype.Number): The parameter initialization type. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: None.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
            Default: True.
        lora_rank (int): The value of rank. Default: 8.
        lora_alpha (int):  The alpha value for lora. Default: 32.
        lora_dropout (float):  The dropout rate for lora. Default: 0.0.
        lora_a_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_a weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'normal'.
        lora_b_init (Union[Tensor, str, Initializer, numbers.Number]):  The trainable lora_b weight_init parameter.
            The values of str refer to the function `initializer`. Default: 'zeros'.

    Inputs:
        - **input_** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        - **output** (Tensor): Result of linear with shape :math:`(*, output\_size)`.
        - **output_bias** (Parameter): Bias parameter when `skip_bias_add=True` with shape :math:`(output\_size)`.

    Supported Platforms:
        ``Ascend``
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
            skip_bias_add,
            stride=1,
            keep_master_weight_for_test=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            bias_init='zeros',
            param_init_dtype=None,
            compute_dtype=None,
            transpose_b=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.0,
            lora_a_init='normal',
            lora_b_init='zeros',
        ):
        super(RowParallelLoRA, self).__init__()
        if skip_bias_add:
            raise NotImplementedError("`skip_bias_add=True` is not supported for now.")
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

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = mint.nn.Dropout(p=lora_dropout)
        self.scaling = Tensor(self.lora_alpha / self.lora_rank)

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
            except AssertionError as ex:
                raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                                   "need be initialized. Please check 'dp' in order when calling "
                                   "initialize_model_parallel.") from ex

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
                "To enable `sequence_arallel`, `input_is_parallel` must be `True`"
            )

        mode = EXPERT_PARALLEL_GENERATOR if self.is_expert and self.expert_parallel else TENSOR_PARALLEL_GENERATOR
        weight_shape = (self.output_size, self.input_size_per_partition) if transpose_b \
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
            self.matmul = P.BatchMatMul(transpose_b=self.transpose_b)

            if self.has_bias:
                if self.use_zero3 and not self.transpose_b:
                    self.output_size = divide(self.output_size, dp_size)
                self.bias = Parameter(
                    initializer(bias_init, (self.output_size), self.param_init_dtype), name="bias"
                )

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )

        lora_a_shape = (lora_rank, weight_shape[1]) if self.transpose_b else \
                       (weight_shape[0], lora_rank)
        lora_b_shape = (weight_shape[0], lora_rank) if self.transpose_b else (lora_rank, weight_shape[1])
        self.lora_a_matmul = P.BatchMatMul(transpose_b=self.transpose_b)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=self.transpose_b)

        self.lora_a = Parameter(initializer(lora_a_init, lora_a_shape, self.param_init_dtype))
        self.lora_b = Parameter(initializer(lora_b_init, lora_b_shape, self.param_init_dtype))

        self.scatter_to_mp_region = ScatterToModelParallelRegion()
        self.reduce_scatter_to_sp_region = ReduceScatterToSequenceParallelRegion(
            need_to_swapaxes=self.config.dataset_config.data_layout == "BSH"
        )
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()

    def construct(self, input_):
        """construct method"""
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = ops.cast(self.weight, self.compute_dtype)
        input_parallel = ops.cast(input_parallel, self.compute_dtype)
        lora_a = ops.cast(self.lora_a, self.compute_dtype)
        lora_b = ops.cast(self.lora_b, self.compute_dtype)
        scaling = ops.cast(self.scaling, self.compute_dtype)
        output_parallel = self.matmul(input_parallel, weight)

        input_parallel = self.lora_dropout(input_parallel)
        lora_parallel = self.lora_a_matmul(input_parallel, lora_a)
        if self.explicit_expert_comm:
            pass
        if self.sequence_parallel:
            lora_parallel = self.reduce_scatter_to_sp_region(lora_parallel)
        else:
            lora_parallel = self.reduce_from_mp_region(lora_parallel)

        lora_output = self.lora_b_matmul(lora_parallel, lora_b)
        lora_output = lora_output * scaling

        output_parallel = ops.cast(output_parallel, origin_dtype)
        lora_output = ops.cast(lora_output, origin_dtype)

        if self.explicit_expert_comm:
            output = output_parallel
        elif self.sequence_parallel:
            output = self.reduce_scatter_to_sp_region(output_parallel)
        else:
            output = self.reduce_from_mp_region(output_parallel)
        output = output + lora_output

        if self.has_bias and not self.skip_bias_add:
            output = mint.add(output, ops.cast(self.bias, self.compute_dtype))
        output = ops.cast(output, origin_dtype)
        output_bias = self.bias if self.has_bias and self.skip_bias_add else None

        return output, output_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        tp_size = get_tensor_model_parallel_world_size()
        w_shard = (1, tp_size) if self.transpose_b else (tp_size, 1)
        lora_a_shard = (1, tp_size) if self.transpose_b else (tp_size, 1)
        lora_b_shard = (1, 1)
        state_dict = {}
        opt_weight_shard_step = get_tensor_model_parallel_world_size() if self.use_zero3 else 0
        opt_weight_shard_size = get_data_parallel_world_size() if self.use_zero3 else 0
        state_dict[self.weight.name] = {
            'shape': self.weight.shape,
            'shard': w_shard,
            'opt_weight_shard_step': opt_weight_shard_step,
            'opt_weight_shard_size': opt_weight_shard_size
        }
        state_dict[self.lora_a.name] = {
            'shape': self.lora_a.shape,
            'shard': lora_a_shard,
            'opt_weight_shard_step': opt_weight_shard_step,
            'opt_weight_shard_size': opt_weight_shard_size
        }
        state_dict[self.lora_b.name] = {
            'shape': self.lora_b.shape,
            'shard': lora_b_shard,
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
