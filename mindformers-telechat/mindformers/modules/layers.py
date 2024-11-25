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
"""
The basic layer of the Transformer Networks. This is an experimental interface that is subject to
change or deletion.
"""
from __future__ import absolute_import

from enum import Enum
from functools import wraps, partial
import inspect
import math
import numpy as np

from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
from mindspore.parallel.shard import Layout

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode

from mindformers.tools.logger import logger
from mindformers.modules.activation import get_activation
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, OpParallelConfig, MoEParallelConfig
from mindformers.version_control import check_valid_gmm_op

__all__ = [
    "FixedSparseAttention",
    "Dropout",
    "LayerNorm",
    "Linear",
    "AlibiTensor",
    "AlibiTensorV2",
    "RotaryEmbedding"
]


def _args_type_validator_check(*type_args, **type_kwargs):
    """Check whether input data type is correct."""

    def type_check(func):
        sig = inspect.signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal bound_types
            bound_values = sig.bind(*args, **kwargs)

            argument_dict = bound_values.arguments
            if "kwargs" in bound_types:
                bound_types = bound_types["kwargs"]
            if "kwargs" in argument_dict:
                argument_dict = argument_dict["kwargs"]
            for name, value in argument_dict.items():
                if name in bound_types:
                    bound_types[name](value, name)
            return func(*args, **kwargs)

        return wrapper

    return type_check


def _valid_type_checks(types, class_name):
    # types should be a list of types, this function check if the type is in the valid dtypes
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name, valid_types=types, prim_name=class_name)
        return partial_check(name, type(value))

    return validator_check_func


def _valid_value_checks(types, class_name):
    # the value should be a list of types, this function check if the value is in the valid dtypes
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name, valid_types=types, prim_name=class_name)
        return partial_check(name, value)

    return validator_check_func


class _LayerInputCheck:
    """
       A input check class for the inputs of the transformer model.
    """

    @staticmethod
    def check_shape_length(input_shape, param_name, func_name, target_len):
        """
        Check the input shape's length is equal to the expected shape
        :param input_shape(list): a list of the tensor shapes.
        :param param_name(str): the name of the checked parameter.
        :param func_name(str): the name of the function.
        :param target_len: the expected length of the shape.
        :return:
        """
        if not isinstance(target_len, list):
            target_len = [target_len]
        matched = False
        for item in target_len:
            if len(input_shape) == item:
                matched = True
        if not matched:
            raise ValueError(f"{func_name} {param_name} shape length must be one of {target_len} dimension, "
                             f"but got shape {input_shape}")
        return True

    @staticmethod
    def check_shape_equal(input_shape, param_name, func_name, target_shape):
        """
        Check the input shape's is equal to the expected shape
        :param input_shape(list): a list of the tensor shapes.
        :param param_name(str): the name of the checked parameter.
        :param func_name(str): the name of the function.
        :param target_shape: the expected shape.
        :return:
        """
        if not isinstance(target_shape[0], list):
            target_shape = [target_shape]
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        _LayerInputCheck.check_shape_length(input_shape, param_name, func_name,
                                            [len(item) for item in target_shape])
        matched = False
        for item in target_shape:
            if item == input_shape:
                matched = True
                break

        if not matched:
            raise ValueError(f"{func_name} {param_name} shape must be one of {target_shape},"
                             f"but got {input_shape}")
        return True

    @staticmethod
    def check_shape_value_on_axis(input_shape, dim, param_name, cls_name, target_value):
        """ Check whether the input_shape[dim] is equal to target value"""
        if input_shape[dim] != target_value:
            raise ValueError(f"{cls_name} {param_name} at {dim} shape must be {target_value},"
                             f"but got {input_shape[dim]}")
        return True

    @staticmethod
    def check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape):
        """
        Check the input shape's is equal to the expected shape, the value on 0-th is viewed as batch, and the
        batch size will not be checked.
        """
        target_shape = target_shape
        length, hidden = target_shape
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        _LayerInputCheck.check_shape_length(input_shape, param_name, func_name,
                                            [len(target_shape), len(target_shape) + 1])
        if input_shape[-1] != hidden:
            raise ValueError(f"For {func_name}, the last dimension of {param_name} shape must be {hidden},"
                             f"but got the last dimension {input_shape[-1]} in {input_shape}.")
        if input_shape[0] == 0:
            raise ValueError(f"For {func_name}, the first dimension of {param_name} shape greater than 0,"
                             f"but got the first dimension {input_shape[0]} in {input_shape}.")
        if len(input_shape) == 2 and input_shape[0] % length != 0:
            raise ValueError(f"For {func_name}, the first dimension of {param_name} shape should be divisible "
                             f"by {length}, "
                             f"but got the first dimension {input_shape[0]} in {input_shape}.")
        return True


@constexpr
def _check_past_none_input_none(use_past, param_name, func_name, default_value, is_tensor, is_default):
    """ If the use_past is True, check whether the inputs is None"""
    if not use_past:
        if is_tensor:
            raise TypeError(f"{func_name} {param_name} must be {default_value}, if use_past is False, but found "
                            f"a tensor")
        if not is_default:
            raise TypeError(f"{func_name} {param_name} must be {default_value}, if use_past is False.")
    else:
        if not is_tensor:
            raise TypeError(f"{func_name} {param_name} must be tensor, if use_past is True")
    return True


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    Validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr
def _check_input_shape(input_shape, param_name, func_name, target_len):
    # check the input length
    _LayerInputCheck.check_shape_length(input_shape, param_name, func_name, target_len)


@constexpr
def _check_shape_equal(input_shape, param_name, func_name, target_shape):
    # check the input length
    _LayerInputCheck.check_shape_equal(input_shape, param_name, func_name, target_shape)


@constexpr
def _check_input_shape_value(input_shape, dim, param_name, cls_name, target_value):
    _LayerInputCheck.check_shape_value_on_axis(input_shape, dim, param_name, cls_name, target_value)


@constexpr
def _check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape):
    _LayerInputCheck.check_shape_equal_without_batch(input_shape, param_name, func_name, target_shape)


class Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout(x)
        return out

    def extend_repr(self):
        return 'keep_prob={}'.format(self.keep_prob)

    def shard(self, strategy):
        self.dropout.shard(strategy)


class LayerNorm(Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32, is_self_defined=False):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16], "
                            "but got the type : {}.".format(type(param_init_type)))
        # Since the mindspore 1.10 version, the layernorm has been changed to P.LayerNorm
        self.is_self_defined = is_self_defined
        if not self.is_self_defined:
            self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                          begin_params_axis=-1,
                                          epsilon=eps)
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.Add()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        if self.is_self_defined:
            mean = self.mean(x, -1)
            diff = self.sub1(x, mean)
            variance = self.mean(self.square(diff), -1)
            variance_eps = self.sqrt(self.add(variance, self.eps))
            output = self.real_div(diff, variance_eps)
            output = self.add2(self.mul(output, self.gamma), self.beta)
        else:
            output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output

    def shard(self, strategy):
        r"""
        Set the shard for the layer norm. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy (tuple): The strategy for the dropout. Should be the same shape as the inputs.
        Examples:
            >>> import mindspore
            >>> net = mindformers.modules.transformer.LayerNorm(normalized_shape=(1024, 10))
            >>> net.shard(((10, 2, 1),))
        """
        if self.is_self_defined:
            self.mean.shard(strategy)
            self.square.shard(strategy)
            self.sqrt.shard(strategy)
            self.sub1.shard((strategy[0], strategy[0]))
            self.sub2.shard((strategy[0], strategy[0]))
            self.add.shard((strategy[0], ()))
            self.mul.shard((strategy[0], (1,)))
            self.add2.shard((strategy[0], (1,)))
            self.real_div.shard((strategy[0], strategy[0]))
        else:
            self.layer_norm.shard((strategy[0], (1,), (1,)))

        return self


class Linear(Cell):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'. Default: None.
        expert_num (int): The number of experts used in this Linear. Here, for the case expert_num > 1, BatchMatMul is
            used and the first dimension in BatchMatMul indicate expert_num. Default: 1.
        outer_batch (int): The replication number of experts. The replication is effective only when MoE is applied.
            Default: 1.
        expert_group_size (int): The number of tokens in each data parallel group. Default: None.
        use_gmm (bool): Implemented GroupedMatmul or not, Default: False.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float16
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    @cell_attr_register
    @_args_type_validator_check(in_channels=Validator.check_positive_int,
                                out_channels=Validator.check_positive_int,
                                has_bias=Validator.check_bool,
                                transpose_b=Validator.check_bool,
                                expert_num=Validator.check_positive_int,
                                outer_batch=Validator.check_positive_int,
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "Linear"),
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "Linear"))
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 expert_group_size=None,
                 use_gmm=False,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 skip_redistribution=False):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not (isinstance(activation, str) or activation is None or issubclass(activation, nn.Cell)):
            raise TypeError(f"For Linear cell, the activation should str type or nn.Cell type, but got {activation}.")

        transpose_b = False if use_gmm else transpose_b
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != weight_shape[0] or
                                                weight_init.shape[1] != weight_shape[1]):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")

        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.expert_group_size = expert_group_size
        self.use_gmm = use_gmm
        self.transpose_b = transpose_b
        if self.expert_num > 1:
            self.expert_flag = True
            self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                    name="weight")
            if self.use_gmm and check_valid_gmm_op():
                from mindspore.ops.auto_generate import GroupedMatmul
                # split_item only supports 0 and 3 now, 0 means the size of tensorlist not equal to 1,
                # 3 means the size of tensorlist is 1.
                # group_type only supports -1 and 0 now, -1 means ungrouped and 0 means split x-axis.
                self.matmul = GroupedMatmul(split_item=3, group_type=0)
            else:
                self.matmul = P.BatchMatMul(transpose_b=transpose_b)
        else:
            self.expert_flag = False
            self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
            self.matmul = P.MatMul(transpose_b=transpose_b)
        self.use_expert_group_size = _get_parallel_mode() not in (ParallelMode.STAND_ALONE,
                                                                  ParallelMode.AUTO_PARALLEL,
                                                                  ParallelMode.SEMI_AUTO_PARALLEL) \
                                     and self.expert_flag is True
        if self.use_expert_group_size is True and self.expert_group_size is None:
            raise ValueError("'expert_group_size' should be configured as an integer in MoEConfig.")
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            if self.expert_flag:
                self.bias = Parameter(initializer(bias_init,
                                                  [1, self.expert_num, 1, out_channels], param_init_type), name="bias")
            else:
                self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = P.Add()
        self.act_name = activation
        if callable(activation):
            self.activation = activation()
        else:
            self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        if skip_redistribution:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.shape = P.Shape()

    def construct(self, x, group_list=None):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        # apply gmm to the inference of moe structural models when use_past=True.
        if self.use_gmm:
            x = self.matmul([x], [weight], None, None, None, None, None, group_list)[0]
        else:
            x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(x)
        x = F.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None, out_strategy_matmul=None):
        r"""
        Set the shard for the linear. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy_matmul (tuple): The strategy for the matmul. Should be the same shape as the inputs.
            strategy_bias (tuple): The strategy for the bias_add. Should be the same shape as the inputs.
            strategy_activation (tuple): The strategy for the strategy_activation. Should be the same shape as
            the inputs.
            out_strategy_matmul (tuple): The out strategy for the matmul. Should be the same shape as the inputs.
        """
        self.matmul.shard(in_strategy=strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag and isinstance(self.act_name, str):
            # some operations has many primitives, need to manually set the shard
            if self.act_name.lower() == "leakyrelu":
                self.activation.select_op.shard((strategy_activation[0], strategy_activation[0]))
            elif self.act_name.lower() == "logsigmoid":
                self.activation.mul.shard((strategy_activation[0], ()))
                self.activation.exp.shard(strategy_activation)
                self.activation.add.shard((strategy_activation[0], ()))
                self.activation.rec.shard(strategy_activation)
                self.activation.log.shard(strategy_activation)
            elif self.act_name.lower() == "logsoftmax":
                raise ValueError("The 'LogSoftmax' function is not supported in semi auto parallel "
                                 "or auto parallel mode.")
            else:
                getattr(self.activation, self.act_name).shard(strategy_activation)
        elif self.activation_flag and isinstance(self.activation, Cell):
            if hasattr(self.activation, 'activation_shard') and strategy_activation:
                shard_tuple = strategy_activation[0]
                if len(shard_tuple) == 2:
                    parallel_config = OpParallelConfig(data_parallel=shard_tuple[0],
                                                       model_parallel=shard_tuple[1])
                elif len(shard_tuple) == 4:
                    parallel_config = MoEParallelConfig(data_parallel=shard_tuple[0],
                                                        expert_parallel=shard_tuple[1],
                                                        model_parallel=shard_tuple[2])
                else:
                    raise ValueError("The user-defined activation function currently only supports the case where the "
                                     "input policy is 2 or 4, so that relevant policies can be extracted from it."
                                     "To avoid this error, you need to add the function of extracting "
                                     "'ParallelConfig' or 'OpParallelConfig' for the incoming strategy_activation ")
                self.activation.activation_shard(parallel_config, self.use_gmm)
            else:
                logger.warning("The user passed the custom defined activation function %s. "
                               "If the user want to enable shard for the activation cell, "
                               "the user should set the shard for each primitives in the cell.", self.activation_flag)
        return self


class FixedSparseAttention(nn.Cell):
    """
    Fixed Sparse Attention Layer.

    This function contains the sparse attention primitives used in Sparse Transformers (see paper)
    `Generating Long Sequences with Sparse Transformers <https://arxiv.org/abs/1904.10509>`_.

    Specifically, it includes the following:

    1. A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused).
    2. An implementation of "strided" and "fixed" attention, as in the Sparse Transformers paper.

    Args:
        batch_size(int): Number of input batch size.
        num_heads(int): Number of attention heads.
        size_per_head(int): An integer determining embedding size of each attention head,
            only supports 64, 128 for now.
        block_size(int): An integer determining the block size. Current implementation of sparse self-attention
            is based on blocked sparse matrices. In which this parameter defines the size of such blocks,
            Block X Block. Only supports 64 for now.
        seq_length(int): length of input sequence, only supports 1024 for now. Default 1024.
        num_different_global_patterns(int): An integer determining the number of different global attentions layouts.
            While global attention can be fixed by which block/s are representative of
            any local window, since there are multi-heads, each head can use a
            different global representative, only supports 4 for now. Default 4.
        parallel_config(OpParallelConfig): The config of parallel setting, see `OpParallelConfig`.
            Default `default_dpmp_config`, an instance of `OpParallelConfig` with
            default args.

    Inputs:
        - **q** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, seq_length, hidden_size]): Sequence of
          queries to query the context.
        - **k** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, seq_length, hidden_size]): Sequence of
          queries to query the context.
        - **v** (Tensor) - Tensor value (:class:`mstype.fp16` [batch size, sequence length, Embedding Size]):
          Sequence of queries to query the context.
        - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp32`, :class:`mstype.fp16`
          [batch_size, seq_length, seq_length]): Lower triangular matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, seq_length, hidden_size]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindformers.modules import FixedSparseAttention
        >>> from mindspore import Tensor
        >>> model = FixedSparseAttention(batch_size=2,
        ...                              num_heads=8,
        ...                              size_per_head=64,
        ...                              block_size=64)
        >>> q = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> k = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> v = Tensor(np.ones((2, 1024, 8*64)), mstype.float16)
        >>> attention_mask = Tensor(np.ones((2, 1024, 1024)), mstype.float32)
        >>> output = model(q, k, v, attention_mask)
        >>> print(output.shape)
        (2, 1024, 512)
    """

    @_args_type_validator_check(batch_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                size_per_head=Validator.check_positive_int,
                                block_size=Validator.check_positive_int,
                                seq_length=Validator.check_positive_int,
                                num_different_global_patterns=Validator.check_positive_int,
                                parallel_config=_valid_type_checks([OpParallelConfig], "FixedSparseAttention"))
    def __init__(self,
                 batch_size,
                 num_heads,
                 size_per_head,
                 block_size,
                 seq_length=1024,
                 num_different_global_patterns=4,
                 parallel_config=default_dpmp_config):
        super(FixedSparseAttention, self).__init__()
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        if num_heads % mp != 0:
            raise ValueError(f"The number of heads {num_heads} must be a "
                             f"multiple of parallel_config.model_parallel {mp}.")
        if batch_size % dp != 0:
            raise ValueError(f"The batch_size {batch_size} must be a "
                             f"multiple of parallel_config.data_parallel {parallel_config.data_parallel}.")
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = size_per_head * num_heads
        self.num_heads = num_heads
        self.block_size = block_size
        self.block_num = seq_length // block_size
        self.size_per_head = size_per_head
        self.global_size = seq_length // 4
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, mp, 1),))
        self.batch_matmul = P.BatchMatMul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.multiply = P.Mul().shard(((dp, 1, 1, 1), (1, 1, 1)))
        self.multiply_data = Tensor([-10000.0], dtype=mstype.float32)
        self.parallel_config = parallel_config
        size_per_head_list = [64, 128]
        if self.seq_length != 1024:
            raise ValueError("For 'FixedSparseAttention', the class variable 'seq_length' must be 1024, "
                             "but got the value : {}.".format(seq_length))
        if self.block_size != 64:
            raise ValueError("For 'FixedSparseAttention', the class variable 'block_size' must be 64, "
                             "but got the value : {}.".format(block_size))
        if num_different_global_patterns != 4:
            raise ValueError("For 'FixedSparseAttention', the class variable 'num_different_global_patterns' "
                             "must be 4, but got the value : {}".format(num_different_global_patterns))
        if self.size_per_head not in size_per_head_list:
            raise ValueError("For 'FixedSparseAttention', the class variable 'size_per_head' only supports {}, "
                             "but got the value : {}.".format(size_per_head_list, self.size_per_head))
        local_ones = np.ones((self.block_size, self.block_size),
                             dtype=np.float16)
        global_mask_original = np.ones((self.seq_length, self.global_size), dtype=np.float16)
        for i in range(self.seq_length):
            for j in range(self.global_size):
                if i // 16 >= (j // 16 + 1) * 4:
                    global_mask_original[i, j] = 0.0

        global_mask_original = -10000 * global_mask_original
        global_mask_fx = global_mask_original.reshape((self.seq_length // 16, 16, self.global_size // 16, 16))
        global_mask = np.transpose(global_mask_fx, (2, 0, 1, 3))
        global_mask = np.repeat(global_mask[np.newaxis, :, :, :, :], self.batch_size, axis=0)
        global_mask = global_mask.reshape((self.batch_size * self.global_size // 16, self.seq_length // 16, 16, 16))
        self.global_mask = Tensor(global_mask, mstype.float32)
        self.local_mask_triangle = Tensor(np.tril(local_ones), mstype.float32)
        self.scale_factor = Tensor((math.sqrt(self.size_per_head)))
        self.matmul_dds = P.MatmulDDS(self.batch_size, self.num_heads).shard(((mp, dp, 1, 1),
                                                                              (mp, dp, 1, 1),
                                                                              (1, dp, 1, 1),
                                                                              (dp, 1, 1, 1)))
        self.matmul_dsd = P.DSDMatmul().shard(((dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1, 1, 1, 1), (dp, mp, 1, 1)))
        self.sub1 = P.Sub().shard(((1,), (dp, 1, 1, 1)))
        self.mul1 = P.Mul().shard(((dp, 1, 1, 1), (1,)))
        self.transpose1 = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose2 = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose3 = P.Transpose().shard(((dp, mp, 1, 1, 1, 1),))
        self.transpose4 = P.Transpose().shard(((dp, mp, 1, 1),))
        self.div = P.RealDiv().shard(((mp, dp, 1, 1), ()))
        self.slice1 = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, q, k, v, attention_mask):
        """Forward process"""
        _check_shape_equal(F.shape(q), "q", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(q), "q", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(k), "k", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(k), "k", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(v), "v", self.cls_name,
                           [self.batch_size, self.seq_length, self.hidden_size])
        _check_input_dtype(F.dtype(v), "v", [mstype.float16], self.cls_name)
        _check_shape_equal(F.shape(attention_mask), "attention_mask", self.cls_name,
                           [self.batch_size, self.seq_length, self.seq_length])
        _check_input_dtype(F.dtype(attention_mask), "attention_mask",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)

        q, k, v = self._transpose_inputs(q, k, v)
        local_mask, global_mask = self._generate_attention_mask(attention_mask)
        q = self.div(q, F.cast(self.scale_factor, F.dtype(q)))
        k = self.div(k, F.cast(self.scale_factor, F.dtype(k)))
        local_prob, global_prob = self.matmul_dds(q, k, local_mask, global_mask)
        attention = self.matmul_dsd(local_prob, global_prob, v)
        attention_merge = self.transpose3(attention, (0, 1, 3, 4, 2, 5))
        attention_merge = F.reshape(
            attention_merge,
            (-1, self.num_heads, self.seq_length, self.size_per_head))
        attention_merge = self.transpose4(attention_merge, (0, 2, 1, 3))
        attention_merge = F.reshape(
            attention_merge,
            (-1, self.seq_length, self.size_per_head * self.num_heads))

        return attention_merge

    def _generate_attention_mask(self, attention_mask):
        """
        generate global attention mask and local attention mask from origin attention mask
        """
        attention_mask = self.reshape(attention_mask, (-1, self.seq_length, self.seq_length))
        input_mask = self.slice1(attention_mask, (0, self.seq_length - 1, 0),
                                 (self.batch_size, self.seq_length, self.seq_length), (1, 1, 1))
        input_mask = self.reshape(input_mask, (-1, self.seq_length))
        input_shape = P.Shape()(input_mask)  # bs, seq_length
        # bs, block_num, 1, block_size
        local_shape_right = (input_shape[0], self.block_num, 1, self.block_size)
        # bs, block_num, block_size, 1
        local_shape_left = (input_shape[0], self.block_num, self.block_size, 1)
        local_mask_left = self.reshape(input_mask, local_shape_left)
        local_mask_right = self.reshape(input_mask, local_shape_right)
        # bs, block_num, block_size, block_size
        local_attention_mask = self.batch_matmul(local_mask_left, local_mask_right)
        lower_triangle = P.ExpandDims()(self.local_mask_triangle, 0)
        local_attention_mask = self.multiply(local_attention_mask, lower_triangle)
        local_multiplied_out = self.sub1(P.Cast()(F.tuple_to_array((1.0,)), mstype.float32),
                                         P.Cast()(local_attention_mask, mstype.float32))
        local_adder = self.mul1(local_multiplied_out, self.multiply_data)
        local_mask_original = self.transpose1(local_adder, (0, 2, 1, 3))
        local_mask_original = self.reshape(
            local_mask_original,
            (self.batch_size * self.block_size, self.block_num * self.block_size))
        local_mask_fx = self.reshape(
            local_mask_original,
            (self.batch_size * self.block_size // 16, 16,
             self.block_num * self.block_size // 16, 16))
        local_mask = self.transpose2(local_mask_fx, (2, 0, 1, 3))
        global_mask = self.global_mask

        return local_mask, global_mask

    def _transpose_inputs(self, q, k, v):
        """
        do reshape and transpose to inputs
        """
        q = self.transpose(
            self.reshape(
                q,
                (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (2, 0, 1, 3))
        k = self.transpose(
            self.reshape(
                k, (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (2, 0, 1, 3))
        v = self.transpose(
            self.reshape(
                v,
                (-1, 16, self.num_heads * self.size_per_head // 16, 16)),
            (0, 2, 3, 1))

        return q, k, v


class AlibiTensor(nn.Cell):
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    Args:
        seq_length(int) - length of sequence
        num_heads(int) - number of heads

    Inputs:
        attention_mask(Tensor) - Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        dtype(mstype) - dtype of the output tensor

    Returns:
        alibi(Tensor), ailibi tensor shaped (batch_size, num_heads, 1, max_seq_len)
    """

    def __init__(self, seq_length, num_heads, parallel_config=default_dpmp_config):
        super(AlibiTensor, self).__init__()
        dp = parallel_config.data_parallel

        self.seq_length = seq_length
        self.num_heads = num_heads
        self.minus_one = Tensor(-np.ones(seq_length), mstype.float32)

        self.expand_2d = P.ExpandDims().shard(((dp, 1),))
        self.expand_3d = P.ExpandDims().shard(((dp, 1, 1),))
        self.cumsum = P.CumSum().shard(((dp, 1),))
        self.add = P.Add().shard(((dp, 1), (1,)))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul_slope = P.Mul().shard(((1, 1), (dp, 1, 1)))

        # build slopes
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = np.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=np.float32)
        powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
        slopes = np.power(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = np.array(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=np.float32
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=np.int32)
            slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

        self.slopes = Tensor(slopes[:, None], mstype.float32)  # (num_heads, 1)

    def construct(self, attention_mask, dtype):
        """
        Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        """
        arange_tensor = self.cumsum(attention_mask, -1)  # (batch_size, seq_len)
        arange_tensor = self.add(arange_tensor, self.minus_one)  # (batch_size, seq_len)
        arange_tensor = self.mul(arange_tensor, attention_mask)  # (batch_size, seq_len)
        arange_tensor = self.expand_2d(arange_tensor, 1)  # (batch_size, 1, seq_len)
        alibi = self.mul_slope(self.slopes, arange_tensor)  # (batch_size, num_heads, seq_len)
        alibi = self.expand_3d(alibi, 2).astype(dtype)  # (batch_size, num_heads, 1, seq_len)
        return alibi


class AlibiTensorV2(nn.Cell):
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    Args:
        seq_length(int) - length of sequence
        num_heads(int) - number of heads

    Inputs:
        attention_mask(Tensor) - Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        dtype(mstype) - dtype of the output tensor

    Returns:
        alibi(Tensor), ailibi tensor shaped (batch_size, num_heads, 1, max_seq_len)
    """

    def __init__(self, num_heads):
        super(AlibiTensorV2, self).__init__()
        self.num_heads = num_heads

        self.expand_2d = P.ExpandDims()
        self.expand_3d = P.ExpandDims()
        self.cumsum = P.CumSum()
        self.add_2d = P.Add()
        self.add_3d = P.Add()
        self.mul = P.Mul()
        self.mul_slope = P.Mul()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.mul_mask = P.Mul()

        # build slopes
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = np.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=np.float32)
        powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
        slopes = np.power(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = np.array(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=np.float32
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=np.int32)
            slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

        self.slopes = Tensor(slopes[None, :, None, None], mstype.float32)  # (num_heads, 1)

    def construct(self, attention_mask, dtype=mstype.float32):
        """
        Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        """
        bs, seqlen = attention_mask.shape
        arange_tensor = self.cumsum(attention_mask, -1)  # (batch_size, seq_len)
        max_pos = -arange_tensor[:, -1:]  # (batch_size, 1)
        arange_tensor = self.add_2d(arange_tensor, max_pos)  # (batch_size, seq_len)
        arange_tensor = self.expand_2d(arange_tensor, 1)  # (batch_size, 1, seq_len)
        diag = -self.transpose(arange_tensor, (0, 2, 1))  # (batch_size, seq_len, 1)
        arange_tensor = self.add_3d(arange_tensor, diag)  # (batch_size, seq_len, seq_len)
        arange_tensor = self.expand_3d(arange_tensor, 1)  # (batch_size, 1, seq_len, seq_len)
        alibi = self.mul_slope(self.slopes, arange_tensor)  # (batch_size, num_heads, seq_len, seq_len)
        alibi_mask = self.mul_mask(alibi, self.reshape(attention_mask, (bs, 1, seqlen, 1)))  # (batch_size, num_heads, seq_len, seq_len)
        alibi_mask = self.mul_mask(alibi_mask, self.reshape(attention_mask, (bs, 1, 1, seqlen)))  # (batch_size, num_heads, seq_len, seq_len)
        return alibi_mask.astype(dtype)

    def shard(self, parallel_config):
        """Parallel strategy configuratiuon interface."""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.expand_2d.shard(((dp, 1),))
        self.expand_3d.shard(((dp, 1, 1),))
        self.cumsum.shard(((dp, 1),))
        self.add_2d.shard(((dp, 1), (dp, 1)))
        self.add_3d.shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul.shard(((dp, 1), (dp, 1)))
        self.mul_slope.shard(((1, 1, 1, 1), (dp, 1, 1, 1)))
        self.transpose.shard(((dp, 1, 1),))
        self.mul_mask.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))


def _get_interleave(n):
    """calculate slopes of alibi tensor"""

    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)

    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return _get_interleave_power_of_2(closest_power_of_2) + \
        _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


def build_alibi_tensor_v2(seq_len, num_heads, return_tensors='ms', dtype=mstype.float32):
    """build alibi tensor"""
    if return_tensors not in ['np', 'ms']:
        raise ValueError(f"return tensors must be 'np' or 'ms', {return_tensors} not support.")
    slopes = _get_interleave(num_heads)
    slopes = np.expand_dims(np.expand_dims(slopes, 1), 1)
    position_point = np.arange(seq_len) - seq_len + 1
    position_point = np.expand_dims(np.expand_dims(position_point, 0), 0)
    position_point = np.tile(position_point, (num_heads, seq_len, 1))
    diag = np.diag(position_point[0])
    diag = np.expand_dims(np.expand_dims(diag, 0), 0)
    diag = np.transpose(diag, (0, 2, 1))
    position_point = position_point - diag
    alibi = slopes * position_point
    alibi = np.expand_dims(alibi, 0)
    if return_tensors == 'np':
        return alibi
    return Tensor(alibi, dtype=dtype)


def _check_llama3_scaling_factor(scaling_factor, max_position_embedding):
    """check llama3 scaling factor"""
    if scaling_factor is None or not isinstance(scaling_factor, dict):
        raise ValueError(f"`scaling_factor` must be a dict for {SeqExtendMethod.LLAMA3.value} rope extend method,"
                         f" but got {scaling_factor}")

    required_keys = {"factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(scaling_factor.keys())

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_factor` for 'extend_method' LLAMA3': {missing_keys}")
    unused_keys = received_keys - received_keys
    if unused_keys:
        raise KeyError(f"Unrecognized keys in `scaling_factor` for 'extend_method' LLAMA3': {unused_keys}")

    factor = scaling_factor["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = scaling_factor["low_freq_factor"]
    high_freq_factor = scaling_factor["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        raise ValueError(f"`scaling_factor`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        raise ValueError(f"`scaling_factor`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor < low_freq_factor:
        raise ValueError(
            "`scaling_factor`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= max_position_embedding:
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={max_position_embedding}"
        )


def _check_yarn_scaling_factor(scaling_factor, max_position_embedding):
    """check YARN scaling factor"""
    if scaling_factor is None or not isinstance(scaling_factor, dict):
        raise ValueError(f"`scaling_factor` must be a dict for {SeqExtendMethod.YARN.value} rope extend method,"
                         f" but got {scaling_factor}")

    required_keys = {"factor", "original_max_position_embeddings", "beta_slow", "beta_fast",
                     "mscale", "mscale_all_dim"}
    received_keys = set(scaling_factor.keys())

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_factor` for 'extend_method' YARN': {missing_keys}")
    unused_keys = received_keys - received_keys
    if unused_keys:
        raise KeyError(f"Unrecognized keys in `scaling_factor` for 'extend_method' YARN': {unused_keys}")

    factor = scaling_factor["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

    beta_slow = scaling_factor["beta_slow"]
    beta_fast = scaling_factor["beta_fast"]
    if beta_slow is None or not isinstance(beta_slow, float):
        raise ValueError(f"`scaling_factor`'s beta_slow field must be a float, got {beta_slow}")
    if beta_fast is None or not isinstance(beta_fast, float):
        raise ValueError(f"`scaling_factor`'s beta_fast field must be a float, got {beta_fast}")
    if beta_fast < beta_slow:
        raise ValueError(
            "`scaling_factor`'s beta_fast field must be greater than beta_slow, got beta_fast="
            f"{beta_fast} and beta_slow={beta_slow}"
        )

    original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= max_position_embedding:
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={max_position_embedding}"
        )


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Inverse dim formula to find dim based on number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dim range bounds based on rotations"""
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=np.float32) - min_) / (max_ - min_)
    ramp_func = np.clip(linear_func, 0, 1, out=None)
    return ramp_func


class SeqExtendMethod(Enum):
    """Stores the acceptable string identifiers for seq length extend method"""
    PI = "PI"
    NTK = "NTK"
    YARN = "YARN"
    NONE = "None"
    LLAMA3 = "LLAMA3"


class FreqsMgr(Cell):
    r"""freqs_cis manager."""

    def __init__(self,
                 head_dim,
                 seq_length=None,
                 max_position_embedding=4096,
                 rotary_dtype=mstype.float16,
                 theta=10000,
                 scaling_factor=1.0,
                 extend_method=SeqExtendMethod.NONE.value,
                 parallel_config=None):
        super().__init__()
        if seq_length is not None and seq_length > max_position_embedding:
            max_position_embedding = seq_length
        if extend_method == SeqExtendMethod.NTK.value:
            theta *= scaling_factor
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (theta ** (freqs_base / head_dim))  # (head_dim // 2, )
        mscale = 1.0

        if extend_method == SeqExtendMethod.YARN.value:
            _check_yarn_scaling_factor(scaling_factor, max_position_embedding)
            factor = scaling_factor["factor"]
            beta_fast = scaling_factor["beta_fast"]
            beta_slow = scaling_factor["beta_slow"]
            base = theta
            original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
            mscale_all_dim = scaling_factor["mscale_all_dim"]
            mscale_ = scaling_factor["mscale"]

            internal_freq_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)
            internal_freq = 1.0 / (factor * theta ** (internal_freq_base / head_dim))

            extra_freq_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)
            extra_freq = 1.0 / (theta ** (extra_freq_base / head_dim))

            low, high = _yarn_find_correction_range(beta_fast, beta_slow, head_dim, base,
                                                    original_max_position_embeddings)
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, head_dim // 2)
            freqs = internal_freq * (1 - inv_freq_mask) + extra_freq * inv_freq_mask
            mscale = float(_yarn_get_mscale(factor, mscale_)
                           / _yarn_get_mscale(factor, mscale_all_dim))

        if extend_method == SeqExtendMethod.LLAMA3.value:
            _check_llama3_scaling_factor(scaling_factor, max_position_embedding)

            factor = scaling_factor["factor"]
            if factor is None or not isinstance(factor, float) or factor < 1.0:
                raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

            factor = scaling_factor["factor"]
            low_freq_factor = scaling_factor["low_freq_factor"]
            high_freq_factor = scaling_factor["high_freq_factor"]
            old_context_len = scaling_factor["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            new_freqs = []
            for freq in freqs:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    if low_freq_wavelen == high_freq_wavelen:
                        raise ValueError(f"low_freq_wavelen should not equal high_freq_wavelen, "
                                         f"but low_freq_wavelen got {low_freq_wavelen},"
                                         f"high_freq_wavelen got {high_freq_wavelen}.")
                    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = np.array(new_freqs, dtype=freqs.dtype)

        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, max_position_embedding / scaling_factor, 1 / scaling_factor).astype(np.float32)
        else:
            t = np.arange(0, max_position_embedding, 1).astype(np.float32)

        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * mscale # (seq_len, head_dim)
        swap_mask = FreqsMgr.get_swap_mask(head_dim)

        if parallel_config is not None and parallel_config.context_parallel > 1:
            self.context_parallel = parallel_config.context_parallel
        else:
            self.context_parallel = 1
        self.head_dim = head_dim
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.swap_mask = Tensor(swap_mask, dtype=rotary_dtype)

        self.reshape = P.Reshape()
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.tile = P.Tile().shard(((1, 1),))

    def construct(self, seq_length):
        freqs_cos = self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim), (1, 1))
        freqs_sin = self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim), (1, 1))
        if self.context_parallel > 1:
            freqs_cos = self.reshape(freqs_cos, (1, 1, seq_length, self.head_dim))
            freqs_sin = self.reshape(freqs_sin, (1, 1, seq_length, self.head_dim))
        return freqs_cos, freqs_sin, self.swap_mask

    def prefill(self, bs, seq_length):
        freqs_cos = self.tile(self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim), (1, 1)), (bs, 1))
        freqs_sin = self.tile(self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim), (1, 1)), (bs, 1))
        return freqs_cos, freqs_sin, self.swap_mask

    def prefill_flatten(self):
        return self.freqs_cos, self.freqs_sin, self.swap_mask

    def increment(self, batch_valid_length):
        indices = batch_valid_length - 1
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    def increment_multi_ids(self, indices):
        indices = indices.reshape(-1)
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask(head_dim):
        """Swap matrix"""
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])


class RotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **use_rope_slice** (dict): - Choose using rope slice. Default False.

    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32, use_rope_slice=False):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.dtype = compute_dtype
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True
        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()
        self.mul_inc = P.Mul()
        self.neg = P.Neg()
        self.slice = P.StridedSlice()
        self.concat = P.Concat(axis=-1)
        self.shape = P.Shape()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        x = self.bmm_swap(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = self.shape(x)
        x1 = self.slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = self.slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = self.concat((self.neg(x2), x1))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis

        mul = self.mul if self.is_first_iteration else self.mul_inc
        if self.use_rope_slice:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.slice_half(xq), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.rotate_half(xq, swap_mask), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.rotate_half(xk, swap_mask), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out

    def shard(self, parallel_config):
        """sharding for rotary embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        strategy_in = (dp, mp, 1, 1)
        if cp > 1:
            layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
            layout_add = (layout("dp", "mp", "cp", "None"), layout("dp", "mp", "cp", "None"))
            layout_bmm_swap = (layout("dp", "mp", "cp", "None"), layout("None", "None"))
            layout_mul = (layout("dp", "mp", "cp", "None"), layout("None", "None", "cp", "None"))
            self.add.shard(in_strategy=layout_add)
            self.bmm_swap.shard(in_strategy=layout_bmm_swap)
            self.mul.shard(in_strategy=layout_mul)
        else:
            self.add.shard((strategy_in, strategy_in))
            self.bmm_swap.shard((strategy_in, (1, 1)))
            self.mul.shard((strategy_in, (1, 1)))
        self.mul_inc.shard((strategy_in, (strategy_in[0], 1, 1, 1)))
        self.neg.shard((strategy_in,))
        self.slice.shard((strategy_in,))
        self.concat.shard((strategy_in, strategy_in))
