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
# ======================
"""mapping"""

import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_tensor_and_expert_parallel_group,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def gather_along_first_dim_expert_parallel(input_):
    world_size = get_expert_model_parallel_world_size()
    if world_size == 1:
        return input_
    group = get_expert_model_parallel_group()
    output = comm_func.all_gather_into_tensor(input_.contiguous(), group=group)[0]
    return output


# pylint: disable=W0622, C0111
def all_to_all_sp2hp(input):
    world_size = get_tensor_model_parallel_world_size()
    input = input.reshape(-1, input.shape[-1])
    split_tensors = ops.split(input, input.shape[-1] // world_size, axis=1)
    concat_tensor = ops.cat(split_tensors, axis=0)
    if world_size > 1:
        tp_group = get_tensor_model_parallel_group()
        all_to_all = AllToAllEven(tp_group, world_size, 0, 0)
        output = all_to_all(concat_tensor)
    else:
        output = concat_tensor
    return output


class AllToAllSP2HP(nn.Cell):
    """implement of AllToAll sp2hp"""
    def __init__(self):
        super(AllToAllSP2HP, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            tp_group = get_tensor_model_parallel_group()
            self.all_to_all = AllToAllEven(tp_group, self.world_size, 0, 0)

    def construct(self, input_):
        """forward process"""
        input_ = input_.reshape(-1, input_.shape[-1])
        split_tensors = ops.split(input_, input_.shape[-1] // self.world_size, axis=1)
        concat_tensor = ops.cat(split_tensors, axis=0)
        if self.world_size == 1:
            output = concat_tensor
        else:
            output = self.all_to_all(concat_tensor)
        return output


# pylint: disable=W0622
def all_to_all_hp2sp(input):
    world_size = get_tensor_model_parallel_world_size()
    if world_size > 1:
        tp_group = get_tensor_model_parallel_group()
        all_to_all = AllToAllEven(tp_group, world_size, 0, 0)
        input_exchanged = all_to_all(input)
    else:
        input_exchanged = input
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = ops.split(input_reshaped, input_reshaped.shape[0] // world_size, axis=0)
    output = ops.cat(split_tensors, axis=-1)
    return output


class AllToAllHP2SP():
    """implement of AllToAll hp2sp"""
    def __init__(self):
        super(AllToAllHP2SP, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            tp_group = get_tensor_model_parallel_group()
            self.all_to_all = AllToAllEven(tp_group, self.world_size, 0, 0)

    def construct(self, input_):
        """forward process"""
        if self.world_size == 1:
            input_exchanged = input_
        else:
            input_exchanged = self.all_to_all(input_)
        input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
        split_tensors = ops.split(input_reshaped, input_reshaped.shape[0] // self.world_size, axis=0)
        output = ops.cat(split_tensors, axis=-1)
        return output


class CopyToModelParallelRegion(nn.Cell):
    "Pass the input to the model parallel region."

    def __init__(self):
        super(CopyToModelParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()

    # pylint: disable=C0303
    def construct(self, input_):
        return ops.stop_gradient(input_)

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        output = comm_func.all_reduce(dout, group=self.tp_group)[0]
        return (output,)


class ScatterToModelParallelRegion(nn.Cell):
    "Split the input and keep only the corresponding chuck to the rank."

    def __init__(self):
        super(ScatterToModelParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.rank = get_tensor_model_parallel_rank()

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        last_dim = input_.ndim - 1
        last_dim_size = input_.shape[last_dim] // self.world_size
        tensor_tuple = ops.split(input_, last_dim_size, axis=last_dim)
        output = tensor_tuple[self.rank]

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        # Size and dimension.
        last_dim = dout.ndim - 1
        output = comm_func.all_gather_into_tensor(dout, group=self.tp_group)[0]
        tensor_list = ops.split(output, dout.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return (output,)


class GatherFromTensorAndExpertParallelRegion(nn.Cell):
    """Gather From Sequence Parallel Region To MOE"""
    def __init__(self):
        super(GatherFromTensorAndExpertParallelRegion, self).__init__()
        self.world_size = get_tensor_and_expert_parallel_world_size()
        if self.world_size > 1:
            self.tp_ep_group = get_tensor_and_expert_parallel_group()


    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        return comm_func.all_gather_into_tensor(input_.contiguous(), group=self.tp_ep_group)[0]

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        output = comm_func.reduce_scatter_tensor(dout.contiguous(), group=self.tp_ep_group)[0]
        return (output,)


class GatherFromModelParallelRegion(nn.Cell):
    "Gather the input from model parallel region and concatinate."

    def __init__(self):
        super(GatherFromModelParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
            self.rank = get_tensor_model_parallel_rank()

    # pylint: disable=C0111
    def construct(self, input_):
        # Size and dimension.
        last_dim = input_.ndim - 1
        if self.world_size == 1:
            output = input_
        else:
            output = comm_func.all_gather_into_tensor(input_, group=self.tp_group)[0]
        tensor_list = ops.split(output, input_.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        last_dim = dout.ndim - 1
        last_dim_size = divide(dout.shape[last_dim], self.world_size)
        # 对按第零维allgather的结果重新按最后一维排列
        tensor_tuple = ops.split(dout, last_dim_size, axis=last_dim)
        output = tensor_tuple[self.rank].contiguous()
        return (output,)


class ReduceFromModelParallelRegion(nn.Cell):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super(ReduceFromModelParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        output = comm_func.all_reduce(input_, group=self.tp_group)[0]
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        return (dout,)


class ReduceScatterToSequenceParallelRegion(nn.Cell):
    "Reduce scatter the input from the model parallel region."

    def __init__(self, need_to_swapaxes):
        super(ReduceScatterToSequenceParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        self.need_to_swapaxes = need_to_swapaxes
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
        self.used_bprop_inputs = [2]

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        if self.need_to_swapaxes:
            input_ = input_.swapaxes(0, 1)
        output = comm_func.reduce_scatter_tensor(input_, group=self.tp_group)[0]
        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, *args):
        dout = args[-1]
        if self.world_size == 1:
            return dout
        if self.need_to_swapaxes:
            dout = dout.swapaxes(0, 1)
        output = comm_func.all_gather_into_tensor(dout, group=self.tp_group)[0]
        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)

        return (output,)


class ReduceScatterToTensorParallelRegion(nn.Cell):
    "Reduce scatter the input from the model parallel region."

    def __init__(self):
        super(ReduceScatterToTensorParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()

    # pylint: disable=C0111
    def construct(self, input_):
        num_dims = input_.ndim
        permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        input_ = ops.transpose(input_, permute_order).contiguous()
        if self.world_size == 1:
            output = input_
        else:
            output = comm_func.reduce_scatter_tensor(input_, group=self.tp_group)[0]

        permute_order = tuple(range(1, num_dims)) + (0,)
        output = ops.transpose(output, permute_order).contiguous()
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        # Size and dimension.
        last_dim = dout.ndim - 1
        output = comm_func.all_gather_into_tensor(dout, group=self.tp_group)[0]
        tensor_list = ops.split(output, dout.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()

        return (output,)


class ScatterToSequenceParallelRegion(nn.Cell):
    """Scatter To Sequence Paralle lRegion"""
    def __init__(self, need_to_swapaxes):
        super(ScatterToSequenceParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.need_to_swapaxes = need_to_swapaxes
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()

    # pylint: disable=C0111
    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        if self.need_to_swapaxes:
            input_ = input_.swapaxes(0, 1)

        dim_size = input_.shape[0]
        if dim_size % self.world_size != 0:
            raise ValueError("First dimension of the tensor should be divisible by tensor parallel size.")
        local_dim_size = dim_size // self.world_size

        dim_offset = self.rank * local_dim_size
        output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)

        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)
        if self.need_to_swapaxes:
            dout = dout.swapaxes(0, 1)
        output = comm_func.all_gather_into_tensor(dout.contiguous(), group=self.tp_group)[0]
        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)
        return (output,)


class GatherFromSequenceParallelRegion(nn.Cell):
    """Gather From Sequence Parallel Region"""
    def __init__(self, need_to_swapaxes, tensor_parallel_output_grad=True):
        super(GatherFromSequenceParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()
        self.rank = get_tensor_model_parallel_rank()
        self.tensor_parallel_output_grad = tensor_parallel_output_grad
        self.need_to_swapaxes = need_to_swapaxes

    def construct(self, input_):
        """define a forward propagate function"""
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        if self.need_to_swapaxes:
            input_ = input_.swapaxes(0, 1)
        output = comm_func.all_gather_into_tensor(input_, group=self.tp_group)[0]
        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)
        return output

    # pylint: disable=W0613
    def bprop(self, x, out, dout):
        """define a backward propagate function"""
        if self.world_size == 1:
            return (dout,)

        if self.need_to_swapaxes:
            dout = dout.swapaxes(0, 1)

        if self.tensor_parallel_output_grad:
            output = comm_func.reduce_scatter_tensor(dout.contiguous(), group=self.tp_group)[0]
        else:
            dim_size = dout.shape[0]
            if dim_size % self.world_size != 0:
                raise ValueError("First dimension of the tensor should be divisible by tensor parallel size.")
            local_dim_size = dim_size // self.world_size

            dim_offset = self.rank * local_dim_size
            output = dout[dim_offset : dim_offset + local_dim_size].contiguous()

        if self.need_to_swapaxes:
            output = output.swapaxes(0, 1)

        return (output,)


class AllGatherFromTensorParallelRegion(nn.Cell):
    """AllGather From Tensor Parallel Region"""
    def __init__(self):
        super(AllGatherFromTensorParallelRegion, self).__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tensor_model_parallel_group()

    # pylint: disable=C0111
    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        # Size and dimension.
        last_dim = input_.ndim - 1
        if self.world_size == 1:
            output = input_
        else:
            output = comm_func.all_gather_into_tensor(input_, group=self.tp_group)[0]
        tensor_list = ops.split(output, input_.shape[0], axis=0)
        output = ops.cat(tensor_list, axis=last_dim).contiguous()
        return output

    # pylint: disable=W0613, C0111
    def bprop(self, x, out, dout):
        num_dims = dout.ndim
        permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
        dout = ops.transpose(dout, permute_order).contiguous()
        if self.world_size == 1:
            output = dout
        else:
            output = comm_func.reduce_scatter_tensor(dout, group=self.tp_group)[0]

        permute_order = tuple(range(1, num_dims)) + (0,)
        output = ops.transpose(output, permute_order).contiguous()
        return (output,)


class AllToAllEven(nn.Cell):
    """All to All"""
    def __init__(self, group, split_count, split_dim, concat_dim):
        super(AllToAllEven, self).__init__()
        self.world_size = get_group_size(group=group)
        if self.world_size > 1:
            self.all_to_all = ops.AlltoAll(split_count, split_dim, concat_dim, group=group)
            self.all_to_all_grad = ops.AlltoAll(split_count, concat_dim, split_dim, group=group)

    def construct(self, input_):
        if self.world_size == 1:
            return ops.stop_gradient(input_)

        return self.all_to_all(input_)

    # pylint: disable=W0613
    def bprop(self, x, out, dout):
        if self.world_size == 1:
            return (dout,)

        output = self.all_to_all_grad(dout)
        return (output,)


class AllToAll(nn.Cell):
    '''
    scatter and gather input with split size to/from all rank, and return result in a single tensor.

       For example:
            assuming rank_size = 2, there are two sample on each rank before dispatch:

            rank0: input_: ["疑", "前", "是", "上", "月", "光", "地", "霜", "床", "明"]

                   output_splits: [[4],
                                   [6]]

                   input_splits:  [4, 6]

                   output: ["疑", "前", "是", "上", "山", "近", "流", "白", "海"]

            rank1: input_: ["山", "近", "流", "白", "海", "日", "依", "黄", "河", "入"]

                   output_splits: [[5],
                                   [5]]

                   input_splits:  [5, 5]

                   output: ["月", "光", "地", "霜", "床", "明", "日", "依", "黄", "河", "入"]
    '''

    def __init__(self, group, output_shape, input_shape, output_splits, input_splits, use_self_defined_alltoall=False):
        """
        Args:
            group (str): The communication group to work on.
            output_shape (List[int]): alltoall output shape.
            input_shape (List[int]): alltoall input shape.
            output_splits (List[int]): output split size.
            input_splits (List[int]): input split size.
            use_self_defined_alltoall (bool): use self defined alltoall or not. Default: False.
        """
        super(AllToAll, self).__init__()

        self.group = group
        self.world_size = get_group_size(group=group)

        self.output_splits = output_splits
        self.input_splits = input_splits
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.alltoall = comm_func.all_to_all_single_with_output_shape
        if use_self_defined_alltoall:
            self.alltoall = all_to_all_self_defined

    # pylint: disable=W0613
    def construct(self, input_):
        """
        Input:
            input_ (Tensor, shape(*, hidden_size)): input

        Output:
            output (Tensor, shape(*, hidden_size)): output
        """
        if self.world_size == 1:
            return ops.stop_gradient(input_)
        output = self.alltoall(self.output_shape,
                               input_,
                               self.output_splits,
                               self.input_splits,
                               self.group)[0]
        return output

    # pylint: disable=W0613
    def bprop(self, input_, output, dout):
        """define a bprop process"""
        if self.world_size == 1:
            return (dout,)
        dout2 = self.alltoall(self.input_shape,
                              dout,
                              self.input_splits,
                              self.output_splits,
                              self.group)[0]
        return (dout2,)


# pylint: disable=W0613
def all_to_all_self_defined(output_shape, input_, output_split_sizes=None, input_split_sizes=None, group=None):
    """define the forward process"""
    ep_world_size = get_expert_model_parallel_world_size()

    if ep_world_size == 1:
        return input_
    rank = get_expert_model_parallel_rank()
    group = get_expert_model_parallel_group()
    hidden_dtype = input_.dtype
    hidden_size = input_.shape[-1]

    group_input_splits = comm_func.all_gather_into_tensor(ms.Tensor(input_split_sizes, dtype=ms.int32), group=group)[0]
    group_input_splits = group_input_splits.reshape(-1, ep_world_size)
    # 1. prepare indices to slice
    zeros = ops.zeros((group_input_splits.shape[0], 1), dtype=group_input_splits.dtype)
    group_inputs_slice_begin_idx = ops.cumsum(ops.cat((zeros, group_input_splits), axis=-1), axis=-1)
    local_inputs_slice_begin_idx = group_inputs_slice_begin_idx[:, rank].asnumpy().tolist()

    group_inputs_split_sizes = ops.stack(ops.split(group_input_splits, ep_world_size, axis=-1), axis=-1)
    local_inputs_split_sizes = group_inputs_split_sizes[:, rank].asnumpy().reshape(-1).tolist()

    group_inputs_sizes = group_input_splits.sum(axis=-1).asnumpy().tolist()

    # 2. gather all input and indices from ep_group rank
    num_group_max_token = max(group_inputs_sizes)
    if input_.shape:
        num_local_token = input_.shape[-2]
        pad_len = num_group_max_token-num_local_token
        # if current token is shorter than max length, pad it to longest length
        padded_local_token = ops.pad(input_, [0, 0, 0, pad_len], value=-100)
    else:
        padded_local_token = ops.fill(type=hidden_dtype,
                                      shape=(num_group_max_token, hidden_size),
                                      value=-100)
    # allgather
    padded_group_inputs = comm_func.all_gather_into_tensor(padded_local_token, group=group)[0].reshape(
        (ep_world_size, num_group_max_token, -1))

    # slice them to origin length
    group_inputs = [x[:group_inputs_sizes[i]] for i, x in enumerate(padded_group_inputs)]

    # 3. perform split
    outputs = ms.Tensor(0)
    for i, row in enumerate(group_inputs):
        if local_inputs_split_sizes[i] > 0:
            begin = local_inputs_slice_begin_idx[i]
            end = begin + local_inputs_split_sizes[i]
            if not outputs.shape:
                outputs = row[begin:end]
            else:
                outputs = ops.cat((outputs, row[begin:end]), axis=0)
    return outputs
