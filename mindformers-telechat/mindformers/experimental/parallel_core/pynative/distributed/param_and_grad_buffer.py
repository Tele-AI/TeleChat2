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
""" Param and grad buffer, bucket implemenatrion. """
import math
from enum import Enum
import numpy as np

from mindspore import ops, mint, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Zero
from mindspore.communication.management import get_rank, get_group_size
import mindspore.communication.comm_func as comm_func

from mindformers.experimental.parallel_core.pynative.utils import divide


__all__ = ['Bucket', 'ParamAndGradBuffer']


class BufferType(Enum):
    PARAM = 0
    GRAD = 1

MEM_ALIGN_SIZE = 512
ALIGN_BYTES = 32


class Bucket:
    """
    Bucket to track a subset of parameters and gradients in the buffer. Bucket records the parameters
    whose gradient has already been computed. It also provide functionality to synchronize gradients among
    data parallel group when all parameters' graidents have been computed.

    Args:
        ddp_config (DistributedDataParallelConfig): The DistributedDataParallelConfig object containing the ddp
            related configurations.
        params (List(Parameters)): Parameters belongs to this bucket.
        param_data (Tensor): A section of buffers' parameter data, coressponding to parameters in this bucket.
        grad_data (Tensor): A section of buffers' gradient data, coressponding to parameters in this bucket.
        offset (int): Start index in the buffer.
        numel_unpadded (int): Number of unpadded elements in bucket.
        data_parallel_group (str): Data parallel group name.
        data_parallel_world_size (int): Data parallel group size.
    """
    def __init__(
            self,
            ddp_config,
            params,
            param_data,
            grad_data,
            offset,
            numel_unpadded,
            data_parallel_group,
            data_parallel_world_size,
            gradient_scaling_factor,
        ):
        self.ddp_config = ddp_config

        self.params_list = params
        self.params = set(params)
        self.params_grad_ready = set()
        self.param_data = param_data
        self.grad_data = grad_data
        self.grad_data_numel = self.grad_data.numel()
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.gradient_scaling_factor = gradient_scaling_factor

        if self.data_parallel_world_size > 1:
            self.grad_reducer = comm_func.reduce_scatter_tensor \
                                if self.ddp_config.use_distributed_optimizer \
                                else comm_func.all_reduce
        self.reset()

    def inplace_reduce_dp(self, src):
        """ conduct all-reduce/reduce-scatter on src tensor and inplace update result into target. """
        self.communication_result, self.communication_handle = \
            self.grad_reducer(src, 'sum', self.data_parallel_group, async_op=self.ddp_config.overlap_grad_reduce)

    def reset(self):
        """ reset bucket for the next iteration. """
        self.params_grad_ready = set()
        self.is_reduce_issued = False
        self.communication_handle = None
        self.communication_result = None

    def issue_grad_reduce(self):
        """ issue grad reduce for the local grad data view. """
        if self.is_reduce_issued:
            raise RuntimeError('The bucket reduce is already issued')

        if self.gradient_scaling_factor != 1.0:
            self.grad_data.copy_(mint.mul(self.grad_data, self.gradient_scaling_factor))

        if self.data_parallel_world_size > 1:
            self.inplace_reduce_dp(self.grad_data)
        self.is_reduce_issued = True

    def final_grad_reduce(self):
        """ finalize grad reduce for the local grad data view. """
        # when using distributed optimizer, reduce-scatter will be conducted
        # on grad data, and only the section of grad_data which current dp rank
        # takes charge will be updated
        if self.ddp_config.use_distributed_optimizer:
            sharded_size = self.grad_data_numel // self.data_parallel_world_size
            dp_rank = get_rank(self.data_parallel_group)
            start_idx = dp_rank * sharded_size
            end_idx = (dp_rank + 1) * sharded_size
        else:
            start_idx = 0
            end_idx = self.grad_data_numel
        target = self.grad_data[start_idx:end_idx]

        if not self.ddp_config.overlap_grad_reduce:
            self.issue_grad_reduce()
            if self.data_parallel_world_size > 1:
                target.copy_(self.communication_result)
                self.communication_result = None
                if self.ddp_config.average_in_collective:
                    target.copy_(mint.div(target, self.data_parallel_world_size))
            return
        if not self.is_reduce_issued:
            raise RuntimeError(f"The bucket reduce has not been issued "
                               f"with only {len(self.params_grad_ready)}/{len(self.params)} params ready")
        if self.data_parallel_world_size > 1:
            self.communication_handle.wait()
            target.copy_(self.communication_result)
            self.communication_result = None
            if self.ddp_config.average_in_collective:
                target.copy_(mint.div(target, self.data_parallel_world_size))

    def register_grad_ready(self, param):
        """ register grad ready and issue bucket grad reduce when the bucket is ready. """
        if param not in self.params:
            raise ValueError('The param to be registered is not in the bucket')
        if param in self.params_grad_ready:
            raise ValueError(f'The param {param} is already registered')
        if not self.ddp_config.overlap_grad_reduce:
            raise RuntimeError('overlap_grad_reduce is not enabled, should not register grad')
        self.params_grad_ready.add(param)
        if len(self.params_grad_ready) == len(self.params):
            self.issue_grad_reduce()

    def __repr__(self):
        return f"Bucket (offset={self.offset}, param_lens={len(self.params)})"


class ParamAndGradBuffer:
    """
    Allocate contiguous memory buffer for given parameters and corresponding gradients. Breaking
    up parameters and gradients buffer into small buckets, which is the unit for all-reduce/reduce-scatter
    communication during back-propagation.

    Args:
        ddp_config (DistributedDataParallelConfig): The DistributedDataParallelConfig object containing the ddp
            related configurations.
        param_dtype (mindspore.dtype): The parameters' datatype.
        grad_dtype (mindspore.dtype): The gradients' datatype.
        params (List(Parameters)): Parameters belongs to this buffer.
        data_parallel_group (str): Data parallel group name.
        bucket_size (int): Bucket size threshold used to partition bucekts.
    """
    # pylint: disable=W0613
    def __init__(
            self,
            ddp_config,
            param_dtype,
            grad_dtype,
            params,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
        ):
        super(ParamAndGradBuffer, self).__init__()
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = get_group_size(group=self.data_parallel_group)
        self.gradient_scaling_factor = gradient_scaling_factor

        self.buckets = []
        self.param_index_map = {}
        self.ddp_config = ddp_config
        self.param_to_bucket = {}
        self.sync_enabled = True

        shard_num = 1 if not self.ddp_config.use_distributed_optimizer else self.data_parallel_world_size

        # helper func
        def _need_new_bucket(bucket_numel):
            return bucket_size is not None and bucket_numel != 0 and bucket_numel >= bucket_size

        def _does_param_need_new_bucket(param):
            return getattr(param, 'shared_embedding', False) and self.ddp_config.use_distributed_optimizer

        # mindspore operations need input tensor to be 512 byte aligned.
        # When allocating parameters' data in a contiguous memory,
        # each parameter should be allocated a 512 byte aligned memory.
        def _pad_param(numel):
            numel_in_bytes = numel * mstype.type_size_in_bytes(self.param_dtype)
            target = (
                (numel_in_bytes + MEM_ALIGN_SIZE + ALIGN_BYTES - 1) // MEM_ALIGN_SIZE
            ) * MEM_ALIGN_SIZE
            padding_in_bytes = target - numel_in_bytes
            padding_numel = divide(padding_in_bytes, mstype.type_size_in_bytes(self.param_dtype))
            return padding_numel

        # When using distributed optimizer, a bucket will be equally sharded to dp_world_size part.
        # Each shard will be passed to communication operations as input, the size of each shard also
        # need to be 512 byte aligned. Which means the whole bucket need to be 512 * dp_world_size byte aligned.
        def _pad_bucket(numel, bucket_align_size):
            numel_in_bytes = numel * mstype.type_size_in_bytes(self.param_dtype)
            if numel_in_bytes % bucket_align_size == 0:
                return 0, 0
            padding_target_in_bytes = math.ceil(
                numel_in_bytes / bucket_align_size
            ) * bucket_align_size - numel_in_bytes
            padding_target_numel = divide(padding_target_in_bytes, mstype.type_size_in_bytes(self.param_dtype))
            padding_in_bytes = padding_target_in_bytes - ALIGN_BYTES + 1 \
                if self.ddp_config.enable_mem_align else padding_target_in_bytes
            padding_tensor_numel = padding_in_bytes // mstype.type_size_in_bytes(self.param_dtype)
            return padding_target_numel, padding_tensor_numel

        bucket_align_size = shard_num * MEM_ALIGN_SIZE if self.ddp_config.enable_mem_align else shard_num

        def _build_bucket():
            nonlocal last_bucket_numel, bucket_align_size, data_start_index, buckets_metadata, \
                bucket_start_index, bucket_params, bucket_id
            # when using distirubted optimizer, bucket needs to be 512 * dp_wold_size byte aligned.
            # Since parameters are 512 byte aligned when allocating memory, the padding size is natural
            # 512 byte aligned.
            padded_numel, padding_tensor_size = _pad_bucket(last_bucket_numel, bucket_align_size) \
                if self.ddp_config.use_distributed_optimizer else (0, 0)
            if padding_tensor_size > 0:
                param_data_list.append(ops.Tensor(shape=(padding_tensor_size), dtype=self.param_dtype, init=Zero()))
            bucket_end_index = data_start_index + padded_numel
            buckets_metadata.append((bucket_start_index, bucket_end_index, padded_numel, bucket_params))
            data_start_index = bucket_end_index
            bucket_start_index = bucket_end_index
            bucket_id = bucket_id + 1
            bucket_params = []

        # get bucket partition metadata
        param_data_list = []
        buckets_metadata = []
        data_start_index = 0
        data_end_index = 0
        bucket_id = 0
        bucket_start_index = 0
        bucket_params = []
        for param in params[::-1]:
            last_bucket_numel = data_start_index - bucket_start_index
            if _need_new_bucket(last_bucket_numel) or \
                (_does_param_need_new_bucket(param) and last_bucket_numel > 0):
                _build_bucket()

            data_end_index = data_start_index + param.numel()
            data_padded_numel = _pad_param(param.numel()) if self.ddp_config.use_distributed_optimizer and \
                self.ddp_config.enable_mem_align else 0
            data_actual_end = data_end_index + data_padded_numel
            bucket_params.append(param)
            param_data_list.append(param)
            self.param_index_map[param] = (data_start_index, data_end_index, bucket_id)
            data_start_index = data_actual_end
            if _does_param_need_new_bucket(param):
                last_bucket_numel = data_start_index - bucket_start_index
                _build_bucket()

        last_bucket_numel = data_start_index - bucket_start_index
        # add bucket for the last few params which do not reach the bucket_size threshold
        if last_bucket_numel > 0:
            # add bucket for the last few params which do not reach the bucket_size threshold
            padded_numel, padding_tensor_size = _pad_bucket(last_bucket_numel, bucket_align_size) \
                if self.ddp_config.use_distributed_optimizer else (0, 0)
            if padding_tensor_size > 0:
                param_data_list.append(ops.Tensor(shape=(padding_tensor_size), dtype=self.param_dtype, init=Zero()))
            bucket_end_index = data_start_index + padded_numel
            buckets_metadata.append((bucket_start_index, bucket_end_index, padded_numel, bucket_params))
            data_start_index = bucket_end_index

        # allocate contiguous memory for parameters and gradients
        self.numel = data_start_index
        self.param_data = None
        if self.ddp_config.use_distributed_optimizer:
            from mindspore.hal.contiguous_tensors_handle import combine_tensor_list_contiguous
            self.param_data = combine_tensor_list_contiguous(param_data_list, \
                                                             enable_mem_align=self.ddp_config.enable_mem_align)
        self.grad_data = Tensor(shape=(self.numel), dtype=self.grad_dtype, init=Zero())
        self.numel_unpadded = 0

        # build bucket instance according to partition metadata
        for (bucket_start_index, bucket_end_index, padded_numel, bucket_params) in buckets_metadata:
            local_param_data = None
            if self.param_data is not None:
                local_param_data = self.param_data[bucket_start_index:bucket_end_index]
            local_grad_data = self.grad_data[bucket_start_index:bucket_end_index]
            self.numel_unpadded += bucket_end_index - bucket_start_index - padded_numel
            bucket = Bucket(ddp_config=self.ddp_config,
                            params=bucket_params,
                            param_data=local_param_data,
                            grad_data=local_grad_data,
                            offset=bucket_start_index,
                            numel_unpadded=bucket_end_index - bucket_start_index - padded_numel,
                            data_parallel_group=self.data_parallel_group,
                            data_parallel_world_size=self.data_parallel_world_size,
                            gradient_scaling_factor=self.gradient_scaling_factor)
            self.buckets.append(bucket)
            for param in bucket_params:
                self.param_to_bucket[param] = bucket

        for param in params:
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param.main_grad = self._get_buffer_slice(param.shape,
                                                     data_start_index,
                                                     BufferType.GRAD)
            if not self.ddp_config.use_distributed_optimizer:
                param.grad = param.main_grad

    def _get_buffer_slice(self, shape, start_index, buffer_type):
        """ get the buffer view with the same shape """
        end_index = start_index + int(np.prod(shape))
        if start_index < 0 or end_index > self.numel:
            raise ValueError("index out of range")
        if buffer_type == BufferType.PARAM:
            if self.param_data is None:
                raise ValueError("param_data can not be None")
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise TypeError("Invalid buffer type for _get_buffer_slice.")
        buffer_tensor = buffer_tensor.view(shape)

        return buffer_tensor

    def reset(self):
        """ reset buffer for the next iteration. """
        self.grad_data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.sync_enabled = True

    def issue_grad_reduce(self):
        """ issue grad reduce for each bucket. """
        for bucket in self.buckets:
            bucket.issue_grad_reduce()

    def final_grad_reduce(self):
        """ finalize grad reduce for each bucket """
        for bucket in self.buckets:
            bucket.final_grad_reduce()

    def register_grad_ready(self, param):
        """ register ready grad in its buckets """
        if not self.ddp_config.overlap_grad_reduce:
            raise RuntimeError('overlap_grad_reduce is not enabled, should not to register grad')
        if self.sync_enabled:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)

    def __repr__(self):
        param_index_with_name = {param.name: index for (param, index) in self.param_index_map.items()}
        return f"Buffer has buckets: \n {self.buckets} \n and param_index_map: \n {param_index_with_name}"
