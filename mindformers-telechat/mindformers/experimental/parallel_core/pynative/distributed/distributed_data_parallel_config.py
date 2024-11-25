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
""" DistributedDataParallelConfig. """
from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel wrapper."""
    # if True, grad buffer will be created in fp32. Grad accumulate and synchronizer will be done in fp32.
    grad_reduce_in_fp32: bool = False

    # enable gradients calculation and communication overlap between buckets.
    overlap_grad_reduce: bool = False

    # use distributed optimizer to achieve optimizer calculation parallelism.
    use_distributed_optimizer: bool = False

    # bucket size for ParamAndGradBuffer. None means all parameters will be assigned to one bucket.
    bucket_size: Optional[int] = None

    # average gradients among data parallel group when communication.
    average_in_collective: bool = False

    # if true, check gradients in buffer are finite after synchronization.
    check_for_nan_in_grad: bool = False

    enable_mem_align: bool = True
