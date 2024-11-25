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
"""Utilities for models"""
import math

from mindspore import Tensor
from mindspore import dtype
from mindspore.common.initializer import initializer, Normal, Zero

__all__ = ['init_method_normal',
           'init_method_zero',
           'scaled_init_method_normal']


def init_method_normal(sigma: float = 0.01, param_init_dtype: dtype = dtype.float32):
    """Init method based on N(0, sigma)."""
    def init_(tensor: Tensor):
        return initializer(Normal(mean=0.0, sigma=sigma), tensor.shape, param_init_dtype)

    return init_


def init_method_zero(param_init_dtype: dtype = dtype.float32):
    """Init method based on zeros"""
    def init_(tensor: Tensor):
        return initializer(Zero(), tensor.shape, param_init_dtype)

    return init_


def scaled_init_method_normal(sigma: float, num_layers: int, param_init_dtype: dtype = dtype.float32):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return initializer(Normal(mean=0.0, sigma=std), tensor.shape, param_init_dtype)

    return init_
