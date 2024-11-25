# Copyright 2023 Huawei Technologies Co., Ltd
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
"""SAM Layers"""
from typing import Type

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.ops import operations as P

from mindformers.modules.layers import Linear


class MLPBlock(nn.Cell):
    """
    Multi-Layer Perceptron (MLP) block.

    Args:
        x (ms.Tensor): Input tensor.

    Returns:
        ms.Tensor: Output tensor.
    """
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Cell] = nn.GELU,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels=embedding_dim,
                           out_channels=mlp_dim,
                           compute_dtype=compute_dtype,
                           param_init_type=param_init_type)
        self.lin2 = Linear(in_channels=mlp_dim,
                           out_channels=embedding_dim,
                           compute_dtype=compute_dtype,
                           param_init_type=param_init_type)
        self.act = act()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Cell):
    """
    Layer Normalization for 2D data.

    Args:
        x (ms.Tensor): Input tensor.

    Returns:
        ms.Tensor: Normalized tensor.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = Parameter(P.Ones()(num_channels, ms.float32))
        self.bias = Parameter(P.Zeros()(num_channels, ms.float32))
        self.eps = eps
        self.pow = P.Pow()
        self.sqrt = P.Sqrt()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        u = x.mean(1, keep_dims=True)
        s = self.pow(x - u, 2).mean(1, keep_dims=True)
        x = (x - u) / self.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
