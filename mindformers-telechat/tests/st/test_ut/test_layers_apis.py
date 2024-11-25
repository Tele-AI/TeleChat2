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
""" test layers."""
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.common.api import _cell_graph_executor
from mindformers.modules.layers import AlibiTensor


def test_alibi_tensor():
    """
    Feature: AlibiTensor
    Description: Test alibi tensor forward
    Expectation: No Exception
    """
    model = AlibiTensor(seq_length=128, num_heads=32)
    input_mask = Tensor(np.ones((10, 128)), dtype.float32)
    mstype = dtype.float32
    _cell_graph_executor.compile(model, input_mask, mstype)
    