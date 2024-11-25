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
"""test loss."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers import CrossEntropyLoss

ms.set_context(mode=1, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cross_entropy_loss():
    """
    Feature: CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional
    Expectation: No Exception
    """
    loss = CrossEntropyLoss()
    input_data = Tensor(np.array([[-0.438346, 0.582246, 0.842038, -0.591126, 0.590775],
                                  [-0.968004, 0.189337, 0.982326, 0.374514, 0.851041],
                                  [0.269876, 0.195891, 0.748780, -0.056748, 0.263205]], dtype=np.float32))
    target_data = Tensor(np.array([1, 4, 0], dtype=np.int32))
    input_mask = Tensor(np.array([1, 1, 1], dtype=np.int32))
    loss_result = loss(input_data, target_data, input_mask).asnumpy()[0]
    loss_result_std = 1.4249968528747559
    error = 1e-8

    assert abs(loss_result - loss_result_std) < error
