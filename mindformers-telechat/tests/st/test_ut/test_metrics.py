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
"""test metric schedule."""
import pytest
import numpy as np

import mindspore as ms
from mindspore.common import Tensor
from mindspore.common import dtype as mstype

from mindformers.core.metric import PromptAccMetric, EmF1Metric

ms.set_context(mode=1, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_prompt_acc_metric():
    """
    Feature: PromptAccMetric
    Description: Test PromptAccMetric
    Expectation: No Exception
    """
    logits = Tensor(np.array([[[[0.4, -0.9],
                                [0.2, 0.7]],
                               [[0.9, 0.09],
                                [-0.4, 0.4]],
                               [[-0.2, 0.05],
                                [0.6, -0.8]],
                               [[0.6, -0.4],
                                [0.18, -0.56]]]]), mstype.float16)
    input_ids = Tensor(np.array([[0, 1], [3, 7], [6, 2], [4, 4]]), mstype.int32)
    input_mask = Tensor(np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), mstype.float32)
    labels = Tensor(np.array([[0]]), mstype.int32)

    prompt_acc_std = 0

    metric = PromptAccMetric()
    metric.clear()
    metric.update(logits, input_ids, input_mask, labels)
    prompt_acc = metric.eval().get("Acc", -1)

    error = 1e-8
    assert abs(prompt_acc - prompt_acc_std) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_emf1_metric():
    """
    Feature: EmF1Metric
    Description: Test EmF1Metric
    Expectation: No Exception
    """
    str_pre = ["I love Beijing, because it's beautiful", "Hello world。"]
    str_label = ["I love Beijing.", "Hello world"]
    metric = EmF1Metric()
    metric.clear()
    for pre, label in zip(str_pre, str_label):
        metric.update([pre], [label])
    result = metric.eval()
    error = 1e-8
    f1_score, em_score = 75.0, 50.0
    assert abs(result.get("F1", 0) - f1_score) < error and abs(result.get("Em", 0) - em_score) < error
