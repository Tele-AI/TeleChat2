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
"""test lr schedule."""
import mindspore as ms
from mindspore.common.tensor import Tensor

from mindformers.experimental.graph.optimizer.lr_scheduler.lr_scheduler import LearningRateScheduler
import troubleshooter as ts


ms.set_context(mode=1, device_target='CPU')


def test_lr_schedule():
    """
    Feature: learning scheduler.
    Description: test learning scheduler.
    Expectation: No Exception
    """
    total_steps = 20
    warmup_steps = 10
    learning_rate = 0.005
    lr_end = 0.0000001

    cosine_lr_std = [0.0005, 0.005, 0.00012236]
    polynomial_lr_std = [0.0005, 0.005, 0.00050009]

    cosine_warmup = LearningRateScheduler(learning_rate=learning_rate,
                                          warmup_steps=warmup_steps,
                                          total_steps=total_steps,
                                          lr_decay_style="cosine")
    polynomial_warmup = LearningRateScheduler(learning_rate=learning_rate,
                                              warmup_steps=warmup_steps,
                                              total_steps=total_steps,
                                              lr_end=lr_end,
                                              lr_decay_style="polynomial")

    cosine_warmup_lr = []
    polynomial_warmup_lr = []

    assert_steps = [1, 10, 19]

    for step in range(0, total_steps):
        if step in assert_steps:
            step = Tensor(step, ms.int32)
            cosine_warmup_lr.append(cosine_warmup(step).asnumpy())
            polynomial_warmup_lr.append(polynomial_warmup(step).asnumpy())

    ts.save("/tmp", Tensor(cosine_warmup_lr))
    ts.save("/tmp", Tensor(polynomial_warmup_lr))

    error = 1e-8
    for i in range(3):
        assert abs(cosine_warmup_lr[i] - cosine_lr_std[i]) < error
        assert abs(polynomial_warmup_lr[i] - polynomial_lr_std[i]) < error
