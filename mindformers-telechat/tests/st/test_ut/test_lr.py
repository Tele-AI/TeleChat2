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
"""test lr schedule."""
import pytest
import mindspore as ms
from mindspore import Tensor
from mindformers.core.lr import (
    LinearWithWarmUpLR, CosineWithWarmUpLR, PolynomialWithWarmUpLR,
    CosineWithRestartsAndWarmUpLR, ConstantWarmUpLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts)

ms.set_context(mode=1, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lr_schedule():
    """
    Feature: LR Schedule
    Description: Test different LR Schedule
    Expectation: No Exception
    """
    total_steps = 20
    warmup_steps = 10
    learning_rate = 0.005
    lr_end = 0.0000001

    t_max = 10
    t_0 = 10
    t_mult = 2

    constant_with_warmup_lr_std = [0.0005, 0.005, 0.005]
    linear_lr_std = [0.0005, 0.005, 0.0005]
    cosine_lr_std = [0.0005, 0.005, 0.00012236]
    cosine_with_restarts_lr_std = [0.0005, 0.005, 0.00012236]
    polynomial_lr_std = [0.0005, 0.005, 0.00050009]
    cosine_annealing_lr_std = [0.00487764, 0.0000001, 0.00487764]
    cosine_annealing_warm_restarts_lr_std = [0.00487764, 0.005, 0.00289113]

    constant_warmup = ConstantWarmUpLR(learning_rate=learning_rate,
                                       warmup_steps=warmup_steps,
                                       total_steps=total_steps)
    linear_warmup = LinearWithWarmUpLR(learning_rate=learning_rate,
                                       warmup_steps=warmup_steps,
                                       total_steps=total_steps)
    cosine_warmup = CosineWithWarmUpLR(learning_rate=learning_rate,
                                       warmup_steps=warmup_steps,
                                       total_steps=total_steps)
    cosine_restarts_warmup = CosineWithRestartsAndWarmUpLR(learning_rate=learning_rate,
                                                           warmup_steps=warmup_steps,
                                                           total_steps=total_steps)
    polynomial_warmup = PolynomialWithWarmUpLR(learning_rate=learning_rate,
                                               warmup_steps=warmup_steps,
                                               total_steps=total_steps,
                                               lr_end=lr_end)
    cosine_annealing = CosineAnnealingLR(base_lr=learning_rate,
                                         t_max=t_max,
                                         eta_min=lr_end)
    cosine_annealing_warm_restarts = CosineAnnealingWarmRestarts(base_lr=learning_rate,
                                                                 t_0=t_0,
                                                                 t_mult=t_mult,
                                                                 eta_min=lr_end)

    constant_warmup_lr = []
    linear_warmup_lr = []
    cosine_warmup_lr = []
    cosine_restarts_warmup_lr = []
    polynomial_warmup_lr = []
    cosine_annealing_lr = []
    cosine_annealing_warm_restarts_lr = []

    assert_steps = [1, 10, 19]

    for step in range(0, total_steps):
        if step in assert_steps:
            step = Tensor(step, ms.int32)
            constant_warmup_lr.append(constant_warmup(step).asnumpy())
            linear_warmup_lr.append(linear_warmup(step).asnumpy())
            cosine_warmup_lr.append(cosine_warmup(step).asnumpy())
            cosine_restarts_warmup_lr.append(cosine_restarts_warmup(step).asnumpy())
            polynomial_warmup_lr.append(polynomial_warmup(step).asnumpy())
            cosine_annealing_lr.append(cosine_annealing(step).asnumpy())
            cosine_annealing_warm_restarts_lr.append(cosine_annealing_warm_restarts(step).asnumpy())

    error = 1e-8
    for i in range(3):
        assert abs(constant_warmup_lr[i] - constant_with_warmup_lr_std[i]) < error
        assert abs(linear_warmup_lr[i] - linear_lr_std[i]) < error
        assert abs(cosine_warmup_lr[i] - cosine_lr_std[i]) < error
        assert abs(cosine_restarts_warmup_lr[i] - cosine_with_restarts_lr_std[i]) < error
        assert abs(polynomial_warmup_lr[i] - polynomial_lr_std[i]) < error
        assert abs(cosine_annealing_lr[i] - cosine_annealing_lr_std[i]) < error
        assert abs(cosine_annealing_warm_restarts_lr[i] - cosine_annealing_warm_restarts_lr_std[i]) < error
