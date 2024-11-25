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
"""Learning rate scheduler"""

from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindformers.core.lr.lr_schedule import CosineWithWarmUpLR, PolynomialWithWarmUpLR

__all__ = ['LearningRateScheduler']


class LearningRateScheduler(LearningRateSchedule):
    """learning scheduler"""
    def __init__(
            self,
            learning_rate: float,
            warmup_steps: int = 0,
            total_steps: int = None,
            lr_end: float = 0.,
            warmup_lr_init: float = 0.,
            warmup_ratio: float = None,
            num_cycles: float = 0.5,
            power: float = 1.0,
            decay_steps: int = None,
            lr_decay_style: str = "cosine",
            **kwargs
    ):
        super(LearningRateScheduler, self).__init__()
        if lr_decay_style == "cosine":
            self.decay_lr_scheduler = CosineWithWarmUpLR(learning_rate, warmup_steps, total_steps, num_cycles, lr_end,
                                                         warmup_lr_init, warmup_ratio, decay_steps, **kwargs)
        elif lr_decay_style == "polynomial":
            self.decay_lr_scheduler = PolynomialWithWarmUpLR(learning_rate, total_steps, warmup_steps, lr_end, power,
                                                             warmup_lr_init, warmup_ratio, decay_steps, **kwargs)
        else:
            raise ValueError("lr_decay_style only support cosine and polynomial")

    def construct(self, global_step):
        return self.decay_lr_scheduler(global_step)
