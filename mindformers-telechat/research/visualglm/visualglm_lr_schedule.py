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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
# ============================================================================
"""AnnealingLR LR Schedule."""
import math
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['AnnealingLR']


@MindFormerRegister.register(MindFormerModuleType.LR)
class AnnealingLR(LearningRateSchedule):
    """ AnnealingLR implementation for visualglm """
    DECAY_STYLES = ["linear", "cosine", "exponential", "constant", "None"]

    def __init__(self, learning_rate, warmup_steps, num_iters, total_steps, decay_style="cosine", last_iter=-1,
                 decay_ratio=0.1, auto_warmup_steps=100, auto_warmup_rate=0.05):
        super(AnnealingLR, self).__init__()
        self.total_steps = total_steps
        self.start_lr = learning_rate
        self.warmup_iter = Tensor(warmup_steps, mstype.float32)
        self.init_step = last_iter
        self.num_iters = Tensor(last_iter + 1, mstype.float32)
        self.end_iter = Tensor(num_iters, mstype.float32)
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = 1 / decay_ratio
        self.auto_warmup_steps = auto_warmup_steps
        self.auto_warmup_rate = auto_warmup_rate

        self.cos = P.Cos()
        self.min = P.Minimum()

    def construct(self, global_step):
        """ method entrance """
        if global_step <= self.init_step + self.auto_warmup_steps:
            auto_lr = float(self.start_lr) * self.auto_warmup_rate
            schedule_lr = float(self.start_lr) * global_step / self.warmup_iter
            return self.min(auto_lr, schedule_lr)

        if self.warmup_iter > 0 and global_step <= self.warmup_iter:
            return float(self.start_lr) * global_step / self.warmup_iter

        if self.decay_style == self.DECAY_STYLES[0]:
            return self.start_lr * ((self.end_iter - (global_step - self.warmup_iter)) / self.end_iter)

        if self.decay_style == self.DECAY_STYLES[1]:
            tmp_decay_step_ratio = (global_step - self.warmup_iter) / self.end_iter
            decay_step_ratio = self.min(1.0, tmp_decay_step_ratio)
            return self.start_lr / self.decay_ratio * (
                (self.cos(math.pi * decay_step_ratio) + 1) * (self.decay_ratio - 1) / 2 + 1)
        if self.decay_style == self.DECAY_STYLES[2]:
            return self.start_lr
        return self.start_lr
