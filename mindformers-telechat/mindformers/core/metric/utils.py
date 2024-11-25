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
"""utils for metric"""
from mindspore import nn
from mindspore.ops import operations as P

from mindformers.core.loss import CrossEntropyLoss


class PerplexityCell(nn.Cell):
    """cell for calculate ppl loss"""
    def __init__(self, is_pipeline_parallel: bool):
        super().__init__()
        self.loss = CrossEntropyLoss()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.sub = P.Sub()
        self.is_pipeline_parallel = is_pipeline_parallel

    def construct(self, logits, labels, input_mask):
        """construct"""
        if self.is_pipeline_parallel:
            # input_mask was added 1 in GPT2LMModel to avoid allgather issue in Mindspore1.10
            input_mask = self.sub(input_mask, 1)
        batch_size, seq_length, _ = logits.shape
        logits = self.reshape(logits[::, :-1, ::], (batch_size * (seq_length - 1), -1))
        labels = self.reshape(labels[::, 1:], (-1,))
        input_mask = self.reshape(input_mask[::, 1:], (-1,))
        loss = self.loss(logits, labels, input_mask)

        return loss
