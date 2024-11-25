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
"""loss function"""
from mindspore import nn

from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.core.loss.loss import CrossEntropyLoss


class VocabParallelCrossEntropy(nn.Cell):
    """calculate cross entropy loss"""

    def __init__(self, parallel_config=default_dpmp_config, **kwargs):
        super(VocabParallelCrossEntropy, self).__init__()
        self.cross_entropy = CrossEntropyLoss(parallel_config, **kwargs)

    def construct(self, vocab_parallel_logits, target, input_mask=None, label_smoothing=None):
        if label_smoothing:
            raise NotImplementedError("label_smoothing is not supported for now in graph mode.")

        return self.cross_entropy(vocab_parallel_logits, target, input_mask)
