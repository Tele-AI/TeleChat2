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
"""utils for moe"""
from mindspore import nn
from mindspore import ops as P
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config

__all__ = [
    "ZLoss"
]


class ZLoss(nn.Cell):
    """Encouraging the routers' logits to remain small, mitigating round-off error to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.
    """
    def __init__(self, parallel_config=default_dpmp_config):
        super(ZLoss, self).__init__()
        dp = parallel_config.data_parallel
        self.mean_ops = P.ReduceMean().shard(((dp, 1),))
        self.sum_ops = P.ReduceSum().shard(((dp, 1, 1),))
        self.exp = P.Exp().shard(((dp, 1, 1),))
        self.ln = P.Log().shard(((dp, 1),))
        self.square = P.Square().shard(((dp, 1),))
        self.cast = P.Cast()
        self.mul = P.Mul()

    def construct(self, logits, weight):
        logits = self.sum_ops(self.exp(logits), -1)
        logits = self.square(self.ln(logits))
        z_loss = self.mul(self.mean_ops(logits), weight)

        return z_loss
