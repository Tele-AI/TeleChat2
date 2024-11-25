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
"""run silu test"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import dtype
from mindspore.communication import init

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.activation import SiLU

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dp',
    default=1,
    required=True,
    type=int,
    help='data_parallel')
parser.add_argument(
    '--cp',
    default=1,
    required=True,
    type=int,
    help='context_parallel')
parser.add_argument(
    '--tp',
    default=1,
    required=True,
    type=int,
    help='tensor_parallel')
args_, rest_args_ = parser.parse_known_args()

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


class MyNet(nn.Cell):
    def __init__(self, config: TransformerConfig):
        super(MyNet, self).__init__()
        self.silu = SiLU(config)

    def construct(self, x):
        output_ = self.silu(x)
        return output_


config_ = TransformerConfig()
config_.data_parallel = args_.dp
config_.tensor_parallel = args_.tp
config_.context_parallel = args_.cp

bs = 2
seq_len = 4096
dim = 8192
input_shape = (bs, seq_len, dim)

net = MyNet(config_)
input_ = ms.tensor(np.random.standard_normal(input_shape), dtype.float32)
output = net(input_)
