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
"""run scale mask softmax"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import dtype
from mindspore.ops import operations as P
from mindspore.communication import init

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.fused_softmax import FusedScaleMaskSoftmax
from mindformers.experimental.graph.transformer.utils import get_attn_mask_func

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
    """MyNet for scale mask softmax"""
    def __init__(self, config: TransformerConfig, seq_len: int):
        super(MyNet, self).__init__()
        self.num_heads = config.num_attention_heads
        self.mask_func = get_attn_mask_func("attn_mask_fill")(config)
        self.softmax = FusedScaleMaskSoftmax(config=config,
                                             mask_func=self.mask_func)
        dp = config.data_parallel
        cp = config.context_parallel
        tp = config.tensor_parallel
        self.batch_matmul = P.BatchMatMul(transpose_b=True).shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.mask = ms.Tensor(np.triu(np.ones((seq_len, seq_len)), 1), dtype.uint8)
        self.reshape = P.Reshape()

    def construct(self, x):
        bs_, seq_len__, dim_ = x.shape
        # [bs, seq_len, dim] -> [bs, seq_len, n_heads, head_dim]
        x = self.reshape(x, (bs_, seq_len__, self.num_heads, dim_ // self.num_heads))
        # [bs, seq_len, n_heads, head_dim] -> [bs, n_heads, seq_len, head_dim]
        x = x.transpose((0, 2, 1, 3))
        # attn_scores[bs, n_heads, seq_len, seq_len]
        attn_scores = self.batch_matmul(x, x)
        output_with_mask = self.softmax(attn_scores, self.mask)
        return output_with_mask


config_ = TransformerConfig()
config_.data_parallel = args_.dp
config_.tensor_parallel = args_.tp
config_.context_parallel = args_.cp
config_.num_attention_heads = 8

bs = 2
seq_len_ = 4096
dim = 8192
input_shape = (bs, seq_len_, dim)

net = MyNet(config_, seq_len_)
input_ = ms.tensor(np.random.standard_normal(input_shape), dtype.float32)
output = net(input_)
