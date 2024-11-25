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
"""run rotary embedding"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.communication import init
from mindspore.ops import operations as P

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb

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


class MyAttention(nn.Cell):
    def __init__(self, config: TransformerConfig):
        super(MyAttention, self).__init__()
        self.config = config

    def construct(self, x, freqs):
        return apply_rotary_pos_emb(x, freqs, self.config)


class MyNet(nn.Cell):
    """MyNet for rotary embedding"""
    def __init__(self, config: TransformerConfig):
        super(MyNet, self).__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = dim // self.n_heads
        self.rotary_embedding = RotaryEmbedding(self.head_dim)
        self.attention = MyAttention(config)
        dp = config.data_parallel
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose_back = P.Transpose().shard(((dp, 1, 1, 1),))
        self.reshape = P.Reshape()

    def construct(self, x: Tensor):
        """Test Net forward

        Args:
            x (Tensor): input tensor
        """
        bs_, seq_len_, dim_ = x.shape
        # [bs, seq_len, dim] -> [bs, seq_len, heads, head_dim]
        x = self.reshape(x, (bs_, seq_len_, self.n_heads, self.head_dim))
        # [bs, seq_len, heads, head_dim] -> [bs, heads, seq_len, head_dim]
        query = self.transpose(x, (0, 2, 1, 3))
        freqs = self.rotary_embedding(seq_len_)

        output = self.attention(query, freqs)

        # [bs, heads, seq_len, head_dim] -> [bs, seq_len, heads, head_dim]
        output = self.transpose_back(output, (0, 2, 1, 3))
        # [bs, seq_len, heads, head_dim] -> [bs, seq_len, dim]
        output = self.reshape(output, (bs_, seq_len_, dim_))
        return output


config_ = TransformerConfig()
config_.data_parallel = args_.dp
config_.tensor_parallel = args_.tp
config_.context_parallel = args_.cp
config_.num_attention_heads = 8

bs = 2
seq_len = 4096
dim = 8192
input_shape = (bs, seq_len, dim)

net = MyNet(config_)
input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
output_ = net(input_)
