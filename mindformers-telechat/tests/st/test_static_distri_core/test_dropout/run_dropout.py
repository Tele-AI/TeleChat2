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
"""run transformer layer test"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import dtype
from mindspore.communication import init
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindformers.experimental.graph.transformer.transformer import ParallelAttention
from mindformers.experimental.graph.transformer import Dropout
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig


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
    default=2,
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


class TestParallelAttention(nn.Cell):
    """A test class for testing features."""

    def __init__(self, config):
        super(TestParallelAttention, self).__init__()
        self.parallel_attn = ParallelAttention(config, layer_number=1)
        self.config = config
        self.reshape = P.Reshape()
        self.tile = P.Tile()

    def construct(self, x, mask):
        if (self.config.use_flash_attn
                and self.config.use_attn_mask_compression is False):
            batch, seq_length, _ = x.shape
            mask = self.reshape(mask, (1, 1, seq_length, seq_length))
            mask = self.tile(mask, (batch, 1, 1, 1))
        out, _ = self.parallel_attn(x, mask)
        return out


def get_config(hidden_size, dp, cp, tp):
    """A get_config function."""
    config = TransformerConfig()
    config.param_init_dtype = mstype.float32
    config.compute_dtype = mstype.float32
    config.group_query_attention = True
    config.num_attention_heads = 8
    config.num_query_groups = 4
    config.hidden_size = hidden_size
    config.use_flash_attn = False
    config.data_parallel = dp
    config.tensor_parallel = tp
    config.context_parallel = cp
    config.qkv_concat = True
    config.use_attn_mask_compression = False
    config.add_qkv_bias = True
    config.add_bias_linear = True
    config.softmax_compute_dtype = mstype.float32
    config.attention_dropout = 0.0
    config.apply_query_key_layer_scaling = True
    config.mask_func_type = 'attn_mask_fill'

    return config



class OpsDropout(nn.Cell):
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, drop_prob=0.5):
        super(OpsDropout, self).__init__()
        keep_prob = 1.0 - drop_prob
        self.keep_prob = keep_prob
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        out, _ = self.dropout(x)
        return out

    def shard(self, strategy):
        self.dropout.shard((strategy,))

class ExtDropout(nn.Cell):
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, drop_prob=0.5):
        super(ExtDropout, self).__init__()
        self.drop_prob = drop_prob
        self.dropout = Dropout(drop_prob=drop_prob)
    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        out, _ = self.dropout(x)
        return out

    def shard(self, strategy):
        self.dropout.shard((strategy,))


batch_s, seq_length_s, hidden_size_s = (32, 1024, 256)
input_shape = (batch_s, seq_length_s, hidden_size_s)
input_ = ms.tensor(np.random.standard_normal(input_shape), dtype.float32)
p = 0.5

#primitive
print(input_.shape)
print("***************************************************************")
net = ExtDropout(drop_prob=p)
output = net(x=input_)
print(output.shape)
print("***************************************************************")
print(output)
