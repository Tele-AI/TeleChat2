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
"""
Test module for testing ParallelAttention
"""
import os
import argparse
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.communication import init
import mindspore.ops.operations as P
from mindformers.experimental.graph.transformer.transformer import ParallelAttention
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

seed = 22
ms.set_seed(seed)
np.random.seed(seed)


class MyNet(nn.Cell):
    """A test class for testing features."""

    def __init__(self, config):
        super(MyNet, self).__init__()
        self.parallel_attn = ParallelAttention(config, layer_number=1)
        self.config = config
        self.reshape = P.Reshape()
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


def run_attention():
    """A run_attention function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=1, required=True, type=int, help='data_parallel')
    parser.add_argument('--cp', default=1, required=True, type=int, help='context_parallel')
    parser.add_argument('--tp', default=1, required=True, type=int, help='tensor_parallel')
    args_, rest_args_ = parser.parse_known_args()
    print("args:", args_)
    print("rest_args:", rest_args_)

    batch, seq_length, hidden_size = (32, 1024, 256)
    config = get_config(hidden_size, args_.dp, args_.cp, args_.tp)

    input_shape = (batch, seq_length, hidden_size)
    input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
    mask = Tensor(np.triu(np.ones((seq_length, seq_length)), 1), mstype.uint8)
    print(mask)

    # core_attention
    mynet = MyNet(config)
    out_0 = mynet(input_, mask)
    print("out_0 shape:", out_0.shape)
    print("test self_attention complete!")

    # flash_attention
    config.use_flash_attn = True
    mynet = MyNet(config)
    out_1 = mynet(input_, mask)
    print("out_1 shape:", out_1.shape)
    print("test flash_attention complete!")


run_attention()
