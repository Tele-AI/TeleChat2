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
from mindspore import dtype, Tensor
from mindspore.communication import init

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.transformer import ParallelTransformerLayer

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
dp = args_.dp
cp = args_.cp
tp = args_.tp

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


def get_default_causal_mask(seq_len__: int) -> Tensor:
    return ms.tensor(np.triu(np.ones((seq_len__, seq_len__)), 1), dtype.uint8)


class MyNet(nn.Cell):
    """MyNet for transformer layer"""
    def __init__(self, config: TransformerConfig, seq_len: int):
        super(MyNet, self).__init__()
        self.num_layers = config.num_layers
        self.seq_len = seq_len
        self.attn_mask = get_default_causal_mask(self.seq_len)

        def build_layer(layer_number):
            return ParallelTransformerLayer(config, layer_number)
        self.layers = nn.SequentialCell(
            [build_layer(i + 1) for i in range(self.num_layers)]
        )

    def construct(self, hidden_states):
        for index in range(self.num_layers):
            hidden_states = self.layers[index](hidden_states, self.attn_mask)
        return hidden_states


config_ = TransformerConfig()
config_.data_parallel = dp
config_.context_parallel = cp
config_.tensor_parallel = tp
config_.num_layers = 2
config_.hidden_size = 4096
config_.ffn_hidden_size = 14336
config_.hidden_dropout = 0.0
config_.normalization = 'RMSNorm'
config_.layernorm_epsilon = 1.0e-5
config_.param_init_dtype = dtype.float32
config_.compute_dtype = dtype.float32
config_.num_attention_heads = 32
config_.group_query_attention = True
config_.use_flash_attn = False
config_.add_qkv_bias = False
config_.add_bias_linear = False
config_.gated_linear_unit = True
config_.mlp_has_gate = False
config_.hidden_act = 'swiglu'
config_.qkv_concat = True
config_.use_attn_mask_compression = False
config_.mask_func_type = 'attn_mask_fill'
config_.apply_residual_connection_post_layernorm = True
config_.attention_dropout = 0.0
config_.kv_num_heads = 8
config_.layernorm_compute_type = dtype.float32
config_.intermediate_size = None
config_.multiple_of = None

bs = 2
seq_len_ = 2048
dim = 4096
input_shape = (bs, seq_len_, dim)

net = MyNet(config_, seq_len_)
input_ = ms.tensor(np.random.standard_normal(input_shape), dtype.float32)
output = net(input_)
