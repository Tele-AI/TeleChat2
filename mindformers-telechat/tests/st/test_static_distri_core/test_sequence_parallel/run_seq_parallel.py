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
"""Test module for sequence parallel"""
import argparse
import os
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication import init
from mindformers.experimental.graph.transformer.transformer import ParallelTransformer
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
        self.model = ParallelTransformer(config)

    def construct(self, x, mask):
        out, _ = self.model(x, mask)
        return out


class ParallelConfig:
    pipeline_stage = 1
    recompute = False
    gradient_aggregation_group = 1


def get_config(hidden_size, seq_length, dp, cp, tp):
    """A get_config function."""
    config = TransformerConfig()
    config.param_init_dtype = mstype.float32
    config.compute_dtype = mstype.float32
    config.group_query_attention = True
    config.num_attention_heads = 8
    config.num_query_groups = 4
    config.seq_length = seq_length
    config.hidden_size = hidden_size
    config.use_flash_attn = True
    config.data_parallel = dp
    config.tensor_parallel = tp
    config.context_parallel = cp
    config.qkv_concat = False
    config.use_attn_mask_compression = False
    config.num_layers = 4
    config.add_qkv_bias = False
    config.add_bias_linear = False
    config.softmax_compute_dtype = mstype.float32
    config.attention_dropout = 0.0
    config.apply_query_key_layer_scaling = False
    config.mask_func_type = 'attn_mask_fill'

    config.offset = 0
    config.parallel_config = ParallelConfig()
    config.pp_interleave_num = 1
    config.layernorm_compute_type = mstype.float32
    config.intermediate_size = None
    config.multiple_of = 256
    config.ffn_dim_multiplier = None
    return config


def run_test():
    """A run_qkv_test function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=1, required=True, type=int, help='data_parallel')
    parser.add_argument('--cp', default=1, required=True, type=int, help='context_parallel')
    parser.add_argument('--tp', default=1, required=True, type=int, help='tensor_parallel')
    args_, _ = parser.parse_known_args()

    batch, seq_length, hidden_size = (4, 256, 64)
    config = get_config(hidden_size, seq_length, args_.dp, args_.cp, args_.tp)

    input_shape = (batch, seq_length, hidden_size)
    input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
    mask = Tensor(np.triu(np.ones((seq_length, seq_length)), 1), mstype.uint8)
    if config.use_flash_attn and config.use_attn_mask_compression is False:
        mask = mask.reshape((1, 1, seq_length, seq_length))
        mask = mask.tile((batch, 1, 1, 1))

    config.sequence_parallel = True
    model_seq_paral_on = MyNet(config)

    config.sequence_parallel = False
    model_seq_paral_off = MyNet(config)
    trainable_param_dict = {}
    for item in model_seq_paral_on.trainable_params():
        trainable_param_dict[item.name] = item.data
    ms.load_param_into_net(model_seq_paral_off, trainable_param_dict)

    out1 = model_seq_paral_on(input_, mask)
    out2 = model_seq_paral_on(input_, mask)

    assert np.allclose(out1.asnumpy(), out2.asnumpy(), rtol=1e-4, atol=1e-4)


run_test()
