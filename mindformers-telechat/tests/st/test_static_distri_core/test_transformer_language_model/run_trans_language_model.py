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
"""Run TransLanguageModel test"""
import argparse
import os
import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore import dtype
from mindspore.communication import init

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.language_model import TransformerLanguageModel

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


class MyNet(nn.Cell):
    """A test class for testing features."""

    def __init__(self, config: TransformerConfig, args):
        super(MyNet, self).__init__()
        self.lang_model = TransformerLanguageModel(config,
                                                   encoder_attn_mask_type=None,
                                                   num_tokentypes=0,
                                                   add_encoder=args.add_encoder,
                                                   add_decoder=args.add_decoder,
                                                   decoder_attn_mask_type=None,
                                                   add_pooler=args.add_pooler,
                                                   pre_process=args.pre_process,
                                                   post_process=args.post_process
                                                   )

    def construct(self, input_tokens, enc_position_ids, enc_attn_mask, prefix_keys_values):
        output_ = self.lang_model(input_tokens, enc_position_ids,
                                  enc_attn_mask, prefix_keys_values=prefix_keys_values)
        return output_


class ParallelConfig:
    pipeline_stage = 1
    recompute = False
    gradient_aggregation_group = 1


def get_config(args):
    """A get_config function."""
    config = TransformerConfig()
    config.param_init_dtype = dtype.float32
    config.compute_dtype = dtype.float32
    config.num_layers = 5
    config.group_query_attention = True
    config.num_attention_heads = 8
    config.num_query_groups = 4
    config.hidden_size = args.hidden_size
    config.ffn_hidden_size = config.hidden_size * 4
    config.seq_length = args.seq_length
    config.padded_vocab_size = args.vocab_size
    config.use_flash_attn = False
    config.data_parallel = args.dp
    config.tensor_parallel = args.tp
    config.context_parallel = args.cp
    config.qkv_concat = True
    config.use_attn_mask_compression = False
    config.add_qkv_bias = True
    config.add_bias_linear = True
    config.softmax_compute_dtype = dtype.float32
    config.attention_dropout = 0.0
    config.apply_query_key_layer_scaling = True
    config.mask_func_type = 'attn_mask_fill'

    config.is_dynamic = False
    config.pad_token_id = 0
    config.layernorm_compute_type = dtype.float32
    config.intermediate_size = None
    config.multiple_of = 256
    config.ffn_dim_multiplier = None
    config.offset = 0
    config.pp_interleave_num = 1
    config.parallel_config = ParallelConfig()

    return config


def run_language_model():
    """A run_transformer_language_model function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=1, required=True, type=int, help='data_parallel')
    parser.add_argument('--cp', default=1, required=True, type=int, help='context_parallel')
    parser.add_argument('--tp', default=1, required=True, type=int, help='tensor_parallel')
    args_, rest_args_ = parser.parse_known_args()
    print("args:", args_)
    print("rest_args:", rest_args_)

    args_.batch, args_.seq_length, args_.hidden_size = (4, 32, 64)
    args_.add_pooler = False
    args_.add_encoder = True
    args_.add_decoder = False
    args_.pre_process = True
    args_.post_process = True
    args_.vocab_size = 10000

    # set config
    config = get_config(args_)

    # initial inputs
    input_shape = (args_.batch, args_.seq_length)
    input_tokens = ms.Tensor(np.random.randint(0, args_.vocab_size, size=input_shape), dtype.int32)

    enc_position_ids = None
    enc_attn_mask = None
    prefix_keys_values = None

    # testcase 0
    mynet = MyNet(config, args_)
    out = mynet(input_tokens, enc_position_ids, enc_attn_mask, prefix_keys_values)
    print("out shape:", out.shape)

    # test case 1
    prefix_shape = (config.num_layers, 2, args_.batch, args_.seq_length,
                    (config.hidden_size // config.num_attention_heads) * config.num_query_groups)
    prefix_keys_values = ms.Tensor(np.random.randint(0, args_.vocab_size, size=prefix_shape),
                                   dtype.int32)
    out = mynet(input_tokens, enc_position_ids, enc_attn_mask, prefix_keys_values)
    print("prefix-tuning out shape:", out.shape)


run_language_model()
