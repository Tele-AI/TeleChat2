# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test SelfAttention"""
import argparse
from collections import namedtuple

import numpy as np
import pytest
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_param_into_net
from mindformers.experimental.graph.transformer.spec_utils import (
    ModuleSpec,
    build_module,
)
from mindformers.experimental.graph.transformer.transformer_config import (
    TransformerConfig
)
from mindformers.experimental.infer.core.transformer import ParallelAttention
from mindformers.experimental.infer.core.flash_attention import FlashAttention
from mindformers.experimental.infer.core.self_attention import (
    CoreAttention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.experimental.infer.core.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

NUM_BLOCKS = 128
BLOCK_SIZE = 64

ms.set_context(
    device_target="Ascend",
    mode=ms.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    }
)

replacement_map = {
    'w_qkv.weight': 'linear_qkv.weight',
    'wo.weight': 'linear_proj.weight'
}

def convert_weight_name(params):
    """Convert weight name."""
    for old_name, param in list(params.items()):
        new_name = replacement_map.get(old_name, old_name)
        param.name = new_name
        if new_name != old_name:
            params.move_to_end(old_name)
            params[new_name] = params.pop(old_name)
    return params

def get_self_attention_config(args_):
    """Generate config for SelfAttention test."""
    config_ = TransformerConfig()
    parallel_config = TransformerConfig()
    parallel_config.use_sequence_parallel = False
    config_.batch_size = args_.batch_size
    config_.seq_length = args_.seq_length
    config_.param_init_type = mstype.float16
    config_.param_init_dtype = mstype.float16
    config_.compute_dtype = mstype.float16
    config_.compute_type = mstype.float16
    config_.qkv_concat = args_.qkv_concat
    config_.num_heads = args_.num_heads
    config_.num_attention_heads = args_.num_heads
    config_.n_kv_heads = args_.num_query_groups
    config_.num_query_groups = args_.num_query_groups
    config_.hidden_size = args_.hidden_size
    config_.parallel_config = parallel_config
    config_.sequence_parallel = False
    config_.head_dim = int(config_.hidden_size / config_.num_attention_heads)
    config_.out_proj_has_bias = False
    config_.qkv_has_bias = False
    config_.attention_dropout_rate = 1
    config_.use_flash_attention = args_.use_flash_attention
    config_.is_prefill = args_.is_prefill
    config_.use_past = True
    config_.num_blocks = NUM_BLOCKS
    config_.block_size = BLOCK_SIZE

    return config_


def run_self_attention_test(config_):
    """Run a comparison between the original and mcore's SelfAttention."""
    slot_mapping = Tensor(np.arange(config_.batch_size * config_.seq_length),
                          mstype.int32)
    batch_valid_length = Tensor(np.ones((config_.seq_length,)),
                                dtype=mstype.int32)

    block_tables = Tensor(
        np.ones((config_.batch_size, BLOCK_SIZE)) * -1, mstype.int32)
    block_tables[0][0] = 0

    input_ = Tensor(
        np.random.uniform(
            0, 1,
            [config_.batch_size, config_.seq_length, config_.hidden_size]),
        mstype.float16)

    if config_.use_flash_attention:
        self_attn = ModuleSpec(module=SelfAttention,
                               submodules=SelfAttentionSubmodules(
                                   core_attention=FlashAttention,
                                   linear_proj=RowParallelLinear,
                                   linear_qkv=ColumnParallelLinear,
                                   linear_q=ColumnParallelLinear,
                                   linear_k=ColumnParallelLinear,
                                   linear_v=ColumnParallelLinear))
    else:
        self_attn = ModuleSpec(module=SelfAttention,
                               submodules=SelfAttentionSubmodules(
                                   core_attention=CoreAttention,
                                   linear_proj=RowParallelLinear,
                                   linear_qkv=ColumnParallelLinear,
                                   linear_q=ColumnParallelLinear,
                                   linear_k=ColumnParallelLinear,
                                   linear_v=ColumnParallelLinear))

    self_attn_mcore = build_module(self_attn, config=config_, layer_number=1)

    parallel_attn = ParallelAttention(config_, 1)

    if config_.is_prefill:
        output = parallel_attn(input_,
                               block_tables=None,
                               slot_mapping=slot_mapping,
                               batch_valid_length=batch_valid_length)

        param_dict = parallel_attn.parameters_dict()
        converted_param_dict = convert_weight_name(param_dict)
        load_param_into_net(self_attn_mcore, converted_param_dict)

        output_mcore = self_attn_mcore(input_,
                                       kv_cache=None,
                                       slot_mapping=slot_mapping,
                                       actual_seq_qlen=batch_valid_length,
                                       actual_seq_kvlen=batch_valid_length)

        ret = np.allclose(output_mcore, output, rtol=1e-2, atol=1e-2)
        assert ret, ("The output mcore ParallerAttention not equels to "
                     "the original one when is_prefill is True")
    else:
        parallel_attn.is_first_iteration = False
        self_attn_mcore.flash_attention.add_flags(is_prefill=False)

        output_1 = parallel_attn(input_,
                                 block_tables=block_tables,
                                 slot_mapping=slot_mapping,
                                 batch_valid_length=batch_valid_length)

        param_dict = parallel_attn.parameters_dict()
        converted_param_dict = convert_weight_name(param_dict)
        load_param_into_net(self_attn_mcore, converted_param_dict)

        output_mcore = self_attn_mcore(input_,
                                       kv_cache=None,
                                       block_tables=block_tables,
                                       slot_mapping=slot_mapping,
                                       batch_valid_length=batch_valid_length,
                                       context_lens_tensor=batch_valid_length)

        ret = np.allclose(output_mcore, output_1, rtol=1e-2, atol=1e-2)
        assert ret, ("The output mcore ParallerAttention not equels to "
                     "the original one when is_prefill is False")
        ms.reset_auto_parallel_context()

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    'batch_size, seq_length, qkv_concat, '
    'num_heads, num_query_groups, hidden_size, '
    'use_flash_attention, is_prefill', (
        (1, 256, True, 16, 4, 256, True, False),
        (1, 256, True, 16, 4, 256, True, True),
        (1, 256, True, 16, 2, 256, False, True),
    )
)
def test_self_attn(batch_size, seq_length, qkv_concat, num_heads,
                   num_query_groups, hidden_size, use_flash_attention,
                   is_prefill):
    """
    Feature: Test SelfAttention under various configurations.
    Description: Run original and MCore SelfAttention and get output.
    Expectation: The accuracy error exceeds 0.01
    """
    Args = namedtuple('Args', [
        'batch_size',
        'seq_length',
        'qkv_concat',
        'num_heads',
        'num_query_groups',
        'hidden_size',
        'use_flash_attention',
        'is_prefill',
    ])
    args_ = Args(batch_size=batch_size,
                 seq_length=seq_length,
                 qkv_concat=qkv_concat,
                 num_heads=num_heads,
                 num_query_groups=num_query_groups,
                 hidden_size=hidden_size,
                 use_flash_attention=use_flash_attention,
                 is_prefill=is_prefill)
    config_ = get_self_attention_config(args_)
    run_self_attention_test(config_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default="1", type=int)
    parser.add_argument('--seq_length', default="4096", type=int)
    parser.add_argument('--qkv_concat', default=True, type=bool)
    parser.add_argument('--num_heads', default="16", type=int)
    parser.add_argument('--num_query_groups', default="2", type=int)
    parser.add_argument('--hidden_size', default="4096", type=int)
    parser.add_argument('--use_flash_attention', default=False, type=bool)
    parser.add_argument('--is_prefill', default=True, type=bool)
    args = parser.parse_args()
    config = get_self_attention_config(args)
    run_self_attention_test(config)
