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
"""test parallel transformer."""

import argparse

import numpy as np
from mindspore import Tensor, set_context
from mindspore.communication import init

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from tests.st.test_infer.test_infer_core.utils import get_config, get_module


def _test_parallel_mlp(net):
    """
    Test case for the ParallelMLP.

    This function generates a random input tensor with shape [batch_size, seq_length, hidden_size],
    passes it through the `net` (ParallelMLP), and verifies that the output tensor
    has the same shape [batch_size, seq_length, hidden_size].
    """
    base_config = get_config()
    batch_size = base_config.batch_size
    seq_length = base_config.seq_length
    hidden_size = base_config.hidden_size

    input_x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    output = net(input_x)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_attention(net):
    """
    Test case for the ParallelAttention.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelAttention), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    base_config = get_config()
    batch_size = base_config.batch_size
    seq_length = base_config.seq_length
    hidden_size = base_config.hidden_size
    head_dim = base_config.hidden_size // base_config.num_heads
    num_blocks = base_config.num_blocks

    input_x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    batch_valid_length = Tensor(np.ones((batch_size, 1)).astype(np.int32))
    block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
    slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))
    attn_mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    freqs_cos = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_sin = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_cis = (freqs_cos, freqs_sin, None)
    output = net(input_x, batch_valid_length, block_tables, slot_mapping, freqs_cis=freqs_cis, attn_mask=attn_mask)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_transformerlayers(net):
    """
    Test case for the ParallelTransformerLayer.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelTransformerLayer), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    base_config = get_config()
    batch_size = base_config.batch_size
    seq_length = base_config.seq_length
    hidden_size = base_config.hidden_size
    head_dim = base_config.hidden_size // base_config.num_heads
    num_blocks = base_config.num_blocks

    x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    batch_valid_length = Tensor(np.ones((batch_size, 1)).astype(np.int32))
    block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
    slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))
    mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    freqs_cos = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_sin = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_cis = (freqs_cos, freqs_sin, None)
    output = net(x, freqs_cis, mask, batch_valid_length, block_tables, slot_mapping)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_transformer(net):
    """
    Test case for the ParallelTransformer.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelTransformer), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    base_config = get_config()
    batch_size = base_config.batch_size
    seq_length = base_config.seq_length
    num_blocks = base_config.num_blocks
    hidden_size = base_config.hidden_size

    tokens = Tensor(np.arange(seq_length).astype(np.int32)).tile((batch_size, 1))
    batch_valid_length = Tensor(np.ones((batch_size, 1)).astype(np.int32))
    block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
    slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))

    output = net(tokens, batch_valid_length=batch_valid_length, batch_index=None, zactivate_len=None,
                 block_tables=block_tables, slot_mapping=slot_mapping, prefix_keys_values=None)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_module(module):
    """main"""
    # set_context
    jit_level = "O0"
    infer_boost = "on"
    set_context(mode=0, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    # init communication
    init()
    initialize_model_parallel(tensor_model_parallel_size=2)

    # test module
    net = get_module(module)
    TEST_FUNC[module](net)


TEST_FUNC = {
    'mlp': _test_parallel_mlp,
    'attention': _test_parallel_attention,
    'transformerlayer': _test_parallel_transformerlayers,
    'transformer': _test_parallel_transformer,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help='test module of parallel transformer')

    args = parser.parse_args()
    _test_module(args.module)
