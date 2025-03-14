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
from mindspore import Tensor, get_context, set_context
from mindspore.communication import init

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from tests.st.test_infer.test_infer_core.utils import (AttentionNet, MLPNet, TransformerLayerNet, TransformerNet,
                                                       get_config)


def _test_parallel_mlp(config):
    """
    Test case for the ParallelMLP.

    This function generates a random input tensor with shape [batch_size, seq_length, hidden_size],
    passes it through the `net` (ParallelMLP), and verifies that the output tensor
    has the same shape [batch_size, seq_length, hidden_size].
    """

    net = MLPNet(config)

    batch_size = config.batch_size
    seq_length = config.seq_length
    hidden_size = config.hidden_size

    input_x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    output = net(input_x)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_attention(config):
    """
    Test case for the ParallelAttention.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelAttention), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """

    net = AttentionNet(config)
    batch_size = config.batch_size
    seq_length = config.seq_length
    hidden_size = config.hidden_size
    head_dim = config.hidden_size // config.num_heads
    num_blocks = config.num_blocks

    use_past = config.use_past
    is_pynative = get_context('mode') == 1

    input_x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    freqs_cos = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_sin = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    swap_mask = Tensor(np.ones((head_dim, head_dim)).astype(np.float16))
    freqs_cis = (freqs_cos, freqs_sin, swap_mask)
    batch_valid_length = None
    block_tables = None
    slot_mapping = None
    attn_mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    if use_past:
        batch_valid_length = Tensor(np.ones((batch_size,)).astype(np.int32))
        block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
        slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))
        if not is_pynative:
            attn_mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    output = net(input_x, batch_valid_length, block_tables, slot_mapping, freqs_cis=freqs_cis, attn_mask=attn_mask)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_transformerlayers(config):
    """
    Test case for the ParallelTransformerLayer.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelTransformerLayer), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    net = TransformerLayerNet(config)
    batch_size = config.batch_size
    seq_length = config.seq_length
    hidden_size = config.hidden_size
    head_dim = config.hidden_size // config.num_heads
    num_blocks = config.num_blocks

    use_past = config.use_past
    is_pynative = get_context('mode') == 1

    x = Tensor(np.random.randn(batch_size, seq_length, hidden_size).astype(np.float16))
    freqs_cos = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    freqs_sin = Tensor(np.ones((seq_length, head_dim)).astype(np.float16))
    swap_mask = Tensor(np.ones((head_dim, head_dim)).astype(np.float16))
    freqs_cis = (freqs_cos, freqs_sin, swap_mask)
    batch_valid_length = None
    block_tables = None
    slot_mapping = None
    attn_mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    if use_past:
        batch_valid_length = Tensor(np.ones((batch_size,)).astype(np.int32))
        block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
        slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))
        if not is_pynative:
            attn_mask = Tensor(np.ones((batch_size, 1, seq_length, seq_length)).astype(np.uint8))
    output = net(x, freqs_cis, attn_mask, batch_valid_length, block_tables, slot_mapping)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_parallel_transformer(config):
    """
    Test case for the ParallelTransformer.

    This function initializes various input tensors based on the configuration,
    passes them through the `net` (ParallelTransformer), and asserts that
    the output tensor has the expected shape [batch_size, seq_length, hidden_size].
    """
    net = TransformerNet(config)
    batch_size = config.batch_size
    seq_length = config.seq_length
    num_blocks = config.num_blocks
    hidden_size = config.hidden_size
    use_past = config.use_past

    tokens = Tensor(np.arange(seq_length).astype(np.int32)).tile((batch_size, 1))
    batch_valid_length = None
    block_tables = None
    slot_mapping = None

    if use_past:
        batch_valid_length = Tensor(np.ones((batch_size,)).astype(np.int32))
        block_tables = Tensor(np.ones((batch_size, num_blocks)).astype(np.int64))
        slot_mapping = Tensor(np.ones((batch_size * seq_length,)).astype(np.int32))

    output = net(tokens, batch_valid_length=batch_valid_length, batch_index=None, zactivate_len=None,
                 block_tables=block_tables, slot_mapping=slot_mapping, prefix_keys_values=None)

    assert output.shape == (batch_size, seq_length, hidden_size)


def _test_module(module, mode):
    """main"""
    # set_context
    jit_level = "O0"
    infer_boost = "on"
    set_context(mode=mode, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    # init communication
    init()
    initialize_model_parallel(tensor_model_parallel_size=2)

    # test module
    config = get_config(use_past=True)
    TEST_FUNC[module](config)

    config = get_config(use_past=False)
    TEST_FUNC[module](config)


TEST_FUNC = {
    'mlp': _test_parallel_mlp,
    'attention': _test_parallel_attention,
    'transformerlayer': _test_parallel_transformerlayers,
    'transformer': _test_parallel_transformer,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help='test module of parallel transformer')
    parser.add_argument('--mode', type=int, default=0, help='test mode of parallel transformer')

    args = parser.parse_args()
    _test_module(args.module, args.mode)
