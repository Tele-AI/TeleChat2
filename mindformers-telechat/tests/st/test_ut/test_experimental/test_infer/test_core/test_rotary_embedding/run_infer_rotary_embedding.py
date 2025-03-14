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
"""run rotary embedding in infer mode"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig


from tests.st.test_ut.test_experimental.test_infer.test_core.test_rotary_embedding.utils import (
    NewRopeNet,
    NewLlama3RopeNet,
    OldRopeNet,
    OldLlama3RopeNet
)


jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE, full_batch=False)

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


def set_config(args_):
    transformer_config = TransformerConfig()
    transformer_config.batch_size = args_.bs
    transformer_config.seq_length = args_.seq
    transformer_config.hidden_size = args_.hidden
    transformer_config.num_attention_heads = args_.num
    transformer_config.max_position_embeddings = args_.position

    return transformer_config


def generate_prefill_inputs(bs, seq_len, hidden_size):
    input_shape = (bs, seq_len, hidden_size)
    # [bs, seq_len, dim]
    query = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    key = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    # [1, bs * seq_len, dim]
    query_padding = query.reshape(1, bs * seq_len, hidden_size)
    key_padding = key.reshape(1, bs * seq_len, hidden_size)
    return query_padding, key_padding


def generate_increment_inputs(bs, hidden_size):
    input_shape = (bs, 1, hidden_size)
    # [bs, 1, dim]
    query = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    key = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    return query, key


def run_test(config_, old_net, new_net):
    """A run_rotary_embedding_test function."""
    bs = config_.batch_size
    seq_len = config_.seq_length
    hidden_size = config_.hidden_size
    is_prefill = new_net.prefill

    batch_valid_length = Tensor(np.ones((bs,)), mstype.int32) * seq_len
    if is_prefill:
        query, key = generate_prefill_inputs(bs, seq_len, hidden_size)
    else:
        batch_valid_length = Tensor(np.random.randint(0, seq_len, (bs,)), mstype.int32)
        query, key = generate_increment_inputs(bs, hidden_size)

    old_rotary_embedding_q, old_rotary_embedding_k = old_net(query, key, batch_valid_length)
    new_rotary_embedding_q, new_rotary_embedding_k = new_net(query, key, batch_valid_length)

    ret1 = np.array_equal(new_rotary_embedding_q.asnumpy(), old_rotary_embedding_q.asnumpy())
    ret2 = np.array_equal(new_rotary_embedding_k.asnumpy(), old_rotary_embedding_k.asnumpy())

    assert ret1 and ret2, (f"Test failed for batch size:{bs}, seq length:{seq_len}, hidden_size:{hidden_size} "
                           f"in {'prefill' if prefill else 'decode'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rope", required=True, type=str, help='test mode of rope')
    parser.add_argument('--bs', required=True, type=int, help='batch_size')
    parser.add_argument('--seq', required=True, type=int, help='seq_length')
    parser.add_argument('--hidden', required=True, type=int, help='hidden_size')
    parser.add_argument('--num', required=True, type=int, help='n_heads')
    parser.add_argument('--position', required=True, type=int, help='max_position_emb')
    parser.add_argument('--prefill', default=1, required=True, type=int, help='prefill')

    args = parser.parse_args()

    mode = args.mode
    prefill = bool(args.prefill)
    config = set_config(args)
    if mode == "rope":
        old_rope_net = OldRopeNet(config, rotary_cos_format=2, prefill=prefill)
        new_rope_net = NewRopeNet(config, rotary_cos_format=2, prefill=prefill)
    elif mode == "llama3rope":
        old_rope_net = OldLlama3RopeNet(config, rotary_cos_format=2, prefill=prefill)
        new_rope_net = NewLlama3RopeNet(config, rotary_cos_format=2, prefill=prefill)
    else:
        raise ValueError(f"Unsupported mode {mode}")

    dynamic_query = Tensor(shape=[None, None, None], dtype=mstype.float16)
    dynamic_key = Tensor(shape=[None, None, None], dtype=mstype.float16)
    dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)

    old_rope_net.set_inputs(dynamic_query, dynamic_key, dynamic_batch_valid_length)
    new_rope_net.set_inputs(dynamic_query, dynamic_key, dynamic_batch_valid_length)

    run_test(config, old_rope_net, new_rope_net)
