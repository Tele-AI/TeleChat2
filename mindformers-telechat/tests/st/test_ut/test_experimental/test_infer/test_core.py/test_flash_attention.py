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
"""Test FlashAttention."""
import argparse
from collections import namedtuple

import numpy as np
import pytest
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor

from mindformers.experimental.graph.transformer.transformer_config import (
    TransformerConfig,
)
from mindformers.experimental.infer.core.flash_attention import FlashAttention as FlashAttentionMcore
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr

NUM_BLOCKS = 128
BLOCK_SIZE = 64

def gen_kv_cache(config_):
    """Generate the cache of key and value."""
    kv_cache_shape = (NUM_BLOCKS, BLOCK_SIZE, config_.num_kv_heads,
                      config_.head_dim)
    key_cache = Parameter(initializer('normal', kv_cache_shape,
                                      mstype.float16),
                          name="key_cache",
                          requires_grad=False)
    value_cache = Parameter(initializer('normal', kv_cache_shape,
                                        mstype.float16),
                            name="value_cache",
                            requires_grad=False)
    return key_cache, value_cache

ms.set_context(
    device_target="Ascend",
    mode=ms.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    }
)


class FlashAttentionMcoreNet(nn.Cell):
    """Construct flashAttention net of mcore interface."""

    def __init__(self, config_: TransformerConfig):
        super(FlashAttentionMcoreNet, self).__init__()
        self.num_heads = config_.num_heads
        self.n_kv_heads = config_.num_kv_heads
        self.head_dim = config_.head_dim

        kv_cache_shape = (
            NUM_BLOCKS, BLOCK_SIZE, config_.num_kv_heads, config_.head_dim
        )

        self.fa = FlashAttentionMcore(
            head_num=self.num_heads,
            kv_cache_shape=kv_cache_shape,
            head_dim=self.head_dim,
            keep_prob=1.0,
            kv_head_num=self.n_kv_heads,
            scale_value=0.25
        )

    def construct(
            self,
            query,
            key,
            value,
            kv_cache=None,
            slot_mapping=None,
            block_tables=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            attn_mask=None,
            alibi_mask=None,
            padding_mask=None,
            prefix=None,
    ):
        """Forward process of FlashAttentionMcoreNet."""
        return self.fa(
            query,
            key,
            value,
            kv_cache,
            slot_mapping,
            block_tables,
            batch_valid_length,
            context_lens_tensor,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            attn_mask=attn_mask,
            alibi_mask=alibi_mask,
            padding_mask=padding_mask,
            prefix=prefix
        )


class FlashAttentionOriginNet(nn.Cell):
    """Generate original FlashAttention net."""

    def __init__(self, config_: TransformerConfig):
        super(FlashAttentionOriginNet, self).__init__()
        self.num_heads = config_.num_heads
        self.input_layout = config_.input_layout

        self.fa = FlashAttention(
            head_num=self.num_heads,
            keep_prob=1.0,
            input_layout=self.input_layout,
            scale_value=0.25,
            use_attention_mask=False,
            use_ring_attention=False
        )

    def construct(
            self, query, key, value, attn_mask, actual_seq_qlen, actual_seq_kvlen
    ):
        """Forward process of FlashAttentionOriginNet."""
        return self.fa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen
        )


class PagedAttentionOriginNet(nn.Cell):
    """Generate original pagedAttention net."""

    def __init__(self, config_: TransformerConfig, kv_shape):
        super(PagedAttentionOriginNet, self).__init__()
        self.n_heads = config_.num_heads
        self.head_dim = config_.head_dim
        self.n_kv_heads = config_.num_kv_heads
        self.kv_shape = kv_shape

        self.paged_attention_mgr = PagedAttentionMgr(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            n_kv_heads=self.n_kv_heads,
            kv_shape=self.kv_shape
        )

    def construct(
            self,
            query,
            key,
            value,
            slot_mapping,
            batch_valid_length,
            block_tables,
            attn_mask=None,
            q_seq_lens=None
    ):
        """Forward process of PagedAttentionOriginNet."""
        self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length)
        return self.paged_attention_mgr.paged_attn(
            query,
            batch_valid_length,
            block_tables,
            attn_mask=attn_mask,
            q_seq_lens=q_seq_lens
        )


def get_fa_config(args_):
    """Generate config for FlashAttention test."""

    config_ = TransformerConfig()
    config_.batch_size = args_.batch_size
    config_.seq_length = args_.seq_length
    config_.num_heads = args_.num_heads
    config_.hidden_size = args_.hidden_size
    config_.num_kv_heads = args_.n_kv_heads
    config_.head_dim = int(config_.hidden_size / config_.num_heads)
    config_.input_layout = "TH"

    return config_


def gen_random_qkv(config_):
    """Generate random qurey/key/value tensor."""
    q_shape = (config_.batch_size, config_.seq_length, config_.hidden_size)
    kv_shape = (
        config_.batch_size, config_.seq_length,
        config_.num_kv_heads * config_.head_dim
    )
    query = Tensor(np.random.uniform(0, 1, q_shape), mstype.float16)
    key = Tensor(np.random.uniform(0, 1, kv_shape), mstype.float16)
    value = Tensor(np.random.uniform(0, 1, kv_shape), mstype.float16)
    return query, key, value


def gen_random_flatten_qkv(query, key, value):
    """Generate random qurey/key/value tensor."""
    q_b, q_s, q_h = query.shape
    query = query.reshape((1, q_b * q_s, q_h))
    kv_b, kv_s, kv_h = key.shape
    key = key.reshape((1, kv_b * kv_s, kv_h))
    value = value.reshape((1, kv_b * kv_s, kv_h))
    return query, key, value

def run_fa_test(fa_config):
    """Run a comparison between the original and mcore's FlashAttention."""

    query, key, value = gen_random_qkv(fa_config)

    slot_mapping = Tensor(
        np.arange(fa_config.batch_size * fa_config.seq_length), mstype.int32
    )

    flash_attn_mask = ops.ones((fa_config.seq_length, fa_config.seq_length),
                               dtype=mstype.float16)
    flash_attn_mask = ops.triu(flash_attn_mask, diagonal=1)

    actual_seq_qlen = Tensor(np.array([fa_config.seq_length]), dtype=mstype.int32)
    actual_seq_kvlen = Tensor(
        np.array([fa_config.seq_length]), dtype=mstype.int32
    )

    fa = FlashAttentionOriginNet(fa_config)
    query_origin = query.reshape((-1, fa_config.hidden_size)
                                 ) if fa_config.input_layout == "TH" else query
    key_origin = key.reshape((-1, fa_config.num_kv_heads * fa_config.head_dim)
                             ) if fa_config.input_layout == "TH" else key
    value_origin = value.reshape((-1, fa_config.num_kv_heads * fa_config.head_dim)
                                 ) if fa_config.input_layout == "TH" else value

    output = fa(
        query_origin,
        key_origin,
        value_origin,
        attn_mask=flash_attn_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen
    )

    fa_mcore = FlashAttentionMcoreNet(fa_config)

    output_mcore = fa_mcore(
        query,
        key,
        value,
        None,
        slot_mapping,
        attn_mask=flash_attn_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen
    )

    ret = np.allclose(output_mcore, output, rtol=1e-2, atol=1e-2)
    assert ret, (
        "The output mcore FlashAttention not equels to the original one."
    )


def run_pa_test(pa_config):
    """Run mcore and original PagedAttention."""

    query, key, value = gen_random_qkv(pa_config)
    query_flatten, key_flatten, value_flatten = gen_random_flatten_qkv(
        query, key, value)
    key_cache, value_cache = gen_kv_cache(pa_config)

    slot_mapping = Tensor(
        np.arange(pa_config.batch_size * pa_config.seq_length), mstype.int32)
    block_tables = Tensor(np.ones((NUM_BLOCKS, BLOCK_SIZE)) * -1, mstype.int32)
    block_tables[0][0] = 0

    q_seq_lens = Tensor(np.ones((pa_config.batch_size,)), mstype.int32)
    batch_valid_length = Tensor(
        np.ones((pa_config.batch_size,)) * (pa_config.seq_length+1), mstype.int32)
    context_lens_tensor = batch_valid_length - q_seq_lens

    kv_cache_shape = (NUM_BLOCKS, BLOCK_SIZE, pa_config.num_kv_heads,
                      pa_config.head_dim)
    pa = PagedAttentionOriginNet(pa_config, kv_cache_shape)
    pa.paged_attention_mgr.key_cache = key_cache
    pa.paged_attention_mgr.value_cache = value_cache
    output = pa(query_flatten,
                key_flatten,
                value_flatten,
                slot_mapping,
                batch_valid_length=batch_valid_length,
                block_tables=block_tables,
                q_seq_lens=q_seq_lens)

    pa_mcore = FlashAttentionMcoreNet(pa_config)
    pa_mcore.fa.add_flags(is_prefill=False)
    pa_mcore.fa.key_cache = key_cache
    pa_mcore.fa.value_cache = value_cache
    output_mcore = pa_mcore(
        query_flatten,
        key_flatten,
        value_flatten,
        None,
        slot_mapping,
        block_tables,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_tensor,
    )

    ret = np.allclose(output_mcore, output, rtol=1e-2, atol=1e-2)
    assert ret, (
        "The output mcore PagedAttention not equels to the original one.")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    'batch_size, seq_length, num_heads, '
    'n_kv_heads, hidden_size', (
        (2, 8, 2, 2, 32),
        (1, 8, 2, 2, 32),
        (1, 8, 2, 1, 32),
    )
)
def test_fa_attn(batch_size, seq_length, num_heads, n_kv_heads, hidden_size):
    """
    Feature: Test FA under various configurations.
    Description: Run original and MCore FA and get output.
    Expectation: The accuracy error exceeds 0.01
    """
    Args = namedtuple('Args', [
        'batch_size', 'seq_length', 'num_heads', 'n_kv_heads',
        'hidden_size'
    ])
    fa_args = Args(batch_size=batch_size,
                   seq_length=seq_length,
                   num_heads=num_heads,
                   n_kv_heads=n_kv_heads,
                   hidden_size=hidden_size)

    fa_config = get_fa_config(fa_args)
    run_fa_test(fa_config)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    'batch_size, seq_length, num_heads, '
    'n_kv_heads, hidden_size', (
        (2, 8, 2, 1, 32),
        (1, 8, 2, 1, 32),
        (2, 8, 2, 2, 32),
    )
)
def test_pa_attn(
        batch_size, seq_length, num_heads, n_kv_heads, hidden_size
):
    """
    Feature: Test PA under various configurations.
    Description: Run original and MCore PA and get output.
    Expectation: The accuracy error exceeds 0.01
    """
    Args = namedtuple(
        'Args', [
            'batch_size', 'seq_length', 'num_heads', 'n_kv_heads',
            'hidden_size'
        ]
    )
    fa_args = Args(
        batch_size=batch_size,
        seq_length=seq_length,
        num_heads=num_heads,
        n_kv_heads=n_kv_heads,
        hidden_size=hidden_size
    )

    fa_config = get_fa_config(fa_args)
    run_pa_test(fa_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default="2", type=int)
    parser.add_argument('--seq_length', default="8", type=int)
    parser.add_argument('--num_heads', default="2", type=int)
    parser.add_argument('--n_kv_heads', default="1", type=int)
    parser.add_argument('--hidden_size', default="32", type=int)
    args = parser.parse_args()
    config = get_fa_config(args)
    run_pa_test(config)
