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
""" test infer attention"""
import math
import os

import pytest
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor

from mindformers import LowerTriangularMaskWithDynamic
from mindformers.modules.block_tables import BlockTables
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.layers import FreqsMgr


def generate_prefill_flatten_input(bsz, seq_len, head_dim, hidden_size, n_kv_head):
    query = Tensor(np.ones((1, seq_len * bsz, hidden_size)), mstype.float16)
    key = Tensor(np.ones((1, seq_len * bsz, n_kv_head * head_dim)), mstype.float16)
    value = Tensor(np.ones((1, seq_len * bsz, n_kv_head * head_dim)), mstype.float16)
    batch_valid_length = Tensor(np.ones((bsz,)), mstype.int32) * seq_len
    return query, key, value, batch_valid_length


def generate_prefill_padding_input(bsz, seq_len, head_dim, hidden_size, n_kv_head):
    query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    key = Tensor(np.ones((bsz, seq_len, n_kv_head * head_dim)), mstype.float16)
    value = Tensor(np.ones((bsz, seq_len, n_kv_head * head_dim)), mstype.float16)
    batch_valid_length = Tensor(np.ones((bsz,)), mstype.int32) * seq_len
    return query, key, value, batch_valid_length


def generate_decode_input(bsz, head_dim, hidden_size, n_kv_head):
    query = Tensor(np.ones((bsz, 1, hidden_size)), mstype.float16)
    key = Tensor(np.ones((bsz, 1, n_kv_head * head_dim)), mstype.float16)
    value = Tensor(np.ones((bsz, 1, n_kv_head * head_dim)), mstype.float16)
    return query, key, value


def set_dynamic_inputs(model):
    dynamic_query = Tensor(shape=[None, None, None], dtype=mstype.float16)
    dynamic_key = Tensor(shape=[None, None, None], dtype=mstype.float16)
    dynamic_value = Tensor(shape=[None, None, None], dtype=mstype.float16)
    dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
    dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
    dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
    dynamic_mask = Tensor(shape=[None, None], dtype=mstype.float16)
    model.set_inputs(dynamic_query,
                     dynamic_key,
                     dynamic_value,
                     dynamic_batch_valid_length,
                     dynamic_block_tables,
                     dynamic_slot_mapping,
                     None,
                     dynamic_mask,
                     None,
                     None,
                     None)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_infer_attention_with_dynamic_shape_th():
    """
    Feature: Test the infer boost attention with TH input layout.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    bsz = 8
    head_num = 80
    seq_len = 512
    max_seq_len = 8192
    head_dim = 128
    n_kv_head = 8
    block_size = 128
    num_blocks = max_seq_len * bsz // block_size
    hidden_size = head_num * head_dim
    is_dynamic = True

    block_mgr = BlockTables(num_blocks, block_size, max_seq_len)
    block_mgr.init_cache_engine(bsz)

    casual_mask = LowerTriangularMaskWithDynamic(seq_length=seq_len,
                                                 compute_type=mstype.float16,
                                                 is_dynamic=is_dynamic,
                                                 use_flash_attention=True)
    infer_attention = InferAttention(head_num,
                                     head_dim,
                                     n_kv_head,
                                     scale_value=1. / math.sqrt(head_dim),
                                     pre_tokens=65536,
                                     next_tokens=0,
                                     block_size=block_size,
                                     num_blocks=num_blocks,
                                     is_dynamic=is_dynamic,
                                     use_rope_rotary_emb=False,
                                     compute_dtype=mstype.float16)
    set_dynamic_inputs(infer_attention)

    attn_mask = casual_mask.prefill()
    # query shape: (1, 2048, 10240)
    # key shape: (1, 2048, 1024)
    # value shape: (1, 2048, 1024)
    # batch_valid_length shape: (8,)
    # block_tables shape: (8, 32)
    # slot_mapping shape: (2048,)
    query, key, value, batch_valid_length = generate_prefill_flatten_input(bsz, seq_len, head_dim, hidden_size,
                                                                           n_kv_head)
    is_finished = [False] * bsz
    block_tables, slot_mapping = block_mgr.assemble_pa_full_inputs(seq_len,
                                                                   batch_valid_length,
                                                                   is_finished)
    block_tables = Tensor.from_numpy(block_tables)
    slot_mapping = Tensor.from_numpy(slot_mapping)
    # start infer prefill
    prefill_output = infer_attention(query,
                                     key,
                                     value,
                                     batch_valid_length,
                                     block_tables,
                                     slot_mapping,
                                     None,
                                     attn_mask,
                                     None,
                                     None,
                                     None)
    assert prefill_output.shape == (1, bsz * seq_len, hidden_size)

    # start infer decode
    infer_attention.is_first_iteration = False
    # query shape: (8, 1, 10240)
    # key shape: (8, 1, 1024)
    # value shape: (8, 1, 1024)
    # batch_valid_length shape: (8,)
    # block_tables shape: (8, 32)
    # slot_mapping shape: (8,)
    query, key, value = generate_decode_input(bsz, head_dim, hidden_size, n_kv_head)

    block_tables, slot_mapping = block_mgr.assemble_pa_inc_inputs(batch_valid_length, is_finished)
    block_tables = Tensor.from_numpy(block_tables)
    slot_mapping = Tensor.from_numpy(slot_mapping)
    decode_output = infer_attention(query,
                                    key,
                                    value,
                                    batch_valid_length,
                                    block_tables,
                                    slot_mapping,
                                    None,
                                    attn_mask,
                                    None,
                                    None,
                                    None)

    assert decode_output.shape == (bsz, 1, hidden_size)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_infer_attention_with_static_shape_bsh():
    """
    Feature: Test the infer boost attention with TH input layout.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    bsz = 8
    head_num = 80
    seq_len = 512
    max_seq_len = 8192
    head_dim = 128
    n_kv_head = 8
    block_size = 128
    num_blocks = max_seq_len * bsz // block_size
    hidden_size = head_num * head_dim
    is_dynamic = False

    block_mgr = BlockTables(num_blocks, block_size, max_seq_len)
    block_mgr.init_cache_engine(bsz)

    freqs_mgr = FreqsMgr(head_dim=head_dim,
                         seq_length=max_seq_len,
                         max_position_embedding=seq_len)
    infer_attention = InferAttention(head_num,
                                     head_dim,
                                     n_kv_head,
                                     scale_value=1. / math.sqrt(head_dim),
                                     pre_tokens=65536,
                                     next_tokens=0,
                                     block_size=block_size,
                                     num_blocks=num_blocks,
                                     is_dynamic=is_dynamic,
                                     compute_dtype=mstype.float16)

    attn_mask = None
    # query shape: (8, 512, 10240)
    # key shape: (8, 512, 1024)
    # value shape: (8, 512, 1024)
    # batch_valid_length shape: (8,)
    # block_tables shape: (8, 32)
    # slot_mapping shape: (2048,)
    query, key, value, batch_valid_length = generate_prefill_padding_input(bsz, seq_len, head_dim, hidden_size,
                                                                           n_kv_head)
    freqs_cis = freqs_mgr.prefill(bsz, seq_len)
    is_finished = [False] * bsz
    block_tables, slot_mapping = block_mgr.assemble_pa_full_inputs(seq_len,
                                                                   batch_valid_length,
                                                                   is_finished)
    block_tables = Tensor.from_numpy(block_tables)
    slot_mapping = Tensor.from_numpy(slot_mapping)
    # start infer prefill
    prefill_output = infer_attention(query,
                                     key,
                                     value,
                                     batch_valid_length,
                                     block_tables,
                                     slot_mapping,
                                     freqs_cis,
                                     attn_mask,
                                     None,
                                     None,
                                     None)
    assert prefill_output.shape == (bsz, seq_len, hidden_size)

    # start infer decode
    infer_attention.is_first_iteration = False
    batch_valid_length = batch_valid_length + 1
    # query shape: (8, 1, 10240)
    # key shape: (8, 1, 1024)
    # value shape: (8, 1, 1024)
    # batch_valid_length shape: (8,)
    # block_tables shape: (8, 32)
    # slot_mapping shape: (8,)
    query, key, value, = generate_decode_input(bsz, head_dim, hidden_size, n_kv_head)
    freqs_cis = freqs_mgr.increment(batch_valid_length)

    block_tables, slot_mapping = block_mgr.assemble_pa_inc_inputs(batch_valid_length, is_finished)
    block_tables = Tensor.from_numpy(block_tables)
    slot_mapping = Tensor.from_numpy(slot_mapping)
    decode_output = infer_attention(query,
                                    key,
                                    value,
                                    batch_valid_length,
                                    block_tables,
                                    slot_mapping,
                                    freqs_cis,
                                    attn_mask,
                                    None,
                                    None,
                                    None)

    assert decode_output.shape == (bsz, 1, hidden_size)
