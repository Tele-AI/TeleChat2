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
import os
import math
import pytest
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor
from mindformers.modules.infer_attention import InferAttention


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_infer_attention():
    """
    Feature: Test the infer attention.
    Description: Test the forward
    Expectation: No exception
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    bsz, head_num, seq_len, head_dim = 1, 16, 4096, 128
    n_kv_head = 16
    block_size = 1024
    num_blocks = 16
    hidden_size = head_num * head_dim
    query = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    key = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    value = Tensor(np.ones((bsz, seq_len, hidden_size)), mstype.float16)
    batch_valid_length = Tensor(np.ones((bsz, 1)), mstype.int32)
    block_tables = Tensor(np.ones((bsz, num_blocks)), mstype.int64)
    slot_mapping = Tensor(np.ones((bsz * seq_len)), mstype.int32)
    attn_mask = Tensor(np.ones((bsz, 1, seq_len, seq_len)), mstype.uint8)
    freqs_cos = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
    freqs_sin = Tensor(np.ones((seq_len, head_dim)), mstype.float16)
    freqs_cis = (freqs_cos, freqs_sin, None)
    infer_attention = InferAttention(head_num,
                                     head_dim,
                                     n_kv_head,
                                     scale_value=1. / math.sqrt(head_dim),
                                     pre_tokens=65536,
                                     next_tokens=0,
                                     block_size=block_size,
                                     num_blocks=num_blocks,
                                     compute_dtype=mstype.float16)

    output = infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping, freqs_cis, attn_mask)
    assert output.shape == (1, 4096, 2048)
