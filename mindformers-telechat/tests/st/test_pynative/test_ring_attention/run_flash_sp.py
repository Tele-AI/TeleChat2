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
"""Test Ring Attention."""
import numpy as np

import mindspore as ms
from mindspore.communication import init
from mindspore.common import dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from mindformers.experimental.parallel_core.pynative.context_parallel.utils import get_sp_chuncks
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.context_parallel.flash_sp import FlashSP


def generate_inputs(b, n1, n2, s1, s2, d1, input_layout, dtype, return_tensor=True):
    '''generate inputs'''
    min_value = -1
    max_value = 1
    np.random.seed(42)
    if input_layout == "BSH":
        query = np.random.uniform(min_value, max_value, [b, s1, n1 * d1])
        key = np.random.uniform(min_value, max_value, [b, s2, n2 * d1])
        value = np.random.uniform(min_value, max_value, [b, s2, n2 * d1])
    elif input_layout == "BNSD":
        query = np.random.uniform(min_value, max_value, [b, n1, s1, d1])
        key = np.random.uniform(min_value, max_value, [b, n2, s2, d1])
        value = np.random.uniform(min_value, max_value, [b, n2, s2, d1])
    elif input_layout == "SBH":
        query = np.random.uniform(min_value, max_value, [s1, b, n1 * d1])
        key = np.random.uniform(min_value, max_value, [s2, b, n2 * d1])
        value = np.random.uniform(min_value, max_value, [s2, b, n2 * d1])
    elif input_layout == "BSND":
        query = np.random.uniform(min_value, max_value, [b, s1, n1, d1])
        key = np.random.uniform(min_value, max_value, [b, s2, n2, d1])
        value = np.random.uniform(min_value, max_value, [b, s2, n2, d1])
    elif input_layout == "TND":
        query = np.random.uniform(min_value, max_value, [b * s1, n1, d1])
        key = np.random.uniform(min_value, max_value, [b * s2, n2, d1])
        value = np.random.uniform(min_value, max_value, [b * s2, n2, d1])
    else:
        raise ValueError(f"input_layout is invalid.")
    alibi_mask = None
    prefix = None
    drop_mask = None
    attn_mask = None
    padding_mask = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), alibi_mask, \
            drop_mask, padding_mask, attn_mask, prefix
    return query, key, value, alibi_mask, drop_mask, padding_mask, attn_mask, prefix


def run_flash_sp():
    """
    Feature: Test FlashSP.
    Description: Test FlashSP functional.
    Expectation: Success.
    """
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()

    # Init parameter
    dp = 2
    sp = 4
    initialize_model_parallel(context_parallel_size=sp, order="cp-dp")

    bs = 16
    n = 8
    s = 4096
    hidden_size = 16
    test_layout = "BSH" # only "BSH" is supported
    test_mask_type = "causal"  # only "full" is supported
    test_dtype = mstype.float16 # or mstype.bfloat16
    query_output, key_output, value_output, alibi_mask_output, drop_mask_output, \
        padding_mask_output, ring_attn_mask_output, prefix_out = generate_inputs(bs, n, n, s, s,
                                                                                 hidden_size, test_layout,
                                                                                 test_dtype)

    if test_layout == "SBH":
        seq_dim_ = 0
    elif test_layout == "BSH":
        seq_dim_ = 1
    elif test_layout == "BNSD":
        seq_dim_ = 2
    q2 = get_sp_chuncks(query_output, test_layout, enable_flash_sp=True)
    k2 = get_sp_chuncks(key_output, test_layout, enable_flash_sp=True)
    v2 = get_sp_chuncks(value_output, test_layout, enable_flash_sp=True)

    flash_sp = FlashSP(head_num=n,
                       input_layout=test_layout,
                       dp=dp,
                       sp=sp)
    flash_sp_output = flash_sp(q2, k2, v2, ring_attn_mask_output, alibi_mask_output, prefix_out,
                               padding_mask_output, test_mask_type)

    flash_attn_mask = ops.ones((query_output.shape[seq_dim_], key_output.shape[seq_dim_]), dtype=mstype.uint8)
    flash_attn_mask = ops.triu(flash_attn_mask, diagonal=1)
    flash_attention = FlashAttentionScore(head_num=n,
                                          input_layout=test_layout)
    _, _, _, flash_attention_output = flash_attention(query_output,
                                                      key_output,
                                                      value_output,
                                                      alibi_mask_output,
                                                      drop_mask_output,
                                                      padding_mask_output,
                                                      flash_attn_mask)

    flash_attention_output = get_sp_chuncks(
        flash_attention_output, test_layout, enable_flash_sp=True)

    flash_attention_output = ms.ops.cast(flash_attention_output, mstype.float16)
    flash_sp_output = ms.ops.cast(flash_sp_output, mstype.float16)
    assert np.allclose(flash_attention_output.asnumpy(), flash_sp_output.asnumpy(), 0.004, 0.004)
    print("end test.")


if __name__ == "__main__":
    run_flash_sp()
