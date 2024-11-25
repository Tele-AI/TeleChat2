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
import math
import numpy as np

import mindspore as ms
from mindspore.communication import init
from mindspore.common import dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.auto_generate.gen_ops_prim import FlashAttentionScore

from mindformers.experimental.parallel_core.pynative.context_parallel.utils import get_sp_chuncks, \
    get_sp_chuncks_attn_mask_general, get_sp_chuncks_general
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.context_parallel.ring_attention import RingAttention


def generate_inputs(b, n1, n2, s1, s2, d1, input_layout,
                    dtype, return_tensor=True):
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


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Statistics error location and ratio"""
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(
        data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    print(
        "data_expected_std:{0}\ndata_me_error:{1}\ngap:{2}\nerror_percent_num:{3}\nerror_percent_max:{4}".format(
            data_expected[greater],
            data_me[greater],
            error[greater],
            str(loss_count / total_count),
            np.max(np.abs(error[greater] / data_expected[greater])),
        )
    )
    assert (
        loss_count / total_count
    ) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\ngap:{2}\nerror_percent:{3}".format(
        data_expected[greater],
        data_me[greater],
        error[greater],
        str(loss_count / total_count),
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol,
                       atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def test_ring_attention():
    """
    Feature: Test RingAttention.
    Description: Test RingAttention functional.
    Expectation: Success.
    """
    ms.set_context(device_target="Ascend",
                   mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()

    # Init parameter
    # dp = 1 # Automatically calculated
    sp = 8
    initialize_model_parallel(context_parallel_size=sp)

    bs = 16
    n = 8
    s = 4096
    hidden_size = 128
    scale = 1.0 / math.sqrt(hidden_size)
    test_layout = "BSH"  # or "BNSD", "SBH"
    test_mask_type = "causal" # or "full", "user_defined"
    test_dtype = mstype.float16  # or mstype.bfloat16
    query_output, key_output, value_output, alibi_mask_output, drop_mask_output, \
        padding_mask_output, ring_attn_mask_output, prefix_out = generate_inputs(bs, n, n, s, s,
                                                                                 hidden_size, test_layout,
                                                                                 test_dtype)
    if test_layout == "SBH":
        # batch_dim_ = 1
        seq_dim_ = 0
    elif test_layout == "BSH":
        # batch_dim_ = 0
        seq_dim_ = 1
    elif test_layout == "BNSD":
        # batch_dim_ = 0
        seq_dim_ = 2
    if test_mask_type == "causal":
        q2 = get_sp_chuncks(query_output, test_layout,
                            enable_dp_shard=True, enable_flash_sp=False)
        k2 = get_sp_chuncks(key_output, test_layout,
                            enable_dp_shard=True, enable_flash_sp=False)
        v2 = get_sp_chuncks(value_output, test_layout,
                            enable_dp_shard=True, enable_flash_sp=False)
    else:
        q2 = get_sp_chuncks_general(query_output, test_layout)
        k2 = get_sp_chuncks_general(key_output, test_layout)
        v2 = get_sp_chuncks_general(value_output, test_layout)

    if test_mask_type == "user_defined":
        np.random.seed(112)
        ring_attn_mask = Tensor(np.random.randint(0, 2, size=(s, s)))
        ring_attn_mask = ring_attn_mask.astype(ms.uint8)

        ring_attn_mask_output = get_sp_chuncks_attn_mask_general(ring_attn_mask)
    ring_attention = RingAttention(head_num=n,
                                   input_layout=test_layout,
                                   scale_value=scale)

    if test_mask_type == "user_defined":
        ring_attention_output = ring_attention(q2, k2, v2, ring_attn_mask_output, alibi_mask_output, prefix_out,
                                               padding_mask_output)
    else:
        ring_attention_output = ring_attention(q2, k2, v2, ring_attn_mask_output, alibi_mask_output, prefix_out,
                                               padding_mask_output, test_mask_type)

    if test_mask_type == "full":
        flash_attn_mask = None
    elif test_mask_type == "causal":
        flash_attn_mask = ops.ones(
            (query_output.shape[seq_dim_], key_output.shape[seq_dim_]), dtype=mstype.uint8)
        flash_attn_mask = ops.triu(flash_attn_mask, diagonal=1)
    else:
        flash_attn_mask = ring_attn_mask
    flash_attention = FlashAttentionScore(
        head_num=n, input_layout=test_layout, scale_value=scale)
    _, _, _, flash_attention_output = flash_attention(query_output,
                                                      key_output,
                                                      value_output,
                                                      alibi_mask_output,
                                                      drop_mask_output,
                                                      padding_mask_output,
                                                      flash_attn_mask)

    if test_mask_type == "causal":
        flash_attention_output = get_sp_chuncks(
            flash_attention_output, test_layout, enable_dp_shard=True, enable_flash_sp=False)
    else:
        flash_attention_output = get_sp_chuncks_general(flash_attention_output, test_layout)
    ring_attention_output = ms.ops.cast(ring_attention_output, mstype.float16)

    tols = dict(atol=1e-3, rtol=1e-3)
    allclose_nparray(
        flash_attention_output.asnumpy(),
        ring_attention_output.asnumpy(),
        **tols,
        equal_nan=True)

    print("Test passed!", flush=True)


if __name__ == "__main__":
    test_ring_attention()
