# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test local block sparse attention accuracy."""
import math
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
import mindspore.ops.operations as P

from mindformers.modules.local_block_sparse_attention import LocalBlockSparseAttention

ms.set_context(mode=0)


class SparseAttention(nn.Cell):
    """ A Local Sparse Attention"""

    def __init__(self, seq_len, size_per_head, local_size, block_size):
        super(SparseAttention, self).__init__()
        self.seq_len = seq_len
        self.local_size = local_size
        self.block_size = block_size
        self.scale_factor = Tensor(math.sqrt(math.sqrt(size_per_head)))
        self.real_div = P.RealDiv()
        self.qk_bmm = P.BatchMatMul(transpose_b=True)
        self.mask_mul = P.Mul()
        self.mask_expand_dims = P.ExpandDims()
        self.mask_add = P.Add()
        self.softmax = P.Softmax()
        self.pv_bmm = ms.ops.BatchMatMul()
        self.multiply_data = Tensor([-10000.0], dtype=ms.float16)
        sparse_mask = self.gen_sparse_mask()
        self.sparse_mask = Tensor(sparse_mask, ms.float16)

    def gen_sparse_mask(self):
        """generate a mask for sparse attention"""
        sparse_mask = np.ones((self.seq_len, self.seq_len))
        block_groups = self.seq_len // self.block_size
        prev_block_num = self.local_size // self.block_size - 1
        for i in range(block_groups):
            row_begin = i * self.block_size
            row_end = row_begin + self.block_size
            col_begin = max(0, i - prev_block_num) * self.block_size
            col_end = col_begin + self.local_size
            sparse_mask[row_begin:row_end, col_begin:col_end] = 0
        return sparse_mask

    def construct(self, q, k, v, attn_mask=None):
        """forward process"""
        factor = P.Cast()(self.scale_factor, P.DType()(q))
        q = self.real_div(q, factor)
        k = self.real_div(k, factor)
        sim = self.qk_bmm(q, k)
        sparse_mask = self.mask_mul(self.sparse_mask, self.multiply_data)
        sparse_mask = self.mask_expand_dims(sparse_mask, 0)
        sparse_mask = self.mask_expand_dims(sparse_mask, 1)
        sim = self.mask_add(sparse_mask, sim)
        if attn_mask is not None:
            adder = self.mask_mul(attn_mask, self.multiply_data)
            adder = self.mask_expand_dims(adder, 1)
            sim = self.mask_add(adder, sim)
        probs = self.softmax(sim)
        out = self.pv_bmm(probs, v)
        return out


def data_compare(ground_truth,
                 predict,
                 diff_thd=0.001,
                 pct_thd=0.001,
                 max_diff_thd=0.1):
    """compare ground truth with predict value"""
    total_count = np.prod(ground_truth.shape)
    greater_than_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") -
               ground_truth.astype("float32")) > diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9))
    greater_than_max_diff_thd_count = np.sum(
        np.abs(predict.astype("float32") -
               ground_truth.astype("float32")) > max_diff_thd *
        (np.abs(ground_truth.astype("float32")) + 1e-9))

    diff_gt_thd_proportion = greater_than_diff_thd_count / total_count
    diff_gt_max_thd_proportion = greater_than_max_diff_thd_count / total_count
    result = "Pass"
    if diff_gt_thd_proportion > pct_thd or diff_gt_max_thd_proportion > 0:
        result = "Failed"
    return result


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_local_block_sparse_attention():
    """
    Feature: Local Block Sparse Attention
    Description: Test Local Block Sparse Attention Accuracy
    Expectation: result == "Pass"
    """
    input_shape = (4, 32, 4096, 128)
    local_size = 1024
    block_size = 128
    q = Tensor(np.random.random(input_shape).astype("float16"))
    k = Tensor(np.random.random(input_shape).astype("float16"))
    v = Tensor(np.random.random(input_shape).astype("float16"))
    batch_size, seq_len = q.shape[0], q.shape[2]
    att_mask = Tensor(1.0 - np.tril(
        np.ones(shape=(batch_size, seq_len, seq_len), dtype=np.float16)))

    size_per_head = q.shape[-1]
    module_ori = LocalBlockSparseAttention(seq_len, size_per_head, local_size,
                                           block_size, dropout_rate=0)
    module_local = SparseAttention(seq_len, size_per_head, local_size, block_size)

    out1 = module_ori(q, k, v, att_mask).asnumpy()
    out2 = module_local(q, k, v, att_mask).asnumpy()

    result = data_compare(out1, out2)
    assert result == "Pass"
