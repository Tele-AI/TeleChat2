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
"""
A Local Block Sparse Attention.
"""
from __future__ import absolute_import

from functools import wraps, partial
import inspect
import math
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore._extends import cell_attr_register
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, OpParallelConfig

__all__ = ["LocalBlockSparseAttention"]

gather_index_inited = False
kv_index = None
mask_index = None


def _args_type_validator_check(*type_args, **type_kwargs):
    """Check whether input data type is correct."""

    def type_check(func):
        sig = inspect.signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal bound_types
            bound_values = sig.bind(*args, **kwargs)

            argument_dict = bound_values.arguments
            if "kwargs" in bound_types:
                bound_types = bound_types["kwargs"]
            if "kwargs" in argument_dict:
                argument_dict = argument_dict["kwargs"]
            for name, value in argument_dict.items():
                if name in bound_types:
                    bound_types[name](value, name)
            return func(*args, **kwargs)

        return wrapper

    return type_check


def _valid_type_checks(types, class_name):
    """types should be a list of types, this function check if the type is in the valid dtypes"""
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name,
                                valid_types=types,
                                prim_name=class_name)
        return partial_check(name, type(value))

    return validator_check_func


def _valid_value_checks(types, class_name):
    """the value should be a list of types, this function check if the value is in the valid dtypes"""
    def validator_check_func(value, name):
        # The args of Validator.check_type_name is (arg_name, arg_type, valid_types, prim_name)
        # as the input of _args_type_validator_check is fixed, so we need to manually change the input order
        partial_check = partial(Validator.check_type_name,
                                valid_types=types,
                                prim_name=class_name)
        return partial_check(name, value)

    return validator_check_func


class InitGatherIndex:
    """
    A self-defined function to init the index of gather operation
    Args:
        seq_len(int): Length of input sequence.
        local_size(int): An integer determining the local size. Current implementation of sparse self-attention
            is based on local sparse matrices, reference sliding window attention in Longformer
        block_size(int): An integer determining the block size. Current implementation of sparse self-attention
            is based on blocked sparse matrices, reference block sparse attention in FlashAttention
    """

    def __init__(self, seq_len, local_size, block_size):
        self.seq_len = seq_len
        self.local_size = local_size
        self.block_size = block_size
        self.group_num = self.seq_len // self.block_size
        self.local_block_num = self.local_size // self.block_size
        self.prev_block_num = self.local_block_num - 1

    def gen_kv_gather_index(self):
        index = np.ones((self.group_num * self.local_size))
        for i in range(self.group_num):
            start = max(0, i - self.prev_block_num) * self.block_size
            end = (max(0, i - self.prev_block_num) +
                   self.local_size // self.block_size) * self.block_size
            index[i * self.local_size:(i + 1) * self.local_size] = np.arange(
                start, end)
        return index

    def gen_mask_gatherd_index(self):
        index = np.ones((1, self.seq_len, self.local_size))
        for i in range(self.group_num):
            start = max(0, i - self.prev_block_num) * self.block_size
            end = (max(0, i - self.prev_block_num) +
                   self.local_block_num) * self.block_size
            index[:, i * self.block_size:(i + 1) *
                  self.block_size, :] = np.arange(start, end)
        return index


def init_gather_index(seq_len, local_size, block_size):
    gather_index = InitGatherIndex(seq_len, local_size, block_size)
    global kv_index
    kv_index = Tensor(gather_index.gen_kv_gather_index(), mstype.int32)
    global mask_index
    mask_index = Tensor(gather_index.gen_mask_gatherd_index(), mstype.int32)


class LocalBlockSparseAttention(nn.Cell):
    """
    Local Block Sparse Attention Layer.

    This function contains the sliding window attention primitives used in Longformer (see paper)
    `Longformer: The Long-Document Transformer <https://arxiv.org/pdf/2004.05150.pdf>` and block
    sparse attention primitives used in Longformer FlashAttention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

    Specifically, it includes the following:

    1. A faster implementation of normal attention (the entire area of the upper triangle and part of the area of the
        lower triangle are not computed, and many operations are fused).
    2. An implementation of "local" and "block" sparse attention, as in the Longformer and FlashAttention.

    Args:
        seq_len(int): Length of input sequence.
        size_per_head(int): The hidden size of the input.
        local_size(int): An integer determining the local size. Current implementation of sparse self-attention
            is based on local sparse matrices, reference sliding window attention in Longformer.
        block_size(int): An integer determining the block size. Current implementation of sparse self-attention
            is based on blocked sparse matrices, reference sliding window attention in FlashAttention.
        dropout_rate(float): The dropout rate of the attention score. Default 0.1.
        softmax_compute_type(mstype.Number): The type of softmax computation module. Default mstype.float16.
            Should be mstype.float32 or mstype.float16.
        parallel_config(OpParallelConfig): The config of parallel setting, see `OpParallelConfig`.
            Default `default_dpmp_config`, an instance of `OpParallelConfig` with default args.

    Inputs:
      - **q** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, head_num, seq_length, size_per_head])
      - **k** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, head_num, seq_length, size_per_head])
      - **v** (Tensor) - Tensor value (:class:`mstype.fp16` [batch_size, head_num, seq_length, size_per_head])
      - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp16` [batch_size, seq_length,
          seq_length]): Lower triangular matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, head_num, seq_length, size_per_head]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindformers.modules import LocalBlockSparseAttention
        >>> from mindspore import Tensor
        >>> model = LocalBlockSparseAttention(seq_len=4096,
        ...                                   size_per_head=128,
        ...                                   local_size=1024,
        ...                                   block_size=128)
        >>> q = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> k = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> v = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.ones((2, 4096, 4096)), mstype.float16)
        >>> output = model(q, k, v, attention_mask)
        >>> print(output.shape)
        (2, 16, 4096, 128)
    """

    @cell_attr_register
    @_args_type_validator_check(
        seq_len=Validator.check_positive_int,
        size_per_head=Validator.check_positive_int,
        local_size=Validator.check_positive_int,
        block_size=Validator.check_positive_int,
        softmax_compute_type=_valid_value_checks(
            [mstype.float16, mstype.float32], "LocalBlockSparseAttention"),
        parallel_config=_valid_type_checks([OpParallelConfig],
                                           "LocalBlockSparseAttention"))
    def __init__(self,
                 seq_len,
                 size_per_head,
                 local_size,
                 block_size,
                 dropout_rate=0.1,
                 softmax_compute_type=mstype.float16,
                 parallel_config=default_dpmp_config):
        super(LocalBlockSparseAttention, self).__init__()
        if seq_len % block_size != 0:
            raise ValueError(
                f"block_size must be divisible by seq_len, "
                f"but got block_size {block_size} is not divisible by seq_len {seq_len}"
            )
        if local_size % block_size != 0:
            raise ValueError(
                f"block_size must be divisible by local_size, "
                f"but got block_size {block_size} is not divisible by local_size {local_size}"
            )
        if local_size >= seq_len:
            raise ValueError(
                f"local_size must be less than seq_len, "
                f"but got local_size {local_size} is not less than seq_len {seq_len}"
            )
        if block_size > local_size:
            raise ValueError(
                f"block_size must be less than (or equal to) local_size, "
                f"but got block_size {local_size} is not less than (or equal to) local_size {local_size}"
            )
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(
                f"dropout probability should be a number in range [0, 1), but got {dropout_rate}"
            )
        global gather_index_inited
        if not gather_index_inited:
            init_gather_index(seq_len, local_size, block_size)
            gather_index_inited = True
        self.scale_factor = Tensor(math.sqrt(math.sqrt(size_per_head)))
        self.local_size = local_size
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.softmax_compute_type = softmax_compute_type
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.real_div = P.RealDiv().shard(((dp, mp, 1, 1), ()))
        self.qk_bmm = ms.ops.BatchMatMul(transpose_b=True).shard(
            ((dp, mp, 1, 1, 1), (dp, mp, 1, 1, 1)))
        self.attn_mask_mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.attn_mask_expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.attn_mask_add = P.Add().shard(
            ((dp, 1, 1, 1, 1), (dp, mp, 1, 1, 1)))
        self.softmax = P.Softmax().shard(((dp, mp, 1, 1, 1),))
        if self.dropout_rate > 1e-5:
            self.attention_dropout = nn.Dropout(keep_prob=1 -
                                                self.dropout_rate)
            self.attention_dropout.dropout.shard(((dp, mp, 1, 1, 1),))
        self.pv_bmm = ms.ops.BatchMatMul().shard(
            ((dp, mp, 1, 1, 1), (dp, mp, 1, 1, 1)))
        self.multiply_data = Tensor([-10000.0],
                                    dtype=self.softmax_compute_type)

        self.reshape = P.Reshape()
        self.gather1 = P.Gather().shard(((dp, mp, 1, 1), (1,)))
        self.gather2 = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))

    def construct(self, q, k, v, attention_mask=None):
        """Forward process"""
        bsz, head_num, seq_len, head_dim = q.shape

        factor = P.Cast()(self.scale_factor, P.DType()(q))
        q = self.real_div(q, factor)
        k = self.real_div(k, factor)
        q = self.reshape(q, (bsz, head_num, seq_len // self.block_size, self.block_size, head_dim))
        # kv: [bsz, head_num, seq_len, head_dim] -> [bsz, head_num, seq_len // block_size, local_size, head_dim]
        k = self.gather1(k, kv_index, 2)
        k = self.reshape(k, (bsz, head_num, -1, self.local_size, head_dim))
        v = self.gather1(v, kv_index, 2)
        v = self.reshape(v, (bsz, head_num, -1, self.local_size, head_dim))
        # q * k.T
        score = self.qk_bmm(q, k)

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, self.softmax_compute_type)

        # softmax
        if attention_mask is not None:
            mask_index_ = P.tile(mask_index, (bsz, 1, 1))
            attention_mask = self.gather2(
                attention_mask, 2, mask_index_)  # [bsz, seq_len, local_size]
            adder = self.attn_mask_mul(
                P.Cast()(attention_mask, self.softmax_compute_type),
                self.multiply_data)
            adder = self.attn_mask_expand_dims(adder, 1)
            adder = self.reshape(
                adder, (bsz, 1, -1, self.block_size, self.local_size))
            score = self.attn_mask_add(adder, score)
        probs = self.softmax(score)
        probs = P.Cast()(probs, ori_dtype)
        if self.dropout_rate > 1e-5:
            probs = self.attention_dropout(probs)
        # p * v
        output = self.pv_bmm(probs, v)
        output = self.reshape(output, (bsz, head_num, seq_len, head_dim))
        return output
