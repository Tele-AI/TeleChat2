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
"""YiZhao Modules."""
from typing import Tuple

import numpy as np
from mindspore import nn, Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from yizhao_config import YiZhaoConfig

from mindformers.modules.layers import Linear
from mindformers.version_control import check_rmsnorm_big_kernel_valid, check_valid_big_kernel


class YiZhaoFreqsMgr(nn.Cell):
    r"""freqs_cis manager."""

    def __init__(self,
                 dim,
                 seq_length=None,
                 rotary_dtype=mstype.float16,
                 base=10000,
                 rope_ratio=1.0
                 ):
        super().__init__()
        self.reshape = P.Reshape()
        base = base * rope_ratio
        theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        seq_idx = np.arange(seq_length, dtype=np.float32)
        idx_theta = np.outer(seq_idx, theta).astype(np.float32)

        freqs = np.expand_dims(idx_theta, 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        emb = emb.reshape(seq_length, dim)
        freqs_cos = np.concatenate((np.cos(emb), np.ones_like(emb)), axis=-1)
        freqs_sin = np.concatenate((np.sin(emb), np.zeros_like(emb)), axis=-1)
        swap_mask = YiZhaoFreqsMgr.get_swap_mask_llama(dim * 2)
        self.seq_length = seq_length
        self.dim = dim

        def rearange(w):
            """
            Args:
                w: weight
            """
            w = np.concatenate(
                [
                    w[..., 0::2],
                    w[..., 1::2],
                ],
                axis=-1
            )
            return w

        freqs_cos = rearange(freqs_cos)
        freqs_sin = rearange(freqs_sin)

        self.head_dim = dim
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.swap_mask = Tensor(swap_mask, dtype=rotary_dtype)

        self.slice = P.StridedSlice().shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.tile = P.Tile().shard(((1, 1),))

    def construct(self, seq_length):
        """Construct for YiZhaoFreqsMgr"""
        freqs_cos = self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim * 2), (1, 1)).reshape(
            (self.seq_length, 1, 2 * self.dim))
        freqs_sin = self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim * 2), (1, 1)).reshape(
            (self.seq_length, 1, 2 * self.dim))
        return freqs_cos, freqs_sin, self.swap_mask

    def prefill(self):
        """ Prefill for YiZhaoFreqsMgr """
        return self.freqs_cos, self.freqs_sin, self.swap_mask

    def increment(self, batch_valid_length):
        """ Increment for YiZhaoFreqsMgr"""
        indices = batch_valid_length - 1
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask_llama(head_dim):
        """Swap matrix"""
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])


class YiZhaoRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32, compute_type=mstype.float32):
        super(YiZhaoRMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)

        if param_init_type == mstype.bfloat16:
            self.weight = Parameter(initializer('ones', (dim,), dtype=mstype.float32).astype(mstype.bfloat16))
        else:
            self.weight = Parameter(initializer('ones', (dim,), dtype=param_init_type))
        self.compute_type = compute_type
        if not check_rmsnorm_big_kernel_valid():
            self.square = P.Square()
            self.mean = P.ReduceMean(keep_dims=True)
            self.add = P.Add()
            self.rsqrt = P.Rsqrt()
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.rms_norm = self._self_norm
            self.self_define = True
        else:
            self.norm = P.RmsNorm(float(eps))
            self.rms_norm = self._rms_norm
            self.self_define = False
            self.cast = P.Cast()
            self.rcast = P.Cast()

    def _self_norm(self, x):
        """ Self norm for RMSNorm"""
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, self.cast(norm_factor, original_type))
        output = self.mul2(output, self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        """Rms norm"""
        original_type = x.dtype
        x = self.cast(x, self.compute_type)
        output = self.norm(x, self.cast(self.weight, self.compute_type))[0]
        return self.rcast(output, original_type)

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)

    def shard(self, strategy):
        """Parallel strategy configuratiuon interface."""
        if self.self_define:
            self.square.shard(strategy)
            self.mean.shard(strategy)
            self.rsqrt.shard(strategy)
            self.add.shard((strategy[0], ()))
            self.mul.shard((strategy[0], strategy[0]))
            self.mul2.shard((strategy[0], (1,)))
        else:
            self.norm.shard((strategy[0], (1,)))


class YiZhaoSiLU(nn.Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """

    def __init__(self):
        super(YiZhaoSiLU, self).__init__()
        if check_valid_big_kernel():
            # pylint: disable=W0212
            self.silu = P._inner_ops.SiLU()
            self.self_define = False
        else:
            self.sigmoid = P.Sigmoid()
            self.mul = P.Mul()
            self.silu = self._self_silu
            self.self_define = True

    def shard(self, strategy):
        """sharding for SiLU"""
        if self.self_define:
            self.sigmoid.shard(strategy)
            self.mul.shard((strategy[0], strategy[0]))
        else:
            self.silu.shard(strategy)

    def _self_silu(self, x):
        """ mul sigmoid for SiLU"""
        return self.mul(x, self.sigmoid(x))

    def construct(self, x):
        """Construct for SiLU"""
        return self.silu(x)


class YiZhaoSwiGLU4ConcatMLP(nn.Cell):
    """SwiGLU activation function for MLP concat."""

    def __init__(self):
        super(YiZhaoSwiGLU4ConcatMLP, self).__init__()
        self.split = P.Split(axis=-1, output_num=2)
        self.silu = YiZhaoSiLU()
        self.mul = P.Mul()

    def construct(self, x):
        """construct for SwiGLU activation function"""
        x0, x1 = self.split(x)
        return self.mul(self.silu(x0), x1)

    def shard(self, strategy):
        """Shard for SwiGLU"""
        self.split.shard(strategy)
        self.silu.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))


class YiZhaoSwiGLU(nn.Cell):
    """SwiGLU activation function."""
    def __init__(self):
        super(YiZhaoSwiGLU, self).__init__()
        self.silu = YiZhaoSiLU()
        self.mul = P.Mul()

    def construct(self, left, right):
        """construct for SwiGLU activation function"""
        return self.mul(self.silu(left), right)

    def shard(self, strategy):
        """Shard for SwiGLU"""
        self.silu.shard(strategy)
        self.mul.shard(strategy * 2)


class YiZhaoConcatMLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: YiZhaoConfig):
        super(YiZhaoConcatMLP, self).__init__()
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )
        self.dense_h_to_4h.shard(
            strategy_matmul=((dp * cp, 1), (mp, 1)),
            strategy_bias=((dp * cp, mp),
                           (mp,)))

        self.activation_func = YiZhaoSwiGLU4ConcatMLP()
        # shard need to be checked.
        self.activation_func.shard(((dp, cp, 1),))

        # Project back to h.
        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )
        self.dense_4h_to_h.shard(
            strategy_matmul=((dp * cp, mp), (1, mp)),
            strategy_bias=((dp * cp, 1), (1,)))

    def construct(self, hidden_states):
        """Construct of concat MLP"""
        # [bs, seq_len, 4 * hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [bs, seq_len, hidden_size]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class YiZhaoMLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    def __init__(self, config: YiZhaoConfig):
        super(YiZhaoMLP, self).__init__()
        self.add_bias = config.add_bias_linear
        self.dense_left = Linear(
            config.hidden_size,
            config.ffn_hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype
        )
        self.dense_right = Linear(
            config.hidden_size,
            config.ffn_hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype
        )
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel
        self.dense_left.shard(strategy_matmul=((dp * cp, 1), (mp, 1)), strategy_bias=((dp * cp, mp), (mp,)))
        self.dense_right.shard(strategy_matmul=((dp * cp, 1), (mp, 1)), strategy_bias=((dp * cp, mp), (mp,)))

        self.activation_func = YiZhaoSwiGLU()
        self.activation_func.shard(((dp, cp, mp),))

        # Project back to h.
        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )
        self.dense_4h_to_h.shard(strategy_matmul=((dp * cp, mp), (1, mp)), strategy_bias=((dp * cp, 1), (1,)))

    def construct(self, hidden_states):
        """Construct for MLP"""
        # [bs, seq_len, 4 * hidden_size]
        intermediate_left = self.dense_left(hidden_states)
        intermediate_right = self.dense_right(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_left, intermediate_right)
        # [bs, seq_len, hidden_size]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class YiZhaoRotaryEmbedding(nn.Cell):
    """Embedding Layer"""
    def __init__(self, compute_dtype=mstype.float32):
        super(YiZhaoRotaryEmbedding, self).__init__()
        self.mul = P.Mul()
        self.bmm_swap = P.BatchMatMul()
        self.add = P.Add()
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = compute_dtype

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  rotary_pos_emb: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Construct for Embedding Layer"""
        # freqs_cos, freqs_sin: [seq_len, head_dim]
        # swap_mask: [head_dim, head_dim]
        freqs_cos, freqs_sin, swap_mask = rotary_pos_emb
        # query, key: [bs, n_head/n_kv_head, seq/1, head_dim]
        original_dtype = query.dtype
        query = self.cast(query, self.dtype)
        key = self.cast(key, self.dtype)
        query = self.add(self.mul(query, freqs_cos),
                         self.mul(self.bmm_swap(query, swap_mask), freqs_sin))
        key = self.add(self.mul(key, freqs_cos),
                       self.mul(self.bmm_swap(key, swap_mask), freqs_sin))
        query = self.cast(query, original_dtype)
        key = self.cast(key, original_dtype)
        return query, key

    def shard(self, strategy):
        """sharding for Embedding"""
        self.add.shard((strategy, strategy))
        self.bmm_swap.shard((strategy, (1, 1)))
        self.mul.shard((strategy, (1, 1)))


class YiZhaoRotaryEmbeddingOpt(nn.Cell):
    """YiZhaoRotaryEmbeddingOpt"""
    def __init__(self, compute_dtype=mstype.float32):
        super(YiZhaoRotaryEmbeddingOpt, self).__init__()
        self.mul = P.Mul()
        self.bmm_swap = P.BatchMatMul()
        self.add = P.Add()
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = compute_dtype

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  rotary_pos_emb: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """construct"""
        # freqs_cos, freqs_sin: [seq_len, head_dim]
        # swap_mask: [head_dim, head_dim]
        freqs_cos, freqs_sin, swap_mask = rotary_pos_emb
        # b, s, n, d
        _, s, _, d = query.shape
        # query, key: [bs, n_head/n_kv_head, seq/1, head_dim]

        original_dtype = query.dtype
        query = self.cast(query, self.dtype)
        key = self.cast(key, self.dtype)

        freqs_cos = F.reshape(freqs_cos, (s, 1, d))
        freqs_sin = F.reshape(freqs_sin, (s, 1, d))
        query = self.add(self.mul(query, freqs_cos),
                         self.mul(self.bmm_swap(query, swap_mask), freqs_sin))
        key = self.add(self.mul(key, freqs_cos),
                       self.mul(self.bmm_swap(key, swap_mask), freqs_sin))
        query = self.cast(query, original_dtype)
        key = self.cast(key, original_dtype)
        return query, key

    def shard(self, strategy):
        _, cp, _, _ = strategy  # dp cp mp
        self.add.shard((strategy, strategy))
        self.bmm_swap.shard((strategy, (1, 1)))
        self.mul.shard((strategy, (cp, 1, 1)))


class GetCompressMask(nn.Cell):
    """GetCompressMask"""

    # pylint: disable=W0613
    def __init__(self, mask_length, parallel_config):
        super(GetCompressMask, self).__init__()
        self.mask_length = mask_length
        tril_dev = np.tril(np.ones((self.mask_length, self.mask_length), dtype=np.int8))
        attention_mask = np.ones((self.mask_length, self.mask_length), dtype=np.int8)
        attention_mask = attention_mask - tril_dev
        self.attention_mask = Tensor(attention_mask, dtype=mstype.uint8)
        self.cast = P.Cast()

    # pylint: disable=W0613
    def construct(self, sequence_start_ids):
        mask = self.cast(self.attention_mask, mstype.uint8)
        return mask


class GetEodResetMask(nn.Cell):
    """GetEodResetMask"""
    def __init__(self, seq_length, parallel_config):
        super(GetEodResetMask, self).__init__()
        dp = parallel_config.data_parallel
        self.seq_length = seq_length
        self.expand_dims = P.ExpandDims().shard(((dp, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.equal = P.Equal().shard(((dp, 1, 1,), (dp, 1, 1)))
        self.tril_op = P.Tril().shard(((dp, 1, 1,),))
        self.sub = P.Sub().shard(((), (dp, 1, 1),))

    def construct(self, eod_vec):
        eod_vec_row = self.expand_dims(eod_vec, 1)
        eod_vec_column = self.expand_dims(eod_vec, 2)
        eod_matrix_1 = self.tile(eod_vec_row, (1, self.seq_length, 1))
        eod_matrix_2 = self.tile(eod_vec_column, (1, 1, self.seq_length))
        eod_matrix = self.equal(eod_matrix_1, eod_matrix_2)
        eod_matrix = F.cast(eod_matrix, mstype.uint8)
        mask = self.tril_op(eod_matrix)
        mask = self.sub(1, mask)
        return mask
