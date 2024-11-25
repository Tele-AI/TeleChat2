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
"""ChatGLM2 Modules."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer

from mindformers.modules.layers import Linear
from mindformers.version_control import check_rmsnorm_big_kernel_valid, check_valid_big_kernel

from .glm2_config import ChatGLM2Config


class FreqsMgr(nn.Cell):
    r"""freqs_cis manager."""

    def __init__(self,
                 dim,
                 seq_length=None,
                 max_position_embedding=4096,
                 rotary_dtype=mstype.float16,
                 base=10000,
                 rope_ratio=1.0
                 ):
        super().__init__()
        if seq_length is not None and seq_length > max_position_embedding:
            max_position_embedding = seq_length
        self.reshape = P.Reshape()
        base = base * rope_ratio
        theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        seq_idx = np.arange(seq_length, dtype=np.float32)
        idx_theta = np.outer(seq_idx, theta).astype(np.float32)

        cache = Tensor(np.stack((np.cos(idx_theta), np.sin(idx_theta)), axis=-1), dtype=rotary_dtype)

        freqs = np.expand_dims(idx_theta, 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        emb = emb.reshape(seq_length, dim)
        freqs_cos = Tensor(np.concatenate((np.cos(emb), np.ones_like(emb)), axis=-1), dtype=rotary_dtype)
        freqs_sin = Tensor(np.concatenate((np.sin(emb), np.zeros_like(emb)), axis=-1), dtype=rotary_dtype)

        self.head_dim = dim
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.cache = Tensor(cache, dtype=rotary_dtype)

        self.slice = P.StridedSlice().shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.tile = P.Tile().shard(((1, 1),))

    def construct(self, seq_length):
        freqs_cos = self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim * 2), (1, 1))
        freqs_sin = self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim * 2), (1, 1))
        return freqs_cos, freqs_sin, self.cache

    def prefill(self, bsz, seq_length):
        freqs_cos = self.tile(self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim * 2), (1, 1)), (bsz, 1))
        freqs_sin = self.tile(self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim * 2), (1, 1)), (bsz, 1))
        return freqs_cos, freqs_sin, self.cache

    def increment(self, batch_valid_length):
        indices = batch_valid_length - 1
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.cache


class ChatGLM2RMSNorm(nn.Cell):
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

    def __init__(self, dim, eps=1e-6, param_init_type=mstype.float32):
        super(ChatGLM2RMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=param_init_type)
        self.weight = Parameter(initializer('ones', (dim,), dtype=param_init_type))
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

    def _self_norm(self, x):
        # shard:(dp, 1, 1)
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(x, norm_factor)
        output = self.mul2(output, self.weight)
        return output

    def _rms_norm(self, x):
        return self.norm(x, self.weight)[0]

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


class ChatGLM2SiLU(nn.Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """

    def __init__(self):
        super(ChatGLM2SiLU, self).__init__()
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
        if self.self_define:
            self.sigmoid.shard(strategy)
            self.mul.shard((strategy[0], strategy[0]))
        else:
            self.silu.shard(strategy)

    def _self_silu(self, x):
        return self.mul(x, self.sigmoid(x))

    def construct(self, x):
        return self.silu(x)


class ChatGLM2SwiGLU(nn.Cell):
    """SwiGLU activation function."""

    def __init__(self):
        super(ChatGLM2SwiGLU, self).__init__()
        self.split = P.Split(axis=-1, output_num=2)
        self.silu = ChatGLM2SiLU()
        self.mul = P.Mul()

    def construct(self, x):
        x0, x1 = self.split(x)
        return self.mul(self.silu(x0), x1)

    def shard(self, strategy):
        self.split.shard(strategy)
        self.silu.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))


class ChatGLM2MLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2MLP, self).__init__()
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )
        self.dense_h_to_4h.shard(
            strategy_matmul=((config.parallel_config.data_parallel, 1), (config.parallel_config.model_parallel, 1)),
            strategy_bias=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                           (config.parallel_config.model_parallel,)))

        self.activation_func = ChatGLM2SwiGLU()
        # shard need to be checked.
        self.activation_func.shard(((config.parallel_config.data_parallel, 1, 1),))

        # Project back to h.
        self.dense_4h_to_h = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
        )
        self.dense_4h_to_h.shard(
            strategy_matmul=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                             (1, config.parallel_config.model_parallel)),
            strategy_bias=((config.parallel_config.data_parallel, 1), (1,)))

    def construct(self, hidden_states):
        # [bs, seq_len, 4 * hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [bs, seq_len, hidden_size]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output
