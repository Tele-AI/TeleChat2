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
"""Rotary position embedding."""

__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]

import mindspore as ms
from mindspore import Tensor, ops, mint

from .module import Module


class RotaryEmbedding(Module):
    r"""
    Rotary positional embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Default: 1.0.
        rotary_interleaved (bool, optional): Determines the method of applying rotary embeddings to the input
            dimensions. Default: False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Default: None.
        rotary_base (int, optional): Base period for rotary position embeddings. Default: 10000.

    Inputs:
        - **max_seq_len** (int) - Max sequence length of inputs.
        - **offset** (int) - The starting point for the position encoding.

    Outputs:
        - **emb** (Tensor) - Embeddings after applying RoPE.

    Raises:
        NotImplementedError: If `rotary_interleaved` is True.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore import ops
        >>> from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
        >>> from mindformers.experimental.graph.transformer.rotary_pos_embedding import (
        ...     RotaryEmbedding,
        ...     apply_rotary_pos_emb
        ... )
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> rank_id = os.environ.get('RANK_ID')
        >>> if rank_id is not None:
        >>>     ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
        >>>     init()
        >>> seed_value = 42
        >>> ms.set_seed(seed_value)
        >>> np.random.seed(seed_value)
        >>> class MyAttention(nn.Cell):
        >>>     def __init__(self, config: TransformerConfig):
        >>>         super(MyAttention, self).__init__()
        >>>         self.config = config
        >>>     def construct(self, x, freqs):
        >>>         return apply_rotary_pos_emb(x, freqs, self.config)
        >>> class MyNet(nn.Cell):
        >>>     def __init__(self, config: TransformerConfig):
        >>>         super(MyNet, self).__init__()
        >>>         self.n_heads = config.num_attention_heads
        >>>         self.head_dim = dim // self.n_heads
        >>>         self.rotary_embedding = RotaryEmbedding(self.head_dim)
        >>>         self.attention = MyAttention(config)
        >>>         dp = config.data_parallel
        >>>         self.transpose = ops.Transpose().shard(((dp, 1, 1, 1),))
        >>>         self.transpose_back = ops.Transpose().shard(((dp, 1, 1, 1),))
        >>>         self.reshape = ops.Reshape()
        >>>     def construct(self, x: Tensor):
        >>>         bs_, seq_len_, dim_ = x.shape
        >>>         # [bs, seq_len, dim] -> [bs, seq_len, heads, head_dim]
        >>>         x = self.reshape(x, (bs_, seq_len_, self.n_heads, self.head_dim))
        >>>         # [bs, seq_len, heads, head_dim] -> [bs, heads, seq_len, head_dim]
        >>>         query = self.transpose(x, (0, 2, 1, 3))
        >>>         freqs = self.rotary_embedding(seq_len_)
        >>>         output = self.attention(query, freqs)
        >>>         # [bs, heads, seq_len, head_dim] -> [bs, seq_len, heads, head_dim]
        >>>         output = self.transpose_back(output, (0, 2, 1, 3))
        >>>         # [bs, seq_len, heads, head_dim] -> [bs, seq_len, dim]
        >>>         output = self.reshape(output, (bs_, seq_len_, dim_))
        >>>         return output
        >>> config_ = TransformerConfig()
        >>> config_.data_parallel = 1
        >>> config_.tensor_parallel = 1
        >>> config_.context_parallel = 1
        >>> config_.num_attention_heads = 8
        >>> bs = 2
        >>> seq_len = 4096
        >>> dim = 8192
        >>> input_shape = (bs, seq_len, dim)
        >>> net = MyNet(config_)
        >>> input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> output_ = net(input_)
        >>> print(output_.shape)
        (2, 4096, 8192)
    """

    def __init__(
            self,
            kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        ):
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        if self.rotary_interleaved:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (mint.arange(0, dim, 2)[: (dim // 2)].astype(ms.float32) / dim)
        )

    def construct(self, max_seq_len, offset=0):
        """ Construct function of rotary embedding. """
        seq = (mint.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset).astype(ms.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = ops.outer(seq, self.inv_freq)
        if not self.rotary_interleaved:
            emb = ops.concat((freqs, freqs), axis=-1)
        else:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        # emb [S, ..., D]
        emb = emb[:, None, None, :]
        return Tensor(emb)


def _rotate_half(x, rotary_interleaved):
    if not rotary_interleaved:
        x1, x2 = mint.split(x, x.shape[-1] // 2, dim=-1)
        return ops.cat((-x2, x1), axis=-1)

    raise NotImplementedError('rotary_interleaved=True is not supported for now.')


def apply_rotary_pos_emb_bnsd(t, freqs, rotary_interleaved=False) -> Tensor:
    """
    Apply rotary positional embedding to input tensor.
    Please check https://kexue.fm/archives/8265 for detailed formulas

    Inputs:
        - **x** (Tensor) - Input tensor x with shape :math:`(B, N, S, D)`.
        - **freqs** (Tensor) - Rotary positional embedding tensor freq with shape :math:`(..., S, D)`.

    Outputs:
        - **output** (Tensor): The input tensor after applying RoPE.

    Supported Platforms:
        ``Ascend``
    """
    cos_ = mint.cos(freqs).to(t.dtype)
    sin_ = mint.sin(freqs).to(t.dtype)

    # rotate
    output = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return output


# pylint: disable=missing-docstring
def _apply_fused_rotary_pos_emb(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    rot_dim = freqs.shape[-1]
    t_shape_last_dim = t.shape[-1]
    if rot_dim != t_shape_last_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = mint.cos(freqs).to(t.dtype)
    sin_ = mint.sin(freqs).to(t.dtype)
    mode = 1 if rotary_interleaved else 0
    t = ops.rotary_position_embedding(t, cos_, sin_, mode=mode)
    if rot_dim == t_shape_last_dim:
        return t
    return mint.cat((t, t_pass), dim=-1)


# pylint: disable=W0613
def apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None) -> Tensor:
    if cu_seqlens is None:
        if config.apply_rope_fusion:
            return _apply_fused_rotary_pos_emb(t, freqs)
        return apply_rotary_pos_emb_bnsd(t, freqs, rotary_interleaved=False)

    raise NotImplementedError('cu_seqlens input for apply_rotary_pos_emb() is not supported for now.')
