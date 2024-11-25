# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
Rotary position embedding for transformer.
"""
from mindspore import nn, Tensor, dtype
from mindspore import ops
from mindspore.ops import operations as P
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

__all__ = [
    "RotaryEmbedding",
    "ApplyRotaryPosEmb",
    "apply_rotary_pos_emb"
]


class RotaryEmbedding(nn.Cell):
    """Rotary embedding for language model.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (int): The base for rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
    """
    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: int = 10000):
        super(RotaryEmbedding, self).__init__()
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_interleaved = rotary_interleaved
        self.inv_freq = 1.0 / (
            rotary_base ** (ops.arange(0, dim, 2, dtype=dtype.float32)[: (dim // 2)] / dim)
        )
        if self.rotary_interleaved:
            self.stack = P.Stack(axis=-1)
        else:
            self.cat = P.Concat(axis=-1)
        self.reshape = P.Reshape()

    def construct(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Generate rotary position embedding.

        Args:
            max_seq_len (int): The maximum sequence length.
            offset (int): The offset for the sequence.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = ops.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = ops.outer(seq, self.inv_freq)

        if self.rotary_interleaved:
            freqs_new_shape = (freqs.shape[0], -1)
            emb = self.reshape(self.stack(self.reshape(freqs, (-1, 1)), self.reshape(freqs, (-1, 1))), freqs_new_shape)
        else:
            emb = self.cat((freqs, freqs))

        # emb[.., seq_length, dim]
        return self.reshape(emb, (1, 1, emb.shape[0], emb.shape[1]))


class ApplyRotaryPosEmb(nn.Cell):
    """Apply rotary positional embedding to input tensor T.

    Args:
        config (TransformerConfig): The transformer configuration
    """
    def __init__(self,
                 config: TransformerConfig):
        super(ApplyRotaryPosEmb, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.cos = P.Cos()
        self.sin = P.Sin()
        self.neg = P.Neg()
        self.split = P.Split(axis=-1, output_num=2)
        self.cat = P.Concat(axis=-1)
        self.stack = P.Stack(axis=-1)
        self.slice = P.StridedSlice()
        self.reshape = P.Reshape()
        self.shard(config)

    def construct(self, t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            t (Tensor): Input tensor is of shape [bs, n_heads, seq_len, head_dim]
            freqs (Tensor): Rotary position embedding frequencies of shape [1, 1, seq_len, head_dim]
            rotary_interleaved (bool): Whether to use interleaved rotary position embedding

        Returns:
            Tensor: Output tensor after applying rotary position embedding
        """
        if t.shape[-1] < freqs.shape[-1]:
            raise ValueError(
                f"Input tensor shape {t.shape} must be greater than freqs shape {freqs.shape} in the last dimension")
        if t.shape[-1] > freqs.shape[-1]:
            rot_dim = freqs.shape[-1]
            t, t_not_rotary = t[..., :rot_dim], t[..., rot_dim:]
        else:
            t_not_rotary = None

        cos_ = self.cos(freqs).astype(t.dtype)
        sin_ = self.sin(freqs).astype(t.dtype)

        output = self.add(self.mul(t, cos_), self.mul(self._rotate_half(t, rotary_interleaved), sin_))
        if t_not_rotary is not None:
            output = self.cat((output, t_not_rotary))
        return output

    def _rotate_half(self, t: Tensor, rotary_interleaved: bool = False) -> Tensor:
        bs, n_heads, seq_len, head_dim = t.shape
        if rotary_interleaved:
            t_1 = self.slice(t, (0, 0, 0, 0), (bs, n_heads, seq_len, head_dim), (1, 1, 1, 2))
            t_2 = self.slice(t, (0, 0, 0, 1), (bs, n_heads, seq_len, head_dim), (1, 1, 1, 2))
            t_rot = self.reshape(self.stack((self.neg(t_2), t_1)), (bs, n_heads, seq_len, -1))
        else:
            t_1, t_2 = self.split(t)
            t_rot = self.cat((self.neg(t_2), t_1))
        return t_rot

    def shard(self, config: TransformerConfig):
        """The multi-head attention naturally supports tensor parallelism by splitting along the head dimension."""
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1

        sin_in_strategy = ((1, 1, 1, 1),)
        cos_in_strategy = ((1, 1, 1, 1),)
        split_in_strategy = ((dp, tp, 1, 1),)
        neg_in_strategy = ((dp, tp, 1, 1),)
        cat_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        stack_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        slice_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        add_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        mul_in_strategy = ((dp, tp, 1, 1), (1, 1, 1, 1))

        self.add.shard(in_strategy=add_in_strategy)
        self.mul.shard(in_strategy=mul_in_strategy)
        self.split.shard(in_strategy=split_in_strategy)
        self.cat.shard(in_strategy=cat_in_strategy)
        self.stack.shard(in_strategy=stack_in_strategy)
        self.slice.shard(in_strategy=slice_in_strategy)
        self.neg.shard(in_strategy=neg_in_strategy)
        self.sin.shard(in_strategy=sin_in_strategy)
        self.cos.shard(in_strategy=cos_in_strategy)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Tensor = None) -> Tensor:
    if cu_seqlens is not None:
        raise NotImplementedError("For apply_rotary_pos_emb, cu_seqlens is not supported for now.")
    rotary_interleaved = config.rotary_interleaved if hasattr(config, 'rotary_interleaved') else False
    apply_func = ApplyRotaryPosEmb(config)
    return apply_func(t, freqs, rotary_interleaved)
