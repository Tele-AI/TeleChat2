# Copyright 2025 Huawei Technologies Co., Ltd
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
from typing import Union
import math

from mindspore import ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P



class RotaryEmbedding(Cell):
    """Rotary embedding for language model.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (int): The base for rotary embedding.
        rotary_cos_format (int): The cos format of ops.ApplyRotaryPosEmb.
        rotary_dtype (mstype): The dtype of rotary embeddings.
    """

    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: int = 10000,
                 rotary_cos_format: int = 0,
                 rotary_dtype: mstype = mstype.float16):
        super(RotaryEmbedding, self).__init__()
        self.dim = kv_channels
        if rotary_percent < 1.0:
            self.dim = int(self.dim * rotary_percent)

        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base

        if self.rotary_interleaved:
            self.stack = P.Stack(axis=-1)
        else:
            self.cat = P.Concat(axis=-1)
        self.reshape = P.Reshape()
        self.cos = P.Cos()
        self.sin = P.Sin()
        self.cast = P.Cast()
        self.gather = P.Gather()

        self.rotary_dtype = rotary_dtype
        self.rotary_cos_format = rotary_cos_format
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(self.rotary_cos_format)

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base ** (ops.arange(0, self.dim, 2, dtype=mstype.float32)[: (self.dim // 2)] / self.dim)
        )
        return inv_freq

    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """
        Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings
        """
        inv_freq = self._compute_inv_freq(self.rotary_base)
        seq = ops.arange(0, max_seq_len, 1, dtype=inv_freq.dtype) + offset

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = ops.outer(seq, inv_freq)

        return freqs

    def get_cos_sin_for_prefill(self, max_seq_len: int, offset: int = 0) -> (Tensor, Tensor):
        """Compute the cos and sin for prefill"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)

        if self.rotary_interleaved:
            freqs_new_shape = (freqs.shape[0], -1)
            emb = self.reshape(self.stack(self.reshape(freqs, (-1, 1)), self.reshape(freqs, (-1, 1))), freqs_new_shape)
        else:
            emb = self.cat((freqs, freqs))

        rotary_pos_cos = self.cast(self.cos(emb), self.rotary_dtype)
        rotary_pos_sin = self.cast(self.sin(emb), self.rotary_dtype)

        return rotary_pos_cos, rotary_pos_sin

    def get_cos_sin_for_decode(self, positions: Tensor, max_seq_len: int, offset: int = 0) -> (Tensor, Tensor):
        """Compute the cos and sin for decode"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)

        if self.rotary_interleaved:
            freqs_new_shape = (freqs.shape[0], -1)
            emb = self.reshape(self.stack(self.reshape(freqs, (-1, 1)), self.reshape(freqs, (-1, 1))), freqs_new_shape)
        else:
            emb = self.cat((freqs, freqs))

        cos = self.cast(self.cos(emb), self.rotary_dtype)
        sin = self.cast(self.sin(emb), self.rotary_dtype)

        rotary_pos_cos = self.gather(cos, positions, 0)
        rotary_pos_sin = self.gather(sin, positions, 0)

        return rotary_pos_cos, rotary_pos_sin

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  rotary_pos_cos: Tensor,
                  rotary_pos_sin: Tensor,
                  seq_lens_tensor: Tensor) -> Tensor:
        """Generate rotary position embedding.

        Args:
            query(Tensor): The query matrix
            key(Tensor): The key matrix
            rotary_pos_cos(Tensor): The precompute freqs cos for rotary position embedding
            rotary_pos_sin(Tensor): The precompute freqs sin for rotary position embedding
            seq_lens_tensor(Tensor): Int32 tensor with shape [batch_size] for valid seq length.

        Returns:
            query: The query matrix
            key: The key matrix
        """
        return self.rotary_embedding_op(query, key, rotary_pos_cos, rotary_pos_sin, seq_lens_tensor)


class Llama3RotaryEmbedding(RotaryEmbedding):
    """Rotary embedding extended with Llama3 method.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (int): The base for rotary embedding.
        rotary_cos_format (int): The cos format of ops.ApplyRotaryPosEmb.
        rotary_dtype (mstype): The dtype of rotary embeddings.
        scaling_factor (float): The scaling factor for sequence length.
        low_freq_factor (float): The low frequency factor for sequence length.
        high_freq_factor (float): The high frequency factor for sequence length.
        orig_max_position (float): The max position for original position embedding.

    """
    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: int = 10000,
                 rotary_cos_format: int = 0,
                 rotary_dtype: mstype = mstype.float16,
                 scaling_factor: float = 1.0,
                 low_freq_factor: float = 1.0,
                 high_freq_factor: float = 4.0,
                 orig_max_position: int = 4096):
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(kv_channels, rotary_percent, rotary_interleaved, seq_len_interpolation_factor,
                         rotary_base, rotary_cos_format, rotary_dtype)

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        inv_freq = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freq
        if not math.isclose(self.low_freq_factor, self.high_freq_factor):
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0
        new_freqs = ops.where(
            wave_len < high_freq_wavelen,
            inv_freq,
            ops.where(
                wave_len > low_freq_wavelen,
                inv_freq / self.scaling_factor,
                (1 - smooth) * inv_freq / self.scaling_factor +
                smooth * inv_freq,
            ),
        )
        return new_freqs
