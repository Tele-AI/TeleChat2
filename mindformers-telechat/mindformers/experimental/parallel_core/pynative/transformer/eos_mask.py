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
"""EosMask"""

import numpy as np
from mindspore import Tensor, nn, ops, mint
import mindspore.common.dtype as mstype


class EosMask(nn.Cell):
    """
    Generate attention mask corresponding to a specific token.

    Args:
        batch_size (int): batch_size from config
        seq_len (int): seq_len from config
        eod_token_id (int): eod_token_id from config
    """
    def __init__(self, batch_size, seq_len, eod_token_id):
        """Cal attention mask in device."""
        super().__init__()
        self.seq_len = seq_len
        self.position_ids = Tensor(np.broadcast_to(np.expand_dims(np.arange(seq_len), 0), (batch_size, seq_len)),
                                   dtype=mstype.int32)
        self.tril = ops.Tril()
        self.cast = ops.Cast()
        self.expand_dim = ops.ExpandDims()
        self.eod_token = eod_token_id

    def construct(self, input_ids):
        """construct method"""
        # input_ids: [bs, seq_len]
        eod_idx = self.cast(mint.eq(input_ids, self.eod_token), mstype.float16)
        attention_mask = mint.cumsum(eod_idx, 1) - eod_idx
        row = self.expand_dim(attention_mask, 1)
        col = self.expand_dim(attention_mask, 2)
        row = mint.tile(row, (1, self.seq_len, 1))
        col = mint.tile(col, (1, 1, self.seq_len))
        mat = mint.eq(row, col)
        mat = self.cast(mat, mstype.uint8)
        mask = self.tril(mat)
        # [bs, seq_len, seq_len]

        return self.position_ids, mint.sub(1, mask)
