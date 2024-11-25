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
"""Dropout"""
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

__all__ = ['Dropout']


class Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.Dropout and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, drop_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if drop_prob < 0 or drop_prob >= 1:
            raise ValueError(
                "dropout probability should be a number in range [0, 1), but got {}".format(drop_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('drop_prob', drop_prob, [float], self.cls_name)
        keep_prob = 1.0 - drop_prob
        self.keep_prob = keep_prob
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        out, _ = self.dropout(x)
        return out

    def extend_repr(self):
        return 'keep_prob={}'.format(self.keep_prob)

    def shard(self, strategy):
        self.dropout.shard((strategy,))
