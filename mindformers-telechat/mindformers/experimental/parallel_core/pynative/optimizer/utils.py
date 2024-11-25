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
""" utils for distri_core optimizer. """

from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common.initializer import Zero
from mindspore.common import dtype as mstype


def _update_adamw_var(optimizer):
    """
    Updata AdamW optimizer variables when using distributed optimizer.
    Since after bucketing, parameters updated by optimizer is sharded chunks of flattened parameters.
    Variables of optimizer should be replaced according to sharded parameter chunks.
    """
    # clean old optimizer parameters
    del optimizer.exp_avg
    del optimizer.exp_avg_sq
    exp_avg = []
    exp_avg_sq = []
    for param in optimizer.parameters:
        exp_avg.append(
            Parameter(
                Tensor(shape=param.shape, dtype=mstype.float32, init=Zero()),
                name='exp_avg.' + param.name
            )
        )
        exp_avg_sq.append(
            Parameter(
                Tensor(shape=param.shape, dtype=mstype.float32, init=Zero()),
                name='exp_avg_sq.' + param.name
            )
        )
    optimizer.exp_avg = ParameterTuple(exp_avg)
    optimizer.exp_avg_sq = ParameterTuple(exp_avg_sq)
