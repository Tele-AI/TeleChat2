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
"""AdamW API"""
from mindformers.core.optim.adamw import AdamW as Adamw


class AdamW(Adamw):
    """adamw optimizer"""
    def __init__(self, params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, cpu_offload=False):
        if cpu_offload:
            raise NotImplementedError("cpu_offload is not supported now.")
        super(AdamW, self).__init__(params, learning_rate, betas, eps, weight_decay=weight_decay)
