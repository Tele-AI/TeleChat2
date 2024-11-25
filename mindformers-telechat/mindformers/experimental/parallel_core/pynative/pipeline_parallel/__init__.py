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

"""mindformers init"""

__all__ = [
    "P2PPrimitive",
    "forward_backward_pipelining_without_interleaving",
    "forward_backward_pipelining_with_interleaving",
]

from .p2p_communication import P2PPrimitive
from .schedules import forward_backward_pipelining_without_interleaving, \
                       forward_backward_pipelining_with_interleaving
