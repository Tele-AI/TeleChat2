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
"""
Note: PET Adapter is the base adapter class for Parameter Efficient Tuning of MindFormers.
"""
from mindspore import nn

from mindpet.graph.freeze_utils import freeze_delta

from ..pet_config import PetConfig
from ..constants import PetType


class PetAdapter:
    r"""
    PetAdapter is the base class of adapter to modify the pretrained model.
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        """Add efficient tuning parameters to ptm."""
        raise NotImplementedError("should implemented by the certain tuning algorithm.")

    @classmethod
    def freeze_pretrained_model(cls, model, pet_type: PetType, freeze_include=None, freeze_exclude=None):
        """
        Freeze the parameters of ptm which no update in the tuning process.

        Notes:
            Refer to mindpet api.
        """
        freeze_delta(model, pet_type, freeze_include, freeze_exclude)
