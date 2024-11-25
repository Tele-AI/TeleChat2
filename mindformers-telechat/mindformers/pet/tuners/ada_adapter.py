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
Note: Adapter algrithm for mindformers' pretrained model.
Reference: https://arxiv.org/pdf/1902.00751.pdf
"""
from mindspore import nn

from .pet_adapter import PetAdapter
from ..pet_config import PetConfig


class AdaAdapter(PetAdapter):
    r"""
        AdaAdapter is the adapter to modify the pretrained model, which uses adapter tuning algorithm.
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        return super().get_pet_model(model, config)
