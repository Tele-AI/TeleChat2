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

"""modules init"""
from .models import (
    LoraModel,
    PrefixTuningModel,
    Ptuning2Model
)
from .pet_config import (
    LoraConfig,
    PetConfig,
    PrefixTuningConfig,
    Ptuning2Config
)
from .tuners import (
    AdaAdapter,
    AdaLoraAdapter,
    LoraAdapter,
    PetAdapter,
    PrefixTuningAdapter,
    Ptuning2Adapter
)
from .pet_model import (
    get_pet_model,
    is_supported_pet_type
)

__all__ = ['LoraConfig', 'PetConfig']
__all__.extend(models.__all__)
