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
"""test pet model."""
from mindformers import GPT2Config, GPT2LMHeadModel
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.pet.models.lora import LoraModel


class TestPetModel:
    """A test class for testing pet model."""
    def test_input_config_is_dict(self):
        """test input config is dict."""
        config = GPT2Config(num_layers=1)
        model = GPT2LMHeadModel(config)
        pet_config = {
            'pet_type': "lora",
            'target_modules': r'.*dense*|.*linear*'
        }
        pet_model = get_pet_model(model, pet_config)
        assert isinstance(pet_model.config.pet_config, LoraConfig)
        assert isinstance(pet_model.pet_model, LoraModel)

    def test_input_config_is_lora_config(self):
        """test input config is LoraConfig."""
        config = GPT2Config(num_layers=1)
        model = GPT2LMHeadModel(config)
        pet_config = LoraConfig(
            target_modules=r'.*dense*|.*linear*'
        )
        pet_model = get_pet_model(model, pet_config)
        assert isinstance(pet_model.config.pet_config, LoraConfig)
        assert isinstance(pet_model.pet_model, LoraModel)
