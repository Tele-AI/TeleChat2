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
"""
Test llama slora predict.
How to run this:
    pytest tests/st/test_model/test_llama_lora_model/test_llama_slora_predict.py
"""
import pytest
import mindspore as ms

from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.pet.pet_config import SLoraConfig
from mindformers.pet import get_pet_model
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


class TestLlamaSLoraPredict:
    """A test class for testing model prediction."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: SLora model predict
        Description: Test llama slora model prediction.
        Expectation: AssertionError
        """
        model_config = LlamaConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512)
        model_config.pet_config = SLoraConfig(lora_num=2, lora_rank=8, lora_alpha=16,
                                              target_modules='.*wq|.*wk|.*wv|.*wo')
        model = LlamaForCausalLM(model_config)
        model = get_pet_model(model, model_config.pet_config)

        args = TrainingArguments(batch_size=1)
        runner = Trainer(task='text_generation',
                         model=model,
                         model_name='llama_7b_slora',
                         args=args)
        runner.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_slora_embedding(self):
        """
        Feature: SLora model predict with SloRAEmbedding and SLoRAHead
        Description: Test llama slora model prediction.
        Expectation: AssertionError
        """
        model_config = LlamaConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512)
        model_config.pet_config = SLoraConfig(lora_num=2, lora_rank=8, lora_alpha=16,
                                              target_modules='.*wq|.*wk|.*wv|.*wo|embed_token|lm_head')
        model = LlamaForCausalLM(model_config)
        model = get_pet_model(model, model_config.pet_config)

        args = TrainingArguments(batch_size=1)
        runner = Trainer(task='text_generation',
                         model=model,
                         model_name='llama_7b_slora',
                         args=args)
        runner.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
