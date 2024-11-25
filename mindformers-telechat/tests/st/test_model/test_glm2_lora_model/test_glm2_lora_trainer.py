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
Test module for testing the glm2_lora interface used for mindformers.
How to run this:
pytest -v tests/st/test_model/test_glm2_lora_model/test_glm2_lora_trainer.py
"""

import numpy as np
import pytest

import mindspore
from mindspore import context
from mindspore.dataset import GeneratorDataset

from mindformers import AutoTokenizer, ChatGLM2Config, ChatGLM2ForConditionalGeneration
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet import get_pet_model
from mindformers import Trainer, TrainingArguments
from mindformers.tools.utils import is_version_ge


def generator_train():
    """train dataset generator"""
    seq_len = 128
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    train_data = (input_ids, labels)
    for _ in range(32):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 127
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    eval_data = (input_ids, labels)
    for _ in range(8):
        yield eval_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGLM2WithLoRATrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        context.set_context(mode=0, device_target="Ascend")

        args = TrainingArguments(num_train_epochs=1, batch_size=2)
        train_dataset = GeneratorDataset(generator_train,
                                         column_names=["input_ids", "labels"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=2)
        eval_dataset = eval_dataset.batch(batch_size=2)

        model_config = ChatGLM2Config(num_layers=2, seq_length=128, hidden_size=32, inner_hidden_size=None,
                                      num_heads=2, position_encoding_2d=True, padded_vocab_size=64793)
        model_config.pet_config = LoraConfig(lora_rank=8, lora_alpha=32, lora_dropout=0.1,
                                             target_modules='.*query_key_value*')
        model = ChatGLM2ForConditionalGeneration(model_config)
        model = get_pet_model(model, model_config.pet_config)

        self.tokenizer = AutoTokenizer.from_pretrained("glm2_6b")
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='glm2_6b_lora',
                                    tokenizer=self.tokenizer,
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        if is_version_ge(mindspore.__version__, "1.11.0"):
            self.task_trainer.predict(input_data="你好", max_length=20)

    @pytest.mark.run(order=2)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        if is_version_ge(mindspore.__version__, "1.11.0"):
            self.task_trainer.finetune()
