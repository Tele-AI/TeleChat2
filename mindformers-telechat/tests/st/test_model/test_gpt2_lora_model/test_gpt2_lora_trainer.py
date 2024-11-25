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
Test module for testing the gpt2_lora interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_gpt2_lora_model/test_gpt2_lora_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers.models.gpt2.gpt2 import GPT2LMHeadModel
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet import get_pet_model
from mindformers import Trainer, TrainingArguments, GPT2Config

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 65
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)

    train_data = (input_ids, input_mask)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 64
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(16):
        yield train_data


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGPT2LoraTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "input_mask"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "input_mask"])
        train_dataset = train_dataset.batch(batch_size=2)
        eval_dataset = eval_dataset.batch(batch_size=2)

        model_config = GPT2Config(num_layers=2, embedding_size=32, num_heads=2, seq_length=64)
        pet_config = LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.05, target_modules='.*dense1.*|.*dense3.*')
        model_config.pet_config = pet_config
        model = GPT2LMHeadModel(model_config)
        model = get_pet_model(model, model_config.pet_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    pet_method='lora',
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune()

    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
