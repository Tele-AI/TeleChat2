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
Test module for testing the glm_lora interface used for mindformers.
How to run this:
pytest -v tests/st/test_model/test_glm_lora_model/test_glm_lora_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindformers.models.glm import GLMForPreTraining, GLMConfig
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet import get_pet_model
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""

    # name is input_ids, shape is (bs, seq_len), dtype is Int32
    # name is labels, shape is (bs, seq_len), dtype is Int32
    # name is position_ids, shape is (bs, 2, seq_len), dtype is Int32
    seq_len = 128
    input_ids = np.random.randint(low=0, high=15, size=(seq_len)).astype(np.int32)
    labels = np.ones(seq_len).astype(np.int32)
    position_ids = np.ones((2, seq_len)).astype(np.int32)
    attention_mask = np.ones((1, seq_len, seq_len)).astype(np.int32)
    train_data = (input_ids, labels, position_ids, attention_mask)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""

    # name is input_ids, shape is (8, 256), dtype is Int32
    # name is labels, shape is (8, 256), dtype is Int32
    seq_len = 128
    input_ids = np.random.randint(low=0, high=15, size=(seq_len)).astype(np.int32)
    labels = np.ones_like(seq_len).astype(np.int32)
    train_data = (input_ids, labels)
    for _ in range(16):
        yield train_data


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGLMWithLoRATrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4)
        train_dataset = GeneratorDataset(generator_train,
                                         column_names=["input_ids", "labels", "position_ids", "attention_mask"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=4)
        eval_dataset = eval_dataset.batch(batch_size=4)

        # set `vocab_size` to prevent generate token_id that out of vocab file
        model_config = GLMConfig(num_layers=2, seq_length=128, vocab_size=120528)
        model_config.pet_config = LoraConfig(lora_rank=8, lora_alpha=32, lora_dropout=0.1,
                                             target_modules='.*query_key_value*')
        model = GLMForPreTraining(model_config)
        model = get_pet_model(model, model_config.pet_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='glm_6b_lora',
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    @pytest.mark.run(order=1)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune()

    @pytest.mark.run(order=2)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
