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
Test module for testing the qwen interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_qwen_model/test_trainer.py
"""
import os
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments

for path in sys.path:
    if path.endswith('/testcases'):
        new_path = os.path.join(path, 'research')
        if new_path not in sys.path:
            sys.path.append(new_path)

    if path.endswith('/research'):
        new_path = os.path.join(path, 'qwen')
        if new_path not in sys.path:
            sys.path.append(new_path)
# pylint: disable=C0413
from research.qwen.qwen_config import QwenConfig
from research.qwen.qwen_model import QwenForCausalLM

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 512
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(16):
        yield train_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestQwenTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "labels"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=4)
        eval_dataset = eval_dataset.batch(batch_size=4)

        model_config = QwenConfig(num_layers=2,
                                  hidden_size=5120,
                                  num_heads=40,
                                  seq_length=512,
                                  pa_block_size=1,
                                  intermediate_size=13696,
                                  emb_dropout_prob=0.0,
                                  pa_num_blocks=1)
        model = QwenForCausalLM(model_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    @pytest.mark.run(order=1)
    def test_eval(self):
        """
        Feature: Trainer.evaluate()
        Description: Test trainer for evaluate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.model.set_train(False)
        self.task_trainer.evaluate()

    @pytest.mark.run(order=2)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
