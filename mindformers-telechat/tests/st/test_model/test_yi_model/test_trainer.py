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
Test module for testing the yi interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_yi_model/test_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms

from mindspore.dataset import GeneratorDataset
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers import Trainer, TrainingArguments
from mindformers.models.llama.llama import LlamaForCausalLM

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 512
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestYiTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)
        eval_dataset = eval_dataset.batch(batch_size=4)

        model_config = LlamaConfig(num_layers=2, hidden_size=4096, num_heads=32, seq_length=512, pa_block_size=1,
                                   pa_num_blocks=1)
        model = LlamaForCausalLM(model_config)

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
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1.3, top_k=40, top_p=0.8)
