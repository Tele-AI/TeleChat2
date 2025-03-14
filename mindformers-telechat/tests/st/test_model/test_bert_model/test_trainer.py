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
Test module for testing the bert interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_bert_model/test_gpt_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms

from mindspore.dataset import GeneratorDataset
from mindformers.models import BertConfig, BertForPreTraining
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)


def generator():
    """dataset generator"""
    data = np.random.randint(low=0, high=15, size=(128,)).astype(np.int32)
    input_mask = np.ones_like(data)
    token_type_id = np.zeros_like(data)
    next_sentence_lables = np.array([1]).astype(np.int32)
    masked_lm_positions = np.array([1, 2]).astype(np.int32)
    masked_lm_ids = np.array([1, 2]).astype(np.int32)
    masked_lm_weights = np.ones_like(masked_lm_ids)
    train_data = (data, input_mask, token_type_id, next_sentence_lables,
                  masked_lm_positions, masked_lm_ids, masked_lm_weights)
    for _ in range(4):
        yield train_data


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestBertTrainerMethod:
    """A test class for testing trainer."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=1, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "segment_ids",
                                                                  "next_sentence_labels", "masked_lm_positions",
                                                                  "masked_lm_ids", "masked_lm_weights"])
        train_dataset = train_dataset.batch(batch_size=1)

        model_config = BertConfig(batch_size=1, num_hidden_layers=2)
        model = BertForPreTraining(model_config)

        self.task_trainer = Trainer(task='fill_mask',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset)

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        # self.task_trainer.train()

    @pytest.mark.run(order=2)
    def test_eval(self):
        """
        Feature: Trainer.evaluate()
        Description: Test trainer for evaluate.
        Expectation: TypeError, ValueError, RuntimeError
        """

    @pytest.mark.run(order=3)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        input_data = [" Hello I am a [MASK] model.",]
        self.task_trainer.predict(input_data=input_data)

    @pytest.mark.run(order=4)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
