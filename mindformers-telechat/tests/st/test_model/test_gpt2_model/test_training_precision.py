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
Test module for testing the gpt2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_gpt2_model/test_training_precision.py
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, PolynomialWithWarmUpLR, FusedAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
from tests.st.training_checker import TrainingChecker

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 1025
    step_num = 20
    batch_size = 4
    vocab_size = 50257
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGpt2TrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = GPT2Config(num_layers=2, use_flash_attention=True)
        model = GPT2LMHeadModel(model_config)

        lr_schedule = PolynomialWithWarmUpLR(learning_rate=1.e-4, lr_end=1.e-5, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FusedAdamWeightDecay(params=group_params, beta1=0.9, beta2=0.95, eps=1.e-8, weight_decay=0.1,
                                         learning_rate=lr_schedule)

        loss_list_std = [10.858289, 10.866613, 10.866228, 10.860546, 10.862840,
                         10.862125, 10.860486, 10.866213, 10.864862, 10.867769,
                         10.862738, 10.863748, 10.869925, 10.862442, 10.864432,
                         10.865078, 10.873352, 10.866808, 10.864880, 10.869852]
        callback = TrainingChecker(loss_list_std=loss_list_std)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset,
                                    callbacks=callback,
                                    optimizers=optimizer)

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: AssertionError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.config.runner_config.sink_mode = False
        self.task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
        self.task_trainer.config.callbacks = self.task_trainer.config.callbacks[:1]
        self.task_trainer.train()
