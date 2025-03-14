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
Test module for testing the wizardcoder interface used for mindformers.
How to run this:
pytest --disable-warnings -vs tests/st/test_model/test_wizardcoder_model/test_training_precision.py
"""
import os
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from tests.st.training_checker import TrainingChecker

ms.set_context(mode=0)


def dir_path(path, times: int):
    if times > 0:
        return dir_path(os.path.dirname(path), times - 1)
    return path


wizardcoder_path = os.path.join(dir_path(__file__, 5), "research/wizardcoder")
sys.path.append(wizardcoder_path)
ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 1025
    step_num = 20
    batch_size = 1
    vocab_size = 49153
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestWizardcoderPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)
        from research.wizardcoder.wizardcoder import WizardCoderLMHeadModel
        from research.wizardcoder.wizardcoder_config import WizardCoderConfig

        args = TrainingArguments(batch_size=1, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=1)

        model_config = WizardCoderConfig(num_layers=2, batch_size=1)
        model = WizardCoderLMHeadModel(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=2.e-5, lr_end=1.e-6, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.95,
                                             eps=1.e-8,
                                             learning_rate=lr_schedule)

        loss_list_std = [10.871237, 10.868160, 10.860825, 10.848734, 10.861235,
                         10.875328, 10.860098, 10.858349, 10.872917, 10.871431,
                         10.861275, 10.871601, 10.874082, 10.858971, 10.849184,
                         10.873372, 10.857426, 10.873322, 10.867041, 10.870938]
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
