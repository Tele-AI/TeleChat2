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
Test module for testing the internlm2 interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_internlm2_model/test_training_precision.py
"""
import sys
import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from tests.st.training_checker import TrainingChecker

for path in sys.path:
    if path.endswith('/testcases'):
        new_path = os.path.join(path, 'research')
        if new_path not in sys.path:
            sys.path.append(new_path)
    if path.endswith('/research'):
        new_path = os.path.join(path, 'internlm2')
        if new_path not in sys.path:
            sys.path.append(new_path)

research_path = os.path.join('/root', 'mindformers', 'research', 'internlm2')
if research_path not in sys.path:
    sys.path.append(research_path)
# pylint: disable=C0413
from research.internlm2.internlm2 import InternLM2ForCausalLM
from research.internlm2.internlm2_config import InternLM2Config

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 513
    step_num = 20
    batch_size = 4
    vocab_size = 103168
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestInternLM2TrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = InternLM2Config(num_layers=2, seq_length=512, use_flash_attention=True)
        model = InternLM2ForCausalLM(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.999,
                                             eps=1.e-8,
                                             learning_rate=lr_schedule)

        loss_list_std = [10.381567, 10.395970, 10.504854, 10.398868, 10.503867,
                         10.484463, 10.413940, 10.292614, 10.488887, 10.386089,
                         10.402554, 10.432349, 10.535288, 10.542078, 10.328922,
                         10.446496, 10.358795, 10.424121, 10.360553, 10.442839]
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
