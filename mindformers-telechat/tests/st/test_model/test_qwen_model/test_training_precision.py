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
pytest tests/st/test_model/test_qwen_model/test_training_precision.py
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
    step_num = 20
    batch_size = 4
    vocab_size = 152064
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestQwenTrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)

        model_config = QwenConfig(num_layers=2,
                                  hidden_size=5120,
                                  num_heads=40,
                                  seq_length=512,
                                  vocab_size=152064,
                                  multiple_of=256,
                                  intermediate_size=13696,
                                  emb_dropout_prob=0.0,
                                  rms_norm_eps=1.0e-6,
                                  use_flash_attention=True)
        model = QwenForCausalLM(model_config)

        lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_ratio=0.01, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.95,
                                             eps=1.e-6,
                                             weight_decay=0.1,
                                             learning_rate=lr_schedule)

        loss_list_std = [12.194469, 12.192232, 12.221778, 12.196572, 12.172647,
                         12.176313, 12.192204, 12.167645, 12.170177, 12.190814,
                         12.179645, 12.165785, 12.175731, 12.174227, 12.206128,
                         12.202451, 12.185367, 12.212145, 12.200205, 12.190893]
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
