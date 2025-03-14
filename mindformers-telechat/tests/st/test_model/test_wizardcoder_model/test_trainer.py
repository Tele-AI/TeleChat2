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
pytest --disable-warnings -vs tests/st/test_model/test_wizardcoder_model/test_trainer.py
"""

import os
import sys
import pytest

import mindspore as ms

from mindformers import Trainer, TrainingArguments


def dir_path(path, times: int):
    if times > 0:
        return dir_path(os.path.dirname(path), times - 1)
    return path


wizardcoder_path = os.path.join(dir_path(__file__, 5), "research/wizardcoder")
sys.path.append(wizardcoder_path)
ms.set_context(mode=0)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestWizardcoderTrainerMethod:
    """A test class for testing trainer."""

    def setup_method(self):
        """init task trainer."""
        from research.wizardcoder.wizardcoder import WizardCoderLMHeadModel
        from research.wizardcoder.wizardcoder_config import WizardCoderConfig
        args = TrainingArguments(batch_size=1, num_train_epochs=1)

        model_config = WizardCoderConfig(num_layers=2, batch_size=1)
        model = WizardCoderLMHeadModel(model_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args)

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
