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
Test module for testing the deepseek interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_deepseek_model/test_trainer.py
"""
import pytest

import mindspore as ms

from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestDeepseekTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)

        model_config = LlamaConfig(num_layers=2, hidden_size=7168, num_heads=56, seq_length=4096)
        model = LlamaForCausalLM(model_config)

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
        self.task_trainer.predict(
            input_data="write a quick sort algorithm", max_length=20, repetition_penalty=1, top_k=3, top_p=1)
