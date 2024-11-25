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
Test module for testing the glm32k interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_glm32k_model/test_trainer.py
"""
import pytest

import mindspore
from mindspore import context

from mindformers import AutoTokenizer, ChatGLM2Config, ChatGLM2ForConditionalGeneration
from mindformers import Trainer, TrainingArguments
from mindformers.tools.utils import is_version_ge


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGLM32kTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        context.set_context(mode=0, device_target="Ascend", jit_config={"jit_level": "O0", "infer_boost": "on"})

        args = TrainingArguments(num_train_epochs=1, batch_size=2)

        model_config = ChatGLM2Config(num_layers=2, seq_length=128, hidden_size=32, inner_hidden_size=None,
                                      num_heads=2, padded_vocab_size=65024,
                                      rope_ratio=50)
        model = ChatGLM2ForConditionalGeneration(model_config)
        self.tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='glm3_6b',
                                    tokenizer=self.tokenizer,
                                    args=args)

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        model_config = ChatGLM2Config(num_layers=1, hidden_size=32, inner_hidden_size=None,
                                      num_heads=2, padded_vocab_size=65024, block_size=128, num_blocks=1024,
                                      seq_length=32768, rope_ratio=50,
                                      use_past=True, use_flash_attention=True, is_dynamic=True)
        model = ChatGLM2ForConditionalGeneration(model_config)
        task_trainer = Trainer(task='text_generation',
                               model=model,
                               tokenizer=self.tokenizer)
        if is_version_ge(mindspore.__version__, "1.11.0"):
            task_trainer.predict(input_data="使用python编写快速排序代码" * 3600, max_length=32768)
