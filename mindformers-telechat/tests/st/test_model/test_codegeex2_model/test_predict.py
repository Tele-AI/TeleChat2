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
Test codegeex2 predict.
How to run this:
    pytest tests/st/test_model/test_codegeex2_model/test_predict.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestCodeGeeX2Predict:
    """A test class for testing model prediction."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base model predict
        Description: Test base model prediction.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='predict', batch_size=1, experiment_mode=False)

        model_config = get_config()
        model_config.batch_size = runner.batch_size  # set batch size for prediction
        model_config.vocab_size = 32000  # default to use llama2 tokenizer
        model_config.padded_vocab_size = model_config.vocab_size

        model = get_model(model_config)

        outputs = 'hello world.?}iclopedia?}?}?}?}?}iclopediaiclopedia?}iclopedia?}?}iclopediaFCFC'
        runner.set_predict(model=model, expect_outputs=outputs)
