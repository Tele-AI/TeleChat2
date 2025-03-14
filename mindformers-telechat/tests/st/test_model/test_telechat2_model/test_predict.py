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
Test telechat2 predict.
How to run this:
    pytest tests/st/test_model/test_telechat2_model/test_predict.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester
from .base_model import get_config, get_model

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


class TestTelechat2Predict:
    """A test class for testing model prediction."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: SLora model predict
        Description: Test llama slora model prediction.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='predict', batch_size=2, experiment_mode=False)

        model_config = get_config()
        model_config.use_past = True
        model_config.run_mode = 'predict'
        model_config.block_size = 128
        model_config.num_blocks = 256
        model_config.is_dynamic = True
        model_config.out_proj_has_bias = True
        model_config.batch_size = runner.batch_size  # set batch size for prediction
        model_config.vocab_size = 32000  # default to use llama2 tokenizer

        model = get_model(model_config)

        outputs = "hello world."
        runner.set_predict(model=model, expect_outputs=outputs)
