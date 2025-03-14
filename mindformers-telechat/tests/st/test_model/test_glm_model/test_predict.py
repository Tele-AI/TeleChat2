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
Test glm predict.
How to run this:
    pytest tests/st/test_model/test_glm_model/test_predict.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)

class TestGLMPredict:
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
        runner = ModelTester(run_mode='predict', batch_size=2, experiment_mode=False)

        model_config = get_config()
        model_config.batch_size = runner.batch_size  # set batch size for prediction

        model = get_model(model_config)

        expect_outputs = ("hello world.出世长时间云平台 arterioles Walter prophylacti"
                          "carabi对白 AbstractGCM是一种幸福 appeal Garethyria各个方面")
        outputs = runner.set_predict(model=model, expect_outputs=expect_outputs, auto_tokenizer='glm_6b')
        assert outputs == expect_outputs, "The outputs are not as expected, outputs: "\
                                          f"{outputs}, expect_outputs: {expect_outputs}"
