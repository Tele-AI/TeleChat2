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
Test glm2, glm3, glm4, glm32k predict.
How to run this:
    pytest tests/st/test_model/test_glm2_model/test_predict.py
"""
import pytest
import mindspore.common.dtype as mstype

from tests.utils.model_tester import ModelTester
from .base_model import get_config, get_model


class TestGLM2Predict:
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

        expect_outputs = ("hello world.的先因此而底部常务理事常务理事┳矸┳做 OW OW┳┳ Thank Thank")
        outputs = runner.set_predict(model=model, expect_outputs=expect_outputs, auto_tokenizer='glm2_6b')
        assert outputs == expect_outputs, "The outputs are not as expected, outputs: "\
                                          f"{outputs}, expect_outputs: {expect_outputs}"


class TestGLM32kPredict:
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

        # change config for glm32k
        model_config.compute_dtype = mstype.float16
        model_config.max_decode_length = 512
        model_config.no_recompute_layers = [20]
        model_config.param_init_type = mstype.float16
        model_config.rope_ratio = 50
        model_config.seq_length = 32768

        model = get_model(model_config)

        expect_outputs = ("[gMASK]sop 你好长了rad罗global服务能力wp本项"
                          "目让您景觀战中 Positive Positive觉 Continental文献 gro")
        outputs = runner.set_predict(model=model, predict_data='你好',
                                     expect_outputs=expect_outputs, auto_tokenizer='glm3_6b')
        assert outputs == expect_outputs, "The outputs are not as expected, outputs: "\
                                          f"{outputs}, expect_outputs: {expect_outputs}"
