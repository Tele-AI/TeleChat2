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
Test telechat2 train.
How to run this:
    pytest tests/st/test_model/test_telechat2_model/test_train.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestTelechat2Train:
    """A test class for testing model training precision."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base model train
        Description: Test base model training precision.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='train', batch_size=4, experiment_mode=False, use_label=True)

        model_config = get_config()
        model_config.num_layers = 2
        model_config.seq_length = 1024

        loss_std = [
            12.023546, 12.040644, 12.049784, 12.041933, 12.041258,
            12.030152, 12.053402, 12.042796, 12.049565, 12.030393,
            12.038810, 12.027491, 12.048979, 12.027491, 12.048134,
            11.995312, 12.021861, 12.047028, 12.028551, 12.069385
        ]
        model = get_model(model_config)

        runner.set_train(model, model_config, loss_std=loss_std)
