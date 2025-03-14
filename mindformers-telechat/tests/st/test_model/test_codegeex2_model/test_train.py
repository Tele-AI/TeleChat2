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
Test codegeex2 train.
How to run this:
    pytest tests/st/test_model/test_codegeex2_model/test_train.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestCodeGeeX2Train:
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
        runner = ModelTester(run_mode='train', batch_size=4, use_label=True, experiment_mode=False)

        model_config = get_config()

        loss_std = [11.298032, 11.289123, 11.289726, 11.276349, 11.286991,
                    11.266462, 11.274662, 11.274948, 11.262068, 11.258616,
                    11.253914, 11.257857, 11.259014, 11.240475, 11.241916,
                    11.242459, 11.257033, 11.243998, 11.252337, 11.258551]

        model = get_model(model_config)

        model_config.seq_length -= 1  # set for generate data
        runner.set_train(model, model_config, loss_std=loss_std)
