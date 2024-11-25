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
Test glm2, glm3, glm4, glm32k train.
How to run this:
    pytest tests/st/test_model/test_glm2_model/test_train.py
"""
import pytest
import mindspore.common.dtype as mstype

from tests.utils.model_tester import ModelTester
from .base_model import get_config, get_model


class TestGLM2Train:
    """A test class for testing model training precision."""

    @pytest.mark.level0
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

        loss_std = [11.282274, 11.302577, 11.284897, 11.290861, 11.287876,
                    11.270002, 11.287689, 11.283995, 11.282846, 11.267147,
                    11.278307, 11.275076, 11.268270, 11.283443, 11.281622,
                    11.283977, 11.255313, 11.278028, 11.272436, 11.288189]

        model = get_model(model_config)

        model_config.seq_length -= 1  # set for generate data
        runner.set_train(model, model_config, loss_std=loss_std)

class TestGLM32kTrain:
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
        runner = ModelTester(run_mode='train', batch_size=1, use_label=True, experiment_mode=False)

        model_config = get_config()

        # change config for glm32k
        model_config.batch_size = 1
        model_config.compute_dtype = mstype.float16
        model_config.max_decode_length = 512
        model_config.no_recompute_layers = [20]
        model_config.param_init_type = mstype.float16
        model_config.rope_ratio = 50
        model_config.seq_length = 8192

        loss_std = [11.281455, 11.286859, 11.274545, 11.250603, 11.245409,
                    11.200018, 11.198628, 11.190212, 11.153057, 11.151159,
                    11.121563, 11.127791, 11.115051, 11.105974, 11.103176,
                    11.095471, 11.086552, 11.091341, 11.095337, 11.101830,]

        model = get_model(model_config)

        model_config.seq_length -= 1  # set for generate data
        runner.set_train(model, model_config, loss_std=loss_std)
