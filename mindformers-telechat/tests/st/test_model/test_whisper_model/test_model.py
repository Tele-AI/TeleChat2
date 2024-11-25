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
Test whisper model.
How to run this:
    pytest tests/st/test_model/test_whisper_model/test_model.py
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as mstype
from mindformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from .base_model import get_config


class TestWhisper:
    """A test class for testing model"""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_foward(self):
        """
        Feature: Whisper model train
        Description: Test model train.
        Expectation: AssertionError
        """
        config = get_config()
        model = WhisperForConditionalGeneration(config)

        feature_size = 128
        input_features = np.random.randn(1, feature_size, 3000)
        input_features = ms.Tensor(input_features)

        decoder_input_ids = np.arange(0, config.max_target_positions)
        decoder_input_ids = np.expand_dims(decoder_input_ids, 0)
        decoder_input_ids = ms.Tensor(decoder_input_ids, dtype=mstype.int32)

        _ = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
