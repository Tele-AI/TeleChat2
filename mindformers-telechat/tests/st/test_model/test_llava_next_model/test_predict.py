# Copyright 2025 Huawei Technologies Co., Ltd
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
Test cogvlm2-video predict.
How to run this:
    pytest tests/st/test_model/test_llava_next_model/test_predict.py
"""
import os

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore as ms
from .base_model import get_config, get_model

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


class TestLlavaNextPredict:
    """A test class for testing model prediction."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_video_model(self):
        """
        Feature: Video model predict
        Description: Test base model prediction.
        Expectation: AssertionError
        """
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        model_config = get_config(video_config=True)
        model = get_model(model_config).network
        input_ids = np.random.randint(0, 128, size=(1, 2048), dtype=np.int32)
        images = Tensor(np.random.random(size=(1, 4, 3, 336, 336)), dtype=mstype.float32)
        image_context_pos = Tensor(np.array([[[[0, i + 3] for i in range(4 * 144)]]], dtype=np.int32))
        _ = model.generate(input_ids=input_ids,
                           images=images,
                           image_context_pos=image_context_pos,
                           max_new_tokens=100)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_image_model(self):
        """
        Feature: image model predict
        Description: Test base model prediction.
        Expectation: AssertionError
        """
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        model_config = get_config(video_config=False)
        model = get_model(model_config).network
        input_ids = np.random.randint(0, 128, size=(1, 3072), dtype=np.int32)
        images = Tensor(np.random.random(size=(1, 1, 3, 336, 336)), dtype=mstype.float32)
        image_patches = Tensor(np.random.random(size=(1, 2, 2, 3, 336, 336)), dtype=mstype.float32)
        image_context_pos = Tensor(np.array([[[[0, i + 3] for i in range(2928)]]], dtype=np.int32))
        _ = model.generate(input_ids=input_ids,
                           images=images,
                           image_patches=image_patches,
                           image_context_pos=image_context_pos,
                           max_new_tokens=100)
