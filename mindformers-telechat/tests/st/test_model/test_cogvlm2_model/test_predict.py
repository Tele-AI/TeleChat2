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
Test cogvlm2-video predict.
How to run this:
    pytest tests/st/test_model/test_cogvlm2_model/test_predict.py
"""
import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype

from .base_model import get_config, get_model, get_image_model, get_image_config

ms.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})


def generate_context_positions(token_mask, target_token_id, batch_index=0):
    context_length = np.sum(token_mask.astype(np.int32))
    pos = np.where(np.array(token_mask) == target_token_id)[0]
    pos = np.expand_dims(pos, axis=0)
    pos = np.insert(pos, 0, batch_index, axis=0)
    pos = np.transpose(pos).reshape((-1, context_length, 2))
    return pos


def get_expert_mask(token_type_ids):
    vision_token_mask = np.zeros_like(token_type_ids).astype(np.bool_)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == 1) & (token_type_ids[:, 1:] == 1)
    language_token_mask = ~vision_token_mask
    if not vision_token_mask.any():
        vision_token_mask = None
    return vision_token_mask, language_token_mask


class TestCogVLM2VideoPredict:
    """A test class for testing model prediction."""

    # @pytest.mark.level0
    # @pytest.mark.platform_arm_ascend910b_training
    # @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Video model predict
        Description: Test base model prediction.
        Expectation: AssertionError
        """
        os.environ['USE_ROPE_SELF_DEFINE'] = 'True'
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        model_config = get_config()
        model = get_model(model_config)
        input_ids = np.random.randint(0, 128, size=(1, 1024), dtype=np.int32)
        input_ids = np.pad(input_ids, ((0, 0), (0, 1024)), 'constant', constant_values=128002)
        images = Tensor(np.random.random(size=(1, 3, 224, 224)), dtype=mstype.float32)
        video_context_pos = Tensor(np.array([[[0, i + 3] for i in range(66)]], dtype=np.int32))
        position_ids = Tensor(np.arange(2048, dtype=np.int32)).expand_dims(axis=0)
        valid_position = np.array([[1]], dtype=np.int32)
        _ = model.generate(input_ids=input_ids,
                           images=images,
                           video_context_pos=video_context_pos,
                           position_ids=position_ids,
                           valid_position=valid_position)


class TestCogVLM2ImagePredict:
    """A test class for testing model prediction."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Image model predict
        Description: Test image model prediction.
        Expectation: AssertionError
        """
        os.environ['USE_ROPE_SELF_DEFINE'] = 'True'
        model_config = get_image_config()
        model = get_image_model(model_config)
        input_ids = np.random.randint(0, 128, size=(1, 4096), dtype=np.int32)
        input_ids[:, 2048:] = 128002
        images = np.random.random(size=(1, 3, 1344, 1344))
        position_ids = np.expand_dims(np.arange(4096, dtype=np.int32), 0)
        token_type_ids = np.zeros((1, 4096))
        token_type_ids[:, 1:2307] = 1
        image_context_pos = generate_context_positions(token_type_ids[0], 1)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        vision_indices = generate_context_positions(vision_token_mask[0], True)
        language_indices = generate_context_positions(language_token_mask[0], True)
        _ = model.generate(input_ids=input_ids,
                           images=images,
                           image_context_pos=image_context_pos,
                           position_ids=position_ids,
                           vision_token_mask=vision_token_mask,
                           language_token_mask=language_token_mask,
                           vision_indices=vision_indices,
                           language_indices=language_indices,
                           max_length=4096)
