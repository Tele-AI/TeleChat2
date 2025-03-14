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
Test llava next video train.
How to run this:
    pytest tests/st/test_model/test_llava_next_model/test_train.py
"""
from functools import partial

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.dataset import GeneratorDataset

from tests.utils.model_tester import ModelTester
from .base_model import get_config, get_model

ms.set_context(mode=0)


def generate_data(seq_len, vocab_size, is_video=True, step_num=20):
    """generate video data for testing model."""
    for _ in range(step_num):
        input_ids = np.random.randint(0, vocab_size, size=(seq_len + 1,), dtype=np.int64)
        image_frames = 16 if is_video else 1
        images = Tensor(np.random.random(size=(image_frames, 3, 336, 336)), dtype=mstype.float32)
        placeholder_length = 144 if is_video else 5832
        image_context_pos = []
        for i in range(image_frames):
            cur_idx = i * (placeholder_length + 1) + 1
            image_context_pos.append(np.array([[[0, j + cur_idx] for j in range(placeholder_length)]], dtype=np.int32))
        image_context_pos = Tensor(np.concatenate(image_context_pos).reshape((1, -1, 2)))
        if is_video:
            yield input_ids, images, image_context_pos
        else:
            image_patches = Tensor(np.random.random(size=(3, 3, 3, 336, 336)), dtype=mstype.float32)
            yield input_ids, images, image_patches, image_context_pos


def get_dataset(seq_len, vocab_size, is_video=True):
    """build dataset for model training."""
    columns_name = ["input_ids", "images", "image_patches", "image_context_pos"]
    if is_video:
        columns_name = ["input_ids", "images", "image_context_pos"]
    prepare_data = partial(generate_data, seq_len=seq_len, vocab_size=vocab_size, is_video=is_video)
    dataset = GeneratorDataset(
        prepare_data, column_names=columns_name)
    dataset = dataset.batch(batch_size=1)
    return dataset


class TestLlavaNextTrain:
    """A test class for testing model training precision."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_video_model(self):
        """
        Feature: Base video model train
        Description: Test base model training precision.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='train', batch_size=1, experiment_mode=True)

        model_config = get_config(video_config=True)
        model_config.use_past = False
        model_config.text_model.model_config.use_past = False
        model_config.is_dynamic = False
        model_config.text_model.model_config.is_dynamic = False
        model = get_model(model_config)
        dataset = get_dataset(seq_len=model_config.seq_length,
                              vocab_size=model_config.text_model.model_config.vocab_size, is_video=True)
        runner.set_train(model, model_config, dataset=dataset, task='multi_modal_to_text_generation')

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_image_model(self):
        """
        Feature: Base image model train
        Description: Test base model training precision.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='train', batch_size=1, experiment_mode=True)

        model_config = get_config(video_config=False)
        model_config.use_past = False
        model_config.text_model.model_config.use_past = False
        model_config.is_dynamic = False
        model_config.text_model.model_config.is_dynamic = False
        model = get_model(model_config)
        dataset = get_dataset(seq_len=model_config.seq_length,
                              vocab_size=model_config.text_model.model_config.vocab_size, is_video=False)
        runner.set_train(model, model_config, dataset=dataset, task='multi_modal_to_text_generation')
