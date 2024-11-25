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
Test cogvlm2-video train.
How to run this:
    pytest tests/st/test_model/test_cogvlm2_model/test_train.py
"""
import os
from functools import partial
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.dataset import GeneratorDataset

from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


def generate_data(seq_len, vocab_size, step_num=20):
    """generate data for testing model."""
    for _ in range(step_num):
        input_ids = np.random.randint(0, vocab_size, size=(seq_len + 1,), dtype=np.int64)
        images = Tensor(np.random.random(size=(24, 3, 224, 224)), dtype=mstype.float32)
        video_context_pos = []
        for i in range(24):
            cur_idx = i * 67 + 1
            video_context_pos.append(np.array([[[0, j + cur_idx] for j in range(66)]], dtype=np.int32))
        video_context_pos = Tensor(np.concatenate(video_context_pos))
        position_ids = Tensor(np.arange(seq_len + 1, dtype=np.int32))
        yield input_ids, images, video_context_pos, position_ids


def get_dataset(seq_len, vocab_size):
    """build dataset for model training."""
    prepare_data = partial(generate_data, seq_len=seq_len, vocab_size=vocab_size)
    dataset = GeneratorDataset(
        prepare_data, column_names=["input_ids", "images", "video_context_pos", "position_ids"])
    dataset = dataset.batch(batch_size=1)
    return dataset


class TestCogVLM2VideoTrain:
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
        os.environ['USE_ROPE_SELF_DEFINE'] = 'True'
        runner = ModelTester(run_mode='train', batch_size=1, experiment_mode=True)

        model_config = get_config()
        # setting use_past = True in training will cause error
        model_config.use_past = False
        model_config.llm_model.model_config.use_past = False
        model_config.is_dynamic = False
        model_config.llm_model.model_config.is_dynamic = False

        model = get_model(model_config)

        dataset = get_dataset(seq_len=model_config.llm_model.model_config.seq_length,
                              vocab_size=model_config.llm_model.model_config.vocab_size)

        runner.set_train(model, model_config, dataset=dataset, task='multi_modal_to_text_generation')
