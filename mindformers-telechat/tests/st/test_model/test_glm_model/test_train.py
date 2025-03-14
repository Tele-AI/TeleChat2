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
Test glm train.
How to run this:
    pytest tests/st/test_model/test_glm_model/test_train.py
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 512
    vocab_size = 130528
    input_ids = np.random.randint(low=0, high=vocab_size, size=(seq_len,)).astype(np.int32)
    label = np.random.randint(low=0, high=vocab_size, size=(seq_len,)).astype(np.int32)
    position_ids = np.ones((2, seq_len)).astype(np.int64)
    attention_mask = np.ones(shape=(1, seq_len, seq_len)).astype(np.int32)
    train_data = (input_ids, label, position_ids, attention_mask)
    for _ in range(80):
        yield train_data


class TestGLMTrain:
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
        runner = ModelTester(run_mode='train', batch_size=4, experiment_mode=False)

        model_config = get_config()

        loss_std = [11.952507, 11.724562, 11.393075, 11.019833, 10.632746,
                    10.241946, 9.851490, 9.464133, 9.083885, 8.718346,
                    8.387368, 8.119299, 7.878355, 7.665306, 7.503242,
                    7.374232, 7.261548, 7.207399, 7.165281, 7.148236,]

        model = get_model(model_config)


        train_dataset = GeneratorDataset(generator_train,
                                         column_names=["input_ids", "label", "position_ids", "attention_mask"])
        train_dataset = train_dataset.batch(batch_size=4)

        runner.set_train(model, model_config, dataset=train_dataset, loss_std=loss_std)
