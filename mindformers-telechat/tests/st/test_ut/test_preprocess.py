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
"""Test PreProcess in GenerationMixin.forward"""
from unittest import mock

import time
import pytest

import numpy as np
from mindspore import set_context, Tensor

from mindformers.generation.text_generator import GenerationMixin

set_context(device_target='CPU')


class TestConfig:
    def __init__(self):
        self.is_encoder_decoder = False


class TestGenerationMixin:
    """
    Test GenerationMixin class
    Mock:
        prepare_inputs_for_generation: defined in model.
        add_flags_custom: defined in model.
        call_perf: get end time of preprocess.
        call_accuracy: get outputs of preprocess.
    """

    def __init__(self):
        self.config = TestConfig()
        self._pre_set_phase = None
    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": Tensor.from_numpy(input_ids)}

    # pylint: disable=W0613
    def add_flags_custom(self, is_first_iteration):
        """Mock it."""
        return

    # pylint: disable=W0613
    def call_perf(self, input_ids, input_position=None, init_reset=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None):
        time_end = time.time()
        return time_end

    def call_accuracy(self, input_ids, input_position=None, init_reset=None, batch_valid_length=None, block_tables=None,
                      slot_mapping=None):
        return {"input_ids": input_ids, "input_position": input_position, "init_reset": init_reset,
                "batch_valid_length": batch_valid_length, "block_tables": block_tables, "slot_mapping": slot_mapping}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.generation.text_generator.GenerationMixin.__init__', TestGenerationMixin.__init__)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.prepare_inputs_for_generation',
            TestGenerationMixin.prepare_inputs_for_generation)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.add_flags_custom',
            TestGenerationMixin.add_flags_custom)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.__call__', TestGenerationMixin.call_perf)
@pytest.mark.parametrize('batch_size', [1, 8, 16])
@pytest.mark.parametrize('seq_length', [256, 512, 1024, 2048])
@pytest.mark.parametrize('prefill', [True, False])
def test_preprocess_perf(batch_size, seq_length, prefill):
    """
    Feature: Test preprocess performance.
    Description: Check the time delay for preprocess.
    Expectation: Success.
    """
    input_ids = np.arange(batch_size * seq_length).reshape(batch_size, seq_length)
    valid_length_each_example = np.random.uniform(1, seq_length, (batch_size,)).astype(np.int32)
    block_tables = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    slot_mapping = np.arange(32 * 256).astype(np.int32)
    generation_mixin = GenerationMixin()

    # Warm Up
    for _ in range(3):
        _, _ = generation_mixin.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                        block_tables=block_tables, slot_mapping=slot_mapping,
                                        prefill=prefill, use_past=True)
    # Infer
    time_list = []
    for _ in range(10):
        time_start = time.time()
        time_end, _ = generation_mixin.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                               block_tables=block_tables, slot_mapping=slot_mapping,
                                               prefill=prefill, use_past=True)
        time_list.append((time_end - time_start) * 1000.0)
    if prefill:
        # Not include add_flags_custom
        assert np.mean(time_list) < 0.4
    else:
        assert np.mean(time_list) < 0.5


def get_expected_outputs(input_ids, valid_length_each_example, prefill):
    """
    Get expected_outputs of preprocess.
    """
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]

    if not prefill and input_ids.shape[-1] != 1:
        # Only valid in Decode
        inputs_tmp = []
        for i, index_value in enumerate(current_index):
            current_index_tmp = (
                int(index_value) - i * input_ids.shape[1]
            )  # multibatch
            # use numpy to slice array to avoid complie ascend slice op
            inputs_tmp.append(input_ids[i][current_index_tmp: current_index_tmp + 1])
        inputs_tmp = np.array(inputs_tmp, dtype=np.int32)
        input_ids = inputs_tmp
    input_position = np.array(current_index).astype(np.int32)
    init_reset = np.array([not prefill], dtype=np.bool_)
    batch_valid_length = np.array([valid_length_each_example], dtype=np.int32)
    return {"input_ids": input_ids, "input_position": input_position, "init_reset": init_reset,
            "batch_valid_length": batch_valid_length}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.generation.text_generator.GenerationMixin.__init__', TestGenerationMixin.__init__)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.prepare_inputs_for_generation',
            TestGenerationMixin.prepare_inputs_for_generation)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.add_flags_custom',
            TestGenerationMixin.add_flags_custom)
@mock.patch('mindformers.generation.text_generator.GenerationMixin.__call__', TestGenerationMixin.call_accuracy)
@pytest.mark.parametrize('batch_size', [1, 8, 16])
@pytest.mark.parametrize('seq_length', [256, 512, 1024, 2048])
@pytest.mark.parametrize('prefill', [True, False])
def test_preprocess_accuracy(batch_size, seq_length, prefill):
    """
    Feature: Test preprocess accuracy.
    Description: Check the accuracy for preprocess.
    Expectation: Success.
    """
    input_ids = np.arange(batch_size * seq_length).reshape(batch_size, seq_length)
    valid_length_each_example = np.random.uniform(1, seq_length, (batch_size,)).astype(np.int32)
    block_tables = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    slot_mapping = np.arange(32 * 256).astype(np.int32)
    generation_mixin = GenerationMixin()

    outputs, _ = generation_mixin.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                          block_tables=block_tables, slot_mapping=slot_mapping,
                                          prefill=prefill, use_past=True)
    expected_outputs = get_expected_outputs(input_ids, valid_length_each_example, prefill)
    assert np.allclose(outputs["input_ids"].asnumpy(), expected_outputs["input_ids"])
    assert np.allclose(outputs["batch_valid_length"].asnumpy(), expected_outputs["batch_valid_length"])
    assert np.allclose(outputs["block_tables"].asnumpy(), block_tables)
    assert np.allclose(outputs["slot_mapping"].asnumpy(), slot_mapping)
