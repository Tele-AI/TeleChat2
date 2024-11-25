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
"""Test PostProcess in Sample."""
import pytest

import numpy as np

import mindspore as ms
from mindspore import set_context, Tensor

from mindformers import build_context
from mindformers import set_context as set_context_mf
from mindformers.generation.logits_process import TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor, \
    FrequencyPenaltyLogitsProcessor, PresencePenaltyLogitsProcessor, \
    TopPLogitsWarper, TopKLogitsWarper, MinLengthLogitsProcessor, MinNewTokensLengthLogitsProcessor, \
    SamplingLogitsProcessor, GreedySearchLogitsProcessor, \
    LogitNormalization, LogitsProcessorList

set_context(device_target="Ascend", mode=ms.GRAPH_MODE)


def check_accuracy(output_np, output_ms, atol=1e-3, rtol=1e-2):
    """Check accuracy with self-defined method."""
    diff_index = np.where(np.abs(output_np - output_ms) > rtol)
    return np.array(diff_index).size < output_ms.size * atol


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_repetition_penalty():
    """
    Feature: Test RepetitionPenalty.
    Description: Check the accuracy of RepetitionPenalty between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    repetition_penalty = float(np.random.uniform(low=0.1))

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = RepetitionPenaltyLogitsProcessor(repetition_penalty=repetition_penalty)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = RepetitionPenaltyLogitsProcessor(repetition_penalty=repetition_penalty)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = RepetitionPenaltyLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   repetition_penalty=repetition_penalty)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_frequency_penalty():
    """
    Feature: Test Frequency Penalty.
    Description: Check the accuracy of Frequency Penalty between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)
    frequency_penalty = float(np.random.uniform(low=0.1))
    output_tokens_counts = np.random.uniform(low=1, high=4096, size=(1, 4096)).astype(np.int32)

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = FrequencyPenaltyLogitsProcessor(frequency_penalty=frequency_penalty,
                                                   output_tokens_counts=output_tokens_counts)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = FrequencyPenaltyLogitsProcessor(frequency_penalty=frequency_penalty,
                                                                   output_tokens_counts=output_tokens_counts)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = FrequencyPenaltyLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   frequency_penalty=frequency_penalty,
                                                                   output_tokens_counts=Tensor(output_tokens_counts))

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_presence_penalty():
    """
    Feature: Test Presence Penalty.
    Description: Check the accuracy of Presence Penalty between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)
    presence_penalty = float(np.random.uniform(low=0.1))
    output_tokens_mask = np.random.uniform(low=1, high=4096, size=(1, 4096)) > 0.5

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = PresencePenaltyLogitsProcessor(presence_penalty=presence_penalty,
                                                  output_tokens_mask=output_tokens_mask)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = PresencePenaltyLogitsProcessor(presence_penalty=presence_penalty,
                                                                  output_tokens_mask=output_tokens_mask)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = PresencePenaltyLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   presence_penalty=presence_penalty,
                                                                   output_tokens_mask=Tensor(output_tokens_mask,
                                                                                             dtype=ms.bool_))

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_temperature():
    """
    Feature: Test Temperature.
    Description: Check the accuracy of Temperature between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    temperature = float(np.random.uniform(low=0.1))

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = TemperatureLogitsWarper(temperature=temperature)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = TemperatureLogitsWarper(temperature=temperature)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = TemperatureLogitsWarper()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   temperature=temperature)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_top_k():
    """
    Feature: Test TopK.
    Description: Check the accuracy of TopK between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    max_top_k = int(np.random.uniform(low=1, high=4096))
    filter_value = float(np.random.uniform(low=-50000.0, high=50000.0))
    min_tokens_to_keep = int(np.random.uniform(low=1, high=4096))
    top_k = np.array([[max(max_top_k, min_tokens_to_keep) - 1]] * scores.shape[0])

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = TopKLogitsWarper(top_k=max_top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = TopKLogitsWarper(top_k=max_top_k, filter_value=filter_value,
                                                    min_tokens_to_keep=min_tokens_to_keep)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = TopKLogitsWarper()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   max_top_k=max_top_k, top_k=Tensor(top_k),
                                                                   filter_value=filter_value,
                                                                   min_tokens_to_keep=min_tokens_to_keep)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_top_p():
    """
    Feature: Test TopP.
    Description: Check the accuracy of TopP between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    top_p = float(np.random.uniform(low=0.1, high=1))
    filter_value = float(np.random.uniform(low=-50000.0, high=50000.0))
    min_tokens_to_keep = int(np.random.uniform(low=1, high=4096))

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = TopPLogitsWarper(top_p=top_p, filter_value=filter_value,
                                                    min_tokens_to_keep=min_tokens_to_keep)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = TopPLogitsWarper()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores), top_p=top_p,
                                                                   filter_value=filter_value,
                                                                   min_tokens_to_keep=min_tokens_to_keep)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sampling():
    """
    Feature: Test MinLength.
    Description: Check the accuracy of MinLength between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(32, 4096).astype(np.float32)
    do_sample = np.random.uniform(size=(32,)) > 0.5
    seed_array = (np.random.rand(32,) * 1000.0).astype(np.int32)

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = SamplingLogitsProcessor(do_sample=do_sample, seed_array=seed_array)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = SamplingLogitsProcessor(do_sample=do_sample, seed_array=seed_array)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = SamplingLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   do_sample=Tensor(do_sample), seed_array=seed_array)

    assert check_accuracy(output_np[0], output_ms_with_attributes[0].asnumpy())
    assert check_accuracy(output_np[1], output_ms_with_attributes[1].asnumpy())
    assert check_accuracy(output_np[0], output_ms_without_attributes[0].asnumpy())
    assert check_accuracy(output_np[1], output_ms_without_attributes[1].asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_min_length():
    """
    Feature: Test MinLength.
    Description: Check the accuracy of MinLength between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    min_length = 5000  # make sure `cur_len < min_length`
    eos_token_id = [int(np.random.uniform(low=-1, high=4096))] * int(np.random.uniform(low=1, high=4096))
    pad_token_id = int(np.random.uniform(low=-1, high=4096))

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_token_id,
                                                            pad_token_id=pad_token_id)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = MinLengthLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   pad_token_id=pad_token_id, min_length=min_length,
                                                                   eos_token_id=eos_token_id)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_min_new_tokens_length():
    """
    Feature: Test MinNewTokensLength.
    Description: Check the accuracy of MinNewTokensLength between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    prompt_length_to_skip = int(np.random.uniform(low=1, high=4096))
    min_new_tokens = 5000  # make sure `new_tokens_length < min_new_tokens`
    eos_token_id = [int(np.random.uniform(low=-1, high=4096))] * int(np.random.uniform(low=1, high=4096))
    pad_token_id = int(np.random.uniform(low=-1, high=4096))

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=prompt_length_to_skip,
                                                     min_new_tokens=min_new_tokens, eos_token_id=eos_token_id,
                                                     pad_token_id=pad_token_id)
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    # Init with attributes
    processor_ms_with_attributes = MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=prompt_length_to_skip,
                                                                     min_new_tokens=min_new_tokens,
                                                                     eos_token_id=eos_token_id,
                                                                     pad_token_id=pad_token_id)
    output_ms_with_attributes = processor_ms_with_attributes(Tensor(input_ids), Tensor(scores))
    # Init without attributes
    processor_ms_without_attributes = MinNewTokensLengthLogitsProcessor()
    output_ms_without_attributes = processor_ms_without_attributes(Tensor(input_ids), Tensor(scores),
                                                                   pad_token_id=pad_token_id,
                                                                   prompt_length_to_skip=prompt_length_to_skip,
                                                                   eos_token_id=eos_token_id,
                                                                   min_new_tokens=min_new_tokens)

    assert check_accuracy(output_np, output_ms_with_attributes.asnumpy())
    assert check_accuracy(output_np, output_ms_without_attributes.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_normalization():
    """
    Feature: Test Normalization.
    Description: Check the accuracy of Normalization between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = LogitNormalization()
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    processor_ms = LogitNormalization()
    output_ms = processor_ms(Tensor(input_ids), Tensor(scores))

    assert check_accuracy(output_np, output_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_greedy_search():
    """
    Feature: Test GreedySearch.
    Description: Check the accuracy of GreedySearch between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(1, 4096)).astype(np.int32)
    scores = np.random.randn(1, 4096).astype(np.float32)

    # Test Numpy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processor_np = GreedySearchLogitsProcessor()
    output_np = processor_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    processor_ms = GreedySearchLogitsProcessor()
    output_ms = processor_ms(Tensor(input_ids), Tensor(scores))

    assert check_accuracy(output_np, output_ms.asnumpy())


def init_logits_processor_list():
    """
    Init LogitsProcessorList.
    """
    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty=0.9))
    processors.append(MinLengthLogitsProcessor(min_length=5000, eos_token_id=[100, 1000, 4000], pad_token_id=1000))
    processors.append(MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=1000, min_new_tokens=5000,
                                                        eos_token_id=[100, 1000, 4000], pad_token_id=1000))
    processors.append(LogitNormalization())
    processors.append(TemperatureLogitsWarper(temperature=0.9))
    processors.append(LogitNormalization())
    processors.append(TopKLogitsWarper(top_k=1000, filter_value=10000.0, min_tokens_to_keep=3000))
    processors.append(TopPLogitsWarper(top_p=0.4, filter_value=10000.0, min_tokens_to_keep=3000))
    return processors


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_logits_processor_list():
    """
    Feature: Test LogitsProcessorList.
    Description: Check the accuracy of LogitsProcessorList between Numpy and MindSpore operators.
    Expectation: Outputs between them are the same.
    """
    input_ids = np.random.uniform(low=0, high=4096, size=(32, 4096)).astype(np.int32)
    scores = np.random.randn(32, 4096).astype(np.float32)

    # Test NumPy
    build_context({"postprocess_use_numpy": True, "context": {}, "parallel": {}})
    processors_np = init_logits_processor_list()
    output_np = processors_np(input_ids.copy(), scores.copy())

    # Test MindSpore
    set_context_mf(postprocess_use_numpy=False)
    processors_ms = init_logits_processor_list()
    output_ms = processors_ms(input_ids.copy(), scores.copy())

    assert check_accuracy(output_np, output_ms)
