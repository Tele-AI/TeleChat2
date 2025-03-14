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
"""test class ModelRunner."""
from unittest import mock

import os
import pytest

import numpy as np
from mindspore import Tensor

from mindformers.model_runner import ModelRunner


class TestModel:
    """
    Test Model.
    """
    def forward(self, input_ids, valid_length_each_example, block_tables, slot_mapping, prefill, use_past,
                position_ids=None, spec_mask=None, q_seq_lens=None, adapter_ids=None, prefill_head_indices=None,
                mindie_warm_up=False, key_cache=None, value_cache=None):
        """
        Check the info of inputs

        Args:
            input_ids (np.ndarray): rank is 2, and data type is int32.
            valid_length_each_example (Union[np.ndarray, list]): rank is 1, and data type is int32.
            block_tables (np.ndarray): rank is 2, and data type is int32.
            slot_mapping (np.ndarray): rank is 1, and data type is int32.
            prefill (bool).
            use_past (bool).
            position_ids (Union[np.ndarray, list]): rank is 1, and data type is int32.
            spec_mask (np.ndarray): rank is 2 or 3, and data type is float16.
            q_seq_lens (Union[np.ndarray, list]): rank is 1, and data type is int32.
            adapter_ids (list): rank is 1, and data type is string
            prefill_head_indices (Union[np.ndarray, list]): rank is 1, and data type is int32.
            mindie_warm_up (bool).

        Return:
            res: (Tensor): given that shape is (2, 16000).
            current_index (list): given that length is 1.
        """
        assert isinstance(input_ids, np.ndarray) and input_ids.ndim == 2 and input_ids.dtype == np.int32
        assert isinstance(valid_length_each_example, (np.ndarray, list))
        if isinstance(valid_length_each_example, np.ndarray):
            assert valid_length_each_example.ndim == 1 and valid_length_each_example.dtype == np.int32
        else:
            assert isinstance(valid_length_each_example[0], int)
        assert isinstance(block_tables, np.ndarray) and block_tables.ndim == 2 and block_tables.dtype == np.int32
        assert isinstance(slot_mapping, np.ndarray) and slot_mapping.ndim == 1 and slot_mapping.dtype == np.int32
        assert isinstance(prefill, bool)
        assert isinstance(use_past, bool)
        assert isinstance(mindie_warm_up, bool)
        assert isinstance(key_cache, type(None))
        assert isinstance(value_cache, type(None))
        if position_ids is not None:
            if isinstance(position_ids, np.ndarray):
                assert position_ids.ndim == 1 and position_ids.dtype == np.int32
            else:
                assert isinstance(position_ids[0], int)
        if spec_mask is not None:
            assert isinstance(spec_mask, np.ndarray) and (2 <= spec_mask.ndim <= 3) and spec_mask.dtype == np.float16
        if q_seq_lens is not None:
            if isinstance(q_seq_lens, np.ndarray):
                assert q_seq_lens.ndim == 1 and q_seq_lens.dtype == np.int32
            else:
                assert isinstance(q_seq_lens[0], int)
        if adapter_ids is not None:
            assert isinstance(adapter_ids, list) and adapter_ids.dtype == np.str

        if prefill_head_indices is not None:
            if isinstance(prefill_head_indices, np.ndarray):
                assert prefill_head_indices.ndim == 1 and prefill_head_indices.dtype == np.int32
            else:
                assert isinstance(prefill_head_indices[0], int)
        res = np.arange(32000).reshape(2, -1)
        current_index = [1]
        return Tensor.from_numpy(res), current_index


class TestMindIEModelRunner:
    """
    Test MindIEModelRunner API.
    1. Check the type of `__init__` attributes.
    2. Check the type of `forward` inputs.
    3. Check the dimension and data type of `forward` inputs if they are `np.ndarray` or `Tensor`.
    4. Check the consistency between the `res` in model and the `logits` in model_runner which should be same
       if the `res` is `Tensor`, otherwise, `res` would be a list[Tensor], the shape of `res[0]` is compared.
    """
    def __init__(self, model_path, config_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                 npu_device_ids=None, plugin_params=None):
        """Test __init__ api"""
        self.warmup_step = 2
        self.is_multi_modal_model = False
        self.model = TestModel()
        assert isinstance(model_path, str)
        assert isinstance(config_path, str)
        assert isinstance(npu_mem_size, int)
        assert isinstance(cpu_mem_size, int)
        assert isinstance(block_size, int)
        assert isinstance(rank_id, int)
        assert isinstance(world_size, int)
        assert isinstance(npu_device_ids, list) and isinstance(npu_device_ids[0], int)
        if plugin_params:
            assert isinstance(plugin_params, str)

        assert config_path == os.path.join(model_path, 'test_config.yaml')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@mock.patch('mindformers.model_runner.MindIEModelRunner.__init__', TestMindIEModelRunner.__init__)
def test_model_runner():
    """
    Feature: Test ModelRunner API.
    Description: Test ModelRunner API.
    Expectation: Success.
    """
    current_file_path = os.path.dirname(__file__)
    model_path = os.path.join(current_file_path, "test_files")
    npu_mem_size = 3
    cpu_mem_size = 1
    block_size = 128
    rank_id = 0
    world_size = 1
    npu_device_ids = [0]
    model_runner = ModelRunner(model_path=model_path, npu_mem_size=npu_mem_size, cpu_mem_size=cpu_mem_size,
                               block_size=block_size, rank_id=rank_id, world_size=world_size,
                               npu_device_ids=npu_device_ids)
    model_runner.use_legacy = True
    input_ids = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    valid_length_each_example = np.arange(32).astype(np.int32)
    block_tables = np.arange(32 * 256).reshape(32, 256).astype(np.int32)
    slot_mapping = np.arange(32 * 256).astype(np.int32)
    # Given that `prefill` is False, the shape of `logits` should be the same as that of `res` in model.forward
    prefill = False
    logits = model_runner.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                  block_tables=block_tables, slot_mapping=slot_mapping, prefill=prefill)
    assert logits.shape == (2, 16000)
    # Given that `prefill` is True,
    # the shape of `logits` should be the same as that of `res[current_index]` in model.forward
    prefill = True
    logits = model_runner.forward(input_ids=input_ids, valid_length_each_example=valid_length_each_example,
                                  block_tables=block_tables, slot_mapping=slot_mapping, prefill=prefill)
    assert logits.shape == (1, 16000)
