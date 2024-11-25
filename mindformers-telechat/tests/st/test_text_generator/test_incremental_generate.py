# Copyright 2023 Huawei Technologies Co., Ltd
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
Test module for testing the incremental generate for text generator.
How to run this:
pytest tests/st/test_text_generator/test_incremental_generate.py
"""
import numpy as np

import mindspore as ms
from mindspore import set_seed

from mindformers import GPT2Config, GPT2LMHeadModel

ms.set_context(mode=0)


class TestIncrementalGenerate:
    """A test class for testing text generate features."""

    def setup_method(self):
        """setup method."""
        set_seed(0)
        np.random.seed(0)

        vocab_size = 50257
        input_length = 8
        seq_length = 64

        model_config = GPT2Config(num_layers=2, seq_length=seq_length, use_past=True, checkpoint_name_or_path="")
        self.model = GPT2LMHeadModel(model_config)

        self.input_ids = np.pad(np.random.randint(low=0, high=vocab_size, size=input_length),
                                (0, seq_length - input_length), mode='constant', constant_values=50256).tolist()

        self.output_ids_std = np.array([27469, 38984, 6921, 38804, 2163, 5072, 37619, 7877, 32803,
                                        38625, 38625, 26625, 22424, 22424, 23882, 23882, 23882, 22402,
                                        30205, 30205, 33862, 48066, 16777, 16777, 16777, 46218, 46218,
                                        46218, 46218, 46218, 25580, 33862, 1493, 19908, 19908, 41949,
                                        41949, 41949, 25580, 29887, 29887, 29887, 3956, 29417, 28231,
                                        29417, 31297, 31297, 3956, 3956, 43313, 15428, 31127, 31127,
                                        3956, 3956, 25580, 9880, 26194, 22404, 44286, 3892, 3892]).astype(np.int32)

    def test_incremental_generate(self):
        """
        Feature: incremental generate.
        Description: Test correctness of incremental generate.
        Expectation: AssertionError
        """
        output_ids = self.model.generate(self.input_ids)

        assert np.array_equal(output_ids, self.output_ids_std), \
            f"output_ids: {output_ids} is not equal to output_ids_std: {self.output_ids_std}"
