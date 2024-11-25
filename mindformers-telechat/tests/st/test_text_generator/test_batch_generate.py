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
Test module for testing the batch infer for text generator.
How to run this:
pytest tests/st/test_text_generator/test_batch_generate.py
"""
import numpy as np

import mindspore as ms
from mindspore import set_seed

from mindformers import GPT2Config, GPT2LMHeadModel

ms.set_context(mode=0)


class TestBatchGenerate:
    """A test class for testing text generate features."""

    def setup_method(self):
        """setup method."""
        set_seed(0)
        np.random.seed(0)

        vocab_size = 50257
        input_length = 8
        seq_length = 64

        model_config = GPT2Config(batch_size=2, num_layers=2, seq_length=seq_length, use_past=True,
                                  checkpoint_name_or_path="")
        self.model = GPT2LMHeadModel(model_config)

        self.input_ids = np.pad(np.random.randint(low=0, high=vocab_size, size=(2, input_length)),
                                ((0, 0), (0, seq_length - input_length)), mode='constant',
                                constant_values=50256).tolist()

        self.output_ids_std = np.array([[27469, 38984, 6921, 38804, 2163, 5072, 37619, 7877, 32803,
                                         32803, 26194, 26625, 41277, 41277, 22424, 1726, 1726, 35640,
                                         46761, 31067, 14103, 11457, 16178, 16777, 16777, 8425, 14103,
                                         29417, 14103, 16777, 16777, 14103, 14103, 29417, 31297, 18492,
                                         18492, 18492, 13132, 14103, 14103, 31297, 45853, 45853, 45853,
                                         45853, 24279, 24279, 22424, 3956, 25580, 27433, 18492, 5788,
                                         3956, 3956, 20097, 20097, 20097, 20097, 14309, 18014, 13813],
                                        [18430, 1871, 7599, 2496, 47954, 24675, 42968, 31921, 34877,
                                         36731, 36731, 48385, 21475, 21475, 48385, 13517, 30242, 30242,
                                         30242, 38670, 39402, 47589, 6326, 30647, 20887, 20887, 40286,
                                         40286, 15759, 5150, 34877, 7568, 7568, 14777, 14777, 14777,
                                         14777, 1727, 1727, 127, 6328, 6328, 5250, 16862, 25537,
                                         10999, 1459, 1459, 6328, 20298, 6328, 6328, 43763, 40928,
                                         27081, 689, 689, 689, 13487, 42267, 42267, 42267, 42267]]).astype(np.int32)

    def test_batch_generate(self):
        """
        Feature: batch generate.
        Description: Test correctness of batch generate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        output_ids = self.model.generate(self.input_ids)

        assert np.array_equal(output_ids, self.output_ids_std), \
            f"output_ids: {output_ids} is not equal to output_ids_std: {self.output_ids_std}"
