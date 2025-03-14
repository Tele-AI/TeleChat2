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
pytest tests/st/test_text_generator/test_dynamic_ntk.py
"""
import numpy as np

import mindspore as ms
from mindspore import set_seed

from research.telechat2.telechat_config import TelechatConfig
from research.telechat2.infer.telechat import ParallelTelechatForCausalLM

ms.set_context(mode=0, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})
ms.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)

class TestIncrementalGenerate:
    """A test class for testing dynamic_ntk feature."""

    def setup_method(self):
        """setup method."""
        set_seed(0)
        np.random.seed(0)

        vocab_size = 128
        seq_length = 64
        max_position_embedding = 32
        input_len_short = 8
        input_len_long = 48

        model_config = TelechatConfig(seq_length=seq_length, max_position_embedding=max_position_embedding,
                                      num_layers=1, num_heads=2, hidden_size=256, vocab_size=vocab_size,
                                      extend_method="DYNAMIC_NTK", use_past=True, use_flash_attention=True,
                                      do_sample=False)
        self.model = ParallelTelechatForCausalLM(model_config)

        input_ids_short = np.random.randint(low=0, high=vocab_size, size=input_len_short)
        self.input_ids_short = np.pad(input_ids_short, (0, input_len_long - input_len_short),
                                      mode='constant', constant_values=model_config.pad_token_id).tolist()
        self.input_ids_long = np.random.randint(low=0, high=vocab_size, size=input_len_long).tolist()
        self.input_ids_short_long = [self.input_ids_short, self.input_ids_long]
        self.input_ids_long_short = [self.input_ids_long, self.input_ids_short]

    def test_incremental_generate(self):
        """
        Feature: generate with dynamic ntk.
        Description: Test correctness of incremental generate.
        Expectation: AssertionError
        """
        self.model.generate(self.input_ids_short)
        output_ids_long = self.model.generate(self.input_ids_long)
        output_ids_short_long = self.model.generate(self.input_ids_short_long)
        output_ids_long_short = self.model.generate(self.input_ids_long_short)

        assert np.array_equal(output_ids_short_long[1], output_ids_long[0]), \
                f"output_ids_short_long[1]: {output_ids_short_long[1]} \
                    is not equal to output_ids_long: {output_ids_long[0]}"
        assert np.array_equal(output_ids_short_long[1], output_ids_long_short[0]), \
                f"output_ids_short_long[1]: {output_ids_short_long[1]} \
                    is not equal to output_ids_long_short[0]: {output_ids_long_short[0]}"
        assert np.array_equal(output_ids_short_long[0], output_ids_long_short[1]), \
                f"output_ids_short_long[0]: {output_ids_short_long[0]} \
                    is not equal to output_ids_long_short[1]: {output_ids_long_short[1]}"
