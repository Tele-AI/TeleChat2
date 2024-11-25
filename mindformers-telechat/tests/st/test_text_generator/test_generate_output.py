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
Test module for testing output for text generator.
How to run this:
pytest tests/st/test_text_generator/test_generate_output.py
"""
import numpy as np

import mindspore as ms
from mindspore import set_seed

from mindformers import GPT2Config, GPT2LMHeadModel
from mindformers.generation.utils import GenerateOutput

ms.set_context(mode=0)


class TestGenerateOutput:
    """A test class for testing text generate features."""

    def setup_method(self):
        """setup method."""
        set_seed(0)
        np.random.seed(0)

        vocab_size = 50257
        input_length = 8
        self.seq_length = 64

        model_config = GPT2Config(num_layers=2, seq_length=self.seq_length, use_past=True, checkpoint_name_or_path="")
        self.model = GPT2LMHeadModel(model_config)

        self.input_ids = np.pad(np.random.randint(low=0, high=vocab_size, size=input_length),
                                (0, self.seq_length - input_length), mode='constant', constant_values=50256).tolist()

        std_list = [27469, 38984, 6921, 38804, 2163, 5072, 37619, 7877, 22424, 22424, 22424, 22424, 22424, 22424,
                    22424, 36387, 36387, 36387]
        self.output_ids_std = np.array(std_list).astype(np.int32)

    def test_generate(self):
        """
        Feature: batch generate.
        Description: Test correctness of batch generate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        max_new_tokens = 10
        output_ids = self.model.generate(self.input_ids, do_sample=False, max_new_tokens=max_new_tokens)
        assert np.array_equal(output_ids[0], self.output_ids_std), \
            f"output_ids: {output_ids} is not equal to output_ids_std: {self.output_ids_std}"

        result = self.model.generate(self.input_ids, do_sample=False, max_new_tokens=max_new_tokens,
                                     return_dict_in_generate=True, output_scores=True, output_logits=True)
        assert isinstance(result, GenerateOutput)
        assert np.array_equal(result["sequences"][0], self.output_ids_std)
        assert isinstance(result["scores"], tuple)
        assert len(result["scores"]) == max_new_tokens
        assert isinstance(result["logits"], tuple)
        assert len(result["logits"]) == max_new_tokens

        result = self.model.generate(self.input_ids, do_sample=False, max_new_tokenss=max_new_tokens,
                                     return_dict_in_generate=True, output_scores=True, output_logits=False)
        assert isinstance(result["scores"], tuple)
        assert len(result["scores"]) == max_new_tokens
        assert result["logits"] is None

    def test_generate_max_length(self):
        """
        Feature: test generate max length.
        Description: Test correctness of generate max length.
        Expectation: AssertionError
        """
        max_length = 100

        output_ids = self.model.generate(self.input_ids, do_sample=False, max_length=max_length)
        assert len(output_ids[0]) <= self.seq_length

        max_new_tokens = 100
        output_ids = self.model.generate(self.input_ids, do_sample=False, max_length=max_length,
                                         max_new_tokens=max_new_tokens, eos_token_id=[-1])
        assert len(output_ids[0]) <= self.seq_length
