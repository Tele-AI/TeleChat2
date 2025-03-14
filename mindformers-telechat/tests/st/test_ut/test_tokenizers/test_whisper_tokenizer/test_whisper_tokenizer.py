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
"""test whisper tokenizer."""
import unittest

import pytest
from tests.utils.model_tester import create_whisper_tokenizer


class TestWhisperTokenizer(unittest.TestCase):
    """ A test class for testing Whisper tokenizer"""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = create_whisper_tokenizer()
        cls.string = "An increasing sequence: one, two, three."
        cls.input_ids = [50258, 50276, 50360, 50364, 7828, 5662, 8310, 25, 472, 11, 732, 11, 1045, 13, 50257]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        cls.token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cls.token_ids_0 = [1, 2, 3, 4]
        cls.token_ids_1 = [5, 6, 7, 8]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call_method(self):
        res = self.tokenizer(self.string)
        assert res.input_ids == self.input_ids
        assert res.attention_mask == self.attention_mask

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenize(self):
        res = self.tokenizer.tokenize(self.string)
        assert res == ['An', 'Ġincreasing', 'Ġsequence', ':', 'Ġone', ',', 'Ġtwo', ',', 'Ġthree', '.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad(self, max_length=20):
        res = self.tokenizer(self.string, max_length=max_length, padding="max_length")
        assert res.input_ids == self.input_ids + [self.tokenizer.pad_token_id] * (max_length - len(self.input_ids))
        assert res.attention_mask == self.attention_mask + [0] * (max_length - len(self.input_ids))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode(self):
        res = self.tokenizer.encode(self.string)
        assert res == self.input_ids

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation(self, max_length=5):
        res = self.tokenizer(self.string, max_length=max_length)
        assert res.input_ids == [50258, 50276, 50360, 50364, 7828, 5662, 8310, 25, 472, 11, 732, 11, 1045, 13, 50257]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_inputs_with_special_tokens(self):
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0)
        assert res == [50258, 50276, 50360, 50364, 1, 2, 3, 4, 50257]
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0, self.token_ids_1)
        assert res == [50258, 50276, 50360, 50364, 1, 2, 3, 4, 5, 6, 7, 8, 50257]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_create_token_type_ids_from_sequences(self):
        res = self.tokenizer.create_token_type_ids_from_sequences(self.token_ids_0, self.token_ids_1)
        assert res == [0, 0, 0, 0, 1, 1, 1, 1]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_special_tokens_mask(self):
        res = self.tokenizer.get_special_tokens_mask(self.token_ids_0)
        assert res == [1, 1, 1, 1, 0, 0, 0, 0, 1]
        res = self.tokenizer.get_special_tokens_mask(self.token_ids_0, self.token_ids_1)
        assert res == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        res = self.tokenizer.get_special_tokens_mask(self.token_ids_0, already_has_special_tokens=True)
        assert res == [0, 0, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res == ('<|startoftranscript|><|hi|><|transcribe|><|notimestamps|>'
                       'An increasing sequence: one, two, three.<|endoftext|>')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        res = self.tokenizer.get_vocab()
        assert len(res) == 51866

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_vocab_size(self):
        res = self.tokenizer.vocab_size
        assert res == 50257

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_tokenization(self):
        res = self.tokenizer.prepare_for_tokenization(self.string, add_prefix_space=True)
        assert res[0] == ' An increasing sequence: one, two, three.'


if __name__ == "__main__":
    unittest.main()
