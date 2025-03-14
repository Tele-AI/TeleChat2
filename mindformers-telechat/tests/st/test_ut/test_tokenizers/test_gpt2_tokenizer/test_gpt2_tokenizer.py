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
"""Test Gpt2 tokenizer."""
import os
import unittest
import copy
import tempfile
import pytest

from mindformers import GPT2Tokenizer
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_bbpe_vocab_model


class TestGPT2Tokenizerr(unittest.TestCase):
    """ A test class for testing Gpt2 tokenizer"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_bbpe_vocab_model("gpt", cls.path)
        cls.vocab_path = os.path.join(cls.path, "gpt_vocab.json")
        cls.merges_path = os.path.join(cls.path, "gpt_merges.txt")
        cls.tokenizer = GPT2Tokenizer(vocab_file=cls.vocab_path, merges_file=cls.merges_path)
        cls.string = "An increasing sequence: one, two, three."
        cls.input_ids = [113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4, 102, 126, 4, 102, 118, 17, 5]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        cls.token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cls.token_ids_0 = [1, 2, 3, 4]
        cls.token_ids_1 = [5, 6, 7, 8]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call_method(self):
        res = self.tokenizer(self.string)
        assert res.input_ids == self.input_ids
        assert res.attention_mask == self.attention_mask
        assert res.token_type_ids == self.token_type_ids

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenize(self):
        res = self.tokenizer.tokenize(self.string)
        assert res == ['An', 'Ġin', 'cre', 'as', 'ing', 'Ġse', 'qu', 'ence', ':',
                       'Ġone', ',', 'Ġt', 'wo', ',', 'Ġt', 'hre', 'e', '.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad(self, max_length=20):
        res = self.tokenizer(self.string, max_length=max_length, padding="max_length")
        assert res.input_ids == [113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4,
                                 102, 126, 4, 102, 118, 17, 5, 200, 200]
        assert res.attention_mask == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode(self):
        res = self.tokenizer.encode(self.string)
        assert res == [113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4, 102, 126, 4, 102, 118, 17, 5]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation(self, max_length=5):
        res = self.tokenizer(self.string, max_length=max_length)
        assert res.input_ids == [113, 163, 116, 114, 191]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_inputs_with_special_tokens(self):
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0)
        assert res == [1, 2, 3, 4]
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0, self.token_ids_1)
        assert res == [1, 2, 3, 4, 5, 6, 7, 8]

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
        assert res == [0, 0, 0, 0]
        res = self.tokenizer.get_special_tokens_mask(self.token_ids_0, self.token_ids_1)
        assert res == [0, 0, 0, 0, 0, 0, 0, 0]
        res = self.tokenizer.get_special_tokens_mask(self.token_ids_0, already_has_special_tokens=True)
        assert res == [0, 0, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_vocabulary(self):
        """Test save vocabulary."""
        res = self.tokenizer.save_vocabulary("not_a_file")
        assert res is None
        res = self.tokenizer.save_vocabulary(self.path)[0]
        assert res == os.path.join(self.path, "vocab.json")
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.vocab_file = "not_a_file"
        res = tokenizer.save_vocabulary(self.path)[0]
        assert res == os.path.join(self.path, "vocab.json")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res == self.string

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        res = self.tokenizer.get_vocab()
        assert len(res) == 201

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_vocab_size(self):
        res = self.tokenizer.vocab_size
        assert res == 200

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_tokenization(self):
        res = self.tokenizer.prepare_for_tokenization(self.string, add_prefix_space=True)
        assert res[0] == ' An increasing sequence: one, two, three.'
