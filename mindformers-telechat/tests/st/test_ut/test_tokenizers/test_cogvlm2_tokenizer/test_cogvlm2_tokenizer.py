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
"""Test cogvlm2 tokenizer."""
import os
import unittest
import tempfile
import pytest

from mindformers import CogVLM2Tokenizer
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_bbpe_vocab_model


class TestCogVLM2Tokenizer(unittest.TestCase):
    """ A test class for testing Cogvlm2 tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_bbpe_vocab_model("cogvlm2", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "cogvlm2_tokenizer.json")
        cls.tokenizer = CogVLM2Tokenizer(vocab_file=tokenizer_model_path)
        cls.string = "An increasing sequence: one, two, three."
        cls.input_ids = [113, 163, 116, 114, 191, 106, 123, 196, 13, 167, 4, 102, 126, 4, 199, 17, 5]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        cls.token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cls.token_ids_0 = [1, 2, 3, 4]
        cls.token_ids_1 = [5, 6, 7, 8]

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
    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res == "AnĠincreasingĠsequence:Ġone,Ġtwo,Ġthree."

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        res = self.tokenizer.get_vocab()
        assert len(res) == 456

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_vocab_size(self):
        res = self.tokenizer.vocab_size
        assert res == 456

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_tokenization(self):
        res = self.tokenizer.prepare_for_tokenization(self.string, add_prefix_space=True)
        assert res[0] == 'An increasing sequence: one, two, three.'
