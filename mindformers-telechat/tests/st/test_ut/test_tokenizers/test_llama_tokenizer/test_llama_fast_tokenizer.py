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
"""test llama fast tokenizer."""
import os
import unittest
import copy
import tempfile

import pytest

from mindformers import LlamaTokenizerFast
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


class TestLlamaFastTokenizer(unittest.TestCase):
    """ A test class for testing Llama fast tokenizer"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_sp_vocab_model("llama", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "llama_tokenizer.model")
        cls.tokenizer = LlamaTokenizerFast(vocab_file=tokenizer_model_path)
        cls.string = "华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three."
        cls.input_ids = [1, 83, 88, 167, 16, 87, 85, 157, 65, 135, 67, 135, 80, 150]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        cls.token_ids_0 = [1, 2, 3, 4]
        cls.token_ids_1 = [5, 6, 7, 8]

    def test_call_method(self):
        res = self.tokenizer(self.string)
        assert res.input_ids == self.input_ids
        assert res.attention_mask == self.attention_mask

    def test_tokenize(self):
        res = self.tokenizer.tokenize(self.string)
        assert res == ['▁华为是一家总部', '位于中国深圳的多元化科技公司', '。', 'An', '▁increasing', '▁sequence', ':', '▁one',
                       ',', '▁two', ',', '▁three', '.']

    def test_pad(self, max_length=20):
        res = self.tokenizer(self.string, max_length=max_length, padding="max_length")
        assert res.input_ids == self.input_ids + [self.tokenizer.pad_token_id] * (max_length - len(self.input_ids))
        assert res.attention_mask == self.attention_mask + [0] * (max_length - len(self.input_ids))

    def test_encode(self):
        res = self.tokenizer.encode(self.string)
        assert res == self.input_ids

    def test_truncation(self, max_length=5):
        res = self.tokenizer(self.string, max_length=max_length)
        assert res.input_ids == self.input_ids[:max_length]

    def test_build_inputs_with_special_tokens(self):
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0)
        assert res == [1, 1, 2, 3, 4]
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0, self.token_ids_1)
        assert res == [1, 1, 2, 3, 4, 1, 5, 6, 7, 8]

    def test_save_vocabulary(self):
        res = self.tokenizer.save_vocabulary("")
        assert res is None
        res = self.tokenizer.save_vocabulary(self.path)[0]
        assert res == os.path.join(self.path, "tokenizer.model")
        with pytest.raises(ValueError):
            tokenizer = copy.deepcopy(self.tokenizer)
            tokenizer.vocab_file = "not_a_file"
            assert tokenizer.save_vocabulary(self.path)

    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res == '<s> 华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three.'

    def test_update_post_processor(self):
        """test update post processor."""
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_eos_token = True
        tokenizer.update_post_processor()
        res = tokenizer(self.string)
        assert res.input_ids == self.input_ids + [self.tokenizer.eos_token_id]
        assert res.attention_mask == self.attention_mask + [1]
        with pytest.raises(ValueError):
            tokenizer.bos_token = None
            assert tokenizer.update_post_processor()
        with pytest.raises(ValueError):
            tokenizer.bos_token = self.tokenizer.bos_token
            tokenizer.eos_token = None
            assert tokenizer.update_post_processor()

    def test_add_bos_token(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_bos_token = False
        res = tokenizer(self.string)
        assert res.input_ids == self.input_ids[1:]
        assert res.attention_mask == self.attention_mask[1:]
