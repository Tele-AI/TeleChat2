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
"""Test Glm3 tokenizer."""
import os
import unittest
import copy
import tempfile
import pytest

from mindformers import ChatGLM3Tokenizer
from mindformers.models.glm2.glm3_tokenizer import SPTokenizer
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


# pylint: disable=W0212
class TestChatGLM3Tokenizer(unittest.TestCase):
    """ A test class for testing Chatglm3 tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_sp_vocab_model("chatglm3", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "chatglm3_tokenizer.model")
        cls.tokenizer = ChatGLM3Tokenizer(vocab_file=tokenizer_model_path)
        cls.string = "华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three."
        cls.input_ids = [201, 203, 83, 88, 167, 16, 87, 85, 157, 65, 135, 67, 135, 80, 150]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
        assert res == ['▁华为是一家总部', '位于中国深圳的多元化科技公司', '。', 'An', '▁increasing', '▁sequence', ':',
                       '▁one', ',', '▁two', ',', '▁three', '.']

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
        assert res.input_ids == [201, 203, 83, 88, 167]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_inputs_with_special_tokens(self):
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0)
        assert res == [201, 203, 1, 2, 3, 4]
        res = self.tokenizer.build_inputs_with_special_tokens(self.token_ids_0, self.token_ids_1)
        assert res == [201, 203, 1, 2, 3, 4, 5, 6, 7, 8, 2]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_tokens_to_string(self):
        tokens = ['▁华为是一家总部', '位于中国深圳的多元化科技公司', '。', 'An', '▁increasing', '▁sequence', ':', '▁one',
                  ',', '▁two', ',', '▁three', '.']
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_tokens("An", special_tokens=True)
        res = tokenizer.convert_tokens_to_string(tokens)
        assert res == '华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three.'

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
        res = self.tokenizer.save_vocabulary(self.path)[0]
        assert res == os.path.join(self.path, "tokenizer.model")
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.vocab_file = os.path.join(self.path, "chatglm3_tokenizer.model")
        res = tokenizer.save_vocabulary(self.path)[0]
        assert res == os.path.join(self.path, "tokenizer.model")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res.strip() == '[gMASK]sop 华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three.'
        res = self.tokenizer._decode([83, 202])
        assert res == '华为是一家总部[sMASK]'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        res = self.tokenizer.get_vocab()
        assert len(res) == 211

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_vocab_size(self):
        res = self.tokenizer.vocab_size
        assert res == 209

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_res(self, token="▁华为是"):
        index = self.tokenizer.get_vocab()[token]
        res = self.tokenizer._convert_token_to_id(token)
        assert res == index
        res = self.tokenizer._convert_id_to_token(index)
        assert res == token

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_token_to_id_with_added_voc(self):
        res = self.tokenizer._convert_token_to_id_with_added_voc(None)
        assert res is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_token_to_id(self):
        res = self.tokenizer._convert_token_to_id('<pad>')
        assert res == 0
        res = self.tokenizer._convert_token_to_id('>')
        assert res == 0


class TestSPTokenizer(unittest.TestCase):
    """ A test class for testing text tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_sp_vocab_model("spt", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "spt_tokenizer.model")
        cls.tokenizer = SPTokenizer(model_path=tokenizer_model_path)
        cls.string = "华为是一家总部位于中国深圳的多元化科技公司。An increasing sequence: one, two, three."

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenize(self):
        res = self.tokenizer.tokenize(self.string, add_dummy_prefix=False)
        assert res == ['▁华为是一家总部', '位于中国深圳的多元化科技公司', '。', 'An', '▁increasing',
                       '▁sequence', ':', '▁one', ',', '▁two', ',', '▁three', '.']
