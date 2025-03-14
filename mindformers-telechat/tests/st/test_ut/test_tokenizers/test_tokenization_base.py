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
"""test tokenizer base."""
import os
import unittest
import tempfile
import yaml
import pytest

from mindformers import AutoTokenizer
from mindformers.models.tokenization_utils import PaddingStrategy, TruncationStrategy
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


# pylint: disable=W0212
class TestTokenizerBase(unittest.TestCase):
    """ A test class for testing base tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        cls.string = "An increasing sequence: one, two, three."
        get_sp_vocab_model("llama2_7b", cls.path)
        cls.tokenizer_model_path = os.path.join(cls.path, "llama2_7b_tokenizer.model")
        create_yaml("llama2_7b", cls.path)
        cls.tokenizer = AutoTokenizer.from_pretrained("llama2_7b")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_origin_pretrained(self):
        """test from origin pretrained."""
        real_tokenizer_model_path = os.path.join(self.path, "tokenizer.model")
        if os.path.exists(self.tokenizer_model_path):
            os.rename(self.tokenizer_model_path, real_tokenizer_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        tokenizer.save_pretrained(self.path)
        tokenizer.save_pretrained(self.path, save_json=True)
        res = tokenizer.encode(self.string)
        assert res == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150]
        res = self.tokenizer.encode(self.string)
        assert res == [1, 530, 10231, 5665, 29901, 697, 29892, 1023, 29892, 2211, 29889]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_special_tokens(self):
        """test add special tokens."""
        special_tokens_dict = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            "additional_special_tokens": ["<additional1>", "<additional2>"]
        }
        res = self.tokenizer.add_special_tokens(special_tokens_dict)
        assert res == 6

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cache_vocab_files(self):
        res = self.tokenizer.cache_vocab_files("llama2_7b")
        assert res['vocab_file'] == './checkpoint_download/llama2/tokenizer.model'

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_origin_pretrained(self):
        """test save origin pretrained."""
        res = self.tokenizer.save_origin_pretrained("notexist")
        assert res is None
        res = self.tokenizer.save_origin_pretrained(self.path, "")
        assert res is None
        with pytest.raises(ValueError):
            assert res == self.tokenizer.save_origin_pretrained(self.path, None, 'txt')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_padding_truncation_strategies(self):
        """test get padding truncation strategies."""
        res = self.tokenizer._get_padding_truncation_strategies(padding=True)
        assert res == (PaddingStrategy.LONGEST, TruncationStrategy.DO_NOT_TRUNCATE, None, {})
        res = self.tokenizer._get_padding_truncation_strategies(padding=True, max_length=2048)
        assert res == (PaddingStrategy.LONGEST, TruncationStrategy.DO_NOT_TRUNCATE, 2048, {})
        res = self.tokenizer(self.string, padding=PaddingStrategy.LONGEST)
        assert res['input_ids'] == [1, 530, 10231, 5665, 29901, 697, 29892, 1023, 29892, 2211, 29889]
        kwargs = {'truncation_strategy': 'only_first'}
        res = (PaddingStrategy.LONGEST, None, 2048, kwargs)
        assert res == (PaddingStrategy.LONGEST, None, 2048, {'truncation_strategy': 'only_first'})

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template(self):
        conversation = "Conversation"
        res = self.tokenizer.apply_chat_template(self, conversation, None, padding=True)
        assert res == [1281, 874, 362]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_sequences(self):
        """test truncate sequences."""
        ids = [1, 2, 3, 4, 5]
        pair_ids = [1, 2, 3]
        res = self.tokenizer.truncate_sequences(ids, None, 1)
        assert res == ([1, 2, 3, 4], None, [5])
        res = self.tokenizer.truncate_sequences(ids, None, 6, TruncationStrategy.ONLY_FIRST)
        assert res == ([1, 2, 3, 4, 5], None, [])
        res = self.tokenizer.truncate_sequences(ids, None, 6, TruncationStrategy.LONGEST_FIRST)
        assert res == ([1, 2, 3, 4, 5], None, [])
        res = self.tokenizer.truncate_sequences(ids, pair_ids, 6, TruncationStrategy.LONGEST_FIRST)
        assert res == ([1], [1], [])
        res = self.tokenizer.truncate_sequences(ids, pair_ids, 1, TruncationStrategy.ONLY_SECOND)
        assert res == ([1, 2, 3, 4, 5], [1, 2], [3])


def create_yaml(model_name, dir_path):
    """create yaml."""
    yaml_content = {
        "processor": {
            "return_tensors": "ms",
            "tokenizer": {
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
                "type": "LlamaTokenizer"
            },
            "type": "LlamaProcessor"
        }
    }
    file_name = f'{dir_path}/{model_name}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
