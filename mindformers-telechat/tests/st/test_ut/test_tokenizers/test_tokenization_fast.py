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
"""test tokenizer fast."""
import os
import unittest
import tempfile
import yaml
import pytest

from mindformers import LlamaTokenizerFast
from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model


# pylint: disable=W0212
class TestTokenizerFast(unittest.TestCase):
    """ A test class for testing base tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        cls.string = "An increasing sequence: one, two, three."
        get_sp_vocab_model("llama2_7b", cls.path)
        cls.tokenizer_model_path = os.path.join(cls.path, "llama2_7b_tokenizer.model")
        create_yaml("llama2_7b", cls.path)
        real_tokenizer_model_path = os.path.join(cls.path, "tokenizer.model")
        if os.path.exists(cls.tokenizer_model_path):
            os.rename(cls.tokenizer_model_path, real_tokenizer_model_path)
        cls.tokenizer = LlamaTokenizerFast.from_pretrained(cls.path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_origin_pretrained(self):
        file_names = tuple("config.json")
        self.tokenizer._save_pretrained(self.path, file_names)
        res = self.tokenizer.encode(self.string)
        assert res == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_train_new_from_iterator(self):
        text_iterator = ["An", "increasing", "sequence"]
        res = self.tokenizer.train_new_from_iterator(text_iterator, 1000)
        assert isinstance(res, LlamaTokenizerFast)


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
                "type": "LlamaTokenizerFast"
            },
            "type": "LlamaProcessor"
        }
    }
    file_name = f'{dir_path}/{model_name}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
