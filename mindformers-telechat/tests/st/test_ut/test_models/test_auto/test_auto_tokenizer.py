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
"""test AutoTokenizer."""
import json
import shutil
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://modelfoundrysh.test.osinfra.cn"

# pylint: disable=C0413
from mindformers import GPT2Tokenizer, BertTokenizer
from mindformers.models.auto import AutoTokenizer
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer


class TestAutoTokenizer:
    """A test class for testing TestAutoTokenizer."""

    def setup_method(self):
        """init test class."""
        self.gpt_model_name = "gpt2"
        self.gpt_repo_id = "mindformersinfra/test_auto_tokenizer_gpt2_ms"
        self.gpt_saved_ori_dir = "./gpt2_origin_save_pretrained"
        self.gpt_saved_exp_dir = "./gpt2_experimental_save_pretrained"
        self.gpt_saved_vocab_dir = "./gpt2_save_vocabulary"

        self.baichuan_repo_id = "mindformersinfra/test_auto_tokenizer_baichuan2_ms"
        self.baichuan_saved_ori_dir = "./baichuan_origin_save_pretrained"
        self.baichuan_saved_exp_dir = "./baichuan_experimental_save_pretrained"
        self.baichuan_saved_vocab_dir = "./baichuan_save_vocabulary"

        self.bert_saved_vocab_dir = "./bert_save_vocabulary"

        self.str_input = "I love Beijing."
        self.gpt_id_output = [40, 1842, 11618, 13]
        self.baichuan_id_output = [92346, 2950, 19801, 72]
        self.bert_id_output = [1045, 2293, 7211, 1012]

    def test_tokenizer_from_model_name(self):
        """test init AutoTokenizer from model name."""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_tokenizer_save_pretrained(self):
        """test AutoTokenizer save_pretrained() method."""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        tokenizer.save_pretrained(self.gpt_saved_ori_dir)

    def test_tokenizer_save_vocabulary(self):
        """test AutoTokenizer save_vocabulary() method."""
        os.makedirs(self.gpt_saved_vocab_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        tokenizer.save_vocabulary(self.gpt_saved_vocab_dir)

    def test_tokenizer_from_saved_dir(self):
        """test init AutoTokenizer from saved dir."""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_saved_ori_dir)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_tokenizer_from_vocab_dir(self):
        """test init AutoTokenizer from saved vocabulary dir."""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_saved_vocab_dir)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_class_tokenizer_from_model_name(self):
        """test init class Tokenizer from model name."""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_tokenizer_from_repo(self):
        """test init AutoTokenizer from repo id."""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_repo_id)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_tokenizer_experimental_save_pretrained(self):
        """test AutoTokenizer save_pretrained() method with save_json=True."""
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        tokenizer.save_pretrained(self.gpt_saved_exp_dir, save_json=True)

    def test_tokenizer_from_saved_experimental_dir(self):
        """test init AutoTokenizer from saved experimental dir."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.gpt_saved_exp_dir, tokenizer_type="gpt2", use_fast=False)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_tokenizer_from_config(self):
        """test init AutoTokenizer from config.json."""
        tokenizer_config_path = os.path.join(self.gpt_saved_exp_dir, "tokenizer_config.json")
        config_path = os.path.join(self.gpt_saved_exp_dir, "config.json")
        os.remove(tokenizer_config_path)

        # test experimental type: del tokenizer_config.json, and build through config.json - tokenizer_class
        config = {"tokenizer_class": "GPT2Tokenizer"}
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(config_path, flags_, 0o750), "w", encoding="utf-8") as w:
            w.write(json.dumps(config))
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_saved_exp_dir)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

        # test experimental type: del tokenizer_config.json, and build through config.json - model_type
        config = {"model_type": "gpt2"}
        with os.fdopen(os.open(config_path, flags_, 0o750), "w", encoding="utf-8") as w:
            w.write(json.dumps(config))
        tokenizer = AutoTokenizer.from_pretrained(self.gpt_saved_exp_dir)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_class_tokenizer_from_repo(self):
        """test init class Tokenizer from repo id."""
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_repo_id)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.gpt_id_output

    def test_plugin_tokenizer_from_repo(self):
        """test init plugin AutoTokenizer from repo id."""
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.baichuan_id_output

    def test_plugin_tokenizer_save_pretrained(self):
        """test plugin AutoTokenizer save_pretrained() method."""
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)
        tokenizer.save_pretrained(self.baichuan_saved_ori_dir)

    def test_plugin_tokenizer_from_saved_dir(self):
        """test init plugin AutoTokenizer from saved dir."""
        try:
            Baichuan2Tokenizer.from_pretrained(self.baichuan_saved_ori_dir)
        except ValueError as e:
            # if it has used save_pretrained(save_json=False) to save plugin tokenizer,
            # you cannot use from_pretrained() to get tokenizer instance.
            print(e)

    def test_plugin_tokenizer_experimental_save_pretrained(self):
        """test plugin AutoTokenizer save_pretrained() method with save_json=True."""
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)
        tokenizer.save_pretrained(self.baichuan_saved_exp_dir, save_json=True)

    def test_plugin_tokenizer_save_vocabulary(self):
        """test plugin AutoTokenizer save_vocabulary() method."""
        os.makedirs(self.baichuan_saved_vocab_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_repo_id, trust_remote_code=True)
        tokenizer.save_vocabulary(self.baichuan_saved_vocab_dir)

    def test_plugin_tokenizer_from_saved_experimental_dir(self):
        """test init plugin AutoTokenizer from saved experimental dir."""
        tokenizer = AutoTokenizer.from_pretrained(self.baichuan_saved_exp_dir, trust_remote_code=True)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.baichuan_id_output

        try:
            AutoTokenizer.from_pretrained(self.baichuan_saved_exp_dir)
        except ValueError as e:
            # when tokenizer class not in mindformers, you can not instantiate it through yaml
            print(e)

    def test_plugin_tokenizer_from_vocab_dir(self):
        """test init plugin AutoTokenizer from saved vocabulary dir."""
        tokenizer = Baichuan2Tokenizer.from_pretrained(self.baichuan_saved_vocab_dir)
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.baichuan_id_output

    def test_class_tokenizer_from_single_vocab_dir(self):
        """test init class Tokenizer from single vocabulary dir."""
        tokenizer = AutoTokenizer.from_pretrained("bert_base_uncased")
        os.makedirs(self.bert_saved_vocab_dir, exist_ok=True)
        tokenizer.save_vocabulary(self.bert_saved_vocab_dir)

        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.bert_saved_vocab_dir, os.listdir(self.bert_saved_vocab_dir)[0]))
        output = tokenizer(self.str_input, add_special_tokens=False)["input_ids"]
        assert output == self.bert_id_output

    def teardown_method(self):
        """delete cache files after testing."""
        shutil.rmtree(self.gpt_saved_ori_dir, ignore_errors=True)
        shutil.rmtree(self.gpt_saved_vocab_dir, ignore_errors=True)
        shutil.rmtree(self.gpt_saved_exp_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_saved_exp_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_saved_vocab_dir, ignore_errors=True)
        shutil.rmtree(self.baichuan_saved_ori_dir, ignore_errors=True)
        shutil.rmtree(self.bert_saved_vocab_dir, ignore_errors=True)
        # shutil.rmtree("./checkpoint_download", ignore_errors=True)    # cannot be deleted directly
