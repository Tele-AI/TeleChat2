# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Module for testing Tokenizer class

How to run this:
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_tokenizer.py
"""
import os
import shutil

import pytest
from mindspore import Tensor

from mindformers import CLIPTokenizer, AutoTokenizer


def generate_fake_vocab(output_path):
    vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(os.path.join(output_path, 'vocab_file.txt'), flags_, 0o750), 'w') as fp:
        for item in vocabs:
            fp.write(item + '\n')


class TestAutoTokenizerMethod:
    """A test class for testing the AutoTokenizer"""
    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output')
        os.makedirs(self.output_path, exist_ok=True)
        generate_fake_vocab(self.output_path)

    def teardown_method(self):
        shutil.rmtree(self.output_path, ignore_errors=True)

    @pytest.mark.run(order=1)
    def test_save_and_load_using_bert_tokenizer(self):
        """
        Feature: The test load and save function for the clip tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer.tokenize("hello world?")
        assert isinstance(clip_tokenizer, CLIPTokenizer)
        assert res == ['hello</w>', 'world</w>', '?</w>']

    @pytest.mark.run(order=2)
    def test_load_from_yaml(self):
        """
        Feature: The test load from yaml and save as the yaml for the tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        tokenizer = AutoTokenizer.from_pretrained("clip_vit_b_32")
        res = tokenizer.tokenize("hello world?")
        assert isinstance(tokenizer, CLIPTokenizer)
        assert res == ['hello</w>', 'world</w>', '?</w>']


class TestClipTokenizerMethod:
    """Test the basic usage of the CLIPTokenizer"""
    def test_padding(self):
        """
        Feature: The CLIPTokenizer test using padding
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to the target.
        """
        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length')
        pad_id = clip_tokenizer.pad_token_id
        assert res == {'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0],
                       'input_ids': [49406, 3306, 1002, 286, 49407, pad_id, pad_id, pad_id]}, f"The res is {res}."

        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length', return_tensors='ms')
        assert len(res) == 2
        for k in res.keys():
            assert isinstance(res[k], Tensor)

        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        batch_inputs = ["hello world?", "Who are you?", "I am find, thank you."]
        res = clip_tokenizer(batch_inputs, max_length=12, padding='max_length')
        assert len(res) == 2
        assert res == {'input_ids': [[49406, 3306, 1002, 286, 49407, pad_id, pad_id, pad_id, pad_id,
                                      pad_id, pad_id, pad_id],
                                     [49406, 822, 631, 592, 286, 49407, pad_id, pad_id, pad_id,
                                      pad_id, pad_id, pad_id],
                                     [49406, 328, 687, 1416, 267, 1144, 592, 269, 49407, pad_id, pad_id, pad_id]],
                       'attention_mask': [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]}
