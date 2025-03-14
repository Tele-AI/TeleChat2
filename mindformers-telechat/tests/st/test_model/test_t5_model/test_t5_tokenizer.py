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
linux:  pytest ./tests/st/test_model/test_t5_model/test_t5_tokenizer.py
"""
from mindformers import T5Tokenizer, AutoTokenizer


class TestT5TokenizerMethod:
    """A test class for testing the BertTokenizer"""
    def setup_method(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5_small')
        self.auto_tokenizer = AutoTokenizer.from_pretrained('t5_small')

    def test_from_pretrained_tokenizer(self):
        """
        Feature: The T5Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        self.tokenizer.show_support_list()
        self.tokenizer("hello world")

    def test_auto_tokenizer(self):
        """
        Feature: The T5Tokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        self.auto_tokenizer("hello world")
