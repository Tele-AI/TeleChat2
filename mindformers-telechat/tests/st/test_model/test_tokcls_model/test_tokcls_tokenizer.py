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
linux:  pytest ./tests/st/test_model/test_tokcls_model/test_tokcls_tokenizer.py
"""

from mindformers import BertTokenizer, AutoTokenizer


class TestBertTokenizerForChineseMethod:
    """A test class for testing the AutoTokenizer"""

    def test_from_pretrained(self):
        """
        Feature: The BertTokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = BertTokenizer.from_pretrained('tokcls_bert_base_chinese')
        tokenizer.show_support_list()
        res = tokenizer.tokenize("你好，世界！")

        assert isinstance(tokenizer, BertTokenizer)
        assert res == ['你', '好', '，', '世', '界', '！']

        tokenizer = BertTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
        tokenizer.show_support_list()
        res = tokenizer.tokenize("你好，世界！")

        assert isinstance(tokenizer, BertTokenizer)
        assert res == ['你', '好', '，', '世', '界', '！']

    def test_auto_tokenizer(self):
        """
        Feature: The BertTokenizer test using auto_class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese')
        tokenizer.show_support_list()
        res = tokenizer.tokenize("你好，世界！")

        assert isinstance(tokenizer, BertTokenizer)
        assert res == ['你', '好', '，', '世', '界', '！']

        tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
        tokenizer.show_support_list()
        res = tokenizer.tokenize("你好，世界！")

        assert isinstance(tokenizer, BertTokenizer)
        assert res == ['你', '好', '，', '世', '界', '！']
