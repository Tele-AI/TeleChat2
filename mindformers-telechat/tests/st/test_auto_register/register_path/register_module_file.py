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
"""Test common module for testing the interface."""

from mindformers.models.llama import LlamaTokenizer


class TestCommonAPI:
    """Test Common API For Auto Register."""
    def __init__(self, config=None):
        self.config = config


class TestProcessor:
    """Test Processor For Auto Register."""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


class TestTokenizer(LlamaTokenizer):
    """Test Tokenizer API For Auto Register."""
    def __init__(self, vocab_file):
        super(TestTokenizer, self).__init__(vocab_file)
        self.vocab_file = vocab_file
