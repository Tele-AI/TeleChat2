# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizer classes for CodeGeex."""
import numpy as np
from mindformers import AutoTokenizer
from mindformers.models.tokenization_utils import AddedToken


def encode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """ Encode whitespaces to extra tokens.

    >>> encode_whitespaces('a\\n  b\\n   c', 10, 10)
    'a\\n<|extratoken_10|>b\\n<|extratoken_11|>c'
    """
    for i in np.arange(max_len, 1, -1):
        text = text.replace(
            " " * i, f"<|extratoken_{start_extra_id + i - 2}|>")
    return text


def decode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """ Decode the whitespace-encoded strings produced by encode_whitespace.

    >>> text = 'a\\n  b\\n   c'
    >>> s, l = 10, 10
    >>> text == decode_whitespaces(encode_whitespaces(text, s, l), s, l)
    True
    """
    for l in range(2, max_len + 1):
        token_id = start_extra_id - 2 + l
        token = f'<|extratoken_{token_id}|>'
        text = text.replace(token, ' ' * l)
    text = text.replace(f'<|endoftext|>', '')
    return text if not text.startswith("vocab_pad_token") else ''


class CodeTokenizer():
    """Tokenizer classes for CodeGeex"""
    def __init__(
            self,
            vocab_size,
            start_extra_id: int = 10,
            max_len: int = 10,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        sp_tokens = [AddedToken(
            f'<|extratoken_{token_id-50256}|>', lstrip=False, rstrip=False) for token_id in range(50257, 50400)]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sp_tokens})
        num_pad = vocab_size - self.tokenizer.vocab_size
        vocab_pad_tokens = ["vocab_pad_token{}".format(
            i) for i in range(1, num_pad + 1)]
        self.tokenizer.add_tokens(vocab_pad_tokens)

        self.start_extra_id = start_extra_id
        self.max_len = max_len
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode_code(self, code: str):
        code = encode_whitespaces(code, self.start_extra_id, self.max_len)
        input_ids = self.tokenizer(code).input_ids
        return input_ids

    def decode_code(self, input_ids):
        texts = self.tokenizer.batch_decode(input_ids)
        output_code = [decode_whitespaces(
            text, self.start_extra_id, self.max_len) for text in texts]
        return output_code
