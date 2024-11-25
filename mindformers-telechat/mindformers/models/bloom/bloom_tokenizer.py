# Copyright 2023 Huawei Technologies Co., Ltd
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
""" Bloom Tokenizer """
import json
import os
from functools import lru_cache
from typing import List, Optional
import regex as re

from mindformers.models.tokenization_utils import PreTrainedTokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook

__all__ = ['BloomTokenizer']


VOCAB_FILES_NAMES = {'vocab_file': 'tokenizer.json'}


@lru_cache()
def bytes_to_unicode():
    """
    bytes to unicode
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class BloomTokenizer(PreTrainedTokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        vocab_file(str): The vocabulary file path.
        unk_token(str): The token that represents the unknown. Default "<|unk|>".
        bos_token(str): The token that represents the begin-of-sentence. Default "<|s|>"".
        eos_token(str): The token that represents the end-of-sentence. Default "<|/s|>".
        pad_token(str): The token that represents the pad. Default "<|pad|>".
        add_prefix_space(bool): whether to add a whitespace in the front of text. Default "False"
        add_bos_token(bool): Whether or not to add the bos_token_id to the left of the input. Default "True"
        add_eos_token(bool): Whether or not to add the eos_token_id to the right of the input. Default "True"
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import BloomTokenizer
        >>> tokenizer = BloomTokenizer.from_pretrained("bloom_560m")
        >>> res = tokenizer("Hello world", add_special_tokens=False)
        >>> print(res)
        {'input_ids': [59414, 8876], 'token_type_ids': [0, 0], 'attention_mask': [1, 1]}

    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    FILE_LIST = ['tokenizer_config.json']
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    _support_list = MindFormerBook.get_tokenizer_support_list()['bloom']

    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            add_prefix_space=False,
            add_bos_token=False,
            add_eos_token=False,
            **kwargs):

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        with open(vocab_file, 'r', encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)["model"]["vocab"]

        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(vocab_file, 'r', encoding="utf-8") as vocab_handle:
            bpe_merges = json.load(vocab_handle)["model"]["merges"]

        self.the_unk_token = unk_token

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}

        self._unk_token_id = 0
        self._bos_token_id = 1
        self._eos_token_id = 2
        self._pad_token_id = 3

        super(BloomTokenizer, self).__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )

    def bpe(self, token):
        """ bpe encode """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(token)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            word = new_word_tuple
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def _tokenize(self, text, **kwargs):
        """ Tokenize a string using bpe encode. """
        text, _ = self.prepare_for_tokenization(text, is_pretokenized=False)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ the index of the tokens in the vocabulary. """
        return self.encoder.get(token, self.encoder.get(self.the_unk_token))

    def _convert_id_to_token(self, index):
        """ return the origin bpe tokens according to ids """
        return self.decoder.get(index)

    def _convert_tokens_to_string(self, tokens):
        """ return a string according to the list of tokens"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors='ignore')
        return text

    def convert_tokens_to_string(self, tokens):
        """Convert the tokens to the string"""
        return self._convert_tokens_to_string(tokens)

    def prepare_for_tokenization(self, text, **kwargs):
        """ whether to add a whitespace in the front of text """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        is_pretokenized = kwargs.pop("is_pretokenized", False)
        if is_pretokenized or add_prefix_space:
            text = " " + text
        return text, kwargs

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """write the word to the files"""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory", save_directory)
            return None

        output_file_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(output_file_path, flags_, 0o750), 'w', encoding="utf-8") as f:
            f.write(json.dumps(self.encoder))

        return (output_file_path,)

    @property
    def vocab_size(self):
        """Get the vocab size of the """
        return len(self.decoder)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Insert the special tokens to the input_ids. Currently"""
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        .. code-block::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
