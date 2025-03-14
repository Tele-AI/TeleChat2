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
"""wizardcoder Tokenizer"""
import json
from functools import lru_cache
from typing import List, Optional
import os
import regex as re

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.tokenization_utils import PreTrainedTokenizer
from mindformers.tools.utils import check_file


__all__ = ['WizardCoderTokenizer']


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
class WizardCoderTokenizer(PreTrainedTokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        vocab_file(str): The vocabulary file path.
        merge_file(str): The merge file path.
        unk_token(str): The token that represents the unknown. Default "<|endoftext|>".
        bos_token(str): The token that represents the begin-of-sentence. Default "<|endoftext|>".
        eos_token(str): The token that represents the end-of-sentence. Default "<|endoftext|>".
        add_prefix_space(bool): whether to add a whitespace in the front of text. Default "False"
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from research.wizardcoder.wizardcoder_tokenizer import WizardCoderTokenizer
        >>> tokenizer = WizardCoderTokenizer("vocab.json", "merges.txt")
        >>> res = tokenizer("Hello world")
        >>> print(res)
    {'input_ids': [8279, 5788], 'token_type_ids': [0, 0], 'attention_mask': [1, 1]}

    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    VOCAB_FILES = {'merge_file': 'merges.txt', 'vocab_file': 'vocab.json'}
    FILE_LIST = ['tokenizer_config.json']

    def __init__(
            self,
            vocab_file,
            merge_file,
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="[PAD]",
            add_prefix_space=False,
            add_bos_token=False,
            add_eos_token=False,
            **kwargs
    ):
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        check_file(vocab_file, "tokenizer")
        with open(vocab_file, 'r', encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merge_file, 'r', encoding="utf-8") as merge_handle:
            bpe_merges = merge_handle.read().split('\n')[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.the_unk_token = unk_token
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}

        super(WizardCoderTokenizer, self).__init__(
            unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs
        )

        self.add_tokens([self.pad_token, unk_token, bos_token, eos_token], special_tokens=True)

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

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence by concatenating and adding special tokens.

        A WizardCoder sequence has the following format:

        - single sequence: ``<bos> X <eos>``
        - pair of sequences: ``<bos> A <eos> B <eos>``

        Args:
            token_ids_0 (List[int]): List of IDs to which the special tokens will be added
            token_ids_1 (List[int], `optional`, defaults to `None`): Optional second list of IDs for sequence pairs.
        """
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def _tokenize(self, text):
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
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.the_unk_token))

    def _convert_tokens_to_ids(self, tokens):
        """ the index of the tokens in the vocabulary. """
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder.get(self.the_unk_token))
        output = []
        for token in tokens:
            output.append(self.encoder.get(token, self.encoder.get(self.the_unk_token)))
        return output

    def _convert_ids_to_tokens(self, ids):
        """ return the origin bpe tokens according to ids """
        if isinstance(ids, int):
            return self.decoder.get(ids)

        if isinstance(ids, list):
            output = []
            for item in ids:
                output.append(self.decoder.get(item))
            return output
        raise TypeError(f"The type of ids should be int or list, but found {type(ids)}.")

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
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
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        is_split_into_words = kwargs.pop("is_split_into_words", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory, filename_prefix):
        """write the word to the files"""
        output_file_path = os.path.join(save_directory, filename_prefix)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(output_file_path, flags_, 0o750), 'w') as fp:
            for k in self.vocab_dict.keys():
                fp.write(k + '\n')
        return output_file_path

    @property
    def vocab_size(self):
        """Get the vocab size of the """
        return len(self.decoder)

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
