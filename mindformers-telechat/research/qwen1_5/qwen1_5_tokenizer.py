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
"""Tokenization classes for Qwen2."""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Dict, Optional, Tuple
import regex as re

from mindspore import log as logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.tokenization_utils import PreTrainedTokenizer
from mindformers.models.tokenization_utils_base import AddedToken
from mindformers.tools.utils import check_file


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "merges.txt"},
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"  # used in Qwen-72B-chat
IMEND = "<|im_end|>"  # used in Qwen-72B-chat
ENDOFTEXTID = 151643
IMSTARTID = 151644
IMENDID = 151645


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                         ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
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
class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Construct a Qwen2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.
            tokenize("<|endoftext|>") = ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.
            tokenize("<|endoftext|>")` will be give `['<', '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only
            supported for `slow` tokenizers for the moment.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors="replace",
                 unk_token="<|endoftext|>",
                 bos_token=None,
                 eos_token="<|endoftext|>",
                 pad_token="<|endoftext|>",
                 clean_up_tokenization_spaces=False,
                 split_special_tokens=False,
                 **kwargs,
                 ):
        # Qwen vocab does not contain control tokens; added tokens need to be special
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False,
                       special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False,
                       special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False,
                       special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False,
                       special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        end_of_text_token = AddedToken(
            ENDOFTEXT, lstrip=False, rstrip=False, special=True, normalized=False)
        im_start_token = AddedToken(
            IMSTART, lstrip=False, rstrip=False, special=True, normalized=False)
        im_end_token = AddedToken(
            IMEND, lstrip=False, rstrip=False, special=True, normalized=False)
        self.special_tokens = {
            ENDOFTEXT: ENDOFTEXTID,
            IMSTART: IMSTARTID,
            IMEND: IMENDID,
        }
        self.end_of_text_id = self.special_tokens[ENDOFTEXT]
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]
        check_file(vocab_file, "tokenizer")
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {}

        self.pat = re.compile(PRETOKENIZE_REGEX)

        if kwargs.get("add_prefix_space", False):
            logger.warning(
                f"{self.__class__.__name__} does not support `add_prefix_space`, setting it to True has no effect."
            )

        self._added_tokens_decoder: Dict[int, AddedToken] = {
            self.end_of_text_id: end_of_text_token,
            self.im_start_id: im_start_token,
            self.im_end_id: im_end_token,
        }

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        """byte pair encoding"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float("inf")))
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

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
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

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c]
                          for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(self,
               token_ids,
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: Optional[bool] = False,
               spaces_between_special_tokens: bool = False,
               **kwargs,
               ) -> str:
        """decode token ids"""
        # `spaces_between_special_tokens` defaults to True for _decode in slow tokenizers
        # and cannot be configured elsewhere, but it should default to False for Qwen2Tokenizer
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """save vocabulary"""
        if not os.path.isdir(save_directory):
            logger.error(
                f"Vocabulary path ({save_directory}) should be a directory")
            return None
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["merges_file"]
        )

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(vocab_file, flags_, 0o750), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with os.fdopen(os.open(merge_file, flags_, 0o750), "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return text, kwargs
