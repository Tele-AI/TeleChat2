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
"""Qwen tokenizer APIs."""

import base64
from typing import Collection, Dict, List, Set, Union

import unicodedata

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import AddedToken, PreTrainedTokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import check_file

try:
    import tiktoken
except ImportError as e:
    raise ImportError("Package 'tiktoken' required to run Qwen. please install it with pip.") from e

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"  # used in Qwen-7B-chat
IMEND = "<|im_end|>"  # used in Qwen-7B-chat
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND,) + EXTRAS


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class QwenTokenizer(PreTrainedTokenizer):
    """Qwen Tokenizer"""
    VOCAB_FILES = {'vocab_file': 'qwen.tiktoken'}
    FILE_LIST = []
    _support_list = MindFormerBook.get_tokenizer_support_list()['qwen']

    def __init__(self,
                 vocab_file="qwen.tiktoken",
                 errors="replace",
                 pad_token="<|endoftext|>",
                 **kwargs):
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.errors = errors  # how to handle errors in decoding
        self.vocab_file = vocab_file
        check_file(vocab_file, "tokenizer")
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        start_index = len(self.mergeable_ranks)
        token_with_index = enumerate(SPECIAL_TOKENS, start=start_index)
        self.special_tokens = {
            token: index for index, token in token_with_index
        }

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        if len(self.mergeable_ranks) + len(self.special_tokens) != enc.n_vocab:
            raise ValueError(f"len(mergeable_ranks)+ len(special_tokens): "
                             f"{len(self.mergeable_ranks) + len(self.special_tokens)} != "
                             f"n_vocab: {enc.n_vocab} in encoding.")
        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

        super().__init__(pad_token=pad_token, **kwargs)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # override Tokenizer.convert_tokens_to_string()
    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    # called by Tokenizer.convert_tokens_to_ids() & SpecialTokensMixin
    def _convert_tokens_to_ids(
            self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> Union[int, List[int]]:
        """Convert the tokens to ids using vocab mapping"""
        if isinstance(tokens, (str, bytes)):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    # required by Tokenizer.convert_ids_to_tokens() of mindformers<=0.6
    def _convert_ids_to_tokens(self, input_id: int):
        return self._convert_id_to_token(input_id)

    # called by Tokenizer.convert_ids_to_tokens()
    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    # pylint: disable=W0613
    def tokenize(
            self,
            text: str,
            allowed_special: Union[Set, str] = "all",
            disallowed_special: Union[Collection, str] = (),
            **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
                text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])
        return tokens

    # pylint: disable=W0613
    def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            errors: str = None,
            **kwargs,
    ) -> str:
        """override Tokenizer._decode(), called by PreTrainedTokenizerBase.decode()"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
