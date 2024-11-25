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
"""cogvlm2 tokenizer APIs."""
from typing import Collection, Dict, List, Set, Union
import base64
import json
import unicodedata

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import AddedToken, PreTrainedTokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

try:
    import tiktoken
except ImportError as e:
    raise ImportError("Package 'tiktoken' required to run CogVLM2. please install it with pip.") from e

__all__ = ['CogVLM2Tokenizer']

PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| " \
          r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def _load_tokenizer_json(json_file):
    with open(json_file, "rb") as f:
        contents = json.loads(f.read())
    return {
        bytes(token, encoding='utf8'): int(rank)
        for token, rank in contents['model']['vocab'].items()
    }


# pylint: disable=W0221
# pylint: disable=W0223
@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class CogVLM2Tokenizer(PreTrainedTokenizer):
    """CogVLM2 Tokenizer."""
    VOCAB_FILES = {'vocab_file': 'tokenizer.json'}
    FILE_LIST = []
    _support_list = MindFormerBook.get_tokenizer_support_list()['llama']
    special_tokens: Dict[str, int]

    def __init__(self,
                 vocab_file,
                 bos_token="<|begin_of_text|>",
                 eos_token="<|end_of_text|>",
                 pad_token="<|reserved_special_token_0|>",
                 add_bos_token=False,
                 add_eos_token=False,
                 errors="replace",
                 num_reserved_special_tokens=256,
                 **kwargs):
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.errors = errors
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        if vocab_file.split('.')[-1] == 'json':
            self.mergeable_ranks = _load_tokenizer_json(vocab_file)
        else:
            self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        num_base_tokens = len(self.mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.tokenizer = tiktoken.Encoding(
            "Llama3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        super().__init__(bos_token=bos_token,
                         eos_token=eos_token,
                         pad_token=pad_token,
                         **kwargs)

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
        if self.add_bos_token:
            tokens.insert(0, self.decoder[self.bos_token_id])

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
                text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])
        if self.add_eos_token:
            tokens.append(self.decoder[self.eos_token_id])
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
            token_ids = [i for i in token_ids if i != self.pad_token_id and i not in self.special_tokens.values()]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
