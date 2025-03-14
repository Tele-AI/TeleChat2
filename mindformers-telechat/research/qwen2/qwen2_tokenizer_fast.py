# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Qwen2 fast tokenizer APIs."""

from typing import Optional, Tuple

from qwen2_tokenizer import Qwen2Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.tokenization_utils_base import AddedToken
from mindformers.models.tokenization_utils_fast import PreTrainedTokenizerFast


__all__ = ["Qwen2TokenizerFast"]

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Qwen2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Note:
        Currently, the qwen2_tokenizer_fast process supports only the 'right' padding mode.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        tokenizer_file (str, optional):
            Tokenizers file (generally has a .json extension) that contains everything needed to load the tokenizer.
            Default: ``None`` .
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.

    Returns:
        Qwen2TokenizerFast, a Qwen2TokenizerFast instance.

    Examples:
        >>> from qwen2_tokenizer_fast import Qwen2TokenizerFast
        >>>
        >>> tokenizer = Qwen2TokenizerFast(tokenizer_file="/path/to/tokenizer.json")
        >>> tokenizer.encode("I love Beijing.")
        [40, 2948, 26549, 13]
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    padding_side = "right"

    def __init__(
            self,
            vocab_file=None,
            merges_file=None,
            tokenizer_file=None,
            unk_token="<|endoftext|>",
            bos_token=None,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            **kwargs,
    ):
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
