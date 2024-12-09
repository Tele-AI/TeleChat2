# coding=utf-8
# Copyright 2023 The Telechat2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is adapted from the Hugging Face Transformers library.
# See https://github.com/huggingface/transformers for more information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

# Update PRETRAINED_VOCAB_FILES_MAP when available
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}


class Telechat2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Telechat2Tokenizer, which uses a SentencePiece model.

    This tokenizer inherits from `PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to the superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<_start>",
        eos_token: str = "<_end>",
        pad_token: str = "<_pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        """
        Args:
            vocab_file (`str`): Path to the SentencePiece `.model` file.
            unk_token (`str`, *optional*, defaults to `<unk>`): The unknown token.
            bos_token (`str`, *optional*, defaults to `<_start>`): The beginning of sentence token.
            eos_token (`str`, *optional*, defaults to `<_end>`): The end of sentence token.
            pad_token (`str`, *optional*, defaults to `<_pad>`): The padding token.
            sp_model_kwargs (`Dict[str, Any]`, *optional*): Additional parameters for SentencePieceProcessor.
            add_bos_token (`bool`, *optional*, defaults to `True`): Whether to add `bos_token` at the start.
            add_eos_token (`bool`, *optional*, defaults to `False`): Whether to add `eos_token` at the end.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`): Whether to clean up spaces.
            **kwargs:
                Additional keyword arguments passed along to `PreTrainedTokenizer`.
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self) -> int:
        """
        `int`: Returns the size of the vocabulary (number of tokens).
        """
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a string using the SentencePiece model.

        Args:
            text (`str`): The input text to tokenize.

        Returns:
            `List[str]`: A list of tokens.
        """
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token (string) into its vocabulary index.

        Args:
            token (`str`): The token to convert.

        Returns:
            `int`: The corresponding token id.
        """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a vocabulary index into its corresponding token.

        Args:
            index (`int`): The token id.

        Returns:
            `str`: The token corresponding to `index`.
        """
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens into a single string.

        This method handles special tokens by not decoding them through the SentencePiece model directly.

        Args:
            tokens (`List[str]`): The sequence of tokens to convert.

        Returns:
            `str`: The decoded string.
        """
        current_sub_tokens = []
        out_string = ""

        for token in tokens:
            if token in self.all_special_tokens:
                # Decode current collected subtokens and then add the special token directly.
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)

        # Decode any remaining subtokens
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer vocabulary (i.e., SentencePiece model file) to a directory.

        Args:
            save_directory (`str`): Directory to save the vocabulary file.
            filename_prefix (`str`, *optional*): Prefix to add to the saved files.

        Returns:
            `Tuple[str]`: Paths to the saved vocabulary files.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by concatenating and adding special tokens.

        A Telechat2 sequence has the following format:
        - If `token_ids_1` is not `None`:
            [BOS] token_ids_0 [EOS] [BOS] token_ids_1 [EOS]
        - If `token_ids_1` is `None`:
            [BOS] token_ids_0 [EOS]

        Args:
            token_ids_0 (`List[int]`): List of IDs to which special tokens will be added.
            token_ids_1 (`List[int]`, *optional*): Optional second sequence to be concatenated.

        Returns:
            `List[int]`: `token_ids_0` and `token_ids_1` with added BOS/EOS tokens.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`): Whether tokens already contain special tokens.

        Returns:
            `List[int]`: A sequence of 0s and 1s where 1 indicates a special token and 0 indicates a regular token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pairs.

        For a Telechat2 sequence pair:
        - If `token_ids_1` is not `None`:
            [BOS] seq0 [EOS] [BOS] seq1 [EOS]
          Corresponding token types: 0s for seq0 part and 1s for seq1 part.

        - If `token_ids_1` is `None`:
            [BOS] seq0 [EOS]
          Corresponding token types: all 0s.

        Args:
            token_ids_0 (`List[int]`): List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*): List of IDs for the second sequence.

        Returns:
            `List[int]`: Token type IDs.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
