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
"""LLaMA fast tokenizer APIs."""

import os
from shutil import copyfile
from typing import Optional, Tuple

from tokenizers import processors
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook
from ..tokenization_utils_fast import PreTrainedTokenizerFast
from .llama_tokenizer import LlamaTokenizer

__all__ = ["LlamaTokenizerFast"]

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    Note:
        Currently, the llama_tokenizer_fast process supports only the 'right' padding mode.
        padding_side = "right"

    Note:
        If you want to change the `bos_token` or the `eos_token`, make sure to specify
        them when initializing the model, or call `tokenizer.update_post_processor()`
        to make sure that the post-processing is correctly done (otherwise the values
        of the first token and final token of an encoded sequence will not be correct).

    Args:
        vocab_file (str, optional): `SentencePiece <https://github.com/google/sentencepiece>`_
            file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer. Default: ``None`` .
        tokenizer_file (str, optional):
            Tokenizers file (generally has a .json extension) that contains everything needed to load the tokenizer.
            Default: ``None`` .
        clean_up_tokenization_spaces (bool, optional):
            Whether to clean-up spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces. Default: ``False`` .
        unk_token (Union[str, tokenizers.AddedToken], optional):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Default: ``"<unk>"`` .
        bos_token (Union[str, tokenizers.AddedToken], optional):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            Default: ``"<s>"`` .
        eos_token (Union[str, tokenizers.AddedToken], optional):
            The end of sequence token. Default: ``"</s>"`` .
        add_bos_token (bool, optional):
            Whether to add an `bos_token` at the start of sequences. Default: ``True`` .
        add_eos_token (bool, optional):
            Whether to add an `eos_token` at the end of sequences. Default: ``False`` .
        use_default_system_prompt (bool, optional):
            Whether the default system prompt for Llama should be used. Default: ``False`` .

    Returns:
        LlamaTokenizer, a LlamaTokenizer instance.

    Examples:
        >>> from transformers import LlamaTokenizerFast
        >>>
        >>> tokenizer = LlamaTokenizerFast(vocab_file="./llama2/tokenizer.model")
        >>> tokenizer.encode("Hello this is a test")
        [1, 15043, 445, 338, 263, 1243]
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    FILE_LIST = ['tokenizer_config.json']
    _support_list = MindFormerBook.get_tokenizer_support_list()['llama']
    slow_tokenizer_class = LlamaTokenizer

    # Currently, the llama_tokenizer_fast process supports only the 'right' padding mode.
    padding_side = "right"

    def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            clean_up_tokenization_spaces=False,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            add_bos_token=True,
            add_eos_token=False,
            use_default_system_prompt=False,
            **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.

        Raises:
            ValueError: Raised if `add_bos_token` or `add_eos_token` is set but the
            corresponding token is `None`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} " \
               f"$B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary to the specified directory. This method is used to
        export the vocabulary file from the slow tokenizer.

        Args:
            save_directory (str): The directory where the vocabulary will be saved.
            filename_prefix (str, optional): The prefix for the saved files. Default: ``None`` .

        Returns:
            A tuple containing the paths of the saved vocabulary files.

        Raises:
            ValueError: Raises this exception if the vocabulary cannot be saved from
            a fast tokenizer, or if the specified save directory does not exist.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory", save_directory)
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # ArthurZ let's rely on the template processor instead, refactor all fast tokenizers
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Insert the special tokens to the input_ids, currently.

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Second list of IDs for sequence pairs. Default: ``None`` , only use one sequence.

        Returns:
            list of the tokens after inserting special tokens.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output
