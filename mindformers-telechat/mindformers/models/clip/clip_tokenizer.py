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

"""
This is a temporary version of clip tokenizer
"""
import gzip
import html
import os
import shutil
from functools import lru_cache
from typing import List, Optional
import ftfy
import regex as re

from mindformers.tools.logger import logger
from mindformers.tools.utils import try_sync_file
from ...mindformer_book import MindFormerBook
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...tools.download_tools import download_with_progress_bar
from ..tokenization_utils import PreTrainedTokenizer


__all__ = ['CLIPTokenizer']


VOCAB_FILES_NAMES = {'vocab_file': 'bpe_simple_vocab_16e6.txt.gz'}


@lru_cache()
def default_bpe():
    r"""Bpe path"""
    path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                        'clip', "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(path):
        url = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/" \
              "XFormer_for_mindspore/clip/bpe_simple_vocab_16e6.txt.gz"
        download_with_progress_bar(url, path)
    try_sync_file(path)
    return path


def get_pairs(input_wd):
    r"""Get_pairs"""
    output = set()
    prev_char = input_wd[0]
    for char in input_wd[1:]:
        output.add((prev_char, char))
        prev_char = char
    return output


@lru_cache()
def bytes_to_unicode():
    r"""Bytes_to_unicode"""
    input_bt = list(range(ord("!"), ord("~") + 1))\
         + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    output_cd = input_bt[:]
    num = 0
    for item in range(2**8):
        if item not in input_bt:
            input_bt.append(item)
            output_cd.append(2**8 + num)
            num += 1
    output_cd = [chr(item) for item in output_cd]
    return dict(zip(input_bt, output_cd))


def whitespace_clean(input_text):
    r"""Whitespace clean"""
    input_text = re.sub(r'\s+', ' ', input_text)
    input_text = input_text.strip()
    return input_text


def basic_clean(input_text):
    r"""Basic_clean"""
    input_text = ftfy.fix_text(input_text)
    input_text = html.unescape(html.unescape(input_text))
    return input_text.strip()


class TempTokenizer:
    r"""Simple Tokenizer"""
    def __init__(self, merges, vocab, flag_dict):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.flag_dict = flag_dict
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

    def tokenize_alg(self, input_tk):
        r"""Bpe"""
        if input_tk in self.flag_dict:
            return self.flag_dict[input_tk]
        word = tuple(input_tk[:-1]) + (input_tk[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return input_tk + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

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
            pairs = get_pairs(word)
        word_str = ' '.join(word)
        self.flag_dict[input_tk] = word_str
        return word_str

    def decode(self, input_ids):
        r"""Decode"""
        output_text = ''.join([self.decoder[input_id] for input_id in input_ids])
        output_text = bytearray([self.byte_decoder[c] for
                                 c in output_text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return output_text

    def encode(self, content):
        r"""Encode"""
        output_ids = []
        content = whitespace_clean(basic_clean(content)).lower()
        for token in re.findall(self.pat, content):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            output_ids.extend(self.encoder[bpe_token] for bpe_token in self.tokenize_alg(token).split(' '))
        print("res is:", output_ids)
        return output_ids


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class CLIPTokenizer(PreTrainedTokenizer):
    r"""
    CLIP Tokenizer

    Args:
        vocab_file(str): The vocabulary file path.
        eos_token(str): The token that represents the end-of-sentence. Default "<|endoftext|>".
        bos_token(str): The token that represents the begin-of-sentence. Default "<|startoftext|>"".
        pad_token(str): The token that represents the pad. Default "<|endoftext|>".
        unk_token(str): The token that represents the unknown. Default "<|endoftext|>".
        add_prefix_space(bool): whether to add a whitespace in the front of text. Default "False"
        add_bos_token(bool): Whether or not to add the bos_token_id to the left of the input. Default "True"
        add_eos_token(bool): Whether or not to add the eos_token_id to the right of the input. Default "True"
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import CLIPTokenizer
        >>> CLIPTokenizer.show_support_list()
            INFO - support list of CLIPTokenizer is:
            INFO -    ['clip_vit_b_32']
            INFO - -------------------------------------
        >>> tokenizer = CLIPTokenizer.from_pretrained('clip_vit_b_32')
        >>> tokenizer("a boy")
            {'input_ids': [49406, 320, 1876, 49407], 'attention_mask': [1, 1, 1, 1]}
    """
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES
    FILE_LIST = ['tokenizer_config.json']
    '''clip tokenizer'''
    _support_list = MindFormerBook.get_tokenizer_support_list()['clip']

    def __init__(self,
                 vocab_file,
                 eos_token="<|endoftext|>",
                 bos_token="<|startoftext|>",
                 pad_token="<|endoftext|>",
                 unk_token="<|endoftext|>",
                 add_bos_token=True,
                 add_eos_token=True,
                 **kwargs
                 ):
        self.path = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        merges = self._read_merge_files(vocab_file)
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend([bos_token, eos_token])

        flag_dict = {bos_token: bos_token, eos_token: eos_token}
        self.tool = TempTokenizer(merges, vocab, flag_dict)
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|
        've|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        super(CLIPTokenizer, self).__init__(
            eos_token=eos_token,
            bos_token=bos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )

    @staticmethod
    def _read_merge_files(text_path, start_pos=1, end_pos=49152 - 256 - 2 + 1):
        r"""Read the merge files"""
        with gzip.open(text_path) as fp:
            data = fp.read()
        merges = data.decode("utf-8").split('\n')
        merges = merges[start_pos: end_pos]
        new_list = []
        for item in merges:
            new_list.append(tuple(item.split()))
        return new_list

    def _tokenize(self, text, **kwargs):
        r"""Tokenize"""
        output_ids = []
        content = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, content):
            token = ''.join(self.tool.byte_encoder[b] for b in token.encode('utf-8'))
            output_ids.extend(self.tool.tokenize_alg(token).split(' '))
        return output_ids

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Insert the special tokens to the input_ids. Currently, we support token_ids_0 is a list of ids.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def save_vocabulary(self, save_directory, filename_prefix=None):
        r"""Save_vocabulary"""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory", save_directory)
            return None

        output_file_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"][0])

        shutil.copy(self.path, output_file_path)
        return (output_file_path,)

    def _convert_token_to_id(self, token):
        r"""Convert_token_to_id"""
        return self.tool.encoder[token]

    def _convert_id_to_token(self, index):
        r"""Convert_id_to_token"""
        return self.tool.decoder[index]

    @property
    def vocab_size(self):
        r"""Get the vocab size"""
        return len(self.tool.encoder)

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
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
