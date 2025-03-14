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
"""llama3 tokenizer APIs."""

from typing import Dict

from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .llama3_2_tokenizer import Llama3Tokenizer

__all__ = ['MllamaTokenizer']

PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| " \
          r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


# pylint: disable=W0223
@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class MllamaTokenizer(Llama3Tokenizer):
    """Mllama Tokenizer"""
    VOCAB_FILES = {'vocab_file': 'tokenizer.json'}
    FILE_LIST = []
    _support_list = MindFormerBook.get_tokenizer_support_list()['mllama']
    special_tokens: Dict[str, int]

    def __init__(self,
                 vocab_file,
                 bos_token="<|begin_of_text|>",
                 eos_token="<|end_of_text|>",
                 pad_token="<|finetune_right_pad_id|>",
                 add_bos_token=False,
                 add_eos_token=False,
                 errors="replace",
                 num_reserved_special_tokens=256,
                 num_reserved_start_pos=2,
                 special_tokens_used_num=9,
                 **kwargs):
        num_reserved_start = num_reserved_start_pos
        num_reserved_end = num_reserved_special_tokens - special_tokens_used_num
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(num_reserved_start, num_reserved_end)
        ] + ["<|image|>"]

        super().__init__(vocab_file,
                         bos_token,
                         eos_token,
                         pad_token,
                         add_bos_token,
                         add_eos_token,
                         errors,
                         num_reserved_special_tokens,
                         special_tokens,
                         num_reserved_start_pos,
                         special_tokens_used_num,
                         **kwargs)
