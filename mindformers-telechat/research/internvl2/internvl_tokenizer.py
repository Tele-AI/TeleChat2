# Copyright 2025 Huawei Technologies Co., Ltd
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
Internvl Tokenizer
"""
from typing import Any, Dict, Optional

from mindformers.models.llama import LlamaTokenizer
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.models.tokenization_utils import AddedToken


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class InternvlTokenizer(LlamaTokenizer):
    """
    Construct a Internvl tokenizer. Based on byte-level Byte-Pair-Encoding.
    The default padding token is unset as there is no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
            this token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
            token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored
            by attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other
            things to set:

              -  `enable_sampling`: Enable subword regularization.
              -  `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

                `nbest_size = {0,1}`: No sampling is performed.
                `nbest_size > 1`: samples from the nbest_size results.
                `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

              -  `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
                BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add an `eos_token` at the end of sequences.
        add_special_tokens (bool, optional):
            Whether to add the special tokens associated with the corresponding model. Default: ``True``.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether to clean up spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        legacy (`bool`, *optional*):
            Whether the `legacy` behavior of the tokenizer should be used.
    """
    def __init__(self,
                 vocab_file,
                 unk_token='<unk>',
                 bos_token='<s>',
                 eos_token='</s>',
                 pad_token='<unk>',
                 sp_model_kwargs: Optional[Dict[str, Any]] = None,
                 add_bos_token=False,
                 add_eos_token=False,
                 add_special_tokens=True,
                 clean_up_tokenization_spaces=False,
                 legacy=True,
                 **kwargs):
        add_tokens = {
            "</box>": 64006,
            "</quad>": 64002,
            "</ref>": 64004,
            "<IMG_CONTEXT>": 64000,
            "<box>": 64005,
            "<quad>": 64001,
            "<ref>": 64003
        }

        self._add_tokens_decoder = {}
        for i in add_tokens:
            self._add_tokens_decoder[add_tokens[i]] = AddedToken(
                i, lstrip=False, normalized=False, rstrip=False, single_word=False, special=True
            )
        kwargs["added_tokens_decoder"] = self._add_tokens_decoder
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_special_tokens=add_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            legacy=legacy,
            **kwargs
        )
