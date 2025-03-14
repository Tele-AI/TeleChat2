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
"""llava tokenizer"""
from typing import Optional, Dict, Any

from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.llama import LlamaTokenizer
from mindformers.models.tokenization_utils import AddedToken
from mindformers.tools.utils import check_file


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlavaTokenizer(LlamaTokenizer):
    """
    Construct a Llava tokenizer. Based on byte-level Byte-Pair-Encoding.
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
        image_tag (`str` or `tokenizers.AddedToken`, *optional*):
            A image token tag means the images tensor will fill in this position in input ids.

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
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether to clean up spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether the default system prompt for Llama should be used.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of
            #24622 and #25224 which includes fixes to properly handle tokens that appear after special tokens.
    """
    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            image_tag="<image>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            legacy=True,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        image_token = AddedToken(image_tag, lstrip=False, rstrip=False, special=True) if isinstance(image_tag,
                                                                                                    str) else image_tag
        self._img_token_id = 32000
        self.legacy = legacy
        self.vocab_file = vocab_file
        check_file(vocab_file, "tokenizer")
        self.add_bos_token = add_bos_token
        self._image_token = image_tag
        self.add_eos_token = add_eos_token
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        added_tokens_decoder = {}
        added_tokens_decoder[self._img_token_id] = image_token
        added_tokens_decoder[32001] = pad_token
        kwargs["added_tokens_decoder"] = added_tokens_decoder
        super().__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            legacy=legacy,
            **kwargs,
        )

    @property
    def image_token(self):
        return self._image_token

    @property
    def img_token_id(self):
        return self._img_token_id
