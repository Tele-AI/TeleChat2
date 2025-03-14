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
"""llava next tokenizer"""
from mindformers import AutoTokenizer
from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.tokenization_utils import AddedToken


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlavaNextTokenizer:
    r"""
    Construct a Llava-Next tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset
    as there is no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the tokenizer files.
        image_tag (`str` or `tokenizers.AddedToken`, *optional*):
            A image token tag means the images tensor will fill in this position in input ids.
    """

    def __new__(
            cls,
            vocab_file,
            image_tag="<image>",
            **kwargs,
    ):
        cls._instance = AutoTokenizer.from_pretrained(vocab_file, **kwargs)
        cls._instance.image_token = image_tag
        if len(cls._instance.tokenize(image_tag)) != 1:
            cls._instance.add_tokens(AddedToken(image_tag, lstrip=False, rstrip=False, special=False),
                                     special_tokens=True)
        cls._instance.img_token_id = cls._instance.convert_tokens_to_ids(image_tag)

        return cls._instance
