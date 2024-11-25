# Copyright 2022 Huawei Technologies Co., Ltd
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
CLIPProcessor
"""
from typing import Optional, Union, List
import numpy as np
import PIL

import mindspore as ms

from mindformers.dataset import (
    BCHW2BHWC, BatchResize, BatchToTensor,
    BatchNormalize, BatchCenterCrop, BatchPILize
)
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.processing_utils import ProcessorMixin
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CLIPImageProcessor(BaseImageProcessor):
    """
    CLIPImageProcessor.

    Args:
        image_resolution (int): The target size.

    Examples:
        >>> from mindformers import CLIPImageProcessor
        >>> processor = CLIPImageProcessor(image_resolution=256)
        >>> type(processor)
        <class 'mindformers.models.clip.clip_processor.CLIPImageProcessor'>
    """
    def __init__(self, image_resolution: Optional[int] = 224, **kwargs):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image,
                                       np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        bchw2bhwc = BCHW2BHWC()
        batch_pilizer = BatchPILize()
        batch_resizer = BatchResize(self.image_resolution)
        batch_crop = BatchCenterCrop(self.image_resolution)
        batch_totensor = BatchToTensor()
        batch_normalizer = BatchNormalize()

        if not self._bhwc_check(images):
            images = bchw2bhwc(images)
        images = batch_pilizer(images)
        images = batch_resizer(images)
        images = batch_crop(images)
        images = batch_totensor(images)
        images = batch_normalizer(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    def _bhwc_check(self, image_batch: Union[ms.Tensor, PIL.Image.Image,
                                             np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CLIPProcessor(ProcessorMixin):
    r"""CLIP Processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (PreTrainedTokenizerBase) for text input.

    Args:
        image_processor (BaseImageProcessor): Used for process image data.
        tokenizer (PreTrainedTokenizerBase): Used for process text data.
        max_length (Optional[int]): The length of text tokens.
        padding (Optional[str]): The padding strategy of tokenizer, [None, "max_length"].
        return_tensors (Optional[str]): The type of returned tensors for tokenizer, [None, "ms"].

    Examples:
        >>> from mindformers import CLIPProcessor
        >>> processor = CLIPProcessor.from_pretrained('clip_vit_b_32')
        >>> type(processor)
        <class 'mindformers.models.clip.clip_processor.CLIPProcessor'>
    """
    _support_list = MindFormerBook.get_processor_support_list()['clip']

    attributes = ["tokenizer", "image_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "CLIPTokenizer"

    def __init__(self, image_processor, tokenizer,
                 max_length=77, padding='max_length', return_tensors='ms'):
        super(CLIPProcessor, self).__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors)

    def __call__(self, image_input=None, text_input=None):
        """call function"""
        output = {}

        if image_input is not None and self.image_processor:
            if not isinstance(self.image_processor, BaseImageProcessor):
                raise TypeError(f"feature_extractor should inherit from the BaseImageProcessor,"
                                f" but got {type(self.image_processor)}.")

            image_output = self.image_processor(image_input)
            output['image'] = image_output

        if text_input is not None and self.tokenizer:
            if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"tokenizer should inherited from the from PreTrainedTokenizerBase,"
                                f" but got {type(self.tokenizer)}.")
            # Format the input into a batch
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input, return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            output['text'] = text_output

        return output
