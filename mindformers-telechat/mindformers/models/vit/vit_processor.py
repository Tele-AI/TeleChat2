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
ViTProcessor
"""
import numpy as np
from PIL import Image

from mindspore import Tensor
from mindspore.dataset.vision import CenterCrop, ToTensor, Normalize

from mindformers.mindformer_book import MindFormerBook
from mindformers.dataset import Resize
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.processing_utils import ProcessorMixin
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class ViTImageProcessor(BaseImageProcessor):
    """
    ViTImageProcessor.

    Args:
        image_resolution (int): the target size.

    Examples:
        >>> from mindformers import ViTImageProcessor
        >>> vit_image_processor = ViTImageProcessor(224)
        >>> type(vit_image_processor)
        <class 'mindformers.models.vit.vit_processor.ViTImageProcessor'>
    """

    def __init__(self,
                 size=224,
                 resize=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 is_hwc=False,
                 interpolation='cubic',
                 **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.resize = resize
        self.mean = mean
        self.std = std
        self.is_hwc = is_hwc
        self.interpolation = interpolation

    def preprocess(self, images, **kwargs):
        """
        Preprocess required by base processor.

        Args:
            images (tensor, PIL.Image, numpy.array, list): a batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = self._format_inputs(images)

        resize = Resize(self.resize, interpolation=self.interpolation)
        center_crop = CenterCrop(self.size)
        to_tensor = ToTensor()
        normalize = Normalize(mean=self.mean, std=self.std, is_hwc=self.is_hwc)

        res = []
        for image in images:
            image = resize(image)
            image = center_crop(image)
            image = to_tensor(image)
            image = normalize(image)
            res.append(image)
        return Tensor(res)

    def _format_inputs(self, inputs):
        """
        Transform image classification inputs into (bz, h, w, c) or (h, w, c) numpy array.

        Args:
             inputs (tensor, numpy.array, PIL.Image, list, BaseDataset):
             for numpy or tensor input, the channel could be (bz, c, h, w), (c, h, w) or (bz, h, w, c), (h, w, c);
             for list, the item could be PIL.Image, numpy.array, Tensor;
             for BaseDataset, return without any operations.

        Return:
             transformed images:
             for PIL.Image, numpy or tensor input, return a numpy array, the channel is (bz, h, w, c) or (h, w, c);
             for list, return a numpy array for each element;
             for BaseDataset, it is returned directly.
        """
        if not isinstance(inputs, (list, Image.Image, Tensor, np.ndarray, BaseDataset)):
            raise TypeError("input type is not Tensor, numpy, Image, list of Image or MindFormer BaseDataset")

        if isinstance(inputs, list):
            return [self._format_inputs(item) for item in inputs]

        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)

        if isinstance(inputs, Tensor):
            inputs = inputs.asnumpy()

        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, 0)
                inputs = self._chw2hwc(inputs)
            elif len(inputs.shape) == 4:
                inputs = self._chw2hwc(inputs)
            else:
                raise ValueError(f"the rank of image_batch should be 3 or 4,"
                                 f" but got {len(inputs.shape)}")
        return inputs

    @staticmethod
    def _chw2hwc(inputs):
        if inputs.shape[-1] != 3:
            inputs = inputs.transpose(0, 2, 3, 1)
        return inputs


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class ViTProcessor(ProcessorMixin):
    """
    Vit processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (PreTrainedTokenizerBase) for text input.
    """
    _support_list = MindFormerBook.get_processor_support_list()['vit']

    attributes = ["image_processor"]
    image_processor_class = "ViTImageProcessor"

    def __init__(self, image_processor=None, return_tensors='ms'):
        super().__init__(
            image_processor=image_processor,
            return_tensors=return_tensors
        )

    def __call__(self, text_input=None, text_pair=None):
        """call function"""
        output = {}
        if not self.tokenizer:
            raise ValueError(f"For {self.__name__}, the `tokenizer` should not be None.")
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"tokenizer should inherited from the PreTrainedTokenizerBase,"
                            f" but got {type(self.tokenizer)}.")
        if text_input:
            # Format the input into a batch
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input, return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            output['text'] = text_output

        if text_pair:
            # Format the input into a batch
            if isinstance(text_pair, str):
                text_input = [text_pair]
            text_output = self.tokenizer(text_pair, return_tensors=self.return_tensors,
                                         max_length=self.tgt_max_length,
                                         padding=self.padding)["input_ids"]
            output['tgt_output'] = text_output

        return output
