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
ViTMAEProcessor
"""
import numpy as np
from PIL import Image

from mindspore import Tensor
from mindspore.dataset.vision.transforms import ToTensor, Normalize

from mindformers.mindformer_book import MindFormerBook
from mindformers.dataset import Resize
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.processing_utils import ProcessorMixin
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['ViTMAEProcessor', 'ViTMAEImageProcessor']


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class ViTMAEImageProcessor(BaseImageProcessor):
    """
    ViTMAEImageProcessor.

    Args:
        image_resolution (int): the target size.
        patch_size(int): patch size.
        mask_ratio(float): mask ratio of image.
    """
    def __init__(self,
                 size=224,
                 patch_size=16,
                 mask_ratio=0.75,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 is_hwc=False,
                 interpolation='cubic',
                 **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mean = mean
        self.std = std
        self.is_hwc = is_hwc
        self.interpolation = interpolation

        if not 0 < mask_ratio < 1:
            raise ValueError('masking ratio must be kept between 0 and 1, but get mask_ratio {mask_ratio}.')
        # seq_length
        self.num_patches = (size // patch_size) ** 2
        # seq masked number
        self.keep_num = int((1 - mask_ratio) * self.num_patches)


    def preprocess(self, images, **kwargs):
        """
        Preprocess required by base processor.

        Args:
            images (tensor, PIL.Image, numpy.array, list): a batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        resize = Resize((self.size, self.size), interpolation=self.interpolation)
        to_tensor = ToTensor()
        normalize = Normalize(mean=self.mean, std=self.std, is_hwc=self.is_hwc)

        images = self._format_inputs(images)

        res = []
        ids_restores = []
        masks = []
        unmask_indexes = []
        for image in images:
            image = resize(image)
            image = to_tensor(image)
            image = normalize(image)
            res.append(image)
            rand_indices = np.argsort(
                np.random.uniform(size=(self.num_patches,)), axis=0).astype(np.int32)
            ids_restore = np.argsort(rand_indices, axis=0).astype(np.int32)
            ids_restores.append(ids_restore)
            mask = np.ones((self.num_patches,)).astype(np.int32)
            mask[:self.keep_num] = 0
            masks.append(mask)
            unmask_index = rand_indices[:self.keep_num]
            unmask_indexes.append(unmask_index)
        return Tensor(res), Tensor(masks), Tensor(ids_restores), Tensor(unmask_indexes)

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
class ViTMAEProcessor(ProcessorMixin):
    """
    ViTMAEProcessor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input.

    Examples:
        >>> from mindformers import MindFormerBook
        >>> from mindformers.models import ViTMAEProcessor
        >>> import os
        >>> yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
        ...                          "mae", "model_config", "mae_vit_base_p16.yaml")
        >>> # build ViTMAEProcessor from pretrained
        >>> pro_a = ViTMAEProcessor.from_pretrained('mae_vit_base_p16')
        >>> type(pro_a)
        <class 'mindformers.models.mae.mae_processor.ViTMAEProcessor'>
    """
    _support_list = MindFormerBook.get_processor_support_list()['mae']

    attributes = ["image_processor"]
    image_processor_class = "ViTMAEImageProcessor"

    def __init__(self, image_processor=None, return_tensors='ms'):
        super(ViTMAEProcessor, self).__init__(
            image_processor=image_processor,
            return_tensors=return_tensors
        )

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
