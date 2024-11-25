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
transform methods for vision models
"""
import numpy as np
from PIL import Image

from mindspore.dataset import vision
import mindspore as ms

from mindspore.dataset.vision.utils import Inter
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = [
    'BatchResize', 'BCHW2BHWC', 'BatchPILize',
    'BatchNormalize', 'BatchCenterCrop', 'BatchToTensor',
    'RandomCropDecodeResize', 'RandomResizedCrop', 'Resize',
    'RandomHorizontalFlip'
]

INTERPOLATION = {'nearest': Inter.NEAREST,
                 'antialias': Inter.ANTIALIAS,
                 'linear': Inter.LINEAR,
                 'cubic': Inter.PILCUBIC,
                 'bicubic': Inter.BICUBIC}


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BCHW2BHWC:
    """
    Transform a batch of image from CHW to HWC.

    Args:
         image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input, the
         channel should be (bz, c, h, w) or (c, h, w). for list, the item should be
        PIL.Image or numpy.array (c, h, w).

    Return:
         transformed image batch: for numpy or tensor input, return a numpy array, the channel
         is (bz, h, w, c) or (h, w, c); for PIL.Image input, it is returned directly.
    """

    def __call__(self, image_batch):
        """the call function"""
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return image_batch.transpose(0, 2, 3, 1)
            if len(image_batch.shape) == 3:
                return image_batch.transpose(1, 2, 0)
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return image_batch
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BatchResize:
    """
    Resize a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
         interpolation (str): interpolate method, default is 'cubic'.
    """

    def __init__(self, image_resolution, interpolation='cubic'):
        self.interpolation = INTERPOLATION.get(interpolation)
        self.resize = vision.c_transforms.Resize(image_resolution, self.interpolation)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the shape should be (bz, h, w, c) or (h, w, c). for list, the item should be
            PIL.Image or numpy.array (h, w, c).

        Returns:
            resized image batch: for numpy or tensor input, return a numpy array;
            for PIL.Image input, it returns PIL.Image.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.resize(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.resize(item)[np.newaxis, :]
                                     for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.resize(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.resize(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BatchCenterCrop:
    """
    CenterCrop a batch of image to the given shape.

    Args:
         image_resolution (int): the target size.
    """

    def __init__(self, image_resolution):
        self.crop = vision.CenterCrop(image_resolution)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the shape should be (bz, h, w, c) or (h, w, c). for list, the item should be
            PIL.Image or numpy.array (h, w, c).

        Returns:
            center cropped image batch: for numpy or tensor input, return a numpy array, the shape
            is (bz, image_resolution, image_resolution, c) or (image_resolution,
            image_resolution, c); for PIL.Image input, it is returned with shape (image_resolution,
            image_resolution).
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.crop(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.crop(item)[np.newaxis, :]
                                     for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.crop(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.crop(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BatchToTensor:
    """Transform a batch of image to tensor and scale to (0, 1)."""

    def __init__(self):
        self.totensor = ms.dataset.vision.ToTensor()

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, PIL.Image, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be PIL.Image or numpy.array.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.totensor(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return np.row_stack([self.totensor(item)[np.newaxis, :]
                                     for item in image_batch])
            if len(image_batch.shape) == 3:
                return self.totensor(image_batch)
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")
        if isinstance(image_batch, Image.Image):
            return self.totensor(image_batch)
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BatchNormalize:
    """Normalize a batch of image."""

    def __init__(
            self,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            is_hwc=False
    ):
        self.normalize = vision.Normalize(mean=mean, std=std, is_hwc=is_hwc)

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be numpy.array.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, list):
            return [self.normalize(item) for item in image_batch]
        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 3:
                return self.normalize(image_batch)
            if len(image_batch.shape) == 4:
                return np.row_stack([self.normalize(item)[np.newaxis, :]
                                     for item in image_batch])
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")
        raise TypeError(f"the type {type(image_batch)} of image_batch is unsupported.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class BatchPILize:
    """transform a batch of image to PIL.Image list."""

    def __call__(self, image_batch):
        """
        The forward process.

        Args:
            image_batch (tensor, numpy.array, list): for tensor or numpy input,
            the rank should be 4 or 3. for list, the item should be PIL.Image.

        Returns:
            return a tensor or a list of tensor.
        """
        if isinstance(image_batch, Image.Image):
            return image_batch

        if isinstance(image_batch, list):
            for item in image_batch:
                if not isinstance(item, Image.Image):
                    raise TypeError("unsupported type in list,"
                                    " when the image_batch is a list,"
                                    " the item in list should be PIL.Image.")
            return image_batch

        if isinstance(image_batch, ms.Tensor):
            image_batch = image_batch.asnumpy()

        if isinstance(image_batch, np.ndarray):
            if len(image_batch.shape) == 4:
                return [Image.fromarray(item.astype(np.uint8)) for item in image_batch]
            if len(image_batch.shape) == 3:
                return Image.fromarray(image_batch.astype(np.uint8))
            raise ValueError(f"the rank of image_batch should be 3 or 4,"
                             f" but got {len(image_batch.shape)}")

        raise ValueError("unsupported input type.")


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class RandomCropDecodeResize(vision.transforms.RandomCropDecodeResize):
    """wrapper of RandomCropDecodeResize"""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='cubic', max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = INTERPOLATION.get(interpolation)
        self.max_attempts = max_attempts
        super(RandomCropDecodeResize, self).__init__(size, scale, ratio, self.interpolation, max_attempts)


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class RandomResizedCrop(vision.transforms.RandomResizedCrop):
    """wrapper of RandomCropDecodeResize"""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='cubic', max_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = INTERPOLATION.get(interpolation)
        self.max_attempts = max_attempts
        super(RandomResizedCrop, self).__init__(size, scale, ratio, self.interpolation, max_attempts)


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class Resize(vision.transforms.Resize):
    """wrapper of Resize"""

    def __init__(self, size, interpolation='cubic'):
        self.size = size
        self.interpolation = INTERPOLATION.get(interpolation)
        self.random = False
        super(Resize, self).__init__(size, self.interpolation)


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class RandomHorizontalFlip(vision.transforms.RandomHorizontalFlip):
    """wrapper of RandomHorizontalFlip"""

    def __init__(self, prob=0.5):
        self.prob = prob
        super(RandomHorizontalFlip, self).__init__(prob)
