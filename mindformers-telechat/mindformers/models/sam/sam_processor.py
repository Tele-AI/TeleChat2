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
"""SAM Processor"""
from copy import deepcopy
from typing import Tuple
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.models.processing_utils import ProcessorMixin
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class SamImageProcessor(BaseImageProcessor):
    """
    Subclass of BaseImageProcessor implementing image preprocessing for SAM model.

    Args:
        img_size (int): The target size for image processing.
        mean (list): Mean values for image normalization.
        std (list): Standard deviation values for image normalization.
    """

    def __init__(self,
                 img_size=1024,
                 mean=(123.675, 116.28, 103.53),
                 std=(58.395, 57.12, 57.375),
                 **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.mean = mean
        self.std = std

    @property
    def transform(self):
        return ResizeLongestSide(self.img_size)

    def preprocess(self, images, **kwargs):
        """
        Preprocess the input image for SAM model.

        Args:
            image (PIL.Image or numpy.array): Input image to be preprocessed.

        Returns:
            image (tensor): Preprocessed image as a tensor.
            input_size (tuple): Size of the input image after preprocessing.
        """
        pixel_mean = ms.Tensor(self.mean).view(-1, 1, 1)
        pixel_std = ms.Tensor(self.std).view(-1, 1, 1)
        transform = ResizeLongestSide(self.img_size)

        images = np.asarray(images)
        images = transform.apply_image(images)
        images = images.transpose(2, 0, 1)[None, :, :, :]
        images = ms.Tensor(images)
        input_size = images.shape[-2:]

        # Normalize colors
        images = (images - pixel_mean) / pixel_std

        # Pad
        h, w = images.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        if images.ndim == 4:
            pad = P.Pad(paddings=((0, 0), (0, 0), (0, padh), (0, padw)))
        elif images.ndim == 3:
            pad = P.Pad(paddings=((0, 0), (0, padh), (0, padw)))
        images = pad(images)

        return images, input_size


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched mindspore tensors.
    """

    def __init__(self, target_length: int) -> None:
        """
        Initialize the ResizeLongestSide class.

        Args:
            target_length (int): Target length of the longest side.
        """
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the resizing transformation to a single image.

        Args:
            image (np.ndarray): Input image as HxWxC numpy array.

        Returns:
            np.ndarray: Resized image as H'xW'xC numpy array.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        pil_image = Image.fromarray(image)
        resize_op = vision.Resize(size=target_size)
        resized_image = np.array(resize_op(pil_image))
        return resized_image

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Apply the resizing transformation to coordinates.

        Args:
            coords (np.ndarray): Input coordinates as a numpy array of shape (N, 2).
            original_size (Tuple[int, int]): Original image size as (H, W).

        Returns:
            np.ndarray: Resized coordinates as a numpy array of shape (N, 2).
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Apply the resizing transformation to bounding boxes.

        Args:
            boxes (np.ndarray): Input boxes as a numpy array of shape (B, 4).
            original_size (Tuple[int, int]): Original image size as (H, W).

        Returns:
            np.ndarray: Resized boxes as a numpy array of shape (B, 4).
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_batch(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the resizing transformation to a batch of images.

        Args:
            image (np.ndarray): Input batched images as BxCxHxW numpy array.

        Returns:
            np.ndarray: Resized batched images as BxC'xH'xW' numpy array.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        resize_ = vision.Resize(target_size, Inter.BILINEAR)
        image = resize_(image)
        return image

    def apply_coords_batch(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Apply the resizing transformation to a batch of coordinates.

        Args:
            coords (np.ndarray): Input batched coordinates as numpy array of shape (B, N, 2).
            original_size (Tuple[int, int]): Original image size as (H, W).

        Returns:
            np.ndarray: Resized batched coordinates as numpy array of shape (B, N, 2).
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_batch(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Apply the resizing transformation to a batch of bounding boxes.

        Args:
            boxes (np.ndarray): Input batched boxes as numpy array of shape (B, N, 4).
            original_size (Tuple[int, int]): Original image size as (H, W).

        Returns:
            np.ndarray: Resized batched boxes as numpy array of shape (B, N, 4).
        """
        boxes = self.apply_coords_batch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.

        Args:
            oldh (int): Original image height.
            oldw (int): Original image width.
            long_side_length (int): Target length of the longest side.

        Returns:
            Tuple[int, int]: Output size as (new_h, new_w).
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class SamProcessor(ProcessorMixin):
    """
    Vit processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (PreTrainedTokenizerBase) for text input.
    """
    _support_list = MindFormerBook.get_processor_support_list()['sam']

    attributes = ["image_processor"]
    image_processor_class = "SamImageProcessor"

    def __init__(self, image_processor=None, return_tensors='ms'):
        super().__init__(
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
