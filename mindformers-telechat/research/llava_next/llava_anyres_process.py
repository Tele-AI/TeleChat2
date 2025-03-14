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
"""
Llava any res method
"""
import math
from typing import List, Union, Tuple, Iterable, Optional

import numpy as np

from mindformers.utils.image_transforms import pad
from mindformers.utils.image_transforms import to_channel_dimension_format, resize
from mindformers.utils.image_utils import ChannelDimension, select_best_resolution, get_image_size, divide_to_patches, \
    infer_channel_dimension_format
from mindformers.utils.image_utils import PaddingMode


def _get_patch_output_size(image, target_resolution, input_data_format):
    """based on scale, chose best resized height and width"""
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class LLavaAnyRes:
    """
    Used in single image to split one image into several small images
    """

    def __init__(self, processor):
        super(LLavaAnyRes, self).__init__()
        self.image_processor = processor

    def __call__(self, images, *args, **kwargs):
        new_images = []
        images = [np.array(image) for image in images]
        input_data_format = infer_channel_dimension_format(images[0])
        for image in images:
            image_pathed = self.get_image_patches(
                image,
                size=(self.image_processor.size["shortest_edge"], self.image_processor.size["shortest_edge"])
                if "shortest_edge" in self.image_processor.size
                else (
                    min(self.image_processor.size["height"], self.image_processor.size["width"]),
                    min(self.image_processor.size["height"], self.image_processor.size["width"])),
                patch_size=self.image_processor.crop_size["height"]
                if getattr(self.image_processor, "crop_size", None) is not None
                else self.image_processor.size["height"],
                data_format=input_data_format,
                input_data_format=input_data_format,
            )
            new_images.append(image_pathed)
        return new_images[0]

    def get_image_patches(
            self,
            image: np.array,
            size: tuple,
            patch_size: int,
            data_format: ChannelDimension,
            input_data_format: ChannelDimension,
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """
        grid_pinpoints = self.image_processor.image_grid_pinpoints
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=self.image_processor.resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]

        resized_original_image = resize(
            image,
            size=size,
            resample=self.image_processor.resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        image_patches = [resized_original_image] + patches

        return image_patches

    def _resize_for_patching(
            self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = resize(image, (new_height, new_width), resample=resample, input_data_format=input_data_format)

        return resized_image

    def _pad_for_patching(
            self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = self.pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def pad(
            self,
            image: np.ndarray,
            padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
            mode: PaddingMode = PaddingMode.CONSTANT,
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.

        Args:
            image (`np.ndarray`):
                The image to pad.
            padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
                Padding to apply to the edges of the height, width axes.
            mode (`PaddingMode`):
                The padding mode to use.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image.
                If unset, will use the inferred format of the input image.

        Returns:
            `np.ndarray`: The padded image.

        """
        constant_values = 0.0
        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        return image
