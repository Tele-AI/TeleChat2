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
Image processor
"""
from typing import Union, Optional, List, Dict

import numpy as np
import PIL
from PIL import Image
import mindspore as ms

from mindformers.models import BaseImageProcessor
from mindformers.models.image_processing_utils import get_size_dict
from mindformers.tools import MindFormerRegister, MindFormerModuleType
from mindformers.utils.image_transforms import to_channel_dimension_format, resize, PILIMAGERESAMPLING, \
    get_resize_output_image_size
from mindformers.utils.image_utils import ChannelDimension, get_image_size
from mindformers.utils.image_utils import infer_channel_dimension_format


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class ClipImageProcessorV2(BaseImageProcessor):
    """
    Constructs a Llava Clip image processor.

    Args:
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        crop_size(Dict[str, int]):
            Size of the output image after applying center_crop
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess`
            method.
        image_grid_pinpoints(List[[int, int]]):
            A list of possible resolutions to use for processing high resolution images. The best resolution is
            selected based on the original size of the image.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
            method. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    def __init__(self,
                 size: Dict[str, int] = None,
                 crop_size: Dict[str, int] = None,
                 resample: PILIMAGERESAMPLING = PILIMAGERESAMPLING.BICUBIC,
                 image_grid_pinpoints: List = None,
                 rescale_factor: Union[int, float] = 1 / 255,
                 do_pad: Optional[bool] = False,
                 image_mean: Optional[Union[float, List[float]]] = None,
                 image_std: Optional[Union[float, List[float]]] = None,
                 **kwargs) -> None:
        super(ClipImageProcessorV2, self).__init__(**kwargs)
        self.resample = resample
        size = size if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
        self.image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.do_pad = do_pad
        self.size = size
        self.crop_size = crop_size
        self.rescale_factor = rescale_factor
        self.expected_ndims = 3
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

    # pylint: disable=W0613
    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
        """image process to get final image tensor"""
        if isinstance(images, PIL.Image.Image):
            images = [images]
        elif isinstance(images, list):
            images = images
        elif images.ndim == self.expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == self.expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {self.expected_ndims + 1} or {self.expected_ndims} "
                f"dimensions, but got {images.ndim} dimensions."
            )
        images = [np.array(image) for image in images]

        input_data_format = infer_channel_dimension_format(images[0])
        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        processed_images = self._preprocess(images, input_data_format=input_data_format)
        output = {"pixel_values": ms.Tensor(processed_images, ms.float32),
                  "image_sizes": image_sizes}
        return output

    def _pad_for_batching(
            self,
            pixel_values: List[np.ndarray],
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            List[`np.ndarray`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
            self,
            images,
            data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:
        """image process to get final image tensor"""

        if isinstance(images, PIL.Image.Image):
            # PIL images are never batched
            return [images]

        all_images = []
        for image in images:
            image = self.resize(image=image, size=self.size, resample=self.resample,
                                input_data_format=input_data_format)

            image = self.center_crop(image=image, size=self.crop_size, input_data_format=input_data_format)

            image = self.rescale(image=image, scale=self.rescale_factor, input_data_format=input_data_format)

            image = self.normalize(
                image=image, mean=self.image_mean, std=self.image_std, input_data_format=input_data_format
            )

            all_images.append(image)
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]
        return images

    def resize(
            self,
            image: np.ndarray,
            size: Dict[str, int],
            resample: PILIMAGERESAMPLING = PILIMAGERESAMPLING.BICUBIC,
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            **kwargs,
    ) -> np.ndarray:
        """resize images to spesific height and width"""

        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class SiglipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SigLIP image processor.

    Args:
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        image_grid_pinpoints(List[[int, int]]):
            A list of possible resolutions to use for processing high resolution images. The best resolution is
            selected based on the original size of the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
            self,
            size: Dict[str, int] = None,
            resample: PILIMAGERESAMPLING = PILIMAGERESAMPLING.BICUBIC,
            rescale_factor: Union[int, float] = 1 / 255,
            image_grid_pinpoints: List = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            **kwargs,
    ) -> None:
        super(SiglipImageProcessor, self).__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        self.image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.expected_ndims = 3

    # pylint: disable=W0613
    def preprocess(self, images, **kwargs):
        """image process to get final image tensor with specific sizes"""
        if isinstance(images, PIL.Image.Image):
            images = [images]
        elif isinstance(images, list):
            images = images
        elif images.ndim == self.expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == self.expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {self.expected_ndims + 1} or {self.expected_ndims} dimensions, "
                f"but got {images.ndim} dimensions."
            )
        images = [np.array(image) for image in images]

        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(images[0])
        height, width = self.size.get("height", None), self.size.get("width", None)
        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        new_images = []
        for image in images:
            image = resize(image, size=(height, width), resample=self.resample, data_format=input_data_format)
            image = self.rescale(image=image, scale=self.rescale_factor, input_data_format=input_data_format)
            image = self.normalize(image=image, mean=self.image_mean, std=self.image_std,
                                   input_data_format=input_data_format)
            image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
            new_images.append(image)

        output = {"pixel_values": ms.Tensor(new_images, ms.float32),
                  "image_sizes": image_sizes}
        return output
