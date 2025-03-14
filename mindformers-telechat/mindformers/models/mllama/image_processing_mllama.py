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
"""MllamaImageProcessor"""
from typing import List, Optional, Tuple, Union
import math
import numpy as np
import PIL

import mindspore as ms

from mindformers.models.image_processing_utils import BaseImageProcessor
from mindformers.utils.image_utils import ChannelDimension, PaddingMode
from mindformers.utils.image_transforms import to_channel_dimension_format, get_image_size, resize, pad, \
    PILIMAGERESAMPLING


def make_list_of_images(images):
    """
    Convert a single image or a list of images to a list of numpy arrays.

    Args:
        images (`ImageInput`):
            A single image or a list of images.

    Returns:
        A list of numpy arrays.
    """
    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        output_images = [[images]]
    # If it's a list of images, it's a single batch, so convert it to a list of lists
    elif isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        output_images = [images]
    # If it's a list of batches, it's already in the right format
    elif (
            isinstance(images, (list, tuple))
            and all(isinstance(images_i, (list, tuple)) for images_i in images)
            and any(is_valid_list_of_images(images_i) for images_i in images)
    ):
        output_images = images
    else:
        raise ValueError(
            "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
        )
    return output_images


def is_valid_list_of_images(images: List):
    return images and all(is_valid_image(image) for image in images)


def is_valid_image(img):
    if isinstance(img, PIL.Image.Image):
        return True
    if isinstance(img, np.ndarray):
        return True
    if isinstance(img, ms.Tensor):
        return True
    return False


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")

    if isinstance(img, PIL.Image.Image):
        return np.array(img)

    if isinstance(img, ms.Tensor):
        return img.asnumpy()
    return img


def get_all_supported_aspect_ratios(max_image_tiles: int) -> List[Tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `List[Tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(4)
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return aspect_ratios


def get_optimal_tiled_canvas(
        image_height: int,
        image_width: int,
        max_image_tiles: int,
        tile_size: int,
) -> Tuple[int, int]:
    r"""
    Determines the best canvas based on image and tile size and maximum number of tiles.

    First, calculates possible resolutions based on the maximum number of tiles and tile size.
    For example for max_image_tiles=2, tile_size=100, possible tile arrangements are:
    [(1, 1), (1, 2), (2, 1)] and corresponding canvas sizes are:
    [(100, 100), (100, 200), (200, 100)]

    Args:
        image_height (`int`):
            The height of the image.
        image_width (`int`):
            The width of the image.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.
        tile_size (`int`):
            The tile size.

    Returns:
        `Tuple[int, int]`: The best canvas resolution [height, width] for the given image.
    """
    possible_tile_arrangements = get_all_supported_aspect_ratios(max_image_tiles)
    possible_canvas_sizes = np.array(possible_tile_arrangements) * tile_size

    # get all possible resolutions heights/widths
    target_heights, target_widths = np.array(possible_canvas_sizes).T

    # get scaling factors to resize the image without distortion
    scale_h = target_heights / image_height
    scale_w = target_widths / image_width

    # get the min scale between width and height (limiting side -> no distortion)
    scales = np.where(scale_w > scale_h, scale_h, scale_w)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if np.any(upscaling_options):
        selected_scale = np.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = np.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_canvas_sizes[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = np.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return optimal_canvas


def get_image_size_fit_to_canvas(
        image_height: int,
        image_width: int,
        canvas_height: int,
        canvas_width: int,
        tile_size: int,
) -> Tuple[int, int]:
    """
    Calculates the new size of an image to fit within a canvas while maintaining aspect ratio.

    This function calculates the optimal size for an image to fit within a canvas defined by
    canvas_height and canvas_width, while ensuring that the image dimensions are not smaller than
    tile_size. If the image is larger than the canvas, the returned size will fit within the canvas.
    If the image already fits within the canvas, the size remains unchanged.
    The aspect ratio of the original image is preserved.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        canvas_height (`int`):
            The height of the canvas.
        canvas_width (`int`):
            The width of the canvas.
        tile_size (`int`):
            The tile size.

    Returns:
        `Tuple[int, int]`: A tuple containing the new height and width of the image.

    """
    # Set target image size in between `tile_size` and canvas_size

    target_width = np.clip(image_width, tile_size, canvas_width)
    target_height = np.clip(image_height, tile_size, canvas_height)

    scale_h = target_height / image_height
    scale_w = target_width / image_width

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(image_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(image_width * scale_h), target_width)

    return new_height, new_width


def pack_images(
        batch_images: List[List[np.ndarray]],
        max_image_tiles: int,
        max_num_images: int
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Stack a list of lists of images with variable lengths into a numpy array, applying zero padding as needed.
    Each list in the input represents a batch sample, and each image within a list is expected to be
    pre-split into tiles. The resulting array will have a shape of
    (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).

    Args:
        batch_images (`List[List[np.ndarray]]`):
            A list of lists of image tiles. Each inner list represents
            a batch sample containing multiple images, where each image is pre-split into tiles.
            The shape of each tile array is (num_tiles, channels, tile_height, tile_width).
        max_image_tiles (int):
            The maximum number of tiles any image was potantially split.

    Returns:
        `Tuple[np.ndarray, List[List[int]]]`: A tuple containing:
            - stacked_images (`np.ndarray`):
                A numpy array of stacked images with shape
                (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
            - all_num_tiles (`List[List[int]]`):
                A list of lists containing the number of tiles
                for each image in each batch sample.
    """

    # Determine output shape
    batch_size = len(batch_images)
    max_num_images = max_num_images if max_num_images else max([len(images) for images in batch_images])
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = shapes[0]

    # Initialize the stacked images array with zeros
    stacked_images = np.zeros(
        (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
        dtype=np.float32,
    )

    # Fill the stacked images array with the tiled images from the batch
    all_num_tiles = []
    for i, images in enumerate(batch_images):
        num_sample_tiles = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            stacked_images[i, j, :num_tiles] = image
            num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def convert_aspect_ratios_to_ids(aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles,
                                 max_num_images) -> np.ndarray:
    """
    Convert aspect ratio tuples to unique ids.

    For batch padding we use 0, because there might be different number of images in each batch.
    The aspect ratio ids start from 1, with 1 corresponding to the first supported aspect ratio.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios for each image in the batch.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`:
            The aspect ratios ids as a numpy array with shape (batch_size, max_num_images).
            Each id corresponds to the index of the aspect ratio in the list of supported aspect ratios,
            offset by 1 (so 0 can be used for padding).
    """
    batch_size = len(aspect_ratios)
    max_num_images = max_num_images if max_num_images else max([len(row) for row in aspect_ratios])
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles)

    aspect_ratios_ids = np.zeros((batch_size, max_num_images), dtype=np.int32)
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_h, num_tiles_w) in enumerate(sample_aspect_ratios):
            aspect_ratios_ids[i, j] = supported_aspect_ratios.index((num_tiles_h, num_tiles_w)) + 1
    return aspect_ratios_ids


def build_aspect_ratio_mask(aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int,
                            max_num_images) -> np.ndarray:
    """
    Builds a mask for the aspect ratios of the images.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of lists containing aspect ratios for each image in the batch.
            Each aspect ratio is represented as a tuple of (width, height) in terms of number of tiles.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`: A 3D numpy array of shape (batch_size, max_num_images, max_image_tiles).
            The mask contains 1s for valid tiles and 0s for padding.
    """
    batch_size = len(aspect_ratios)
    max_num_images = max_num_images if max_num_images else max([len(row) for row in aspect_ratios])

    aspect_ratio_mask = np.zeros((batch_size, max_num_images, max_image_tiles), dtype=np.int32)

    # Set the first tile to 1 for all aspect ratios
    # because in original implementation aspect ratios are padded with (1, 1),
    # but original code examples are not built to handle batches, so we might remove it later
    aspect_ratio_mask[:, :, 0] = 1

    # Set the aspect ratio mask for the rest of the tiles
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_w, num_tiles_h) in enumerate(sample_aspect_ratios):
            aspect_ratio_mask[i, j, : num_tiles_w * num_tiles_h] = 1

    return aspect_ratio_mask


def split_to_tiles(image: np.ndarray, num_tiles_height: int, num_tiles_width: int) -> np.ndarray:
    """
    Split an image into a specified number of tiles along its width and height dimensions.

    Args:
        image (`np.ndarray`):
            Input image with shape (num_channels, height, width).
        num_tiles_height (`int`):
            Number of tiles to split the image into along its height.
        num_tiles_width (`int`):
            Number of tiles to split the image into along its width.

    Returns:
        `np.ndarray`:
            Array of image tiles with shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width).
    """
    num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)

    # Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    image = image.transpose(1, 3, 0, 2, 4)

    # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    image = image.reshape(num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

    return np.ascontiguousarray(image)


class MllamaImageProcessor(BaseImageProcessor):
    """
    Constructs a MllamaImageProcessor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image.
        size (`int`, *optional*, defaults to `self.size`):
            Size of the image tile. The height and width values should be equal.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to 0.0):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch.
        max_image_tiles (`int`, *optional*, defaults to 4):
            The maximum number of tiles to split the image into.
        max_num_images (`int`, *optional*, defaults to 1):
            The maximum number of images the conversations contained.
    """

    def __init__(
            self,
            do_resize: bool = True,
            size: int = None,
            do_rescale: bool = True,
            rescale_factor: float = 1 / 255,
            do_normalize: bool = True,
            image_mean: List[float] = None,
            image_std: List[float] = None,
            do_pad: bool = True,
            max_image_tiles: int = 4,
            max_num_images: int = 1,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else 560
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.max_image_tiles = max_image_tiles
        self.max_num_images = max_num_images

    # pylint: disable=W0221
    def preprocess(self,
                   images_input: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]],
                   input_data_format=None):

        images_list = make_list_of_images(images_input)

        images_list = [[to_numpy_array(img) for img in bz_images] for bz_images in images_list]

        batch_images = []
        batch_aspect_ratios = []

        for images in images_list:
            sample_images = []
            sample_aspect_ratios = []

            for image in images:
                data_format = ChannelDimension.FIRST
                image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

                image, aspect_ratio = self.resize(
                    image=image,
                    size=self.size,
                    max_image_tiles=self.max_image_tiles,
                    input_data_format=data_format,
                    data_format=data_format
                )

                image = self.pad(
                    image=image,
                    size=self.size,
                    aspect_ratio=aspect_ratio,
                    input_data_format=data_format,
                    data_format=data_format
                )

                if self.do_rescale:
                    image = self.rescale(
                        image=image,
                        scale=self.rescale_factor,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                if self.do_normalize:
                    image = self.normalize(
                        image=image,
                        mean=self.image_mean,
                        std=self.image_std,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                num_tiles_height, num_tiles_width = aspect_ratio
                image = split_to_tiles(image, num_tiles_height, num_tiles_width)

                sample_images.append(image)
                sample_aspect_ratios.append((num_tiles_height, num_tiles_width))
            batch_images.append(sample_images)
            batch_aspect_ratios.append(sample_aspect_ratios)

        images, num_tiles = pack_images(batch_images, self.max_image_tiles, self.max_num_images)
        aspect_ratio_ids = convert_aspect_ratios_to_ids(batch_aspect_ratios, self.max_image_tiles, self.max_num_images)
        aspect_ratio_mask = build_aspect_ratio_mask(batch_aspect_ratios, self.max_image_tiles, self.max_num_images)

        return images, aspect_ratio_ids, aspect_ratio_mask, num_tiles

    def resize(
            self,
            image: np.ndarray,
            size: int,
            max_image_tiles: int,
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Union[np.ndarray, Tuple[int, int]]:
        """
        Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
        The optimal canvas size is calculated based on the maximum number of tiles and the tile size.

        The function first determines the best tile arrangement for the image, then resizes the image
        to fit within this canvas. The resized image and the number of tiles along the height and width
        dimensions are returned.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`int`):
                Size of the output image.
            max_image_tiles (`int`):
                The maximum number of tiles to split the image into.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `Union[np.ndarray, Tuple[int, int]]`: The resized image and a tuple containing the number of tiles
            along the height and width dimensions.
        """
        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        tile_size = size
        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=max_image_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size

        new_height, new_width = get_image_size_fit_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

        image = resize(
            image,
            (new_height, new_width),
            PILIMAGERESAMPLING.BILINEAR,
            data_format=data_format,
            input_data_format=input_data_format
        )

        return image, (num_tiles_height, num_tiles_width)

    def pad(
            self,
            image: np.ndarray,
            size: int,
            aspect_ratio: Tuple[int, int],
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to the `size` x `aspect_ratio`.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`int`):
                Size of the output image.
            aspect_ratio (`Tuple[int, int]`):
                The aspect ratio of the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The padded image.
        """

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        num_tiles_height, num_tiles_width = aspect_ratio
        padded_height = num_tiles_height * size
        padded_width = num_tiles_width * size
        pad_size = ((0, padded_height - image_height), (0, padded_width - image_width))

        image = pad(
            image,
            pad_size,
            mode=PaddingMode.CONSTANT,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image
