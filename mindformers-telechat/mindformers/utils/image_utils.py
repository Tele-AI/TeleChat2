# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Image utils."""
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class ChannelDimension(ExplicitEnum):
    """
    Enum for channel dimensions.
    """
    FIRST = "channels_first"
    LAST = "channels_last"


def infer_channel_dimension_format(
        image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    if image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    if channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    raise ValueError(f"Unsupported data format: {channel_dim}")
