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
Test module for testing the image transform used for MindFormers.
How to run this:
pytest tests/st/test_multi_modal/test_image_transform.py
"""
import numpy as np
import pytest

from mindformers.utils.image_transforms import resize, PILIMAGERESAMPLING, \
    get_resize_output_image_size, ChannelDimension, pad
from mindformers.utils.image_utils import PaddingMode


@pytest.mark.level1
class TestImageTransform:
    """A test class for testing image transform."""

    def test_resize(self):
        """
        Feature: Resize for image transforms
        Description: Test resize of input image size.
        Expectation: AssertionError
        """
        image_size = np.random.randint(1, 255, size=(1208, 684, 3))
        image_size_last = resize(image_size,
                                 (336, 336),
                                 PILIMAGERESAMPLING.BICUBIC,
                                 ChannelDimension.LAST,
                                 ChannelDimension.LAST)
        image_size_first = resize(image_size,
                                  (336, 336),
                                  PILIMAGERESAMPLING.BICUBIC,
                                  ChannelDimension.FIRST,
                                  ChannelDimension.LAST)
        assert image_size_first.shape == (3, 336, 336)
        assert image_size_last.shape == (336, 336, 3)

    def test_get_resize_output_image_size(self):
        """
        Feature: Resize for image transforms
        Description: Test resize of input image size.
        Expectation: AssertionError
        """
        image = np.random.randint(1, 255, size=(1208, 668, 3))
        image_size = get_resize_output_image_size(image,
                                                  336,
                                                  True,
                                                  None,
                                                  ChannelDimension.LAST)
        image_size_max = get_resize_output_image_size(image,
                                                      336,
                                                      False,
                                                      458,
                                                      ChannelDimension.LAST)
        assert image_size == (336, 336)
        assert image_size_max == (458, 253)

    def test_pad(self):
        """
        Feature: padding for image transforms
        Description: Test padding of input image size.
        Expectation: AssertionError
        """
        image = np.random.randint(1, 255, size=(1208, 668, 3))
        paded_image = pad(image, (10, 20), PaddingMode.CONSTANT, ChannelDimension.LAST, ChannelDimension.LAST)
        assert paded_image.shape == (1238, 698, 3)
