# Copyright 2025 Huawei Technologies Co., Ltd
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
pytest tests/st/test_multi_modal/test_llava_next/test_multi_modal_processor.py
"""
import numpy as np
import pytest

from research.llava_next.llava_next_multi_modal_processor import ClipImageProcessorV2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestMultiModalProcessor:
    """A test class for testing  image processor."""

    def test_image_processor(self):
        """
        Feature: Resize for image transforms
        Description: Test resize of input image size.
        Expectation: AssertionError
        """

        self.size = {"shortest_edge": 4}
        self.crop_size = {"height": 5, "width": 4}
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.image_grid_pinpoints = [[6, 12], [12, 6], [12, 12], [24, 6], [6, 24]]
        clip_grid_image_processor = ClipImageProcessorV2(size=self.size, crop_size=self.crop_size,
                                                         image_grid_pinpoints=self.image_grid_pinpoints,
                                                         do_pad=True, image_mean=self.image_mean,
                                                         image_std=self.image_std)
        np.random.seed(0)
        image = np.random.randint(0, 255, (100, 200, 3))
        res = clip_grid_image_processor.preprocess(image)
        pixel_values = res.get("pixel_values").asnumpy()
        expect_pixel_values = [
            [[[0.10553299, 0.07633615, 0.01794233, 0.07633615], [0.09093457, 0.01794233, 0.07633615, 0.0617376],
              [0.0617376, 0.09093457, 0.09093457, 0.04713918], [0.10553299, 0.07633615, 0.09093457, 0.04713918]],
             [[0.15388943, 0.16889732, 0.16889732, 0.13888167], [0.18390508, 0.13888167, 0.15388943, 0.18390508],
              [0.16889732, 0.21392061, 0.1238739, 0.15388943], [0.24393615, 0.19891284, 0.16889732, 0.1238739]],
             [[0.3257286, 0.3257286, 0.28306842, 0.28306842], [0.35416883, 0.33994877, 0.28306842, 0.36838892],
              [0.3257286, 0.38260898, 0.33994877, 0.26884836], [0.31150854, 0.38260898, 0.39682904, 0.26884836]]]]
        np.testing.assert_allclose(expect_pixel_values, pixel_values, rtol=1e-5)
