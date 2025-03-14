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
Test module for testing the image processor used for MindFormers.
How to run this:
pytest tests/st/test_multi_modal/test_mllama/test_image_processor.py
"""
import numpy as np
import pytest

from mindformers.models.mllama.image_processing_mllama import MllamaImageProcessor


class TestImageProcessor:
    """A test class for testing image processor."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_processor(self):
        """
        Feature: image processors
        Description: Test processor of MllamaImageProcessor.
        Expectation: AssertionError
        """
        np.random.seed(0)
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.max_image_tiles = 2
        self.image_size = 3
        self.image_processor = MllamaImageProcessor(
            size=self.image_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
            max_image_tiles=self.max_image_tiles,
        )
        image = np.random.randint(1, 255, size=(2, 2, 3)).astype(np.int32)

        pixel_values, _, _, _ = self.image_processor(image)

        expect_pixel_values = np.zeros((1, 1, 2, 3, 3, 3)).astype(np.float32)
        expect_pixel_values[0][0][0] = [[[0.7332653, 0.8792495, 1.0252337],
                                         [0.9084464, 1.0398322, 1.171218],
                                         [1.0690291, 1.1858164, 1.3026038]],
                                        [[-1.0317243, -0.8816466, -0.7315688],
                                         [-0.6115067, -0.83662325, -1.0767475],
                                         [-0.1912892, -0.8066077, -1.4219263]],
                                        [[0.19774802, 1.1504925, 2.103237],
                                         [-0.57013553, 0.7381106, 2.0463567],
                                         [-1.3380191, 0.3257286, 1.9752562]]]

        np.testing.assert_array_almost_equal(pixel_values, expect_pixel_values)
