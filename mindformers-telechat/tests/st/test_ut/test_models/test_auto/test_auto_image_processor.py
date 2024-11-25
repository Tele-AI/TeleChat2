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
""" test AutoImageProcessor."""
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
from pathlib import Path
import tempfile
import pytest

from mindformers.models.auto.image_processing_auto import AutoImageProcessor
from mindformers.models.blip2.blip2_processor import Blip2ImageProcessor

TEST_REPO1 = 'mindformersinfra/test_blip2'
TEST_REPO2 = 'mindformersinfra/test_no_preprocessor_json'
TEST_REPO3 = 'mindformersinfra/test_image_processor'
TEST_REPO4 = 'mindformersinfra/test_image_processor_dynamic'
LOCAL_DIR = 'test_dir'


class TestAutoImageProcessor:
    """A test class for testing AutoImageProcessor."""

    def test_from_pretrained(self):
        """test init the image processor with from_pretrained."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # test load from_repo
            cache_dir = Path(tmp_dir) / 'cache'
            image_processor = AutoImageProcessor.from_pretrained(TEST_REPO1, cache_dir=cache_dir)
            assert isinstance(image_processor, Blip2ImageProcessor)

            # test load from local dir
            save_dir = Path(tmp_dir) / LOCAL_DIR
            image_processor.save_pretrained(save_directory=save_dir)
            image_processor_local = AutoImageProcessor.from_pretrained(save_dir)
            assert isinstance(image_processor_local, Blip2ImageProcessor)

    def test_missing_preprocessing_json(self):
        """test init the image processor with from_pretrained when missing preprocessing.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / 'cache'
            with pytest.raises(Exception):
                _ = AutoImageProcessor.from_pretrained(TEST_REPO2, cache_dir=cache_dir)

    def test_instantiate_from_config_json(self):
        """test init the image processor with from_pretrained from config.json."""
        # if missing image_processor_type in preprocessor_config.json, will to load from config.json
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / 'cache'
            image_processor = AutoImageProcessor.from_pretrained(TEST_REPO3, cache_dir=cache_dir)
            assert isinstance(image_processor, Blip2ImageProcessor)

    def test_load_by_plugin_mode(self):
        """test load an image processor from repo in plugin mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / 'cache'
            image_processor = AutoImageProcessor.from_pretrained(
                TEST_REPO4,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            assert image_processor.__class__.__name__ == 'TestImageProcessor'
