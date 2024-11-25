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
"""test image processor."""
import os
from pathlib import Path
import tempfile

from research.visualglm.visualglm_processor import VisualGLMImageProcessor
from mindformers.models import (
    CLIPImageProcessor,
    ViTMAEImageProcessor,
    SwinImageProcessor,
    SamImageProcessor,
    ViTImageProcessor
)
from mindformers.models.blip2 import Blip2ImageProcessor
from mindformers.tools.image_tools import load_image


TEST_IMAGE = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
DIFF_THRESHOLD = 0.1

VISUALGLM_REMOTE_PATH = 'mindformersinfra/test_visualglm'
BLIP2_REMOTE_PATH = 'mindformersinfra/test_blip2'
CLIP_REMOTE_PATH = 'mindformersinfra/test_clip'
MAE_REMOTE_PATH = 'mindformersinfra/test_mae'
SWIN_REMOTE_PATH = 'mindformersinfra/test_swin'
SAM_REMOTE_PATH = 'mindformersinfra/test_sam'
VIT_REMOTE_PATH = 'mindformersinfra/test_vit'

VISUALGLM_OUTPUT_SHAPE = (1, 3, 224, 224)
BLIP2_OUTPUT_SHAPE = (1, 3, 224, 224)
CLIP_OUTPUT_SHAPE = (1, 3, 224, 224)
MAE_OUTPUT_SHAPE = [(1, 3, 224, 224), (1, 196), (1, 196), (1, 49)]
SWIN_OUTPUT_SHAPE = (1, 3, 224, 224)
SAM_OUTPUT_1_SHAPE = (1, 3, 1024, 1024)
VIT_OUTPUT_SHAPE = (1, 3, 224, 224)

VISUALGLM_OUTPUT_SUM = -54167.4
BLIP2_OUTPUT_SUM = -54167.402
CLIP_OUTPUT_SUM = -40270.586
MAE_OUTPUT_SUM = [-64566.43, 147, 19110, 4720]
SWIN_OUTPUT_SUM = -26959.375
SAM_OUTPUT_1_SUM = -1019078.3
SAM_OUTPUT_2 = (773, 1024)
VIT_OUTPUT_SUM = -26959.375


def run_processor(
        name,
        processor,
        remote_path,
        output_shape,
        output_sum
):
    """run processor func"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # test init processor from repo
        cache_dir = Path(tmp_dir) / name
        save_dir = Path(tmp_dir) / 'save_dir'
        image_processor = processor.from_pretrained(remote_path, cache_dir=cache_dir)
        assert isinstance(image_processor, processor)

        # test save_pretrained
        image_processor.save_pretrained(save_directory=save_dir)
        assert os.path.exists(save_dir / 'preprocessor_config.json')

        # test init processor from local dir
        image_processor_local = processor.from_pretrained(save_dir)
        assert isinstance(image_processor_local, processor)

        # test preprocess image
        output = image_processor(TEST_IMAGE)
        if name == 'mae':
            for i, _ in enumerate(output):
                assert output[i].shape == output_shape[i]
            for i in range(len(output) - 1):
                assert abs(output[i].sum() - output_shape[i]) < DIFF_THRESHOLD
        elif name == 'sam':
            assert output[0].shape == output_shape[0]
            assert abs(output[0].sum() - output_sum) < DIFF_THRESHOLD
            assert output[1] == output_shape[1]
        else:
            assert output.shape == output_shape
            assert abs(output.sum() - output_sum) < DIFF_THRESHOLD


class TestImageProcessor:
    """A test class for testing image processor."""

    def test_visualglm_image_processor(self):
        """test visualglm image processor."""
        run_processor(
            name='visualglm',
            processor=VisualGLMImageProcessor,
            remote_path=VISUALGLM_REMOTE_PATH,
            output_shape=VISUALGLM_OUTPUT_SHAPE,
            output_sum=VISUALGLM_OUTPUT_SUM
        )

    def test_blip2_image_processor(self):
        """test blip2 image processor."""
        run_processor(
            name='blip2',
            processor=Blip2ImageProcessor,
            remote_path=BLIP2_REMOTE_PATH,
            output_shape=BLIP2_OUTPUT_SHAPE,
            output_sum=BLIP2_OUTPUT_SUM
        )

    def test_clip_image_processor(self):
        """test clip image processor."""
        run_processor(
            name='clip',
            processor=CLIPImageProcessor,
            remote_path=CLIP_REMOTE_PATH,
            output_shape=CLIP_OUTPUT_SHAPE,
            output_sum=CLIP_OUTPUT_SUM
        )

    def test_vit_image_processor(self):
        """test vit image processor."""
        run_processor(
            name='vit',
            processor=ViTImageProcessor,
            remote_path=VIT_REMOTE_PATH,
            output_shape=VIT_OUTPUT_SHAPE,
            output_sum=VIT_OUTPUT_SUM
        )

    def test_swin_image_processor(self):
        """test swin image processor."""
        run_processor(
            name='vit',
            processor=SwinImageProcessor,
            remote_path=SWIN_REMOTE_PATH,
            output_shape=SWIN_OUTPUT_SHAPE,
            output_sum=SWIN_OUTPUT_SUM
        )

    def test_mae_image_processor(self):
        """test mae image processor."""
        run_processor(
            name='mae',
            processor=ViTMAEImageProcessor,
            remote_path=MAE_REMOTE_PATH,
            output_shape=MAE_OUTPUT_SHAPE,
            output_sum=MAE_OUTPUT_SUM
        )

    def test_sam_image_processor(self):
        """test sam image processor."""
        run_processor(
            name='sam',
            processor=SamImageProcessor,
            remote_path=MAE_REMOTE_PATH,
            output_shape=[SAM_OUTPUT_1_SHAPE, SAM_OUTPUT_2],
            output_sum=MAE_OUTPUT_SUM
        )
