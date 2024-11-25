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
"""test build config."""
from mindformers import MindFormerConfig
from mindformers.models.build_config import build_model_config
from mindformers.models.llama import LlamaConfig
from mindformers.models.blip2 import Blip2Config
from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.models.vit.vit_config import ViTConfig
from mindformers.models.sam import SamConfig


class TestBuildModelConfig:
    """A test class for testing build_model_config() method."""

    def test_build_llama_config(self):
        """test build llama config from yaml."""
        config = MindFormerConfig("configs/llama/run_llama_7b.yaml")
        model_config = build_model_config(config.model.model_config)
        assert isinstance(model_config, LlamaConfig)

    def test_build_blip2_config(self):
        """test build blip2 config from yaml."""
        config = MindFormerConfig("configs/blip2/run_blip2_stage2_vit_g_llama_7b_910b.yaml")
        model_config = build_model_config(config.model.model_config)
        assert isinstance(model_config, Blip2Config)
        assert isinstance(model_config.vision_config, ViTConfig)
        assert isinstance(model_config.qformer_config, QFormerConfig)
        assert isinstance(model_config.text_config, LlamaConfig)

    def test_build_sam_config(self):
        """test build sam config from yaml."""
        config = MindFormerConfig("configs/sam/run_sam_vit-b.yaml")
        model_config = build_model_config(config.model.model_config)
        assert isinstance(model_config, SamConfig)
        assert isinstance(model_config.image_encoder, MindFormerConfig)
        assert isinstance(model_config.prompt_config, MindFormerConfig)
        assert isinstance(model_config.decoder_config, MindFormerConfig)
