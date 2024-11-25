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
"""test config utils."""
import tempfile
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
import pytest

from mindformers import LlamaConfig, MindFormerConfig, Blip2Config, ViTConfig
from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.models.utils import CONFIG_NAME

RMS_NORM_EPS = 1.0e-6
BATCH_SIZE = 4
REPO_ID = "mindformersinfra/test_llama"


class TestLlamaConfig:
    """A test class for testing llama config utils."""

    def setup_method(self):
        """init test class."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def test_build_llama_config_from_yaml(self):
        """test build llama config from yaml."""
        config = LlamaConfig.from_pretrained("configs/llama/run_llama_7b.yaml")
        assert isinstance(config, LlamaConfig)
        assert config.rms_norm_eps == RMS_NORM_EPS

    def test_build_llama_config_from_name(self):
        """test build llama config from name."""
        config = LlamaConfig.from_pretrained("llama_7b")
        assert isinstance(config, LlamaConfig)
        assert config.rms_norm_eps == RMS_NORM_EPS
        assert os.path.exists("./checkpoint_download/llama/llama_7b.yaml")

    def test_build_llama_config_from_wrong_name(self):
        """test build llama config from wrong name."""
        with pytest.raises(ValueError):
            _ = LlamaConfig.from_pretrained("xxxx")

        with pytest.raises(ValueError):
            _ = LlamaConfig.from_pretrained("mindspore/xxxx")

    def test_build_llama_config_from_repo(self):
        """test build llama config from repo."""
        config = LlamaConfig.from_pretrained(REPO_ID)
        assert isinstance(config, LlamaConfig)
        assert config.rms_norm_eps == RMS_NORM_EPS

    def test_build_llama_config_from_wrong_repo(self):
        """test build llama config from wrong repo."""
        with pytest.raises(RuntimeError):
            _ = LlamaConfig.from_pretrained("repo/xxxx")

    def test_build_llama_config_from_dir_or_json(self):
        """test build llama config from dir or json."""
        config = LlamaConfig(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_json=True)

        new_config = LlamaConfig.from_pretrained(self.path)
        assert isinstance(new_config, LlamaConfig)
        assert new_config.batch_size == BATCH_SIZE

        json_path = self.path + "/" + CONFIG_NAME
        new_config = LlamaConfig.from_pretrained(json_path)
        assert isinstance(new_config, LlamaConfig)
        assert new_config.batch_size == BATCH_SIZE

    def test_llama_save_pretrained(self):
        """test llama config save pretrained."""
        config = LlamaConfig(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        assert os.path.exists(yaml_path)
        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.model_config.batch_size == BATCH_SIZE


class TestBlip2Config:
    """A test class for testing blip2 config utils."""

    def setup_method(self):
        """init test class."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def test_build_blip2_config_from_yaml(self):
        """test build blip2 config from yaml."""
        config = Blip2Config.from_pretrained("configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml")
        assert isinstance(config, Blip2Config)
        assert isinstance(config.vision_config, ViTConfig)
        assert isinstance(config.qformer_config, QFormerConfig)
        assert isinstance(config.text_config, LlamaConfig)

    def test_build_blip2_config_from_name(self):
        """test build blip2 config from name."""
        config = Blip2Config.from_pretrained("blip2_stage2_vit_g_llama_7b")
        assert isinstance(config, Blip2Config)
        assert isinstance(config.vision_config, ViTConfig)
        assert isinstance(config.qformer_config, QFormerConfig)
        assert isinstance(config.text_config, LlamaConfig)

    def test_blip2_save_pretrained(self):
        """test blip2 config save pretrained."""
        config = Blip2Config.from_pretrained("blip2_stage2_vit_g_llama_7b")
        config.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        assert os.path.exists(yaml_path)
        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.arch.type == "Blip2Llm"
        assert mf_config.model.model_config.type == "Blip2Config"
        assert mf_config.model.model_config.vision_config.type == "ViTConfig"
        assert mf_config.model.model_config.text_config.type == "LlamaConfig"
        assert mf_config.model.model_config.qformer_config.vocab_size == 44728
