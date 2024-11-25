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
"""test base model."""
import os
import tempfile

from mindformers import MindFormerConfig
from mindformers.models.llama import LlamaConfig, LlamaModel
from mindformers.models.blip2 import Blip2Config, Blip2Llm
from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.models.vit.vit_config import ViTConfig

NUM_LAYERS = 1


class TestBaseModel:
    """A test class for testing model.save_pretrained() method."""

    def setup_method(self):
        """init test class."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def test_llama(self):
        """test llama save pretrained."""
        config = LlamaConfig(num_layers=NUM_LAYERS)
        model = LlamaModel(config)
        model.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        model_path = self.path + "/" + "mindspore_model.ckpt"
        assert os.path.exists(yaml_path)
        assert os.path.exists(model_path)

        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.model_config.num_layers == NUM_LAYERS

    def test_blip2(self):
        """test blip2 save pretrained."""
        vit_config = ViTConfig(num_hidden_layers=1)
        llama_config = LlamaConfig(num_layers=1)
        qformer_config = QFormerConfig(num_hidden_layers=1)
        config = Blip2Config(
            vision_config=vit_config,
            qformer_config=qformer_config,
            text_config=llama_config
        )
        model = Blip2Llm(config)
        model.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        model_path = self.path + "/" + "mindspore_model.ckpt"
        assert os.path.exists(yaml_path)
        assert os.path.exists(model_path)

        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.arch.type == "Blip2Llm"
        assert mf_config.model.model_config.type == "Blip2Config"
        assert mf_config.model.model_config.vision_config.type == "ViTConfig"
        assert mf_config.model.model_config.text_config.type == "LlamaConfig"
        assert mf_config.model.model_config.qformer_config.vocab_size == 30523
