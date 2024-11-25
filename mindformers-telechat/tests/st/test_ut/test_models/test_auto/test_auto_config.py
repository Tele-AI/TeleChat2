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
"""test AutoConfig."""
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
import tempfile

from mindformers import LlamaConfig
from mindformers.models.auto import AutoConfig
from mindformers.models import GPT2Config

REPO_ID = "mindformersinfra/test_auto_config_ms"
NUM_LAYERS = 4

DYNAMIC_REPO_ID = "mindformersinfra/test_dynamic_config"
DYNAMIC_CLASS_NAME = "Baichuan2Config"

MODEL_TYPE = "gpt2"


class TestAutoConfig:
    """A test class for testing AutoConfig."""

    def setup_method(self):
        """init test class."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name
        self.config_path = os.path.join(self.path, "config.json")

    def test_autoconfig_from_repo(self):
        """test init AutoConfig from usr_name/repo_name."""
        config = AutoConfig.from_pretrained(REPO_ID)
        assert config.num_layers == NUM_LAYERS
        assert isinstance(config, GPT2Config)

    def test_autoconfig_with_saved_config(self):
        """test save_pretrained method and init AutoConfig from saved config."""
        config = GPT2Config(num_layers=NUM_LAYERS)
        config.save_pretrained(self.path, save_json=True)

        # load config form saved config dir
        config = AutoConfig.from_pretrained(self.path)
        assert config.num_layers == NUM_LAYERS
        assert isinstance(config, GPT2Config)

        # load config form saved config file path
        config = AutoConfig.from_pretrained(self.config_path)
        assert config.num_layers == NUM_LAYERS
        assert isinstance(config, GPT2Config)

    def test_autoconfig_from_dynamic_repo(self):
        """test init AutoConfig from dynamic repo."""
        config = AutoConfig.from_pretrained(DYNAMIC_REPO_ID, trust_remote_code=True)
        assert config.__class__.__name__ == DYNAMIC_CLASS_NAME

    def test_autoconfig_from_model_name(self):
        """test init AutoConfig from model name."""
        config = AutoConfig.from_pretrained("llama_7b")
        assert isinstance(config, LlamaConfig)

    def test_autoconfig_from_yaml(self):
        """test init AutoConfig from yaml."""
        config = AutoConfig.from_pretrained("configs/llama/run_llama_7b.yaml")
        assert isinstance(config, LlamaConfig)

    def test_autoconfig_for_model(self):
        """test init AutoConfig for model."""
        config = AutoConfig.for_model(MODEL_TYPE)
        assert isinstance(config, GPT2Config)
