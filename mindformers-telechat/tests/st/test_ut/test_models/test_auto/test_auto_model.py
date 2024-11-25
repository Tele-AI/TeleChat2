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
"""test AutoModel."""
import shutil
import os
os.environ["OPENMIND_HUB_ENDPOINT"] = "https://openmind.test.osinfra.cn"
# pylint: disable=C0413
import pytest
import numpy as np

from mindspore import nn

from mindformers import MindFormerConfig
from mindformers.models.gpt2 import GPT2Config
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Model

REPO_ID = "mindformersinfra/test_gpt2_layer2"
MODEL_NAME = "gpt2"
YAML_NAME = "mindspore_model.yaml"
CONFIG_NAME = "config.json"
DEFAULT_MODEL_NAME = "mindspore_model.ckpt"

LOCAL_REPO = "./gpt2_local_repo_ut"
LOCAL_DIR = "./gpt2_local_dir_ut"


def get_state_dict(model):
    """get state dict"""
    if isinstance(model, dict):
        return model
    if isinstance(model, nn.Cell):
        state_dict = {}
        for item in model.get_parameters():
            state_dict[item.name] = item.data
        return state_dict
    raise ValueError("model must be dict or nn.Cell")


def compare_state_dict(model_1, model_2):
    """compare state dict"""
    state_dict_1 = get_state_dict(model_1)
    state_dict_2 = get_state_dict(model_2)

    assert state_dict_1.keys() == state_dict_2.keys()
    for key in state_dict_1:
        value1 = state_dict_1[key].asnumpy()
        value2 = state_dict_2[key].asnumpy()
        assert np.array_equal(value1, value2), f"The value of {key} is not match!"


class TestAutoModel:
    """A test class for testing AutoModel."""

    def test_automodel_load_from_model_name(self):
        """test init AutoModel from model name."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.save_pretrained(LOCAL_DIR)

        file_names = os.listdir(LOCAL_DIR)
        assert YAML_NAME in file_names, f"{YAML_NAME} not found!"
        assert DEFAULT_MODEL_NAME in file_names, f"{DEFAULT_MODEL_NAME} not found!"
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_local_dir(self):
        """test init AutoModel from local dir contains yaml and ckpt."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)
        model_2 = AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_repo(self):
        """test init AutoModel from repo."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        model_1.save_pretrained(LOCAL_REPO, save_json=True)

        file_names = os.listdir(LOCAL_REPO)
        assert CONFIG_NAME in file_names, f"{CONFIG_NAME} not found!"
        assert DEFAULT_MODEL_NAME in file_names, f"{DEFAULT_MODEL_NAME} not found!"
        shutil.rmtree(LOCAL_REPO)

    def test_automodel_load_from_local_repo(self):
        """test init AutoModel from local repo."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)  # load from repo
        model_1.save_pretrained(LOCAL_REPO, save_json=True)
        model_2 = AutoModelForCausalLM.from_pretrained(LOCAL_REPO)  # load from local path

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_REPO)

    def test_automodel_load_from_local_dir_with_yaml_json_ckpt(self):
        """test init AutoModel from local dir which contains yaml, ckpt and json."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        model_1.save_pretrained(LOCAL_DIR, save_json=True)
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)
        model_2 = AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_yaml(self):
        """test init AutoModel from yaml."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)

        yaml_path = os.path.join(LOCAL_DIR, YAML_NAME)
        model_2 = AutoModelForCausalLM.from_config(yaml_path)

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_mindformer_config(self):
        """test init AutoModel from MindFormerConfig."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)

        yaml_path = os.path.join(LOCAL_DIR, YAML_NAME)
        config = MindFormerConfig(yaml_path)
        model_2 = AutoModelForCausalLM.from_config(config)

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_autoconfig_from_repo(self):
        """test init AutoModel from AutoConfig loaded with repo id."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        config = AutoConfig.from_pretrained(REPO_ID)
        model_2 = AutoModelForCausalLM.from_config(config)

        compare_state_dict(model_1, model_2)

    def test_automodel_load_from_autoconfig_from_model_name(self):
        """test init AutoModel from AutoConfig loaded with model name."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model_2 = AutoModelForCausalLM.from_config(config)

        compare_state_dict(model_1, model_2)

    def test_automodel_load_from_pretrained_config(self):
        """test init AutoModel from PretrainedConfig."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        model_1.save_pretrained(LOCAL_REPO, save_json=True)

        ckpt = os.path.join(LOCAL_REPO, DEFAULT_MODEL_NAME)
        config = GPT2Config(checkpoint_name_or_path=ckpt, num_layers=2)
        model_2 = AutoModelForCausalLM.from_config(config)

        compare_state_dict(model_1, model_2)
        shutil.rmtree(LOCAL_REPO)

    def test_automodel_and_automodel_for_causallm_from_pretrained(self):
        """test init AutoModel and AutoModelForCausalLM from pretrained."""
        model_1 = AutoModel.from_pretrained(REPO_ID)
        model_2 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        assert isinstance(model_1, GPT2Model), "AutoModel only supports base model."
        assert isinstance(model_2, GPT2LMHeadModel), "AutoModelForCausalLM only supports model for causal."

    def test_automodel_and_automodel_for_causallm_from_config(self):
        """test init AutoModel and AutoModelForCausalLM from config."""
        config = GPT2Config()
        model_1 = AutoModel.from_config(config)
        model_2 = AutoModelForCausalLM.from_config(config)
        assert isinstance(model_1, GPT2Model), "AutoModel only supports base model."
        assert isinstance(model_2, GPT2LMHeadModel), "AutoModelForCausalLM only supports model for causal."

    def test_automodel_load_from_error_model_name(self):
        """test init AutoModel from error model name."""
        with pytest.raises(ValueError):
            AutoModelForCausalLM.from_pretrained("xxxx")

    def test_automodel_load_from_error_model_name_prefix_mindspore(self):
        """test init AutoModel from error prefix name."""
        with pytest.raises(ValueError):
            AutoModelForCausalLM.from_pretrained("mindspore/xxxx")

    def test_automodel_load_from_dir_or_repo_not_exist(self):
        """test init automodel from dir or repo not exist."""
        with pytest.raises(RuntimeError):
            AutoModelForCausalLM.from_pretrained("xxxx/xxxx")

    def test_pretrained_model_load_from_local_dir_empty(self):
        """test inti AutoModel from empty local dir."""
        os.makedirs(LOCAL_DIR)
        with pytest.raises(FileNotFoundError):
            AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_local_dir_only_ckpt(self):
        """test init AutoModel from dir only contains ckpt."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)
        os.remove(os.path.join(LOCAL_DIR, YAML_NAME))
        with pytest.raises(FileNotFoundError):
            AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_local_dir_only_yaml(self):
        """test init AutoModel from dir only contains yaml."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)
        os.remove(os.path.join(LOCAL_DIR, DEFAULT_MODEL_NAME))
        with pytest.raises(FileNotFoundError):
            AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_local_dir_only_json(self):
        """test init AutoModel from dir only contains json."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        model_1.save_pretrained(LOCAL_REPO, save_json=True)
        os.remove(os.path.join(LOCAL_REPO, DEFAULT_MODEL_NAME))
        with pytest.raises(EnvironmentError):
            AutoModelForCausalLM.from_pretrained(LOCAL_REPO)  # load from local path
        shutil.rmtree(LOCAL_REPO)

    def test_automodel_load_from_local_dir_only_yaml_and_json(self):
        """test init AutoModel from dir only contains yaml and json."""
        model_1 = AutoModelForCausalLM.from_pretrained(REPO_ID)
        model_1.save_pretrained(LOCAL_DIR, save_json=True)
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)
        os.remove(os.path.join(LOCAL_DIR, DEFAULT_MODEL_NAME))
        with pytest.raises(FileNotFoundError):
            AutoModelForCausalLM.from_pretrained(LOCAL_DIR)  # load from local path
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_yaml_not_exist(self):
        """test init AutoModel from yaml not exist."""
        with pytest.raises(ValueError):
            AutoModelForCausalLM.from_config("xxxx.yaml")

    def test_automodel_load_from_mindformer_config_without_model(self):
        """test init AutoModel from MindFormerConfig without model."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)

        yaml_path = os.path.join(LOCAL_DIR, YAML_NAME)
        config = MindFormerConfig(yaml_path)
        config.model = None
        with pytest.raises(AttributeError):
            AutoModelForCausalLM.from_config(config)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_mindformer_config_without_model_config(self):
        """test init AutoModel from MindFormerConfig without model config."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)

        yaml_path = os.path.join(LOCAL_DIR, YAML_NAME)
        config = MindFormerConfig(yaml_path)
        config.model.model_config = None
        with pytest.raises(AttributeError):
            AutoModelForCausalLM.from_config(config)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_mindformer_config_without_model_arch(self):
        """test init AutoModel from MindFormerConfig without model arch."""
        model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_1.save_pretrained(LOCAL_DIR)

        yaml_path = os.path.join(LOCAL_DIR, YAML_NAME)
        config = MindFormerConfig(yaml_path)
        config.model.arch = None
        with pytest.raises(TypeError):
            AutoModelForCausalLM.from_config(config)
        shutil.rmtree(LOCAL_DIR)

    def test_automodel_load_from_lora_config(self):
        """test init AutoModel from lora config."""
        from mindformers.pet.pet_config import LoraConfig
        from mindformers.pet.models.lora import LoraModel
        config = GPT2Config()
        pet_config = LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.05, target_modules='.*dense1.*|.*dense3.*')
        config.pet_config = pet_config
        model = AutoModel.from_config(config)
        assert isinstance(model, LoraModel)
