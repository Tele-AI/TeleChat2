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
Test module for testing verifying llama configs content.
How to run this:
pytest tests/st/test_model/test_llama_model/test_config.py
"""
import os
import sys
# from glob import glob
from multiprocessing.pool import Pool

import mindspore as ms

from mindformers import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig

ms.set_context(mode=0, device_target='CPU')


def build_model(config):
    """build model from yaml"""
    model_config = MindFormerConfig(os.path.realpath(config))
    model_config = LlamaConfig(**model_config.model.model_config)
    model_config.num_layers = 2
    model_config.checkpoint_name_or_path = None
    try:
        _ = LlamaForCausalLM(model_config)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(e)
        logger.error(f"Create Model with {config} Failed. with type error or value error or runtime error.")
        raise AssertionError from e
    except Exception as e:
        logger.error(e)
        logger.error(f"Create Model with {config} Failed with unknown error.")
        raise AssertionError from e


def get_yaml_files(prefix):
    """get all yaml files for testing"""
    files = os.listdir(prefix)

    configs = list()
    remove_list = [
        "finetune_llama2_70b_bf16_32p.yaml",
        "pretrain_llama2_70b_bf16_32p.yaml",
        "pretrain_llama2_70b_bf16_32p.yaml",
        "finetune_llama2_7b_prefixtuning.yaml",
        "finetune_llama2_7b_ptuning2.yaml",
        "finetune_llama3_1_8b.yaml",
        "predict_llama3_1_8b.yaml"]
    for file in files:
        if file in remove_list or not file.endswith('.yaml'):
            continue
        configs.append(os.path.join(prefix, file))
    return configs


def test_configs():
    """
    Feature: Model configs
    Description: Test configs by instantiating model
    Expectation: No exception
    """
    configs = list()
    for path in sys.path:
        if path.endswith('/site-packages'):
            config_path = os.path.join(path, 'configs/llama2')
            configs = get_yaml_files(config_path)

            # config_path = os.path.join(path, 'configs/llama2/*.yaml')
            # configs += glob(config_path)
            if configs:
                break
    assert configs

    with Pool(20) as pl:
        _ = list(pl.imap(build_model, configs))
