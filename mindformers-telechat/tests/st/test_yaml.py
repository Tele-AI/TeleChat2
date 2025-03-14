# Copyright 2022 Huawei Technologies Co., Ltd
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
Test module for testing yaml tools, MindFormerConfig.

How to run this:
windows:  pytest .\\tests\\st\\test_yaml.py
linux:  pytest ./tests/st/test_yaml.py

Note:
    the name of model yaml file should start with model.
    the name of model ckpt file should be same with yaml file.
example:
    clip_vit_b_32.yaml starts with clip,
    and clip_vit_b_32.ckpt is model ckpt file.
"""
import os
import pytest
from mindformers import MindFormerBook
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.logger import logger


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training  # add for ci
@pytest.mark.env_onecard
def test_yaml():
    """
    Feature: MindFormerConfig
    Description: Test to transform yaml file as MindFormerConfig
    Expectation: TypeError
    """
    yaml_path = os.path.join(
        MindFormerBook.get_project_path(), 'configs', 'llama2', 'predict_llama2_7b.yaml')
    yaml_content = MindFormerConfig(yaml_path)

    logger.info(yaml_content)
    assert isinstance(yaml_content, MindFormerConfig)
    assert isinstance(yaml_content, dict)
