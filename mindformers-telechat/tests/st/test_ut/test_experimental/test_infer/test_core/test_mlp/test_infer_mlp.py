# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test mlp in infer mode"""
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_infer_mlp():
    """
    Feature: MLP for prediction
    Description: Test MLP for prediction
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ['MS_ENABLE_LCCL'] = "off"
    ret = os.system(f"python {sh_path}/run_infer_mlp.py")
    assert ret == 0
