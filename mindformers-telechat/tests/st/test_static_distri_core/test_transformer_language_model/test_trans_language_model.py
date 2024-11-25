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
"""test TransformerLanguageModel"""
import os
import pytest


class TestTransformerLanguageModel:
    """A test class for testing TransformerLanguageModel"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_trans_language_model_single(self):
        """
        Feature: TransformerLanguageModel
        Description: Test TransformerLanguageModel
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        dp, cp, tp = (1, 1, 1)
        ret = os.system(f"python {sh_path}/run_trans_language_model.py --dp {dp} --cp {cp} --tp {tp}")
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_trans_language_model_parallel(self):
        """
        Feature: TransformerLanguageModel
        Description: Test TransformerLanguageModel
        Exception: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        device_num = 8
        dp, cp, tp = (2, 2, 2)
        ret = os.system(f"bash {sh_path}/msrun_launch.sh {device_num} {dp} {cp} {tp}")
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0
