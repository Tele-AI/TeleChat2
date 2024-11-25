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
Test module for testing resume training from specified checkpoint.
How to run this:
pytest tests/st/test_resume/test_parallel_resume.py
"""
import os
import pytest

from tests.utils.resume_train import extract_loss_values


class TestResumeTraining:
    """A test class for testing pipeline."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel trainer for train.
        Expectation: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"bash {sh_path}/msrun_launch.sh 8")
        os.system("grep -E 'ERROR|error' {sh_path}/msrun_log/worker_7.log -C 3")

        assert ret == 0
        loss = extract_loss_values("msrun_log/worker_7.log")
        assert abs(loss[4] - loss[8]) < 0.005
        assert abs(loss[5] - loss[9]) < 0.005
        assert abs(loss[6] - loss[10]) < 0.005
        assert abs(loss[7] - loss[11]) < 0.005

        assert abs(loss[12] - loss[10]) < 0.005
        assert abs(loss[13] - loss[11]) < 0.005

        assert abs(loss[14] - loss[12]) < 0.005
