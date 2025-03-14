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
pytest tests/st/test_grace_exit_save_ckpt/test_parallel_grace_exit_save_ckpt.py
"""
import os
import pytest

class TestSaveTtpCkpt:
    """A test class for testing save_ttp_ckpt."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel grace_exit_save_ckpt for train.
        Expectation: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"bash {sh_path}/msrun_launch.sh 8")
        assert ret == 0
        checkpoint_dir = "./grace_ckpt"
        path = "./msrun_log/worker_0.log"
        flag = False
        assert os.path.exists(path)
        with open(path, 'r') as file:
            content = file.read()
            if "Graceful exit is triggered, stop training" in content:
                flag = True
                assert flag
        for _, _, filenames in os.walk(checkpoint_dir):
            for filename in filenames:
                assert filename.endswith('.ckpt')
        for _, _, filenames in os.walk(checkpoint_dir):
            for filename in filenames:
                if os.path.exists(filename):
                    os.remove(filename)
