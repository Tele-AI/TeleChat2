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
Test module for testing saving ttp_checkpoint.
How to run this:
pytest tests/st/test_ttp_save_ckpt/test_parallel_save_ttp_ckpt.py
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
        Description: Test parallel save_ttp_ckpt for train.
        Expectation: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"bash {sh_path}/msrun_launch.sh 8")
        assert ret == 0
        path = "./logs/ttp_log.log"
        assert os.path.exists(path)
        os.remove(path)
        path2 = "./msrun_log/worker_0.log"
        flag = False
        assert os.path.exists(path2)
        with open(path2, 'r') as file:
            while True:
                content = file.read()
                if "Training Over!" in content:
                    if "MindIO TFT" in content and "starting heartbeat thread" in content:
                        flag = True
                        assert flag
                        break
