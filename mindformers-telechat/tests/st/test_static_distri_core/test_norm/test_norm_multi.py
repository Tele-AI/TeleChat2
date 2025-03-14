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
"""Test Normalization"""
import os
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))

base_command = ('msrun --worker_num={device_num} --local_worker_num={device_num} '
                '--master_port=61371 --log_dir=msrun_log --join=True --cluster_time_out=300 '
                'run_norm.py --dp {dp} --cp {cp} --tp {tp}')

def build_msrun_command(device_num, dp, cp, tp):
    command = base_command.format(device_num=device_num, dp=dp, cp=cp, tp=tp)
    return command


class TestNormalizationMulti:
    """A test class for testing LayerNorm/FusedLayerNorm/RMSNorm/FusedRMSNorm."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_run_norm_multi(self):
        """
        Feature: get_norm()
        Description: Test get_norm on four cards
        Expectation: AssertionError
        """
        device_num = 4
        dp, cp, tp = (2, 2, 1)
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(build_msrun_command(device_num, dp, cp, tp))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0
