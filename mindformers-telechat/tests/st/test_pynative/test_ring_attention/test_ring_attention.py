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
"""Test RingAttention"""
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestRingAttention:
    """A test class for testing ring attention."""
    @pytest.mark.run(order=1)
    def test_ring_attention(self):
        """
        Feature: test RingAttention pynative
        Description: accuracy test for RingAttention
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_ring_attention.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8118 " + \
                    f"--log_dir=msrun_log_ring_attention " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_ring_attention/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_ring_attention/worker_*.log"

    @pytest.mark.run(order=2)
    def test_flash_sp(self):
        """
        Feature: test FlashSP pynative
        Description: accuracy test for FlashSP
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_flash_sp.py"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8118 " + \
                    f"--log_dir=msrun_log_flash_sp " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_flash_sp/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_flash_sp/worker_*.log"
