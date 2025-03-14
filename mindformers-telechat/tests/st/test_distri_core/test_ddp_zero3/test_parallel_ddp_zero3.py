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
"""Test ParallelDDPZeRO3"""
import os

import pytest


class TestParallelDDPZeRO3:
    """A test class for testing DDPZeRO3."""
    #os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "4,5,6,7"
    os.environ['HCCL_DETERMINISTIC'] = "true"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_ddp_zero3_pynative(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode ddp zero3 in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ddp_zero3.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8199 "+\
                    f"--log_dir=msrun_log_pynative_ddp_zero3 "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_ddp_zero3/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_ddp_zero3/worker_*.log"
