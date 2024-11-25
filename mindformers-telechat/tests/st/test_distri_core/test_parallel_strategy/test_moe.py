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
"""Test MoE"""
import os

import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestMoEParallelTransform:
    """A test class for testing MoE parallel transformation."""
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip")
    def test_moe_pynative_dp2_src(self):
        """
        Feature: test_moe_pynative
        Description: run pynative mode moe to generate pynative loss
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_moe.py"
        device_num = 2

        rm_list = ["npy_pynative_dp2*", "msrun_log_pynative_dp2*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3721 "+\
                    f"--log_dir=msrun_log_pynative_dp2_src "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"

    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip")
    def test_moe_pynative_dp4_dst(self):
        """
        Feature: test_moe_pynative
        Description: run pynative mode moe to generate pynative loss
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_moe.py"
        device_num = 4

        rm_list = ["npy_pynative_dp2*", "msrun_log_pynative_dp2*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --dp=4 --ep=4 "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3721 "+\
                    f"--log_dir=msrun_log_pynative_dp4_dst "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"
