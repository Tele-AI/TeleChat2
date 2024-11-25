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
"""Test ParallelMLP"""
import os

import pytest

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestAllToAll:
    """A test class for testing AllToAll."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "2,3"
    @pytest.mark.run(order=0)
    def test_alltoall_forward(self):
        """
        Feature: test_alltoall_forward
        Description: test_alltoall_forward
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_alltoall.py"
        device_num = 2

        rm_list = ["msrun_log_pynative*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_forward"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3722 "+\
                    f"--log_dir=msrun_log_pynative "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"

    @pytest.mark.run(order=1)
    def test_alltoall_bprop(self):
        """
        Feature: test_alltoall_self_defined_bprop
        Description: test_alltoall_self_defined_bprop
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_alltoall.py"
        device_num = 2

        rm_list = ["msrun_log_pynative*", "kernel_meta*"]
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
                    f"--master_port=3722 "+\
                    f"--log_dir=msrun_log_pynative "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"

    @pytest.mark.run(order=0)
    def test_alltoall_with_permute_forward(self):
        """
        Feature: test_alltoall_forward
        Description: test_alltoall_forward
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_alltoall_with_permute.py"
        device_num = 2

        rm_list = ["msrun_log_pynative*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_forward"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3722 "+\
                    f"--log_dir=msrun_log_pynative "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"

    @pytest.mark.run(order=1)
    def test_alltoall_with_permute_bprop(self):
        """
        Feature: test_alltoall_self_defined_bprop
        Description: test_alltoall_self_defined_bprop
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_alltoall_with_permute.py"
        device_num = 2

        rm_list = ["msrun_log_pynative*", "kernel_meta*"]
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
                    f"--master_port=3722 "+\
                    f"--log_dir=msrun_log_pynative "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"
