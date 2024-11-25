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
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0,1,2,3"

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestParallelMLP:
    """A test class for testing MLP Shard Strategy in dp/tp mode."""
    @pytest.mark.run(order=1)
    def test_mlp_pynative_src(self):
        """
        Feature: test mlp pynative src
        Description: run pynative mode mlp to generate pynative src loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_mlp.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_src_strategy"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=mlp_src_log "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/mlp_src_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check mlp_src_log/worker_*.log"

    @pytest.mark.run(order=2)
    def test_mlp_pynative_dst(self):
        """
        Feature: test mlp pynative dst
        Description: run pynative mode mlp to generate pynative dst loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_mlp.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=mlp_dst_log "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/mlp_dst_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check mlp_dst_log/worker_*.log"
