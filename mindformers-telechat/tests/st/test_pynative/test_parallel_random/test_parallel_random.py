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
"""Test ParallelRandom"""
import os
import pytest
#os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1,2,3"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestParallelRandom:
    """A test class for testing Random."""

    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip rng tracer testcase")
    def test_random_parallel(self):
        """
        Feature: test random parallel
        Description: test random parallel in different mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_random.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --testcase 0 "
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8899 " + \
                    f"--log_dir=msrun_log_random " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_random/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_random/worker_*.log"

    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip recompute with rng tracer fork")
    def test_recompute_parallel(self):
        """
        Feature: test random parallel
        Description: test random parallel in different mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_random.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --testcase 1 "
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8899 " + \
                    f"--log_dir=msrun_log_random " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_random/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_random/worker_*.log"
