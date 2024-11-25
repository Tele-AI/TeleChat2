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
""" Test ParamAndGradBuffer """
import os
import pytest


class TestDistributedOptimizer:
    """A test class for testing ParamAndGradBuffer."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0,1"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_optimizer_golden(self):
        """
        Feature: DistributedOptimizer golden case.
        Description: DistributedOptimizer golden case.
        Expectation: test success.
        """
        scripts_name = "run_distributed_optimizer.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_optimizer_golden "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_optimizer_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_optimizer_golden/worker_*.log"


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_distributed_optimizer(self):
        """
        Feature: test DistributedOptimizer.
        Description: test DistributedOptimizer.
        Expectation: test success.
        """
        scripts_name = "run_distributed_optimizer.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_distributed_optimizer "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_distributed_optimizer/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_distributed_optimizer/worker_*.log"
