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
"""Test Optimizer Parameter Scheduler."""
import os
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestOptimizerParameterScheduler:
    """A test class for learning rate"""

    @pytest.mark.platform_arm_ascend910b_training
    def test_lr_base_iteration_tarining(self):
        """
        Feature: lr adjustment policy base on iteration tarining
        Description: run pynative mode pipeline net to adjust lr
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_lr_scheduler.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "lr_base_iteration_tarining_log"
        scripts_cmd = f"{scripts_path} --yaml_name test_iteration_tarining.yaml"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    def test_lr_base_sample_tarining(self):
        """
        Feature: lr adjustment policy base on sample tarining
        Description: run pynative mode pipeline net to adjust lr
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_lr_scheduler.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "lr_base_sample_tarining_log"
        scripts_cmd = f"{scripts_path}  --yaml_name test_sample_tarining.yaml"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"
