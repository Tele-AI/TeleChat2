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
"""Test GlobalNorm"""
import os

import pytest


class TestGlobalNorm:
    """A test class for testing GlobalNorm."""
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "4,5"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_transformer_pynative(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "150"
        scripts_name = "run_global_norm.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8128 " + \
              f"--log_dir=msrun_log_global_norm " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_global_norm/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_global_norm/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_transformer_pynative_sp(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "150"
        scripts_name = "run_global_norm.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8138 " + \
              f"--log_dir=msrun_log_global_norm_sp " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd} --use_sequence_parallel"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_global_norm_sp/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_global_norm_sp/worker_*.log"
