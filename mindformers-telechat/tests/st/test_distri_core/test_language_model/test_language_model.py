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
"""Test Language Model"""
import os
import numpy as np
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log


class TestLanguageModel:
    """A test class for language model."""

    @pytest.mark.level0
    @pytest.mark.skip(reason="Get golden loss from records")
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def generate_goloden_loss(self):
        """
        Feature: generate goloden loss.
        Description: run pynative mode language model to generate goloden loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_language_model.py"
        device_num = 1
        log_dir = "language_model_goloden_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8132 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_language_model_loss(self):
        """
        Feature: test language model.
        Description: run pynative mode language model to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_language_model.py"
        device_num = 1
        log_dir = "language_model_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8132 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level0
    @pytest.mark.env_single
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between test loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        log_path = './language_model_log/worker_0.log'
        language_model_loss = read_loss_from_log(log_path)

        language_model_loss = np.array(language_model_loss, np.float32)
        golden_loss = np.array([2482804, 2438391], np.float32)

        print(f"language model loss: {language_model_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(language_model_loss, golden_loss, atol=1e-3), "Language model " \
                                                                "loss accuracy test fail !"
        print("============== Language model loss accuracy test pass !!! ==============")
