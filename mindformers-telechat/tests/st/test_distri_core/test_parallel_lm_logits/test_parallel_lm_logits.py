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
"""Test Parallel LM Logits"""
import os
import subprocess
import numpy as np
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestLanguageModel:
    """A test class for Parallel LM Logits."""

    @pytest.mark.run(order=1)
    def test_parallel_lm_logits_loss_tp2(self):
        """
        Feature: test Parallel LM Logits.
        Description: run pynative mode Parallel LM Logits to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_lm_logits.py"
        device_num = 2
        log_dir = "parallel_lm_logits_tp2_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --tp=2"
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

    @pytest.mark.run(order=2)
    def test_compare_loss_tp2(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between test loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        log_path = './parallel_lm_logits_tp2_log/worker_0.log'
        cmd = "grep Loss: " + log_path + " | cut -d ',' -f 4 | cut -d ':' -f 2"
        ret = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        parallel_lm_logits_loss = []
        for line in ret.stdout.readlines():
            parallel_lm_logits_loss.append(float(line.decode().strip()))
        ret.stdout.close()
        del ret

        parallel_lm_logits_loss = np.array(parallel_lm_logits_loss, np.float32)
        golden_loss = np.array([4.8449183, 4.843995], np.float32)

        print(f"Parallel LM Logits loss: {parallel_lm_logits_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(parallel_lm_logits_loss, golden_loss, atol=1e-3), "Parallel LM Logits " \
                                                                "loss accuracy test fail !"
        print("============== Parallel LM Logits loss accuracy test pass !!! ==============")

    @pytest.mark.run(order=1)
    def test_parallel_lm_logits_loss_tp2_parallel_output(self):
        """
        Feature: test Parallel LM Logits.
        Description: run pynative mode Parallel LM Logits to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_lm_logits.py"
        device_num = 2
        log_dir = "parallel_lm_logits_tp2_parallel_output_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --tp=2 --parallel_output"
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

    @pytest.mark.run(order=2)
    def test_compare_loss_tp2_parallel_output(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between test loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        log_path = './parallel_lm_logits_tp2_parallel_output_log/worker_0.log'
        cmd = "grep Loss: " + log_path + " | cut -d ',' -f 4 | cut -d ':' -f 2"
        ret = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        parallel_lm_logits_loss = []
        for line in ret.stdout.readlines():
            parallel_lm_logits_loss.append(float(line.decode().strip()))
        ret.stdout.close()
        del ret

        parallel_lm_logits_loss = np.array(parallel_lm_logits_loss, np.float32)
        golden_loss = np.array([4.8449183, 4.843995], np.float32)

        print(f"Parallel LM Logits loss: {parallel_lm_logits_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(parallel_lm_logits_loss, golden_loss, atol=1e-3), "Parallel LM Logits " \
                                                                "loss accuracy test fail !"
        print("============== Parallel LM Logits loss accuracy test pass !!! ==============")

    @pytest.mark.run(order=1)
    def test_parallel_lm_logits_loss_dp2(self):
        """
        Feature: test Parallel LM Logits.
        Description: run pynative mode Parallel LM Logits to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_lm_logits.py"
        device_num = 2
        log_dir = "parallel_lm_logits_dp2_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --dp=2"
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

    @pytest.mark.run(order=2)
    def test_compare_loss_dp2(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between test loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        log_path = './parallel_lm_logits_dp2_log/worker_0.log'
        cmd = "grep Loss: " + log_path + " | cut -d ',' -f 4 | cut -d ':' -f 2"
        ret = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        parallel_lm_logits_loss = []
        for line in ret.stdout.readlines():
            parallel_lm_logits_loss.append(float(line.decode().strip()))
        ret.stdout.close()
        del ret

        parallel_lm_logits_loss = np.array(parallel_lm_logits_loss, np.float32)
        golden_loss = np.array([4.8449183, 4.843995], np.float32)

        print(f"Parallel LM Logits loss: {parallel_lm_logits_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(parallel_lm_logits_loss, golden_loss, atol=1e-3), "Parallel LM Logits " \
                                                                "loss accuracy test fail !"
        print("============== Parallel LM Logits loss accuracy test pass !!! ==============")
