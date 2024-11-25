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
"""Test Parallel CrossAttention"""
import os

import numpy as np
import pytest


class TestParallelCrossEntropyLoss:
    """A test class for testing cross entropy loss."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_cross_entropy_loss(self):
        """
        Feature: generate cross entropy loss
        Description: run nn.CrossEntropyLoss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=CrossEntropyLoss --tp=2"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_vocab_parallel_cross_entropy_loss(self):
        """
        Feature: test vocab parallel cross entropy loss
        Description: run vocab parallel cross entropy loss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=VocabParallelCrossEntropy --tp=2"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log_VocabParallelCrossEntropyLoss " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_VocabParallelCrossEntropyLoss/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_VocabParallelCrossEntropyLoss/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = 'msrun_log/worker_0.log'
        pynative_log_path = 'msrun_log_VocabParallelCrossEntropyLoss/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
            "please run test_parallel_cross_entropy_loss.py to generate them by running below command: \n" + \
            "`pytest -sv test_parallel_cross_entropy_loss.py::TestParallelCrossEntropyLoss`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    golden_loss.append(float(line.split(' ')[-1]))
        print(golden_loss)
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    print(line)
                    pynative_loss.append(float(line.split(' ')[-1]))
        print(pynative_loss)
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
            "relative error between pynative loss and golden loss exceeds 1e-3, please your code."

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_cross_entropy_loss_single(self):
        """
        Feature: generate cross entropy loss
        Description: run nn.CrossEntropyLoss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=CrossEntropyLoss"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_single_log " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_single_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_single_log/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_vocab_parallel_cross_entropy_loss_single(self):
        """
        Feature: test vocab parallel cross entropy loss
        Description: run vocab parallel cross entropy loss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=VocabParallelCrossEntropy"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_single_log_Parallel " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_single_log_Parallel/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_single_log_Parallel/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss_single(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = 'msrun_single_log/worker_0.log'
        pynative_log_path = 'msrun_single_log_Parallel/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
            "please run test_parallel_cross_entropy_loss.py to generate them by running below command: \n" + \
            "`pytest -sv test_parallel_cross_entropy_loss.py::TestParallelCrossEntropyLoss`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    golden_loss.append(float(line.split(' ')[-1]))
        print(golden_loss)
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    print(line)
                    pynative_loss.append(float(line.split(' ')[-1]))
        print(pynative_loss)
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
            "relative error between pynative loss and golden loss exceeds 1e-3, please your code."

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_cross_entropy_loss_dp2(self):
        """
        Feature: generate cross entropy loss
        Description: run nn.CrossEntropyLoss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=CrossEntropyLoss --dp=2"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_dp2_log " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_dp2_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_dp2_log/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_vocab_parallel_cross_entropy_loss_dp2(self):
        """
        Feature: test vocab parallel cross entropy loss
        Description: run vocab parallel cross entropy loss to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_cross_entropy_loss.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --loss_func_type=VocabParallelCrossEntropy --dp=2"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_dp2_log_Parallel " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_dp2_log_Parallel/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_dp2_log_Parallel/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss_dp2(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = 'msrun_dp2_log/worker_0.log'
        pynative_log_path = 'msrun_dp2_log_Parallel/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
            "please run test_parallel_cross_entropy_loss.py to generate them by running below command: \n" + \
            "`pytest -sv test_parallel_cross_entropy_loss.py::TestParallelCrossEntropyLoss`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    golden_loss.append(float(line.split(' ')[-1]))
        print(golden_loss)
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    print(line)
                    pynative_loss.append(float(line.split(' ')[-1]))
        print(pynative_loss)
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
            "relative error between pynative loss and golden loss exceeds 1e-3, please your code."
