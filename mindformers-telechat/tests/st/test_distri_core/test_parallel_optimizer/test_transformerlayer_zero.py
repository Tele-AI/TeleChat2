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

import numpy as np
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTransformerLayerWithZeroOptimizer:
    """A test class for testing Linear."""

    @pytest.mark.run(order=1)
    def test_transformerlayer_adamw_golden(self):
        """
        Feature: generate transformerlayer adamw golden
        Description: run transformerlayer to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_adamw_golden "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_adamw_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_adamw_golden/worker_*.log"

    @pytest.mark.run(order=2)
    def test_transformerlayer_adamw_zero1(self):
        """
        Feature: test transformerlayer with adamw zero2
        Description: run transformerlayer with adamw zero2 to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --zero_level z1"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8118 " + \
              f"--log_dir=msrun_log_adamw_zero1 " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_adamw_zero1/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_adamw_zero1/worker_*.log"


    @pytest.mark.run(order=2)
    def test_transformerlayer_adamw_zero2(self):
        """
        Feature: test transformerlayer with adamw zero2
        Description: run transformerlayer with adamw zero2 to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --zero_level z2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_adamw_zero2 "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_adamw_zero2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_adamw_zero2/worker_*.log"

    @pytest.mark.run(order=3)
    def test_adamw_zero_with_golden_loss(self):
        """
        Feature: test_columnparallellinear_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = f'msrun_log_adamw_golden/worker_0.log'
        for zero_level in [1, 2]:
            pynative_log_path = f'msrun_log_adamw_zero{zero_level}/worker_0.log'

            assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
                f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                    "please run run_transformerlayer_zero.py to generate them by running below command: \n" + \
                    "`pytest -sv test_transformerlayer_zero.py::TestTransformerLayerWithZeroOptimizer`"

            golden_loss = []
            with open(golden_log_path, "r") as fp:
                for line in fp:
                    print(line)
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        golden_loss.append(float(line.split(' ')[-1]))
            golden_loss = np.array(golden_loss)

            pynative_loss = []
            with open(pynative_log_path, "r") as fp:
                for line in fp:
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        print(line)
                        pynative_loss.append(float(line.split(' ')[-1]))
            pynative_loss = np.array(pynative_loss)
            assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
                   f"relative error between zero{zero_level} loss and golden loss exceeds 1e-3, please check your code."

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTransformerLayerWithZeroOptimizerZ3:
    """A test class for testing Linear."""

    @pytest.mark.run(order=1)
    def test_transformerlayer_adamw_golden(self):
        """
        Feature: generate transformerlayer adamw golden
        Description: run transformerlayer to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_adamw_golden "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_adamw_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_adamw_golden/worker_*.log"


    @pytest.mark.run(order=2)
    def test_transformerlayer_adamw_zero3(self):
        """
        Feature: test transformerlayer with adamw zero3
        Description: run transformerlayer with adamw zero3 to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --zero_level z3"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8118 " + \
              f"--log_dir=msrun_log_adamw_zero3 " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_adamw_zero3/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_adamw_zero3/worker_*.log"

    @pytest.mark.run(order=3)
    def test_adamw_zero_with_golden_loss(self):
        """
        Feature: test_columnparallellinear_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = f'msrun_log_adamw_golden/worker_0.log'
        for zero_level in [3]:
            pynative_log_path = f'msrun_log_adamw_zero{zero_level}/worker_0.log'

            assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
                f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                    "please run run_transformerlayer_zero.py to generate them by running below command: \n" + \
                    "`pytest -sv test_transformerlayer_zero.py::TestTransformerLayerWithZeroOptimizer`"

            golden_loss = []
            with open(golden_log_path, "r") as fp:
                for line in fp:
                    print(line)
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        golden_loss.append(float(line.split(' ')[-1]))
            golden_loss = np.array(golden_loss)

            pynative_loss = []
            with open(pynative_log_path, "r") as fp:
                for line in fp:
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        print(line)
                        pynative_loss.append(float(line.split(' ')[-1]))
            pynative_loss = np.array(pynative_loss)
            assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
                   f"relative error between zero{zero_level} loss and golden loss exceeds 1e-3, please check your code."

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTransformerLayerWithCame:
    """A test class for testing Linear."""

    @pytest.mark.run(order=1)
    def test_transformerlayer_came_golden(self):
        """
        Feature: generate transformerlayer adamw golden
        Description: run transformerlayer to generate loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden --came"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_came_golden "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_came_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_came_golden/worker_*.log"


    @pytest.mark.run(order=2)
    def test_transformerlayer_came_zero2(self):
        """
        Feature: test transformerlayer with adamw zero3
        Description: run transformerlayer with adamw zero3 to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_transformerlayer_zero.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --zero_level z2 --came"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8118 " + \
              f"--log_dir=msrun_log_came_zero2 " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_came_zero2/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_came_zero2/worker_*.log"

    @pytest.mark.run(order=3)
    def test_came_zero_with_golden_loss(self):
        """
        Feature: test_columnparallellinear_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = f'msrun_log_came_golden/worker_0.log'
        for zero_level in [2]:
            pynative_log_path = f'msrun_log_came_zero{zero_level}/worker_0.log'

            assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
                f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                    "please run run_transformerlayer_zero.py to generate them by running below command: \n" + \
                    "`pytest -sv test_transformerlayer_zero.py::TestTransformerLayerWithZeroOptimizer`"

            golden_loss = []
            with open(golden_log_path, "r") as fp:
                for line in fp:
                    print(line)
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        golden_loss.append(float(line.split(' ')[-1]))
            golden_loss = np.array(golden_loss)

            pynative_loss = []
            with open(pynative_log_path, "r") as fp:
                for line in fp:
                    if "| Loss [" in line:
                        line = line.strip().replace('[', '').replace(']', '')
                        print(line)
                        pynative_loss.append(float(line.split(' ')[-1]))
            pynative_loss = np.array(pynative_loss)
            assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
                   f"relative error between zero{zero_level} loss and golden loss exceeds 1e-3, please check your code."
