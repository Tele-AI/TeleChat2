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
"""Test Router"""
import os

import numpy as np
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestRouter:
    """A test class for testing MoE router."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "2,3"
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip graph st")
    def test_router_golden(self):
        """
        Feature: test_router_golden
        Description: test_router_golden
        Exception: AssertionError
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_router.py"
        device_num = 1

        rm_list = ["npy_golden*", "msrun_log_graph*", "kernel_meta*", "golden_router*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden --dp=1 --batch_size=2 --dataset_size=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3722 "+\
                    f"--log_dir=msrun_log_graph "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_graph/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_graph/worker_*.log"


    @pytest.mark.run(order=2)
    def test_router_pynative_dp1(self):
        """
        Feature: test_router_pynative_dp1
        Description: test_router_pynative_dp1
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_router.py"
        device_num = 1

        rm_list = ["npy_pynative_dp1*", "msrun_log_pynative_single*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --dp=1 --batch_size=2 --dataset_size=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3721 "+\
                    f"--log_dir=msrun_log_pynative_dp1 "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"


    @pytest.mark.run(order=4)
    def test_router_pynative_dp1_with_golden_loss(self):
        """
        Feature: test_router_pynative_dp1_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Exception: AssertionError
        """
        pynative_log_path = f'msrun_log_pynative_dp1/worker_0.log'
        assert os.path.exists(pynative_log_path), \
               f"{pynative_log_path} did not exits, " + \
               "please run test_router.py to generate them by running below command: \n" + \
               "`pytest -sv test_router.py::TestRouter::test_router_pynative_dp1`"

        golden_input_and_loss_path = f"./data/golden_router_input_and_loss.npy"
        assert os.path.exists(golden_input_and_loss_path), \
            f"'{golden_input_and_loss_path}' did not exits, please run generate_golden() to generate one"

        input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
        golden_loss = input_and_loss['loss']

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
               f"relative error between pynative loss and golden loss exceeds 1e-3, please your code."


    @pytest.mark.run(order=4)
    def test_router_pynative_dp2(self):
        """
        Feature: test_router_pynative_dp2
        Description: run pynative mode router to generate pynative loss
        Exception: AssertionError
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_router.py"
        device_num = 2

        rm_list = ["npy_pynative_dp2*", "msrun_log_pynative_dp2*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --dp=2 --batch_size=1 --dataset_size=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=3721 "+\
                    f"--log_dir=msrun_log_pynative_dp2 "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative/worker_*.log"


    @pytest.mark.run(order=4)
    def test_router_pynative_dp2_with_golden_loss(self):
        """
        Feature: test_router_pynative_dp2_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Exception: AssertionError
        """
        pynative_log_path = f'msrun_log_pynative_dp1/worker_0.log'
        assert os.path.exists(pynative_log_path), \
               f"{pynative_log_path} did not exits, " + \
               "please run test_router.py to generate them by running below command: \n" + \
               "`pytest -sv test_router.py::TestRouter::test_router_pynative_dp2`"

        golden_input_and_loss_path = f"./data/golden_router_input_and_loss.npy"
        assert os.path.exists(golden_input_and_loss_path), \
            f"'{golden_input_and_loss_path}' did not exits, please run generate_golden() to generate one"

        input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
        golden_loss = input_and_loss['loss']

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if "loss_all is " in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    print(line)
                    pynative_loss.append(float(line.split(' ')[-1]))
        print(pynative_loss)
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
               f"relative error between pynative loss and golden loss exceeds 1e-3, please your code."
