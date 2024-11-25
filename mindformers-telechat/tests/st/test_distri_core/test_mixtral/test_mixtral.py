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
"""Test Pallel Mixtral"""
import os

import pytest
import numpy as np

class TestMixtral:
    """A test class for testing Linear."""

    env_list = {
        # 'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3',
        # 'ASCEND_RT_VISIBLE_DEVICES': '4,5,6,7',
        # 'ASCEND_GLOBAL_LOG_LEVEL': '3',
        # 'ASCEND_SLOG_PRINT_TO_STDOUT': '1',
        # 'ASCEND_GLOBAL_EVENT_ENABLE': '1',
        # 'GLOG_v': '1',
        # 'PYTHONPATH': f"/path/to/your/mindspore:{os.getenv('PYTHONPATH')}",
        }
    for k, v in env_list.items():
        os.environ[k] = v
    # os.system("ps -ef|grep pytest |grep -v grep|cut -c 9-16|xargs kill -9")
    def extract_loss_from_log(self, pynative_log_path: str):
        '''extract loss from log_path'''

        assert os.path.exists(pynative_log_path), f"{pynative_log_path} did not exits"

        # check loss with golden loss
        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", Loss: " in line:
                    line = line.strip().replace('[', '').replace(']', '').replace(',', '')
                    line = line.split(' ')
                    i = 0
                    for i, s in enumerate(line):
                        if "Loss:" in s:
                            print(f"{i}: {s} {line[i+1]}")
                            break
                    loss = float(line[i+1])
                    pynative_loss.append(loss)
        pynative_loss = np.array(pynative_loss)

        return pynative_loss

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=0)
    @pytest.mark.skip(reason="skip ep1 st")
    def test_mixtral_pynative_ep1tp1pp1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 1
        postfix = "_ep1tp1pp1"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_small.yaml --ep=1 --tp=1 --pp=1 --bs=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8119 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)

        golden_loss = [4.14965, 4.1490374, 4.148424, 4.1478105, 4.1471977,
                       4.1465855, 4.1459723, 4.1453605, 4.1447477, 4.1441364]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip ep2 st")
    def test_mixtral_pynative_ep2tp1pp1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_mixtral.py"
        device_num = 2
        postfix = "_ep2tp1pp1"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_small.yaml --ep=2 --tp=1 --pp=1"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8116 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [4.14965, 4.1490374, 4.148424, 4.1478105, 4.1471977,
                       4.1465855, 4.1459723, 4.1453605, 4.1447477, 4.1441364]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip ep2tp2 st")
    def test_mixtral_pynative_ep2tp2pp1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 4
        postfix = "_ep2tp2pp1"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_small.yaml --ep=2 --tp=2 --pp=1 --sp"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8119 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [4.149672, 4.149058, 4.148445, 4.147832, 4.147219,
                       4.146605, 4.145993, 4.145381, 4.144769, 4.144156]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_mixtral_pynative_ep2tp2pp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 8
        postfix = "_ep2tp2pp2"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_small.yaml --ep=2 --tp=2 --pp=2 --sp"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8119 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_4.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [4.1485944, 4.1479816, 4.1473684, 4.146756, 4.146144,
                       4.145531, 4.144919, 4.144307, 4.143695, 4.1430836]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_ep2tp1pp1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_mixtral.py"
        device_num = 2
        postfix = "_large_ep2tp1pp1"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=1 --pp=1"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8120 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.44597, 10.44503, 10.44437, 10.44345, 10.44254,
                       10.44159, 10.44079, 10.43993, 10.43908, 10.43867]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_ep2tp2pp1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 4
        postfix = "_large_ep2tp2pp1"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=2 --pp=1 --sp"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8119 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        assert os.path.exists(pynative_log_path), \
               f"{pynative_log_path} did not exits, " + \
               "please run test_mixtral.py to generate them by running below command: \n" + \
               "`pytest -sv test_mixtral.py::TestMixtral`"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.44596, 10.44488, 10.44394, 10.44302, 10.44252,
                       10.44193, 10.44106, 10.44001, 10.43891, 10.43805]
        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_ep2tp1pp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 4
        postfix = "_large_ep2tp1pp2"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=1 --pp=2 --mbn=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=1921 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        # pp st should check last pp stage log
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_2.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.44597, 10.44503, 10.44437, 10.44345, 10.44254,
                       10.44159, 10.44079, 10.43993, 10.43908, 10.43867]
        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_layer4_ep2tp2pp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 8
        postfix = "_large_layer4_ep2tp2pp2"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=2 --pp=2 --sp " + \
                      "--mbn=2 --num_layers=4 --checkpoint_dir=data/golden_mixtral_large_layer4.pt"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=1921 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        # pp st should check last pp stage log
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_4.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.584751, 10.582008, 10.578396, 10.574947, 10.57238,
                       10.569126, 10.56565, 10.563158, 10.559839, 10.557177]
        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_layer4_ep2tp2pp2vpp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 8
        postfix = "_large_layer4_ep2tp2pp2vpp2"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=2 --pp=2 --vpp2 --sp " + \
                      "--mbn=2 --num_layers=4 --checkpoint_dir=data/golden_mixtral_large_layer4.pt"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=1921 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        # pp st should check last pp stage log
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_4.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.584751, 10.582008, 10.578396, 10.574947, 10.57238,
                       10.569126, 10.56565, 10.563158, 10.559839, 10.557177]
        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip large st")
    def test_mixtral_pynative_large_ep2tp2pp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 8
        postfix = "_large_ep2tp2pp2"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --config_path=./config_mixtral_large.yaml --ep=2 --tp=2 --pp=2 --sp"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=1921 "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        # pp st should check last pp stage log
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_4.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [10.44596, 10.44488, 10.44394, 10.44302, 10.44252,
                       10.44193, 10.44106, 10.44001, 10.43891, 10.43805]
        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."
