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


class TestParallelAttention:
    """A test class for testing attention."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='no need to run graph mode')
    @pytest.mark.run(order=1)
    def test_attention_golden(self):
        """
        Feature: generate attention golden
        Description: run graph mode linaer to generate golden ckpt and loss
        Expectation: test success
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_attention.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    "--master_port=8118 "+\
                    "--log_dir=msrun_log_graph_attention "+\
                    "--join=True "+\
                    "--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_graph_attention/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_graph_attention/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='need adapt')
    @pytest.mark.run(order=1)
    def test_attention_golden_with_fa(self):
        """
        Feature: generate attention golden
        Description: run graph mode linaer to generate golden ckpt and loss
        Expectation: test success
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_attention.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden --use_fa"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    "--master_port=8118 "+\
                    "--log_dir=msrun_log_graph_attention_with_fa "+\
                    "--join=True "+\
                    "--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_graph_attention_with_fa/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_graph_attention_with_fa/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_attention_pynative(self):
        """
        Feature: test ParallelAttention pynative
        Description: run pynative mode attention to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_attention.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    "--master_port=8118 "+\
                    "--log_dir=msrun_log_pynative_attention "+\
                    "--join=True "+\
                    "--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_attention/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_attention/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_attention_pynative_with_gqa(self):
        """
        Feature: test ParallelAttention pynative
        Description: run pynative mode attention to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_attention.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --group_query_attention"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    "--master_port=8118 "+\
                    "--log_dir=msrun_log_pynative_attention "+\
                    "--join=True "+\
                    "--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_attention/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_attention/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_attention_pynative_with_fa(self):
        """
        Feature: test ParallelAttention pynative
        Description: run pynative mode attention to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_attention.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_fa"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    "--master_port=8118 "+\
                    "--log_dir=msrun_log_pynative_attention_with_fa "+\
                    "--join=True "+\
                    "--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_attention_with_fa/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_attention_with_fa/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='no graph mode run')
    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = 'msrun_log_graph_attention/worker_0.log'
        pynative_log_path = 'msrun_log_pynative_attention/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                "please run test_attention.py to generate them by running below command: \n" + \
                "`pytest -sv test_attention.py::TestParallelAttention`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss [" in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    golden_loss.append(float(line.split(' ')[-1]))
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss [" in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    pynative_loss.append(float(line.split(' ')[-1]))
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
               "relative error between pynative loss and golden loss exceeds 1e-3, please check your code."

    @pytest.mark.level1
    @pytest.mark.skip(reason='need adapt')
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason='no graph mode run')
    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss_with_fa(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = 'msrun_log_graph_attention_with_fa/worker_0.log'
        pynative_log_path = 'msrun_log_pynative_attention_with_fa/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                "please run test_attention.py to generate them by running below command: \n" + \
                "`pytest -sv test_attention.py::TestParallelAttention`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss [" in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    golden_loss.append(float(line.split(' ')[-1]))
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss [" in line:
                    line = line.strip().replace('[', '').replace(']', '')
                    pynative_loss.append(float(line.split(' ')[-1]))
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
               "relative error between pynative loss and golden loss exceeds 1e-3, please check your code."
