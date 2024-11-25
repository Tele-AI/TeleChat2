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
"""Test Sequence Parallel"""
import os
import numpy as np
import pytest


class TestSequenceParallel:
    """A test class for sequence parallel."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_generate_golden_net(self):
        """
        Feature: generate sequence parallel net golden loss
        Description: run pynative mode sequence parallel net to generate golden loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "test_sequence_parallel_net.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "golden_log"
        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8027 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_sequence_parallel_net(self):
        """
        Feature: test pynative sequence parallel net
        Description: run pynative mode sequence parallel net to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "test_sequence_parallel_net.py"
        device_num = 2
        log_dir = "sp_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8115 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between sequence parallel loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_numpy_path = f'./golden_loss.npy'
        sp_numpy_path = f'./use_sequence_parallel_loss.npy'

        sp_loss = np.load(sp_numpy_path)
        golden_loss = np.load(golden_numpy_path)

        print(f"sp loss: {sp_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(sp_loss, golden_loss, atol=1e-3), "sequence parallel loss " \
                                                                "accuracy test fail !"
        print("============== sequence parallel loss accuracy test pass !!! ==============")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_sequence_parallel_golden_net(self):
        """
        Feature: generate sequence parallel net golden loss
        Description: run pynative mode sequence parallel net to generate golden loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "test_sequence_parallel_net.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "sp_overlap_log"
        scripts_cmd = f"{scripts_path} --overlap_grad_reduce"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8027 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_grad_acc_sequence_parallel_net(self):
        """
        Feature: test pynative sequence parallel net using grad acc
        Description: run pynative mode sequence parallel net to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "test_sequence_parallel_net.py"
        device_num = 4
        log_dir = "sp_overlap_grad_scc_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --overlap_grad_reduce --gradient_accumulation_fusion"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8115 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_grad_acc_compare_loss(self):
        """
        Feature: test_grad_acc_compare_loss
        Description: compare relative error between sequence parallel loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_numpy_path = f'./sp_overlap.npy'
        sp_numpy_path = f'./sp_overlap_grad_scc.npy'

        sp_loss = np.load(sp_numpy_path)
        golden_loss = np.load(golden_numpy_path)

        print(f"sp loss: {sp_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(sp_loss, golden_loss, atol=1e-3), "sequence parallel loss " \
                                                                "accuracy test fail !"
        print("============== sequence parallel loss accuracy test pass !!! ==============")
