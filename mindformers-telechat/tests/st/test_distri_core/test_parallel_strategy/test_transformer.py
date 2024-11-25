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


class TestParallelTransformerCkpt:
    """A test class for testing Linear."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip")
    def test_transformer_pynative_src(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_transformer.py"
        device_num = 4
        tp_size = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_mode='transform_src' --tp_size={tp_size} "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_pynative_transformer_src "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_transformer_src/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_transformer_src/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip")
    def test_transformer_pynative_dst(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_transformer.py"
        device_num = 4
        tp_size = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_mode='transform_dst' --tp_size={tp_size} "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_pynative_transformer_dst "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_transformer_dst/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_transformer_dst/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.run(order=3)
    @pytest.mark.skip(reason="skip")
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between two parallel mode
        Expectation: relative error smaller than 1e-3
        """
        src_numpy_path = f'./loss_transform_src.npy'
        dst_numpy_path = f'./loss_transform_dst.npy'
        src_loss = np.load(src_numpy_path)
        dst_loss = np.load(dst_numpy_path)
        print(f"loss src: {src_loss}", flush=True)
        print(f"loss dst: {dst_loss}", flush=True)
        assert np.allclose(src_loss, dst_loss, atol=1e-3), "loss weight accuracy test fail !"

        print("============== loss accuracy test pass !!! ==============")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=4)
    @pytest.mark.skip(reason="skip")
    def test_transformer_pynative_random_src(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_transformer.py"
        device_num = 2
        tp_size = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_mode='rng_check' --tp_size={tp_size} --rng_mode='save'"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_pynative_transformer_dst "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_transformer_dst/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_transformer_dst/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=5)
    @pytest.mark.skip(reason="skip")
    def test_transformer_pynative_random_dst(self):
        """
        Feature: test ParallelTransformer pynative
        Description: run pynative mode transformer to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_transformer.py"
        device_num = 2
        tp_size = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --test_mode='rng_check' --tp_size={tp_size} --rng_mode='load'"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_pynative_transformer_dst "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_transformer_dst/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_transformer_dst/worker_*.log"
