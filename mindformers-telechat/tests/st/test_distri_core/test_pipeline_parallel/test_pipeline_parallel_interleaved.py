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
"""Test Pipeline Parallel Interleaved """
import os
import numpy as np
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestPipelineParallel:
    """A test class for pipeline parallel interleaved. """

    @pytest.mark.skip(reason="Get golden loss from records")
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=1)
    def test_generate_pipeline_net_golden(self):
        """
        Feature: generate pipeline net golden loss
        Description: run pynative mode pipeline net to generate golden loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "standalone_log"
        scripts_cmd = f"{scripts_path} --run_mode standalone_with_share"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8120 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=2)
    def test_interleaved_pipeline_net_loss(self):
        """
        Feature: test pynative pipeline interleaved.
        Description: run pynative mode pipeline net with shared weight to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 4
        log_dir = "pp_interleaved"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --run_mode pp_interleaved"
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

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pipeline loss and golden loss which with shared weight
        Expectation: relative error smaller than 1e-3
        """
        pp_log_path = './pp_interleaved/worker_3.log'
        pp_loss = read_loss_from_log(pp_log_path)

        pp_loss = np.array(pp_loss, np.float32)
        golden_loss = np.array([6.1131716, 6.1253047], np.float32)

        print(f"interleaved staged pp loss: {pp_loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(pp_loss, golden_loss, atol=1e-3), "Interleaved pipeline net " \
                                                                "loss accuracy test fail !"
        print("============== Interleaved staged pipeline net loss accuracy test pass !!! ==============")
