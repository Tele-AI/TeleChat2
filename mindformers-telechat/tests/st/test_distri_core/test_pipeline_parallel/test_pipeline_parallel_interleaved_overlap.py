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

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=1)
    def test_interleaved_pipeline_net_loss_overlap_ddp(self):
        """
        Feature: test pynative pipeline interleaved.
        Description: run pynative mode pipeline net with shared weight to generate pynative loss, overlap ddp
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 4
        log_dir = "pp_interleaved_with_overlap"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --run_mode pp_with_ddp_overlap"
        cmd = f"msrun --worker_num={device_num} "+ \
              f"--local_worker_num={device_num} "+ \
              f"--master_port=8132 "+ \
              f"--log_dir={log_dir} "+ \
              f"--join=True "+ \
              f"--cluster_time_out=300 "+ \
              f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=2)
    def test_interleaved_pipeline_net_loss_overlap_ddp_delay(self):
        """
        Feature: test pynative pipeline interleaved.
        Description: run pynative mode pipeline net with shared weight to generate pynative loss, overlap ddp, delay
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 4
        log_dir = "pp_interleaved_with_overlap_delay"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --run_mode pp_with_ddp_overlap_delay"
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
        pp_log_path1 = './pp_interleaved_with_overlap/worker_3.log'
        pp_loss1 = read_loss_from_log(pp_log_path1)
        pp_loss1 = np.array(pp_loss1, np.float32)
        pp_log_path2 = './pp_interleaved_with_overlap_delay/worker_3.log'
        pp_loss2 = read_loss_from_log(pp_log_path2)
        pp_loss2 = np.array(pp_loss2, np.float32)
        golden_loss = np.array([6.1131716, 6.1253047], np.float32)
        print(f"=======pp_loss1:{pp_loss1}, pp_loss2:{pp_loss2}", flush=True)
        assert np.allclose(pp_loss1, golden_loss, atol=1e-3), "Interleaved pipeline net " \
                                                                "loss accuracy test fail !"
        assert np.allclose(pp_loss2, golden_loss, atol=1e-3), "Interleaved pipeline net " \
                                                              "loss accuracy test fail !"
        print("============== Interleaved staged pipeline with overlap ddp "
              "net loss accuracy test pass !!! ==============")
