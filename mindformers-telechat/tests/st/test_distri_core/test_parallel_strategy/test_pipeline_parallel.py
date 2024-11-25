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
"""Test Pipeline Parallel Shard Strategy"""
import os
import numpy as np
import pytest
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestPipelineParallel:
    """A test class for pipeline parallel shard strategy in dp/pp mode. """
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip pp st")
    def test_pipeline_net_src(self):
        """
        Feature: test pynative pipeline net
        Description: run pynative mode pipeline net to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 4
        log_dir = "pp_src_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --generate_src_strategy"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"
    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip pp st")
    def test_pipeline_net_dst(self):
        """
        Feature: test pynative pipeline net
        Description: run pynative mode pipeline net o generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 8
        log_dir = "pp_dst_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir={log_dir} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"
    @pytest.mark.run(order=3)
    @pytest.mark.skip(reason="skip pp st")
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pipeline loss
        Expectation: relative error smaller than 1e-3
        """
        src_numpy_path = f'./pp_src_log/pp_loss.npy'
        dst_numpy_path = f'./pp_dst_log/pp_loss.npy'
        src_pp_loss = np.load(src_numpy_path)
        dst_pp_loss = np.load(dst_numpy_path)
        print(f"pp loss src: {src_pp_loss}", flush=True)
        print(f"pp loss dst: {dst_pp_loss}", flush=True)
        assert np.allclose(src_pp_loss, dst_pp_loss, atol=1e-3), "Pipeline parallel loss " \
                                                                "weight accuracy test fail !"

        print("============== Pipeline parallel loss accuracy test pass !!! ==============")
