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
""" Test create comm """
import os
import pytest

test_parallel_order = ['tp-cp-ep-dp-pp', 'tp-ep-dp-cp-pp']
class TestCreateComm:
    """A test class for testing creating communication."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0,1,2,3"
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.parametrize("order", test_parallel_order)
    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip comm group test")
    def test_create_comm(self, order):
        """
        Feature: boundary test for creating communication
        Description: test communication group creation
        Expectation: test success
        """
        scripts_name = "run_create_comm.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --order={order}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
