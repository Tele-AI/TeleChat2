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
"""Test selective recompute and gradient checkpointed recompute"""

import os
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestRecompute:
    """A test class for testing selective recompute and gradient checkpoint."""

    @pytest.mark.run(order=1)
    @pytest.mark.skip(reason="skip selective recompute testcase")
    def test_selective_recompute(self):
        """
        Feature: test selective recompute
        Description: run selective recompute on ParallelTransformerLayer
        Expectation: test success
        """
        os.environ["HCCL_BUFFSIZE"] = "1"
        scripts_name = "run_recompute.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --selective"
        cmd = (
            f"msrun --worker_num={device_num} "
            + f"--local_worker_num={device_num} "
            + "--master_port=8118 "
            + "--log_dir=msrun_log_selective_recompute "
            + "--join=True "
            + "--cluster_time_out=300 "
            + f"{scripts_cmd}"
        )
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_selective_recompute/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_selective_recompute/worker_*.log"

    @pytest.mark.run(order=2)
    @pytest.mark.skip(reason="skip checkpointed recompute testcase")
    def test_gradient_checkpoint(self):
        """
        Feature: test gradient_checkpoint
        Description: run gradient_checkpoint on ParallelTransformer
        Expectation: test success
        """
        os.environ["HCCL_BUFFSIZE"] = "1"
        scripts_name = "run_recompute.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = (
            f"msrun --worker_num={device_num} "
            + f"--local_worker_num={device_num} "
            + "--master_port=8118 "
            + "--log_dir=msrun_log_gradient_checkpoint "
            + "--join=True "
            + "--cluster_time_out=300 "
            + f"{scripts_cmd}"
        )
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_gradient_checkpoint/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_gradient_checkpoint/worker_*.log"
