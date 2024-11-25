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
"""Test parallel reducer on grads, is_finite and overflow status reduce"""

import os
import pytest

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_grad_reduce():
    """
    Feature: test grads reduce
    Description: run grads reduce on ParallelTrainingReducer
    Expectation: test success
    """
    os.environ["HCCL_BUFFSIZE"] = "1"
    scripts_name = "run_parallel_reducer.py"
    device_num = 4

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --grad"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8118 "
        + "--log_dir=msrun_log_grad_reduce "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_grad_reduce/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_log_grad_reduce/worker_*.log"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_overflow_reudce():
    """
    Feature: test overflow reduce
    Description: run overflow reduce on ParallelTrainingReducer
    Expectation: test success
    """
    os.environ["HCCL_BUFFSIZE"] = "1"
    scripts_name = "run_parallel_reducer.py"
    device_num = 4

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path}"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8118 "
        + "--log_dir=msrun_overflow_reduce "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_overflow_reduce/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_overflow_reduce/worker_*.log"
