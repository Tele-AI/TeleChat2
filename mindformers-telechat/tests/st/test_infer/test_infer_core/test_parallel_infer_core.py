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
"""test parallel transformer."""

import os
from multiprocessing.pool import Pool

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_parallel_infer_core():
    """
    Feature: Core of Parallel for prediction.
    Description: Test parallel core for prediction.
    Expectation: AssertionError.
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ['MS_ENABLE_LCCL'] = "off"
    commands = [
        (f"export ASCEND_RT_VISIBLE_DEVICES=0,1 && "
         f"msrun --worker_num=2 --local_worker_num=2 --master_port=8110 --log_dir=test_log_mlp --join=True "
         f"{cur_dir}/run_parallel_infer_core.py --module mlp", 'test_log_mlp/worker_0.log'),
        (f"export ASCEND_RT_VISIBLE_DEVICES=2,3 && "
         f"msrun --worker_num=2 --local_worker_num=2 --master_port=8111 --log_dir=test_log_attention --join=True "
         f"{cur_dir}/run_parallel_infer_core.py --module attention", 'test_log_attention/worker_0.log'),
        (f"export ASCEND_RT_VISIBLE_DEVICES=4,5 && "
         f"msrun --worker_num=2 --local_worker_num=2 --master_port=8123 --log_dir=test_log_transformerlayer "
         f"--join=True {cur_dir}/run_parallel_infer_core.py --module transformerlayer",
         'test_log_transformerlayer/worker_0.log'),
        (f"export ASCEND_RT_VISIBLE_DEVICES=6,7 && "
         f"msrun --worker_num=2 --local_worker_num=2 --master_port=8124 --log_dir=test_log_transformer --join=True "
         f"{cur_dir}/run_parallel_infer_core.py --module transformer", 'test_log_transformer/worker_0.log')
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)
