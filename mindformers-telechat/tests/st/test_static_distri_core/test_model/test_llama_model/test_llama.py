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
"""
Test module for testing the llama interface used for mindformers.
"""
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


class TestLlama:
    """A test class for testing llama."""

    @staticmethod
    def setup_method():
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_single_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for training
        Expectation: AssertionError
        """
        commands = [
            (f"python {cur_dir}/run_tests.py --mode single_train --use-new-loss", '')
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_parallel_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel trainer for training
        Expectation: AssertionError
        """
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && "
             f"msrun --worker_num=4 --local_worker_num=4 --master_port=61374 --log_dir=log_train_dp2_mp2 --join=True "
             f"{cur_dir}/run_tests.py --mode parallel_train_dp2_mp2", 'log_train_dp2_mp2/worker_0.log'),
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5 && "
             f"msrun --worker_num=2 --local_worker_num=2 --master_port=61375 --log_dir=log_train_mp2 --join=True "
             f"{cur_dir}/run_tests.py --mode parallel_train_mp2", 'log_train_mp2/worker_0.log'),
            (f"export ASCEND_RT_VISIBLE_DEVICES=6,7 && "
             f"msrun --worker_num=2 --local_worker_num=2 --master_port=61376 --log_dir=log_train_dp2 --join=True "
             f"{cur_dir}/run_tests.py --mode parallel_train_dp2", 'log_train_dp2/worker_0.log')
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)
