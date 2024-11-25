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
Test module for testing the paralleled mixtral interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_mixtral_model/test_parallel_train.py
"""
import os
# pylint: disable=W0611
import pytest


class TestMixtralParallelTrain:
    """A test class for testing pipeline."""

    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel trainer for train.
        Expectation: AssertionError
        """
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"bash {sh_path}/msrun_launch_mixtral.sh 8 test_train")
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_7.log -C 3")
        assert ret == 0
