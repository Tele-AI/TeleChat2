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
"""Test module for testing router for graph or pynative"""
import os
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TestRouter:
    """A test class for testing router for graph or pynative."""

    @staticmethod
    def setup_method():
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        os.environ['MS_MEMORY_POOL_RECYCLE'] = '1'

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_graph_router(self):
        """
        Feature: graph/pynative router
        Description: Test graph route for base api
        Expectation: AssertionError
        """
        ret = os.system(f'msrun --worker_num=1 --local_worker_num=1 --master_port=61374 '
                        f'--log_dir=log_router --join=True {cur_dir}/run_router.py --graph')
        assert ret == 0

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_pynative_router(self):
        """
        Feature: graph/pynative router
        Description: Test pynative route for base api
        Expectation: AssertionError
        """
        ret = os.system(f'msrun --worker_num=1 --local_worker_num=1 --master_port=61374 '
                        f'--log_dir=log_router --join=True {cur_dir}/run_router.py --pynative')
        assert ret == 0
