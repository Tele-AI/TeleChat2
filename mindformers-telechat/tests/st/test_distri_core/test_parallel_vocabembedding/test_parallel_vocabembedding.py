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
"""Test ParallelVocabEmbedding"""
import os

import numpy as np
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestParallelVocabEmbedding:
    """A test class for testing VocabEmbedding."""
    # os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1"
    @pytest.mark.run(order=1)
    def test_vocabembedding_golden(self):
        """
        Feature: generate vocab embedding golden
        Description: run graph mode vocab embedding to generate golden ckpt and loss
        Expectation: test success
        """
        os.environ['GRAPH_OP_RUN'] = "1"
        scripts_name = "run_parallel_vocabembedding.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --generate_golden"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_graph_vocabembedding "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_graph_vocabembedding/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_graph_vocabembedding/worker_*.log"

    @pytest.mark.run(order=2)
    def test_vocabembedding_pynative(self):
        """
        Feature: test vocab embedding pynative
        Description: run pynative mode vocab embedding to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_parallel_vocabembedding.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port=8118 "+\
                    f"--log_dir=msrun_log_pynative_vocabembedding "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_vocabembedding/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_vocabembedding/worker_*.log"

    @pytest.mark.run(order=3)
    def test_pynative_with_golden_loss(self):
        """
        Feature: test_pynative_with_golden_loss
        Description: compare relative error between pynative loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        golden_log_path = f'msrun_log_graph_vocabembedding/worker_0.log'
        pynative_log_path = f'msrun_log_pynative_vocabembedding/worker_0.log'

        assert os.path.exists(golden_log_path) and os.path.exists(pynative_log_path), \
            f"{golden_log_path} or {pynative_log_path} did not exits, " + \
                "please run test_parallel_vocabembedding.py to generate them by running below command: \n" + \
                "`pytest -sv test_parallel_vocabembedding.py::TestParallelVocabEmbedding`"

        golden_loss = []
        with open(golden_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    golden_loss.append(float(line.split(", loss ")[-1]))
        golden_loss = np.array(golden_loss)

        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", loss " in line:
                    pynative_loss.append(float(line.split(", loss ")[-1]))
        pynative_loss = np.array(pynative_loss)

        assert np.allclose(golden_loss[-1], pynative_loss[-1], rtol=1e-3), \
               f"relative error between pynative loss and golden loss exceeds 1e-3, please check your code."
