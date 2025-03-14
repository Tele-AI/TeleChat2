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
Test module for testing the memory calculation of pipeline balancing.
How to run this:
pytest tests/st/test_pipeline_balance/test_sapp_solver.py
"""
import os
import pytest

from toolkit.pipeline_balance.utils.layer import generate_layers_list
from toolkit.pipeline_balance.sapp.sapp_pipeline import SappPipeline


class TestSappSolver:
    """A test class for testing sapp solver."""

    def test_sapp_solver_vpp1(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the sapp solver pipeline when vpp=1.
        Expectation: Correct result
        """
        layers = generate_layers_list(
            os.path.dirname(os.path.abspath(__file__)), "test"
        )
        pipe = SappPipeline(
            model_name="test",
            num_of_stage=16,
            num_of_micro_batch=16,
            max_memory=53000,
            layers=layers,
            num_of_interleave=1,
            vpp_less_memory=True,
        )
        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=20)
        total_time = pipe.simulate(show=False, file_name=None)
        mem_par = pipe.get_memory_parameter()
        mem_act = pipe.get_memory_activation()
        assert pytest.approx(total_time, abs=5) == 36870
        assert mem_par[0] == pytest.approx(
            [
                18915.0,
                9498.0,
                6332.0,
                6332.0,
                9498.0,
                7915.0,
                9498.0,
                9498.0,
                9498.0,
                11081.0,
                11081.0,
                11081.0,
                11081.0,
                11081.0,
                11081.0,
                11971.0,
            ],
            abs=2,
        )
        assert mem_act[0] == pytest.approx(
            [
                1748.0,
                2574.0,
                3304.0,
                3304.0,
                3368.0,
                3336.0,
                4162.0,
                4162.0,
                4956.0,
                5782.0,
                5782.0,
                5782.0,
                5782.0,
                5782.0,
                5782.0,
                4956.0,
            ],
            abs=2,
        )

    def test_sapp_simulate_vpp3(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the sapp solver simulation when vpp=3.
        Expectation: Correct result
        """
        layers = generate_layers_list(os.path.dirname(os.path.abspath(__file__)), "sim")
        pipe = SappPipeline(
            model_name="sim",
            num_of_stage=16,
            num_of_micro_batch=16,
            max_memory=53000,
            layers=layers,
            num_of_interleave=3,
            vpp_less_memory=True,
        )
        pipe.construct_problem(solver="pulp")
        total_time = pipe.simulate_yaml(
            {
                "offset": [
                    [-2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
                ],
                "recompute": [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                ],
                "select_recompute": [
                    [1, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 1, 0, 0, 0, 1, 0, 0],
                    [3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 1, 0, 0, 1, 1, 0, 0],
                ],
                "select_comm_recompute": [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                ],
            },
            show=False,
            interleave_num=3,
            file_name=None,
        )

        assert total_time == 51394.8
