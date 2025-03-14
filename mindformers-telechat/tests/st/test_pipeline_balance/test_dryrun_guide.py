# Copyright 2025 Huawei Technologies Co., Ltd
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
Test module for testing the dryrun guide of pipeline balancing.
How to run this:
pytest tests/st/test_pipeline_balance/test_dryrun_guide.py
"""

import numpy as np

from toolkit.pipeline_balance.sapp.sapp_solver import SappSolver
from toolkit.pipeline_balance.utils.config import Recompute
from toolkit.pipeline_balance.utils.config import generate_solvable_config


class TestDryunGuide:
    """A test class for testing dryrun guide."""

    def test_dryrun_guide_one_round(self):
        """
        Feature: TestDryunGuide.
        Description: Test the dryrun guide.
        Expectation: correct
        """
        considered_rec = [Recompute.TYPE.BOTH, Recompute.TYPE.SLCT, Recompute.TYPE.COMM]
        offset_config_list, rec_config_list = generate_solvable_config(16, 17, considered_rec)
        activation_nums = SappSolver.compute_activation_nums(16, 1, 0)[0]
        layer_per_stage = 1
        coef_matrix = []
        rounds = len(offset_config_list)
        for round_ in range(rounds):
            for stage in range(16):
                if stage not in [0, 16 - 1]:
                    coef_matrix.append(
                        [1, layer_per_stage + offset_config_list[round_][stage]]
                        + Recompute.to_list(
                            {
                                rec: rec_config_list[round_][rec][stage]
                                     * activation_nums[stage]
                                for rec in considered_rec
                            }
                        )
                    )
                if len(coef_matrix) == 2 + len(considered_rec):
                    coef_rank = np.linalg.matrix_rank(coef_matrix)
                    assert coef_rank == len(considered_rec) + 2


    def test_dryrun_guide_multi_rounds(self):
        """
        Feature: TestDryunGuide.
        Description: Test the dryrun guide.
        Expectation: correct
        """
        considered_rec = [Recompute.TYPE.BOTH, Recompute.TYPE.SLCT, Recompute.TYPE.COMM]
        offset_config_list, rec_config_list = generate_solvable_config(4, 5, considered_rec)
        activation_nums = SappSolver.compute_activation_nums(4, 1, 0)[0]
        layer_per_stage = 1
        coef_matrix = []
        rounds = len(offset_config_list)
        for round_ in range(rounds):
            for stage in range(4):
                if stage not in [0, 4 - 1]:
                    coef_matrix.append(
                        [1, layer_per_stage + offset_config_list[round_][stage]]
                        + Recompute.to_list(
                            {
                                rec: rec_config_list[round_][rec][stage]
                                     * activation_nums[stage]
                                for rec in considered_rec
                            }
                        )
                    )
                if len(coef_matrix) == 2 + len(considered_rec):
                    coef_rank = np.linalg.matrix_rank(coef_matrix)
                    assert coef_rank == len(considered_rec) + 2
