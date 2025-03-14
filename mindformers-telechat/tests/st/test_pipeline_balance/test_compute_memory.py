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
pytest tests/st/test_pipeline_balance/test_compute_memory.py
"""
import os

from toolkit.pipeline_balance.utils.compute_memory import ComputeMemory
from toolkit.pipeline_balance.utils.config import memory_parser
from toolkit.pipeline_balance.utils.stage import Stage
import toolkit.pipeline_balance.utils.recompute as Recompute


class TestComputeMemory:
    """A test class for testing compute memory."""

    def test_compute_memory(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the computation of memory of pipeline balancing.
        Expectation: correct
        """
        num_stage = 16
        per_stage_layer_num = 6
        stage_head = Stage(
            sid=0,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 1, Recompute.TYPE.FULL: 1},
            memory_usage=80267,
        )
        stage_1 = Stage(
            sid=1,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=71519,
        )
        stage_2 = Stage(
            sid=2,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=67376,
        )
        stage_3 = Stage(
            sid=3,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 2},
            memory_usage=52962,
        )
        stage_4 = Stage(
            sid=9,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 2, Recompute.TYPE.FULL: 0},
            memory_usage=39373,
        )
        stage_tail = Stage(
            sid=num_stage - 1,
            nb_stage=num_stage,
            nb_layer=per_stage_layer_num,
            nb_layer_rec={Recompute.TYPE.COMM: 0, Recompute.TYPE.FULL: 1},
            memory_usage=16386,
        )

        stages_a = [stage_1, stage_2, stage_3, stage_4, stage_head, stage_tail]

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)
        memory_head = int(comp_mem.get_memory_head())
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_select_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_recompute = int(comp_mem.get_memory_activation(Recompute.TYPE.FULL))
        assert memory_head == 9785, "memory_head: wrong answer"
        assert memory_parameter == 1562, "memory_parameter: wrong answer"
        assert memory_activation == 822, "memory_activation: wrong answer"
        assert memory_tail == 2868, "memory_tail: wrong answer"
        assert memory_recompute == 32, "memory_recompute: wrong answer"
        assert memory_select_comm == 498, "memory_select_comm: wrong answer"

    def test_compute_memory_with_const(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the computation of memory of pipeline balancing.
        Expectation: correct
        """
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "./mem.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)
        memory_head = int(comp_mem.get_memory_head())
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_both = int(comp_mem.get_memory_activation(Recompute.TYPE.BOTH))
        memory_slct = int(comp_mem.get_memory_activation(Recompute.TYPE.SLCT))
        memory_const = int(comp_mem.get_memory_const())

        assert memory_head == 4245, "memory_head: wrong answer"
        assert memory_parameter == 2141, "memory_parameter: wrong answer"
        assert memory_activation == 852, "memory_activation: wrong answer"
        assert memory_tail == 1203, "memory_tail: wrong answer"
        assert memory_both == 511, "memory_both: wrong answer"
        assert memory_comm == 535, "memory_comm: wrong answer"
        assert memory_slct == 767, "memory_slct: wrong answer"
        assert memory_const == -1896, "memory_const: wrong answer"

    def test_compute_memory_pp4(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the computation of memory of pipeline balancing.
        Expectation: correct
        """
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "./mem_pp4.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_full = int(comp_mem.get_memory_activation(Recompute.TYPE.FULL))
        memory_head = int(comp_mem.get_memory_head())

        assert memory_head == 1459, "memory_head: wrong answer"
        assert memory_parameter == 400, "memory_parameter: wrong answer"
        assert memory_activation == 743, "memory_activation: wrong answer"
        assert memory_full == 131, "memory_full: wrong answer"
        assert memory_tail == 3329, "memory_tail: wrong answer"

    def test_compute_memory_lstsq(self):
        """
        Feature: TestPipelineBalance.
        Description: Test the computation of memory of pipeline balancing.
        Expectation: correct
        """
        work_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(work_path, "./mem_lstsq.yaml")
        num_stage, stages_a, _ = memory_parser(file_path)

        comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)
        memory_parameter = int(comp_mem.get_memory_parameter())
        memory_tail = int(comp_mem.get_memory_tail())
        memory_activation = int(comp_mem.get_memory_activation(Recompute.TYPE.NONE))
        memory_slct = int(comp_mem.get_memory_activation(Recompute.TYPE.SLCT))
        memory_comm = int(comp_mem.get_memory_activation(Recompute.TYPE.COMM))
        memory_both = int(comp_mem.get_memory_activation(Recompute.TYPE.BOTH))

        memory_head = int(comp_mem.get_memory_head())

        assert memory_head == 4614, "memory_head: wrong answer"
        assert memory_parameter == 2217, "memory_parameter: wrong answer"
        assert memory_activation == 1002, "memory_activation: wrong answer"
        assert memory_both == 423, "memory_both: wrong answer"
        assert memory_slct == 783, "memory_slct: wrong answer"
        assert memory_comm == 695, "memory_comm: wrong answer"
        assert memory_tail == 5885, "memory_tail: wrong answer"
