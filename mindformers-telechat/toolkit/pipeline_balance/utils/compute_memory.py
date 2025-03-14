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
"""compute memory"""

import numpy as np

from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.stage import Stage, filter_stage_id
from toolkit.pipeline_balance.utils.layer import Layer
import toolkit.pipeline_balance.utils.recompute as Recompute


class ComputeMemory:
    """
    ComputeMemory class to compute the different memories with stages information running (dry) log

    stage{A|B} means stage with different configuration A and B
    stage{1|2} means stage same configuration but different id (can be id other than 1 or 2)

    number_of_stage_ (int): number of stages for the LLM
    stagesA_ (list[Stage]): list of dry run stages information, with all the same configuration A,
            required at least staged 0, 1, (n-2), (n-1)
            Don't set directly stagesA_, but use set_stagesA
    stagesB_ (list[Stage]): list of dry run stages information, with all the same configuration B,
            different from config A required at least staged 0, 1, (n-2), (n-1)
            Don't set directly stagesB_, but use set_stagesB
    memory_parameter_ (float): memory_parameter_ of the BODY layer, memory required to run the layer
    memory_activation_rec_ (dict[Recompute.TYPE, float]) activation memory per recompute types
    recompute_considered_ (dict[Recompute.TYPE, bool]) recomputation types taken into consideration
    memory_const_ (float): constant memory required for each stages
    memory_head_ (float): memory required to run the head layer
    memory_tail_ (float): memory required to run the tail layer
    """

    number_of_stage_: int
    stages_a: list[Stage]
    stages_b: list[Stage]
    memory_parameter_: float
    memory_activation_rec_: dict[Recompute.TYPE, float]
    recompute_considered_: dict[Recompute.TYPE, bool]
    memory_const_: float
    memory_head_: float
    memory_tail_: float

    def __init__(self, number_of_stage: int, stages_A: list[Stage] = None,
                 stages_B: list[Stage] = None,):
        """
        number_of_stage: total number of stage for the LLM
        stagesA: some dry run stages information required at least head and tail stages
                and 2 other stages (stage: 0, i, j, (n-1)) with all the same configuration A
        stagesB: some dry run stages information required at least head and tail stages and 2
                other stages (stage: 0, i, j, (n-1)) with all the same configuration B,
                but different from A
        """
        self.number_of_stage_ = number_of_stage
        self.set_stages_a(stages_A)
        self.set_stages_b(stages_B)
        # number_of_stage != len(stages) can be true
        self.memory_parameter_ = None
        self.memory_activation_rec_ = {r: None for r in Recompute.TYPE}
        self.find_recompute_considered()
        self.memory_const_ = None
        self.memory_head_ = None
        self.memory_tail_ = None

    def set_stages_a(self, stages: list[Stage]):
        """set stage A"""
        if stages is None:
            self.stages_a = []
            return
        for stage1 in stages:
            for stage2 in stages:
                if not stage1.same_global_config(stage2):
                    logger.error(
                        "Cannot set stagesA, all elements don't have the same configuration",)
                    self.stages_a = []
                    return
        self.stages_a = stages

    def set_stages_b(self, stages: list[Stage]):
        """set stage B"""
        if stages is None:
            self.stages_b = []
            return
        for stage1 in stages:
            for stage2 in stages:
                if not stage1.same_global_config(stage2):
                    logger.error(
                        "Cannot set stagesB, all elements don't have the same configuration")
                    self.stages_b = []
                    return
            for stage_a in self.stages_b:
                if stage1.same_global_config(stage_a):
                    logger.error(
                        "Cannot set stagesB, an elements have the same configuration than stagesA")
                    self.stages_b = []
                    return
        self.stages_b = stages

    def find_recompute_considered(self):
        """Finds what type of recomputation can be considered"""
        self.recompute_considered_ = {r: False for r in Recompute.TYPE}
        self.recompute_considered_[Recompute.TYPE.NONE] = True

        for stage in self.stages_a:
            for rec in Recompute.TYPE:
                if stage.nb_layer_rec_[rec] > 0:
                    self.recompute_considered_[rec] = True

    def _compute_memory_parameter_local_(self, stage1: Stage, stage2: Stage) -> float:
        """
        Given 2 stages information with the same configuration, and different id,
        Compute the memory_parameter
        """
        if stage1.same_config(stage2):
            if stage1.id_ != stage2.id_:
                res = stage1.memory_usage_ * (stage1.nb_stage_ - stage1.id_)
                res -= stage2.memory_usage_ * (stage2.nb_stage_ - stage2.id_)
                res /= stage1.id_ - stage2.id_
                res = abs(res)
                res /= stage1.nb_layer_
                return res
            logger.error(
                "stage with same characteristic, BUT SAME ID too, cannot compute memory_parameter")
            return 0
        logger.error("stage with different characteristic, cannot compute memory_parameter")
        return 0

    def _compute_memory_parameter_(self, multi_run=False) -> float:
        """Compute memory_parameter
            With all available stages compute all combinations of memory parameter
            and return the mean of all the memory_parameter found
        BEWARE: can update memory_parameter_ & memory_activation_rec_
                because of _compute_memories_layers_()
        return: memory_parameter
        """
        if multi_run or (len(self.stages_a) < 5 and len(self.stages_b) < 5):
            memory_parameter_list = []
            for stage1 in self.stages_a:
                if stage1.id_ not in [0, (self.number_of_stage_ - 1)]:
                    mem_param = self._compute_memory_parameter_local_(stage1, stage2)
                    for stage2 in self.stages_a:
                        if (stage2.id_ not in [0, (self.number_of_stage_ - 1),
                                               stage1.id_] and mem_param != 0):
                            memory_parameter_list.append(mem_param)
            for stage1 in self.stages_b:
                if stage1.id_ not in [0, (self.number_of_stage_ - 1)]:
                    for stage2 in self.stages_b:
                        mem_param = self._compute_memory_parameter_local_(stage1, stage2)
                        if (stage2.id_ not in [0, (self.number_of_stage_ - 1),
                                               stage1.id_] and mem_param != 0):
                            memory_parameter_list.append(mem_param)
            return np.mean(memory_parameter_list)
        if self._compute_memories_layers_():
            return self.memory_parameter_
        logger.error("Issue with _compute_memory_parameter_!!!")
        return 0

    def _compute_memory_activation_(self, rec, multi_run=False) -> float:
        """
        Compute memory_activation for a given recomputation type
        return: memory_activation
        """
        if multi_run or (len(self.stages_a) < 5 and len(self.stages_b) < 5):
            # look at solution 4 stages
            logger.error("Not implemented yet!!!")
            return 0
        if self._compute_memories_layers_():
            return self.memory_activation_rec_[rec]
        logger.error("Issue with _compute_memory_activation_!!!")
        return 0

    def zero_offset(self):
        nb_layer = self.stages_a[0].nb_layer_
        for s in self.stages_a:
            if s.nb_layer_ != nb_layer:
                return False
        return True

    def _compute_memories_layers_(self) -> bool:
        """check if enough stage number is provided"""
        used_rec = Recompute.get_used_list(self.recompute_considered_)
        used_rec_num = len(used_rec)
        stage_num = len(self.stages_a)
        if stage_num == used_rec_num + 3:
            return self._compute_memories_layer_bodies_(False)
        if stage_num >= used_rec_num + 4:
            logger.info("Enabled const memory component because enough stages were given")
            if self.zero_offset():
                logger.error(
                    "The number of layer per stage cannot be the same for all stages "
                    "when const component is enabled. Some offset must be used"
                )
                return False
            return self._compute_memories_layer_bodies_(True)

        logger.error(f"{stage_num} stages found and ({used_rec_num}) recomputation considered"
                     "is not coherent. There should be 3 or 4 more stages than recomputation considered")
        return False

    def _compute_memories_layer_bodies_local_(
            self, unused_rec: list[Recompute.TYPE],
            stages: list[Stage]) -> tuple[float, float, float]:
        """Compute memory_parameter & memory activation for all recomputation types
        Require at least 3 Stages different from first and last stage
        """
        variable_factor_list = []
        constant_memory_list = []
        unused_rec.sort(reverse=True)
        for stage in stages:
            if stage.id_ not in [0, self.number_of_stage_ - 1]:
                variable_factor_list.append(stage.get_index_memory_var())
                for rec_i in unused_rec:
                    variable_factor_list[-1].pop(1 + rec_i)
                constant_memory_list.append(stage.memory_usage_)
        solution = list(
            np.linalg.solve(np.array(variable_factor_list),
                            np.array(constant_memory_list)))
        memory_param = solution.pop(0)
        memory_act_rec = Recompute.assign_used(solution, unused_rec)
        return (memory_param, memory_act_rec)



    def _compute_memories_layer_bodies_local_with_fix_(
            self, unused_rec: list[Recompute.TYPE],
            stages: list[Stage]) -> tuple[float, float, float]:
        """Compute memory_const, memory_parameter & memory activation for all recomputation types
        Require at least 4 Stages different from first and last stage
        """
        variable_factor_list = []
        constant_memory_list = []
        unused_rec.sort(reverse=True)
        for stage in stages:
            if stage.id_ not in [0, self.number_of_stage_ - 1]:
                variable_factor_list.append([1] + stage.get_index_memory_var())
                for rec_i in unused_rec:
                    variable_factor_list[-1].pop(2 + rec_i)
                constant_memory_list.append(stage.memory_usage_)
        logger.debug("solve("
                     f"\n {np.array(variable_factor_list)}, "
                     f"\n {np.array(constant_memory_list)}) ")
        used_rec = Recompute.get_used_list(self.recompute_considered_)
        used_rec_num = len(used_rec)

        if len(stages) == used_rec_num + 4:
            solution = list(
                np.linalg.solve(np.array(variable_factor_list),
                                np.array(constant_memory_list)))
        if len(stages) > used_rec_num + 4:
            logger.warning("Stages given are more than needed, switch to least sqaures method")
            solution = list(np.linalg.lstsq(np.array(variable_factor_list),
                                            np.array(constant_memory_list), rcond=None)[0])

        memory_const = solution.pop(0)
        memory_param = solution.pop(0)
        memory_act_rec = Recompute.assign_used(solution, unused_rec)
        return (memory_const, memory_param, memory_act_rec)

    def _compute_memories_layer_bodies_(self, with_fix: bool) -> bool:
        """
        Compute memory_parameter, memory_recompute, memory_activation
        Require at least 3 Stages different from first and last stage
        BEWARE: can update memory_parameter_, memory_recompute_, memory_activation_
        return True if success to update memory_parameter_, memory_recompute_, memory_activation_
        """

        memory_const_a = None
        memory_parameter_a = None
        memory_recompute_a = {r: None for r in Recompute.TYPE}

        memory_const_b = None
        memory_parameter_b = None
        memory_recompute_b = {r: None for r in Recompute.TYPE}

        unused_rec = Recompute.get_unused_list(self.recompute_considered_)
        logger.info(f"unused recomputation: {unused_rec}")

        if with_fix:
            if len(self.stages_a) >= 5:
                (memory_const_a,
                 memory_parameter_a,
                 memory_recompute_a) = (self._compute_memories_layer_bodies_local_with_fix_(
                     unused_rec, self.stages_a))
            if len(self.stages_b) >= 5:
                (memory_const_b,
                 memory_parameter_b,
                 memory_recompute_b) = (self._compute_memories_layer_bodies_local_with_fix_(
                     unused_rec, self.stages_b))

            return self._average_if_needed_fix(
                memory_const_a,
                memory_parameter_a,
                memory_recompute_a,
                memory_const_b,
                memory_parameter_b,
                memory_recompute_b,
            )
        if len(self.stages_a) >= 5:
            (memory_parameter_a,
             memory_recompute_a) = (self._compute_memories_layer_bodies_local_(
                 unused_rec, self.stages_a))
        if len(self.stages_b) >= 5:
            (memory_parameter_b,
             memory_recompute_b) = (self._compute_memories_layer_bodies_local_(
                 unused_rec, self.stages_b))

        return self._average_if_needed(
            memory_parameter_a,
            memory_recompute_a,
            memory_parameter_b,
            memory_recompute_b,
        )

    def _average_if_needed_fix(
            self,
            memory_const_a,
            memory_parameter_a,
            memory_recompute_a,
            memory_const_b,
            memory_parameter_b,
            memory_recompute_b,
    ):
        """check if average is needed"""
        if memory_parameter_a is not None and memory_parameter_a != 0:
            if memory_parameter_b is not None and memory_parameter_b != 0:
                self.memory_const_ = (memory_const_a +
                                      memory_const_b) / 2
                self.memory_parameter_ = (memory_parameter_a +
                                          memory_parameter_b) / 2
                Recompute.average([memory_recompute_a, memory_recompute_b])
            else:
                self.memory_const_ = memory_const_a
                self.memory_parameter_ = memory_parameter_a
                self.memory_activation_rec_ = memory_recompute_a

        elif memory_parameter_b is not None and memory_parameter_b != 0:
            self.memory_const_ = memory_const_b
            self.memory_parameter_ = memory_parameter_b
            self.memory_activation_rec_ = memory_recompute_b
        else:
            logger.error("failed to compute memories")
            return False
        return True

    def _average_if_needed(self, memory_parameter_a, memory_recompute_a, memory_parameter_b,
                           memory_recompute_b,):
        """check if average is needed"""
        if memory_parameter_a is not None and memory_parameter_a != 0:
            if memory_parameter_b is not None and memory_parameter_b != 0:
                self.memory_parameter_ = (memory_parameter_a + memory_parameter_b) / 2
                Recompute.average([memory_recompute_a, memory_recompute_b])
            else:
                self.memory_parameter_ = memory_parameter_a
                self.memory_activation_rec_ = memory_recompute_a

        elif memory_parameter_b is not None and memory_parameter_b != 0:
            self.memory_parameter_ = memory_parameter_b
            self.memory_activation_rec_ = memory_recompute_b
        else:
            logger.error("failed to compute memories")
            return False
        return True

    def _compute_memory_head_(self) -> float:
        """compute the memory for the head"""
        head_stages = filter_stage_id(self.stages_a, 0)
        head_stages += filter_stage_id(self.stages_b, 0)
        memory_head_list = []
        mem_parameter = self.get_memory_parameter()
        for head in head_stages:
            head_memory = head.memory_usage_
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec] is True:
                    head_memory -= (head.nb_layer_rec_[rec] * self.get_memory_activation(
                        rec) * self.number_of_stage_)
            head_memory -= (head.nb_layer_) * mem_parameter
            memory_head_list.append(head_memory)
        return np.mean(memory_head_list)

    def _compute_memory_tail_(self) -> float:
        """compute the memory for the tail"""
        tail_stages = filter_stage_id(self.stages_a, self.number_of_stage_ - 1)
        tail_stages += filter_stage_id(self.stages_b, self.number_of_stage_ - 1)
        memory_tail_list = []
        for tail in tail_stages:
            tail_memory = tail.memory_usage_
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec] is True:
                    tail_memory -= (tail.nb_layer_rec_[rec] * self.get_memory_activation(rec) * 1)
            tail_memory -= (tail.nb_layer_) * self.get_memory_parameter()
            memory_tail_list.append(tail_memory)
        return np.mean(memory_tail_list)

    def get_memory_const(self) -> float:
        return self.memory_const_

    def get_memory_parameter(self, force_recompute=False) -> float:
        """get the parameter memory"""
        if force_recompute or self.memory_parameter_ is None:
            self.memory_parameter_ = self._compute_memory_parameter_()
        return self.memory_parameter_

    def get_memory_activation(self, rec, force_recompute=False) -> float:
        """get the activation memory"""
        if force_recompute or self.memory_activation_rec_[rec] is None:
            self.memory_activation_rec_[rec] = self._compute_memory_activation_(rec)
        return self.memory_activation_rec_[rec]

    def get_memory_head(self, force_recompute=False) -> float:
        if force_recompute or self.memory_head_ is None:
            self.memory_head_ = self._compute_memory_head_()
        return self.memory_head_

    def get_memory_tail(self, force_recompute=False) -> float:
        if force_recompute or self.memory_tail_ is None:
            self.memory_tail_ = self._compute_memory_tail_()
        return self.memory_tail_


def compute_memories(layers: list[Layer], memory_folder: str, number_of_stage: int) -> list[Layer]:
    """compute memories"""
    filename = ""
    # Put some meta information in a predefine .json file like layers info?
    with open(memory_folder + filename, encoding="utf-8"):
        pass
    cm = ComputeMemory(number_of_stage=number_of_stage, stages_A=[], stages_B=[])

    for layer in layers:
        if layer.type_ == Layer.type_enum.HEAD:
            layer.memory_parameter_ = cm.get_memory_head()
        elif layer.type_ == Layer.type_enum.TAIL:
            layer.memory_parameter_ = cm.get_memory_tail()
        elif layer.type_ == Layer.type_enum.BODY:
            layer.memory_parameter_ = cm.get_memory_parameter()
            for rec in Recompute.TYPE:
                layer.memory_activation_rec_[rec] = cm.get_memory_activation(rec)
    return layers
