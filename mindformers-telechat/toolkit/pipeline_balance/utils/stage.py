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
"""stage"""

from toolkit.pipeline_balance.utils.error import _assert_sapp
import toolkit.pipeline_balance.utils.recompute as Recompute


class Stage:
    """Stage Class to describe a run from a log

    id_ (int): stage id of the run
    nb_stage_ (int): total number of stage present
    nb_layer_ (int): total number of layer present to distribute for this stage id
    nb_layer_rec_ (dict[Recompute.Type, int]): number of recomputed layer
        per recomputation type for this stage id
    memory_usage_ (int): memory usage of the run for this stage

    Properties:
    nb_layer_ == (nb_recompute+nb_select_rec+nb_norecompute)
    id_ < nb_stage_
    """

    id_: int
    nb_stage_: int
    nb_layer_: int
    nb_layer_rec_: dict[Recompute.TYPE, int]
    memory_usage_: int

    def __init__(self, sid: int, nb_stage: int, nb_layer: int,
                 nb_layer_rec: dict[Recompute.TYPE, int], memory_usage: int):
        self.id_ = sid
        self.nb_stage_ = nb_stage
        self.nb_layer_ = nb_layer
        self.nb_layer_rec_ = self.complete_nb_layer_rec_(nb_layer_rec)
        self.memory_usage_ = memory_usage
        _assert_sapp(nb_layer == Recompute.sums(nb_layer_rec),
                     "init stage, nb_layer == (nb_recompute+nb_norecompute)")
        _assert_sapp(sid < nb_stage, "init stage, id < nb_stage")

    def complete_nb_layer_rec_(self, nb_layer_rec: dict[Recompute.TYPE, int]):
        """Complete the number of layers recomputed from partial filling"""
        sum_layers = 0
        for r in Recompute.TYPE:
            if r is not Recompute.TYPE.NONE:
                if r not in nb_layer_rec:
                    nb_layer_rec[r] = 0
                else:
                    sum_layers += nb_layer_rec[r]

        if Recompute.TYPE.NONE not in nb_layer_rec:
            nb_layer_rec[Recompute.TYPE.NONE] = self.nb_layer_ - sum_layers

        return nb_layer_rec

    def same_config(self, other: 'Stage') -> bool:
        """
        Check if self and other have same configuration
        same number of total layers, number of total stages, recompute layers and no recompute
        layers
        """
        return (
            self.nb_layer_ == other.nb_layer_ and self.nb_stage_ == other.nb_stage_ and
            self.nb_layer_rec_ == other.nb_layer_rec_)

    def same_global_config(self, other: 'Stage') -> bool:
        """
        Check if self and other have same configuration
        same number of total layers and number of total stages
        """
        return self.nb_stage_ == other.nb_stage_

    def get_index_memory_var(self) -> list[int]:
        """
        Returns memory factors for parameter and
        activation for all recomputation types
        """
        diff = self.nb_stage_ - self.id_
        return [self.nb_layer_] + Recompute.to_list(
            {r: self.nb_layer_rec_[r] * diff for r in Recompute.TYPE})


def filter_stage_id(stages: list[Stage], stage_id: int) -> list[Stage]:
    """Filters all stages of stage_id in stages."""
    kept_stages = []
    for s in stages:
        if s.id_ == stage_id:
            kept_stages.append(s)
    return kept_stages
