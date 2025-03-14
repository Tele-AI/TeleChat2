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
"""layer"""

import os
import json
from enum import Enum

from toolkit.pipeline_balance.utils.logger import logger
import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.computation_analyzer import (ComputationAnalyzer,)


class Layer:
    """
    Mandatory parameter:
    name_ (str): name of the layer
    type_ (LayerType): type of the layer 'HEAD', 'BODY', 'TAIL'
    nb_layer_ (int): number of layer to schedule
    time_ (float): total time that a layer take

    Optional (auto-compute) parameter:
    forward_time_ (float): forward time for the layer (1/3 of time)
    backward_time_rec_ (dict[Recompute.Type, float]): backward time (2/3 of time) per recomputation
    recompute_considered_: dict[Recompute.Type, bool] set recomputations when considered

    Optional memory parameter (for recompute):
    memory_parameter_ (float): memory used by the layer (all kind)
    memory_activation_rec_ (dict[Recompute.Type, float]): activation memory per recomputation

    Not manage yet parameter (for multimodal):
    model_name_ (str): name of the model the layer be part of (for multimodal)
    """

    type_enum = Enum('LayerType', ['UNKNOWN', 'HEAD', 'BODY', 'TAIL'])
    backward_default_ratio = 2 # of forward time
    name_: str
    model_name_: str
    type_: type_enum
    nb_layer_: int
    time_: float
    memory_parameter_: float
    memory_activation_rec_: dict[Recompute.TYPE, float]
    forward_time_: float
    backward_time_rec_: dict[Recompute.TYPE, float]
    backward_coef_rec_: dict[Recompute.TYPE, float]
    recompute_considered_: dict[Recompute.TYPE, bool]

    def __init__(self, model_name: str = "misc", name: str = "misc", ltype: type_enum = type_enum.UNKNOWN,
                 nb_layer: int = 0, time: float = 0.0, forward_time: float = 0.0,
                 backward_time_rec: dict[Recompute.TYPE, float] = {r: 0 for r in Recompute.TYPE},
                 backward_coef_rec: dict[Recompute.TYPE, float] = {r: 0 for r in Recompute.TYPE},
                 memory_parameter: float = 0.0,
                 memory_activation_rec: dict[Recompute.TYPE, float] = {r: 0.0 for r in Recompute.TYPE}):
        self.name_ = name
        self.model_name_ = model_name
        self.type_ = ltype
        self.nb_layer_ = nb_layer
        self.time_ = time
        self.memory_activation_rec_ = memory_activation_rec
        self.memory_parameter_ = memory_parameter
        self.backward_time_rec_ = backward_time_rec
        self.backward_coef_rec_ = backward_coef_rec
        self.forward_time_ = forward_time
        self.recompute_considered_ = self.find_recompute_considered()
        self.compute_internal_time()

    def __str__(self) -> str:
        result = 'Layer Description:\n'
        result += '  name             = ' + self.name_ + '\n'
        result += '  model_name       = ' + str(self.model_name_) + '\n'
        result += '  type             = ' + self.type_.name + '\n'
        result += '  nb_layer         = ' + str(self.nb_layer_) + '\n'
        result += '  time             = ' + str(self.time_) + '\n'
        result += '  memory_parameter = ' + str(self.memory_parameter_) + '\n'
        for r in Recompute.TYPE:
            if self.recompute_considered_[r]:
                result += '  ' + Recompute.JSON_MEMORY_NAME[r] + ' = '
                result += str(self.memory_activation_rec_[r]) + '\n'
        result += '  forward_time     = '
        result += str(self.forward_time_) + '\n'
        for r in Recompute.TYPE:
            if self.recompute_considered_[r]:
                result += '  ' + Recompute.JSON_TIME_NAME[r] + ' = '
                result += str(self.backward_time_rec_[r]) + '\n'
        return result

    def dump(self, dump_file: str):
        """Dump json file for this specific layer"""
        logger.error("dump file (%s) Not implemented yet!!!", dump_file)

    def to_json(self):
        """Generate json representation of this object"""
        logger.error("Not implemented yet!!!")

    def find_recompute_considered(self):
        """Find the recomputation types considered"""
        recompute_considered = {i: False for i in range(len(Recompute.TYPE))}

        for rec in Recompute.TYPE:
            if (self.memory_activation_rec_[rec] is not None and self.memory_activation_rec_[
                    rec] > 0):
                recompute_considered[rec] = True

        return recompute_considered

    def compute_internal_time(self, back_ratio: float = (backward_default_ratio),
                              force_fb: bool = False):
        """Auto compute internal time if not already present"""
        if force_fb or self.forward_time_ is None:
            self.forward_time_ = self.time_
        self.backward_time_ = back_ratio * self.time_

        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                if (self.backward_time_rec_[rec] is None or self.backward_time_rec_[rec] == 0):
                    if self.backward_coef_rec_[rec] is None:
                        self.backward_time_rec_[rec] = (1 + Recompute.DEFAULT_COEF[
                            rec]) * self.backward_time_
                    else:
                        self.backward_time_rec_[rec] = (1 + self.backward_coef_rec_[
                            rec]) * self.backward_time_

    def update_internal_time_for_seqpp(self, back_ratio: float = (backward_default_ratio),
                                       force_fb: bool = False):
        """update compute internal time for seqpipe"""
        if force_fb or self.forward_time_ is None:
            self.forward_time_ = (1 - back_ratio) * self.time_
        self.backward_time_ = back_ratio * self.time_

        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                if self.backward_coef_rec_[rec] is None:
                    self.backward_time_rec_[rec] = (1 + Recompute.DEFAULT_COEF[
                        rec]) * self.backward_time_
                else:
                    self.backward_time_rec_[rec] = (1 + self.backward_coef_rec_[
                        rec]) * self.backward_time_

    def compute_timer(self, timeline_folder: str = './timeline', tmp_layer_info=None):
        """Compute the time information from the timeline logs"""
        layer_time = ComputationAnalyzer(timeline_folder, self.model_name_, num_of_stage=0,
                                         layer_list=tmp_layer_info)
        self.time_ = layer_time.layer_with_cost_list.get(self.name_)
        self.compute_internal_time(force_fb=True)

    def compute_memory(self, memory_folder: str = './memory'):
        """ "Compute the memory information from the (dry) run logs"""
        logger.error("compute_memory (%s) Not implemented yet!!!", memory_folder)


# Helper functions on layer list


def generate_layers_list(layer_folder: str, model_name: str) -> list[Layer]:
    """ "Parse layer_folder/model_name.json to generate a list of layer"""
    layers = []
    json_layer = os.path.join(layer_folder, model_name + '.json')
    with open(json_layer, encoding="utf-8") as json_file:
        layer_data_json = json.load(json_file)
        if "layers_description" in layer_data_json:
            for layer_data in layer_data_json["layers_description"]:
                new_layer = Layer(name=layer_data["name"], ltype=Layer.type_enum[layer_data["type"]],
                                  nb_layer=layer_data["nb_layer"], time=layer_data["time"],
                                  model_name=layer_data.get("model_name"),
                                  forward_time=layer_data.get("forward_time"),
                                  backward_time_rec={r: layer_data.get(Recompute.JSON_TIME_NAME[r])
                                                     for r in Recompute.TYPE},
                                  backward_coef_rec={r: layer_data.get(Recompute.JSON_COEF_NAME[r])
                                                     for r in Recompute.TYPE},
                                  memory_activation_rec={r: layer_data.get(Recompute.JSON_MEMORY_NAME[r])
                                                         for r in Recompute.TYPE},
                                  memory_parameter=layer_data.get("memory_parameter"),)
                new_layer.compute_internal_time()
                layers.append(new_layer)
        else:
            logger.error(
                'ERROR: File "%s" doesn\'t have layers_description to parse.\n', json_layer)
    return layers


def filter_layer_type(layers: list[Layer], layer_type: Layer.type_enum) -> list[Layer]:
    """Filters all layers of layer_type in layers."""
    kept_layers = []
    for layer in layers:
        if layer.type_ == layer_type:
            kept_layers.append(layer)
    return kept_layers


def aggregate(layers: list[Layer]) -> Layer:
    """Aggregate all layers into one."""

    def add_none(a, b):
        """Auxiliary function for aggregation with None values."""
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    def add_rec_dict(a, b):
        return {i: a[i] + b[i] for i in Recompute.TYPE}

    aggregation = layers[0]
    layers.pop(0)
    for layer in layers:
        aggregation.time_ += layer.time_
        aggregation.backward_time_rec_ = add_rec_dict(aggregation.backward_time_rec_,
                                                      layer.backward_time_rec_)
        aggregation.memory_activation_rec_ = add_rec_dict(aggregation.memory_activation_rec_,
                                                          layer.memory_activation_rec_)
        aggregation.memory_parameter_ = add_none(aggregation.memory_parameter_, layer.memory_parameter_)
        aggregation.nb_layer_ = add_none(aggregation.nb_layer_, layer.nb_layer_)
    return aggregation
