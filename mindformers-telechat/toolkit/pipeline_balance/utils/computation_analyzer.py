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

"""Computation cost analyzer for pipeline balancing."""

import json
import os
import re
import sys
from itertools import chain

from tqdm import tqdm

from toolkit.pipeline_balance.utils.logger import logger

UNSTABLE_STEPS = 2


class ComputationAnalyzer:
    """Parser & Analyzer for profiling timelines"""

    is_msprof_file = False

    def __init__(self, timeline_folder_path, model_name, num_of_micro_batch, layer_list=None):
        self.timeline_folder_path = timeline_folder_path
        self.model_name = model_name
        self.num_of_micro_batch = num_of_micro_batch
        self.counted_steps = 0
        self.step_time = 0.0
        if layer_list:
            self.layer_list = layer_list
        else:
            self.layer_list = self._get_layer_list()
        self.timeline_data = self._get_timeline_data()
        logger.info("parsing layer objs")
        self.auto_partition_layer_objects, self.pre_defined_layer_objects = (
            self._parse_layer_objects())
        logger.info("parsing auto partition layer name")
        self.auto_partition_layer_name_list = (self.parse_auto_partition_layer_name_list())
        logger.info("parse layer with computation time list")
        self.layer_with_computation_time_list = (self.parse_layer_with_computation_time_list())
        logger.info("transform layer with cost list")
        self.layer_with_cost_list = self.transform_layer_with_cost_list()

    def _get_layer_list(self):
        """Return cfgs from model config file"""

        model_config_file = os.path.join(os.getcwd(), "cfgs", "model_layers.json")
        with open(model_config_file) as json_file:
            model_layers_data = json.load(json_file)
            for layer_list in model_layers_data:
                if self.model_name in layer_list["name"]:
                    return layer_list
        logger.info("ERROR: Not found model in model config file")
        return False

    def _get_timeline_data(self):
        """Return timeline objects from json file."""
        logger.info("loading timeline data")
        timeline_data = []
        for file_name in [file for file in os.listdir(self.timeline_folder_path) if
                          file.endswith(".json")]:
            if file_name.startswith("trace_view"):
                self.is_msprof_file = True
            elif file_name.startswith("msprof"):
                self.is_msprof_file = True
            else:
                self.is_msprof_file = False
                logger.error("ERROR: Not support timeline file type")
            with open(os.path.join(self.timeline_folder_path, file_name)) as json_file:
                timeline_data.append(json.load(json_file))
        return timeline_data

    def _parse_step_duration(self, timeline_data):
        """Return timeline objects during a training step."""

        op_name = ""
        step_start = 0.0
        step_end = 0.0
        cpt = 0
        for obj in timeline_data:
            if "MatMul-op" in obj["name"]:
                op_name = obj["name"]
                break
        for obj in timeline_data:
            if obj["name"] == op_name:
                cpt += 1
                if cpt == self.select_step_number:
                    step_start = float(obj["ts"])
                if cpt == (self.select_step_number + 1):
                    step_end = float(obj["ts"])
        step_time = step_end - step_start
        self.step_time = step_time
        return (step_start, step_end)

    def _load_json_data(self, file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    def _initialize_step_duration(self, timeline_data, step_start, step_end):
        if step_start == 0 or step_end == 0:
            step_start, step_end = self._parse_step_duration(timeline_data)
        return step_start, step_end

    def _add_layer_object(self, objects_list, condition, obj):
        if condition and obj not in objects_list:
            objects_list.append(obj)

    def _is_counted(self, default_table: list, step_start, step_end, cell_object):
        """Check if cell in under forward scope"""
        if float(cell_object["ts"]) < step_start or float(cell_object["ts"]) + float(cell_object["dur"]) > step_end:
            return False

        is_counted = False
        for duration in default_table:
            start = float(cell_object["ts"])
            end = float(cell_object["ts"]) + float(cell_object["dur"])
            if start >= duration[0] and end <= duration[1]:
                is_counted = True
                break
        return is_counted

    def _forward_parser(self, timeline_data):
        """Parse time range of forward operators"""
        logger.info("parsing forward scope")
        scope_pid = 3
        default_durations = []
        cell_durations = {}
        step_range = []
        op_name = ""
        for obj in timeline_data:
            if obj["name"] == "Scope Layer":
                scope_pid = obj["pid"]
                break
        for obj in tqdm(timeline_data):
            if op_name == "" and "MatMul-op" in obj["name"]:
                op_name = obj["name"]
            if obj["name"] == op_name:
                step_range.append(float(obj["ts"]))
            if obj["pid"] != scope_pid:
                continue
            if obj["name"] == "Default" and obj["tid"] == 0:
                start = float(obj["ts"])
                end = float(obj["ts"]) + float(obj["dur"])
                default_durations.append((start, end))
                continue
            for layer_name in chain(self.layer_list["pre_defined_layer"], self.layer_list["auto_partition_layer"]):
                if layer_name in obj["name"]:
                    layer_time = cell_durations.get(layer_name)
                    if layer_time is None:
                        cell_durations[layer_name] = []
                    cell_durations[layer_name].append(obj)

        # step times of first 2 steps are not stable
        # so we don't consider them when enough steps are given
        steps = len(step_range)
        logger.info(f"There are {steps} steps in given timeline data")
        if steps == 0:
            raise ValueError("Failed to parse timeline")
        if steps == 1:
            select_step_start = 0.0
            select_step_end = sys.float_info.max
        else:
            select_step_start = step_range[min(len(step_range) - UNSTABLE_STEPS, UNSTABLE_STEPS)]
            select_step_end = step_range[-1]
        logger.info("select_step_start: %f", select_step_start)
        logger.info("select_step_end: %f", select_step_end)
        self.counted_steps = max(len(step_range) - (UNSTABLE_STEPS + 1), 1)
        logger.info("counted_steps: %f", self.counted_steps)
        return default_durations, cell_durations, select_step_start, select_step_end

    def _process_timelines(self, timeline_data, step_start, step_end, pre_defined_layer_objects,
                           auto_partition_layer_objects):
        """_process_file"""
        logger.info("processing timeline. %d objects in it.", len(timeline_data))
        if not self.is_msprof_file:
            step_start, step_end = self._initialize_step_duration(self.timeline_data, step_start, step_end)
        default_durations, cell_durations, step_start, step_end = self._forward_parser(timeline_data)
        for cell_name, cell_objs in cell_durations.items():
            for obj in cell_objs:
                if not self._is_counted(default_durations, step_start, step_end, obj):
                    continue
                for layer_name in self.layer_list["pre_defined_layer"]:
                    if layer_name in cell_name:
                        self._add_layer_object(pre_defined_layer_objects, True, obj)
                for layer_name in self.layer_list["auto_partition_layer"]:
                    if layer_name in cell_name:
                        self._add_layer_object(auto_partition_layer_objects, True, obj)
        return step_start, step_end

    def _parse_layer_objects(self):
        auto_partition_layer_objects = []
        pre_defined_layer_objects = []
        step_start, step_end = 0, 0
        for timeline in self.timeline_data:
            step_start, step_end = self._process_timelines(timeline, step_start, step_end,
                                                           pre_defined_layer_objects,
                                                           auto_partition_layer_objects)
        return auto_partition_layer_objects, pre_defined_layer_objects

    def parse_auto_partition_layer_name_list(self):
        """example: [42-TransformerEncoderLayer,43-TransformerEncoderLayer]"""
        auto_partition_layer_name_list = []
        for auto_partition_name in self.layer_list["auto_partition_layer"]:
            for obj in [item for timeline in self.timeline_data for item in timeline]:
                object_name = obj["name"]
                if auto_partition_name in object_name:
                    if self.is_msprof_file:
                        find_layer_name = re.findall(r"[0-9]*-" + auto_partition_name,
                                                     object_name)
                        layer_name = find_layer_name[0]
                    else:
                        layer_name = object_name
                    if layer_name not in auto_partition_layer_name_list:
                        auto_partition_layer_name_list.append(layer_name)
        return auto_partition_layer_name_list

    def parse_layer_with_computation_time_list(self):
        """
        Map each layer_name with its duration time. 例
        如[46-TransformerEncoderLayer':
        37.24729124999999, '47-TransformerEncoderLayer': 37.36572429687501]
        """
        layer_with_computation_time_list = {}
        for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
            layer_with_computation_time_list[pre_defined_layer_name] = 0
        for auto_partition_layer_name in self.auto_partition_layer_name_list:
            layer_with_computation_time_list[auto_partition_layer_name] = 0

        for obj in self.pre_defined_layer_objects:
            for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
                if pre_defined_layer_name in obj["name"]:
                    layer_with_computation_time_list[pre_defined_layer_name] += (float(obj["dur"]) / 1000)
        for obj in self.auto_partition_layer_objects:
            for auto_partition_layer_name in self.auto_partition_layer_name_list:
                # if auto_partition_layer_name in obj["name"]:
                if re.search(r"\b" + re.escape(auto_partition_layer_name), obj["name"]):
                    layer_with_computation_time_list[auto_partition_layer_name] += (
                        float(obj["dur"]) / 1000)
        return layer_with_computation_time_list

    def transform_layer_with_cost_list(self):
        """calculating the average value of layer cost"""
        total_cost_auto_partition_layer = {}
        number_of_auto_partition_layer = {}
        transform_layer_with_cost_list = {}
        for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
            transform_layer_with_cost_list[pre_defined_layer_name] = 0

        for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
            total_cost_auto_partition_layer[auto_partition_layer_name] = 0
            number_of_auto_partition_layer[auto_partition_layer_name] = 0
            transform_layer_with_cost_list[auto_partition_layer_name] = 0

        # test
        for layer_name in self.layer_with_computation_time_list:
            for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
                if pre_defined_layer_name in layer_name:
                    var_tmp = (
                        self.layer_with_computation_time_list[layer_name]
                        / self.counted_steps
                        / self.num_of_micro_batch
                    )
                    transform_layer_with_cost_list[pre_defined_layer_name] += var_tmp
            for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
                if auto_partition_layer_name in layer_name:
                    # assuming that the duration time of a layer can not exceed 10% of step time
                    # in order to
                    # avoid some specific long time layers that caused from the error caused by
                    # time_line.json
                    if (not self.is_msprof_file and self.layer_with_computation_time_list[
                            layer_name] > self.step_time / 1000 / 10):
                        continue
                    total_cost_auto_partition_layer[auto_partition_layer_name] += \
                        self.layer_with_computation_time_list[layer_name]
                    number_of_auto_partition_layer[auto_partition_layer_name] += 1

        for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
            transform_layer_with_cost_list[auto_partition_layer_name] = (
                total_cost_auto_partition_layer[auto_partition_layer_name] /
                number_of_auto_partition_layer[auto_partition_layer_name] /
                self.counted_steps / self.num_of_micro_batch)
        return transform_layer_with_cost_list


if __name__ == "__main__":
    path = "/your/path/here"
    example_model_name = "LLaMA_prof"
    comp1 = ComputationAnalyzer(path, example_model_name, 8)
    logger.info(comp1.layer_with_computation_time_list)
    logger.info(comp1.layer_with_cost_list)
