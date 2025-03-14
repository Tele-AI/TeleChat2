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
"""interactive"""

from collections import namedtuple

from toolkit.pipeline_balance.utils.error import _check_in_bounds
from toolkit.pipeline_balance.utils.layer import Layer
import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.sapp.sapp_pipeline import SappPipeline
from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.config import print_dryrun_config, generate_solvable_config

YES_OR_NO = "[y/n]? "

OPTIONAL = " if you wish: "

GLOBALARGUMENTS = namedtuple('GLOBALARGUMENTS', ['stage_num', 'micro_batch', 'interleave', 'max_memory'])


def default_v(d):
    return " (" + str(d) + " if none): "


def is_yes(s: str):
    """Returns true if yes or similar is entered"""
    return s.lower().startswith('y') or s == "1"


def is_empty(s: str):
    """Returns true if nothing or similar is entered"""
    return len(s.strip()) == 0 or (s.strip() == '*')


def global_arguments():
    """Set global arguments"""
    stage_num = 4
    micro_batch = 8
    interleave = 1
    max_memory = 56000
    s = input("Please enter the pipeline stage number" + default_v(stage_num))
    if not is_empty(s):
        stage_num = int(s)
        _check_in_bounds(stage_num, "Pipeline stage number", 1, 10000)

    s = input("Please enter the micro batch number" + default_v(micro_batch))
    if not is_empty(s):
        micro_batch = int(s)
        _check_in_bounds(micro_batch, "Micro batch number", 1, 10000)

    s = input("Please enter the pipeline interleave number" + default_v(interleave))
    if not is_empty(s):
        interleave = int(s)
        _check_in_bounds(interleave, "Interleave", 1, 10)

    s = input("Please enter maximum memory" + default_v(max_memory))
    if not is_empty(s):
        max_memory = int(s)
        _check_in_bounds(max_memory, "Maximum memory", 1, 1000000)

    return GLOBALARGUMENTS(stage_num, micro_batch, interleave, max_memory)


def make_layer(t: Layer.type_enum, model_name):
    """enter necessary information of a layer"""
    nb_layer = 1
    layer_time = 0
    memory_parameter = 0
    memory_activation_rec = {r: None for r in Recompute.TYPE}
    layer_name = "misc_" + t.name
    s = input("\tEnter the layer name" + OPTIONAL)
    if not is_empty(s):
        layer_name = s
    s = input("\tEnter the layer execution time: ")
    if not is_empty(s):
        layer_time = int(s)
    if t is Layer.type_enum.BODY:
        s = input("\tEnter the number of such layer: ")
        if not is_empty(s):
            nb_layer = int(s)
        s = input("\tEnter the layer parameter memory (MB): ")
        if not is_empty(s):
            memory_parameter = int(s)
        for r in Recompute.TYPE:
            s = input("\tEnter the layer " + Recompute.JSON_MEMORY_NAME[r] + OPTIONAL)
            if not is_empty(s):
                memory_activation_rec[r] = int(s)
    else:
        s = input("\tEnter the layer memory (MB): ")
        if not is_empty(s):
            memory_parameter = int(s)

    return Layer(name=layer_name, ltype=t, nb_layer=nb_layer, time=layer_time,
                 model_name=model_name, memory_activation_rec=memory_activation_rec,
                 memory_parameter=memory_parameter,)


def dryrun_guide():
    """offer dryrun guidance"""
    considered_rec = []
    stage_num = 0
    num_layers = 0
    s = input("Please enter the pipeline stage number" + default_v(stage_num))
    if not is_empty(s):
        stage_num = int(s)
        _check_in_bounds(stage_num, "Pipeline stage number", 1, 10000)
    else:
        return

    s = input("Please enter the number of layers" + default_v(num_layers))
    if not is_empty(s):
        num_layers = int(s)
        _check_in_bounds(num_layers, "Micro batch number", 1, 10000)
    else:
        return

    s = input("Do you consider full recomputation?" + YES_OR_NO)
    if is_yes(s):
        considered_rec.append(Recompute.TYPE.FULL)

    s = input("Do you consider select recomputation?" + YES_OR_NO)
    if is_yes(s):
        considered_rec.append(Recompute.TYPE.SLCT)

    s = input("Does your communication recomputation co-work with select recomputation?" + YES_OR_NO)
    if is_yes(s):
        considered_rec.append(Recompute.TYPE.BOTH)

    s = input("Do you consider extra communication recomputation?" + YES_OR_NO)
    if is_yes(s):
        considered_rec.append(Recompute.TYPE.COMM)

    offset_config_list, rec_config_list = generate_solvable_config(stage_num, num_layers, considered_rec)
    print_dryrun_config(offset_config_list, rec_config_list)


def main():
    s = input(
        "No arguments were given. Would you like to proceed to the interactive mode " + YES_OR_NO)
    if not is_yes(s):
        return

    global_args = global_arguments()
    number_of_stage = global_args.stage_num
    number_of_micro_batch = global_args.micro_batch
    interleave_degree = global_args.interleave
    max_memory = global_args.max_memory

    model_name = "misc"
    s = input("\tEnter the model name" + OPTIONAL)
    if not is_empty(s):
        model_name = s

    layers = []
    for ltype in Layer.type_enum:
        if ltype is not Layer.type_enum.UNKNOWN:
            logger.info("Please enter information of your network %s", ltype.name)
            layers.append(make_layer(ltype, model_name))

    pipe = SappPipeline(model_name=model_name, num_of_stage=number_of_stage,
                        num_of_micro_batch=number_of_micro_batch, max_memory=max_memory,
                        layers=layers, num_of_interleave=interleave_degree,)

    for layer in layers:
        logger.info("%s", layer)

    pipe.construct_problem(solver="pulp")
    pipe.solve_problem(time_limit=40, dump_folder="output")
    pipe.print_yaml_results()
    pipe.simulate(show=True)
