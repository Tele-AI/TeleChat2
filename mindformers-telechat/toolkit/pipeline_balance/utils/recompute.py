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
"""support recomputation"""

from enum import IntEnum

from toolkit.pipeline_balance.utils.logger import logger

TYPE = IntEnum("RecomputeType", ["NONE", "SLCT", "COMM", "BOTH", "FULL"], start=0)
OFFSET = "offset"

DEFAULT_COEF = {
    TYPE.NONE: 0,
    TYPE.SLCT: 0.04,
    TYPE.COMM: 0.125,
    TYPE.BOTH: 0.165,
    TYPE.FULL: 0.5,
}  # old

YAML_NAME = {
    TYPE.NONE: "",
    TYPE.COMM: "select_comm_recompute",
    TYPE.SLCT: "select_recompute",
    TYPE.BOTH: "both_comm_select",
    TYPE.FULL: "recompute",
}

JSON_MEMORY_NAME = {
    TYPE.NONE: "memory_activation",
    TYPE.COMM: "memory_select_comm",
    TYPE.BOTH: "memory_both_comm_select",
    TYPE.SLCT: "memory_select_rec",
    TYPE.FULL: "memory_recompute",
}

JSON_MEMORY_NAME_ALIGNED = {
    TYPE.NONE: "memory_activation ",
    TYPE.COMM: "memory_select_comm",
    TYPE.BOTH: "memory_both_comm_select",
    TYPE.SLCT: "memory_select_rec ",
    TYPE.FULL: "memory_recompute  ",
}


JSON_TIME_NAME = {
    TYPE.NONE: "backward_time",
    TYPE.COMM: "select_comm_time",
    TYPE.BOTH: "both_comm_select_time",
    TYPE.SLCT: "select_rec_time",
    TYPE.FULL: "recompute_time ",
}

JSON_COEF_NAME = {
    TYPE.NONE: "backward_coef",
    TYPE.SLCT: "select_rec_coef",
    TYPE.BOTH: "both_comm_select_coef",
    TYPE.COMM: "select_comm_coef",
    TYPE.FULL: "recompute_coef",
}


def sums(rec_dict):
    x = 0
    for r in TYPE:
        x += rec_dict[r]
    return x


def zero_if_none(v):
    if v is not None:
        return int(v)
    return 0


def yaml_from_internal(vpp, pp, lp_variables, nass):
    """covert internal format to mindformers yaml format"""
    slct_is = 0
    comm_is = 0
    both_is = 0
    full_is = 0

    yaml = {
        OFFSET: [],
        YAML_NAME[TYPE.FULL]: [],
        YAML_NAME[TYPE.SLCT]: [],
        YAML_NAME[TYPE.COMM]: [],
    }
    logger.debug(f"pp = {pp}, vpp = {vpp}")
    for i in range(vpp):
        for _, v in yaml.items():
            v.append([])
        for s in range(pp):
            gass_i_s = 0
            for r in TYPE:
                gass_i_s += zero_if_none(lp_variables[r][i][s].varValue)
            slct_is = zero_if_none(lp_variables[TYPE.SLCT][i][s].varValue)
            comm_is = zero_if_none(lp_variables[TYPE.COMM][i][s].varValue)
            both_is = zero_if_none(lp_variables[TYPE.BOTH][i][s].varValue)
            full_is = zero_if_none(lp_variables[TYPE.FULL][i][s].varValue)
            yaml[OFFSET][i].append(gass_i_s - nass[i][s])
            yaml[YAML_NAME[TYPE.FULL]][i].append(full_is)
            yaml[YAML_NAME[TYPE.SLCT]][i].append(slct_is + both_is + full_is)
            yaml[YAML_NAME[TYPE.COMM]][i].append(comm_is + both_is + full_is)

    logger.debug(f"yaml = {yaml}")
    return yaml


def internal_from_yaml(vpp, pp, yaml, nass):
    """covert mindformers yaml format to internal format"""
    slct_is = 0
    comm_is = 0
    full_is = 0
    layer_per_recompute = {r: [] for r in TYPE}
    if yaml[OFFSET] == 0:
        yaml[OFFSET] = [[0] * pp] * vpp

    for rec in [TYPE.SLCT, TYPE.COMM, TYPE.FULL]:
        if (
                YAML_NAME[rec] not in yaml
                or yaml[YAML_NAME[rec]] is False
                or yaml[YAML_NAME[rec]] == 0
        ):
            yaml[YAML_NAME[rec]] = [[0] * pp] * vpp
        if yaml[YAML_NAME[rec]] is True:
            yaml[YAML_NAME[rec]] = [
                [a + b for a, b in zip(list1, list2)]
                for list1, list2 in zip(nass, yaml[OFFSET])
            ]

    for i in range(vpp):
        for _, v in layer_per_recompute.items():
            v.append([])
        for s in range(pp):
            slct_is = zero_if_none(yaml[YAML_NAME[TYPE.SLCT]][i][s])
            comm_is = zero_if_none(yaml[YAML_NAME[TYPE.COMM]][i][s])
            full_is = zero_if_none(yaml[YAML_NAME[TYPE.FULL]][i][s])
            layer_per_recompute[TYPE.FULL][i].append(full_is)
            layer_per_recompute[TYPE.BOTH][i].append(
                max(min(slct_is - full_is, comm_is - full_is), 0)
            )
            layer_per_recompute[TYPE.SLCT][i].append(
                max(slct_is - full_is - layer_per_recompute[TYPE.BOTH][i][s], 0)
            )
            layer_per_recompute[TYPE.COMM][i].append(
                max(comm_is - full_is - layer_per_recompute[TYPE.BOTH][i][s], 0)
            )
            layer_per_recompute[TYPE.NONE][i].append(
                (
                    yaml[OFFSET][i][s]
                    + nass[i][s]
                    - layer_per_recompute[TYPE.FULL][i][s]
                    - layer_per_recompute[TYPE.BOTH][i][s]
                    - layer_per_recompute[TYPE.SLCT][i][s]
                    - layer_per_recompute[TYPE.COMM][i][s]
                )
            )

    logger.debug(f"layer_per_recompute = {layer_per_recompute}")
    return layer_per_recompute


def to_list(rec_dict):
    return list(rec_dict.values())


def right_extend(ll, n):
    all_l = []
    for i in range(n):
        for sublist in ll:
            all_l += [sublist + [i]]
    return all_l


def make_all_indexes_local(used_rec, num_of_interleave, all_indexes: list[list[int]], r: TYPE):
    """make indexes of used recomputation based on the number of interleaves"""
    if r >= len(TYPE) - 1:
        if used_rec[r]:
            all_indexes = right_extend(all_indexes, num_of_interleave)
        return all_indexes
    if used_rec[r]:
        return make_all_indexes_local(
            used_rec,
            num_of_interleave,
            right_extend(all_indexes, num_of_interleave),
            TYPE(r + 1),
        )
    return make_all_indexes_local(used_rec, num_of_interleave, all_indexes, TYPE(r + 1))


def make_all_indexes(used_rec, num_of_interleave):
    return make_all_indexes_local(used_rec, num_of_interleave, [[]], TYPE.NONE)


def recomputes_from_indexes(used_rec, indexes):
    recomputes = []
    for idx in indexes:
        recompute = {r: None for r in TYPE}
        for r in TYPE:
            if used_rec[r]:
                recompute[r] = idx[0]
                idx.pop(0)
        recomputes.append(recompute)
    return recomputes


def average(rec_list):
    """return the average number of each type of  recomputation"""
    num = len(rec_list)
    if num == 0:
        return rec_list
    rec_1 = rec_list.pop(0)
    for rec_i in rec_list:
        for r in TYPE:
            if rec_1[r] is not None and rec_i[r] is not None:
                rec_1[r] = rec_1[r] + rec_i[r]
            elif not (rec_1[r] is None and rec_i[r] is None):
                logger.warning(
                    "WARNING: Recomputation %s is not taken ",
                    "into consideration by all body layers",
                    r.name,
                )
    for r in TYPE:
        if rec_1[r] is not None:
            rec_1[r] = rec_1[r] / num
    return rec_1


def assign_used(values, unused_rec):
    """find the assignment of each recomputation"""
    assignment = {r: None for r in TYPE}
    value_idx = 0
    for r in TYPE:
        if r not in unused_rec:
            assignment[r] = values[value_idx]
            value_idx += 1
    return assignment


def get_used_list(recompute_considered) -> list[TYPE]:
    """get a list of which recomputation is used"""
    used_rec = []
    for rec in TYPE:
        if recompute_considered[rec]:
            used_rec.append(rec)
    return used_rec


def get_unused_list(recompute_considered) -> list[TYPE]:
    unused_rec = []
    for rec in TYPE:
        if rec not in recompute_considered or not recompute_considered[rec]:
            unused_rec.append(rec)
    return unused_rec


def least_recomputed(recompute_considered):
    rec = TYPE.NONE
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
            break
    return rec


def most_recomputed(recompute_considered):
    rec = TYPE.FULL
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
    return rec
