# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Check Model Input Config."""
import json
import re
import numpy as np
import mindspore.common.dtype as mstype
import mindspore as ms
from ..version_control import get_lazy_inline, get_predict_lazy_inline
from ..tools.logger import logger

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
PROCESSOR_NAME = "processor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
MAX_INT32 = 2147483647

str_to_ms_type = {
    "float16": mstype.float16,
    "float32": mstype.float32,
    "bfloat16": mstype.bfloat16,
    "int8": mstype.int8
}


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    ms_type = str(ms_type).lower()
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    if ms_type == "int8":
        return mstype.int8
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16, int8], but get {ms_type}")


def reverse_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        if v in new_d:
            raise ValueError(f"Different keys in dict have same values.")
        new_d[v] = k
    return new_d


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def check_fine_grain_interleave_valid(fine_grain_interleave, parallel_config):
    """Check the fine grain interleave condition"""
    if fine_grain_interleave is None or parallel_config is None:
        return False
    return fine_grain_interleave > 1 and parallel_config.model_parallel > 1


ms_type_to_str = reverse_dict(str_to_ms_type)

lazy_inline = get_lazy_inline
predict_lazy_inline = get_predict_lazy_inline


class LayerSetting:
    r"""
    Class for setting offset, pipeline stage and select recompute for each transformer layer.
    Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            num_layers(int) - The total layers in the model.
            offset(Union[int, List[int], List[List[int]]]) - The layer offset for each (mini) stage.
            parallel_config(dict) - Parallel Config.
            pp_interleave_num(int) - The number of pp-interleave. When it is larger than 1,
                each stage will be divided into mini stages by pp_interleave_num.
    """

    def __init__(self, num_layers, offset, parallel_config, pp_interleave_num=1):
        default_patterns = [r'feed_forward\.mul', r'feed_forward\.w1\.activation\.silu']
        default_comm_patterns = [r'.*\.norm']
        try:
            use_pp_interleave = ms.get_auto_parallel_context("pipeline_interleave")
        except ValueError:
            logger.warning(f"Current MindSpore version do not pipeline interleave. `pp_interleave_num` is set to 1.")
            use_pp_interleave = False
        self.num_layers = num_layers
        self.pp = parallel_config.pipeline_stage
        self.recompute = parallel_config.recompute
        self.gradient_aggregation_group = parallel_config.gradient_aggregation_group
        self.pp_interleave_num = pp_interleave_num if use_pp_interleave else 1
        self.offset = np.array(offset, np.int64)
        self._check_inputs()
        self.offset = np.broadcast_to(self.offset, (self.pp_interleave_num, self.pp))

        avg_layer = self.num_layers // (self.pp * self.pp_interleave_num)
        self.layer_list = np.ones((self.pp_interleave_num, self.pp), np.int64) * avg_layer + self.offset
        interleave_sum = np.insert(np.cumsum(np.sum(self.layer_list, axis=1))[:-1], 0, 0)
        self.layer_accu = np.cumsum(self.layer_list, axis=1) + interleave_sum.reshape(-1, 1)
        self.pp_ids = [np.searchsorted(self.layer_accu.reshape(-1), i + 1) % self.pp for i in range(self.num_layers)]
        self.interleave_ids = [np.where(i < self.layer_accu)[0][0] for i in range(self.num_layers)]
        logger.info(f"num_layers per stage: {self.layer_list.tolist()}")
        logger.info(f"Accumulated num_layers per stage: {self.layer_accu.tolist()}")
        logger.info(f"Pipeline id list: {self.pp_ids}")
        logger.info(f"Interleave id list: {self.interleave_ids}")
        pre_pad = np.array([[0] + [self.layer_accu[i, -1] for i in range(len(self.layer_accu) - 1)]])
        self.layer_accu_mod = np.concatenate((pre_pad.T, self.layer_accu), axis=-1)

        if not isinstance(self.recompute, bool):
            self.layer_recompute = self._format_recompute_list(self.recompute.recompute)
            self.select_recompute = self._format_recompute_dict(
                self.recompute.select_recompute, default_patterns)
            self.select_comm_recompute = self._format_recompute_dict(
                self.recompute.select_comm_recompute, default_comm_patterns)
            logger.info(f"Formative layer_recompute: {self.layer_recompute}")
            logger.info(f"Formative select_recompute: {self.select_recompute}")
            logger.info(f"Formative select_comm_recompute: {self.select_comm_recompute}")

    def set(self, layer, layer_id):
        """Set pipeline stage and recompute for each layer with a layer_id."""
        pp_id = int(self.pp_ids[layer_id])
        layer.pipeline_stage = pp_id
        dis = max(int((self.num_layers + 1) / self.gradient_aggregation_group), 1)
        if self.pp > 1:
            layer.set_comm_fusion(2)
        else:
            layer.set_comm_fusion(int((layer_id + self.offset[0, 0]) / dis) + 1)

        if isinstance(self.recompute, bool):
            if self.recompute:
                layer.recompute()
            return
        if self.recompute.recompute:
            if isinstance(self.recompute.recompute, bool):
                if self.recompute.recompute:
                    layer.recompute(recompute_slice_activation=self.recompute.recompute_slice_activation)
            else:
                if self._check_layer_rule(layer_id):
                    layer.recompute()
                    logger.info(f"Set full recompute at layer {layer_id}")
                else:
                    self._set_select_recompute(layer, layer_id, False)
                    self._set_select_recompute(layer, layer_id, True)
        else:
            self._set_select_recompute(layer, layer_id, False)
            self._set_select_recompute(layer, layer_id, True)

    def _alloc_recompute_layer(self, select_recompute):
        """Average allocate recompute layer among different interleave."""
        res = np.zeros_like(self.layer_list)
        layer_by_stage = np.sum(self.layer_list, axis=0)
        for j, num in enumerate(select_recompute):
            num = min(num, layer_by_stage[j])
            while num:
                for i in range(self.pp_interleave_num):
                    if not num:
                        break
                    if res[i, j] < self.layer_list[i, j]:
                        res[i, j] += 1
                        num -= 1
        return res

    def _format_recompute_list(self, select_recompute):
        """Format recompute inputs into a list."""
        if isinstance(select_recompute, bool):
            if select_recompute:
                return self.layer_list.tolist()
            return np.zeros_like(self.layer_list).tolist()
        if isinstance(select_recompute, (list, tuple)):
            if all(isinstance(item, (int, bool)) for item in select_recompute):
                for i, item in enumerate(select_recompute):
                    if isinstance(item, bool):
                        select_recompute[i] = MAX_INT32 if item else 0
                return self._alloc_recompute_layer(select_recompute).tolist()
            if all(isinstance(item, str) for item in select_recompute):
                return self.layer_list.tolist()
            for i, sub in enumerate(select_recompute):
                if isinstance(sub, bool):
                    select_recompute[i] = self.layer_list[i] if sub else np.zeros_like(self.layer_list[i])
                elif isinstance(sub, (list, tuple)):
                    for j, item in enumerate(sub):
                        if isinstance(item, bool):
                            select_recompute[i][j] = self.layer_list[i][j] if item else 0
                        elif isinstance(item, int):
                            select_recompute[i][j] = min(select_recompute[i][j], self.layer_list[i, j])
                        else:
                            raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
                else:
                    raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
            return select_recompute
        raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")

    def _format_recompute_dict(self, select_recompute, default_patterns):
        """Format select_recompute inputs into a dict"""
        dic = {}
        if isinstance(select_recompute, (list, tuple)) and all(isinstance(item, str) for item in select_recompute):
            parttern = select_recompute
        elif isinstance(select_recompute, dict):
            parttern = select_recompute.keys()
        else:
            parttern = default_patterns
        for p in parttern:
            value = select_recompute[p] if isinstance(select_recompute, dict) else select_recompute
            dic[p] = self._format_recompute_list(value)
        return dic

    def _check_layer_rule(self, layer_id):
        """Check whether a layer should be set full layer recompute."""
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < self.layer_recompute[v_id][pp_id]:
            return True
        return False

    def _set_select_recompute(self, layer, layer_id, add_prim_attr=False):
        """Set select recompute for a layer."""
        select_recompute = self.select_comm_recompute if add_prim_attr else self.select_recompute
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        log_ops = []
        for pattern, layers_dict in select_recompute.items():
            if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < layers_dict[v_id][pp_id]:
                log = LayerSetting.set_pattern_recompute(layer, pattern.split(r'\.'), add_prim_attr)
                if log:
                    log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info(f"Set select {comm}recompute at layer {layer_id}: {log_ops_str}")

    def _check_inputs(self):
        """Check the inputs of offset."""
        if self.offset.ndim >= 1 and self.offset.shape[-1] != self.pp:
            raise ValueError(f"offset.shape[-1] should equal to `pp` ({self.pp}), "
                             f"but got ({self.offset.shape[-1]}). `offset`: {self.offset}")
        if self.offset.ndim >= 2 and self.offset.shape[-2] != self.pp_interleave_num:
            raise ValueError(f"offset.shape[-2] should equal to `pp_interleave_num` ({self.pp_interleave_num}), "
                             f"but got ({self.offset.shape[-2]}). `offset`: {self.offset}")
        if self.offset.sum() != self.num_layers % (self.pp * self.pp_interleave_num):
            r = self.num_layers % (self.pp * self.pp_interleave_num)
            raise ValueError(f"The sum of `offset` ({self.offset.sum()}) should equal to remainder of `num_layers` "
                             f"({self.num_layers}) % (pp ({self.pp}) * pp_interleave_num ({self.pp_interleave_num})) "
                             f"= {r}")

    @staticmethod
    def set_pattern_recompute(layer, p_list, add_prim_attr=False, info=''):
        """Set an operator recompute for a given key-value pair in select_recompute dict."""
        log_list = []
        log = ''
        if p_list:
            p = p_list.pop(0)
        else:
            return info
        if p_list:
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    log = LayerSetting.set_pattern_recompute(cell, p_list, add_prim_attr, info + f'.{name}')
                    if log:
                        log_list.append(log[1:])
        else:
            for attr in dir(layer):
                if re.fullmatch(p, attr):
                    operator = getattr(layer, attr)
                    if add_prim_attr:
                        operator.add_prim_attr("recompute_comm_op", True)
                        log = f"{info}.{attr}"
                    elif hasattr(operator, 'recompute'):
                        operator.recompute()
                        log = f"{info}.{attr}"
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    if not add_prim_attr:
                        cell.recompute()
                        log = f"{info}.{name}"
        p_list.insert(0, p)
        if log_list:
            return " " + ", ".join(log_list)
        return log

    def __call__(self, layer, layer_id):
        self.set(layer, layer_id)
