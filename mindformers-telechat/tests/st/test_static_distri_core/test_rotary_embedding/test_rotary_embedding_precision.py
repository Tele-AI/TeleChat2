# pylint: skip-file
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
"""test rotary embedding"""

from pathlib import Path
import pytest
import json
from mindformers.experimental.graph.transformer.rotary_pos_embedding import RotaryEmbedding
import mindspore as ms
import numpy as np

def create_testconfig(path: str):
    with open(path) as f:
        raw_data = json.dump(f)
    return {k: [tuple(s.values()) if len(s)>1 else tuple(s.values())[0] for s in v] for k, v in raw_data.items()}

class TestRotaryPosEmbedding:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("rotary_param, chatglm, rotary_base, seq, expected", test_config["test_rotary_pos_embedding"])
    def test_rotary_pos_embedding(self, rotary_param, chatglm, rotary_base, seq, expected):
        if rotary_base is not None:
            rotary_param["rotary_base"]=rotary_base
        rotary = RotaryEmbedding(**rotary_param)
        out = rotary(seq)
        expected = ms.Tensor(expected)
        assert ms.ops.isclose(out, expected, rtol=1e-4, atol=1e-4).all()


