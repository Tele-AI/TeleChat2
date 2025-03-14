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
"""test infer mlp utils"""
import mindspore.nn as nn
from mindspore import Tensor

from mindformers.models.llama import LlamaConfig
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.core.mlp import MLP, MLPSubmodules
from mindformers.models.llama.llama_transformer import LlamaFeedForward
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec, build_module


def get_mlp_spec():
    """Construct test mlp spec."""
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
    )


def convert_weight_name(params):
    """Convert weight name."""
    replacement_map = {
        'mlp.w1.weight': 'mlp.gating.weight',
        'mlp.w2.weight': 'mlp.linear_fc2.weight',
        'mlp.w3.weight': 'mlp.linear_fc1.weight'
    }
    for name, param in list(params.items()):
        new_name = replacement_map.get(name, name)
        param.name = new_name
        if new_name != name:
            params.move_to_end(name)
            params[new_name] = params.pop(name)
    return params


class NewMLPNet(nn.Cell):
    """A model class of new mlp."""
    def __init__(self, config: TransformerConfig):
        super(NewMLPNet, self).__init__()
        self.mlp = build_module(
            get_mlp_spec(),
            config=config
        )

    def construct(self, input_x: Tensor):
        return self.mlp(input_x)


class OldMLPNet(nn.Cell):
    """A model class of old mlp."""
    def __init__(self, config: LlamaConfig):
        super(OldMLPNet, self).__init__()
        self.mlp = LlamaFeedForward(
            dim=config.hidden_size,
            intermediate_size=config.intermediate_size,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            ffn_concat=config.ffn_concat
        )

    def construct(self, input_x: Tensor):
        return self.mlp(input_x)
