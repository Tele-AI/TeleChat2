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
"""Transformer MLP"""
from typing import Union

from mindspore import mint, nn, ops

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.experimental.infer.core import get_act_func
from mindformers.experimental.infer.core.layers import ColumnParallelLinear
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec, build_module

__all__ = [
    "MLPSubmodules",
    "MLP"
]


class MLPSubmodules:
    """
    The MLPSubmodules class defines two submodules for a Multi-Layer Perceptron (MLP):

    Args:
        linear_fc1 (Union[ModuleSpec, type], optional): The module definition for the first fully connected layer.
            Defaults to None.
        linear_fc2 (Union[ModuleSpec, type], optional): The module definition for the second fully connected layer.
            Defaults to None.
    """

    def __init__(self, linear_fc1: Union[ModuleSpec, type] = None, linear_fc2: Union[ModuleSpec, type] = None):
        self.linear_fc1 = linear_fc1
        self.linear_fc2 = linear_fc2


class MLP(nn.Cell):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (dict): Configuration.
        submodules (MLPSubmodules): Submodules.
        is_expert (bool): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            is_expert=False):
        super().__init__(config)
        if is_expert:
            raise NotImplementedError("For MLP, `is_expert` is not supported for now.")
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = 4 * self.hidden_size
        if config.intermediate_size is not None:
            self.ffn_hidden_size = config.intermediate_size
        else:
            if config.ffn_dim_multiplier is not None:
                self.ffn_hidden_size = int((config.ffn_dim_multiplier + 0.01) * self.ffn_hidden_size)
            self.ffn_hidden_size = int(2 * self.ffn_hidden_size / 3)
            self.ffn_hidden_size = config.multiple_of * (
                (self.ffn_hidden_size + config.multiple_of - 1) // config.multiple_of
            )

        self.mlp_has_gate = getattr(config, 'mlp_has_gate', False)
        self.gated_linear_unit = self.config.ffn_concat

        tp_group_size = get_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(self.ffn_hidden_size, tp_group_size)

        if self.gated_linear_unit:
            self.mapping_ffn_hidden_size = self.ffn_hidden_size * 2
        else:
            self.mapping_ffn_hidden_size = self.ffn_hidden_size

        if self.mlp_has_gate:
            if not self.gated_linear_unit:
                self.gating = ColumnParallelLinear(
                    self.hidden_size,
                    self.ffn_hidden_size,
                    config=self.config,
                    bias=self.has_bias,
                    transpose_b=True,
                    gather_output=False,
                    is_expert=is_expert,
                    param_init_type=self.config.param_init_type,
                    compute_dtype=self.config.compute_dtype,
                )
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.mapping_ffn_hidden_size,
            config=self.config,
            bias=self.has_bias,
            transpose_b=True,
            gather_output=False,
            is_expert=is_expert,
            param_init_type=self.config.param_init_type,
            compute_dtype=self.config.compute_dtype,
        )

        self.act_type = self.config.hidden_act
        self.act_func = get_act_func(self.act_type)

        # Project back to h.
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            config=self.config,
            bias=self.has_bias,
            transpose_b=True,
            is_expert=is_expert,
            param_init_type=self.config.param_init_type,
            compute_dtype=self.config.compute_dtype,
        )
        self.mul = ops.Mul()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """ Construct function of mlp block. """
        # [T, H] -> [T, ffn_H]
        if self.mlp_has_gate:
            if self.gated_linear_unit:
                gate_hidden_out = self.linear_fc1(x)
                bs, seq_len, _ = gate_hidden_out.shape
                reshape_out = self.reshape(gate_hidden_out,
                                           (bs, seq_len, self.ffn_hidden_size_per_partition, 2))
                gate, hidden = mint.split(reshape_out,
                                          (1, 1), -1)
                gate = self.reshape(gate, (bs, seq_len, self.ffn_hidden_size_per_partition))
                hidden = self.reshape(hidden, (bs, seq_len, self.ffn_hidden_size_per_partition))
            else:
                gate = self.gating(x)
                hidden = self.linear_fc1(x)
            gate = self.act_func(gate)
            hidden = mint.mul(hidden, gate)
        else:
            hidden = self.linear_fc1(x)
            hidden = self.act_func(hidden)

        # [T, ffn_H] -> [T, H]
        output = self.linear_fc2(hidden)
        return output
