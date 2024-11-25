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
"""MLP Module"""

__all__ = [
    "ParallelMLP",
]

from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from mindformers.experimental.parallel_core.pynative.tensor_parallel.lora_layers import (
    ColumnParallelLoRA,
    RowParallelLoRA
)

from .activation import get_act_func
from .module import Module


class ParallelMLP(Module):
    r"""
    Implementation of parallel feedforward block.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        is_expert (bool): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.transformer import ParallelMLP
        >>> init()
        >>> initialize_model_parallel()
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
        >>> config = TransformerConfig      #The config of Transformer model. For details, please refer to TransformerConfig
        >>> mlp = ParallelMLP(config=config)
        >>> input = Tensor(np.random.random((3, 8, 16)).astype(np.float32))
        >>> output, _ = mlp(input)
        >>> print(output)
    """

    def __init__(self, config, is_expert=False):
        super(ParallelMLP, self).__init__(config)
        self.config = config
        self.add_bias = config.add_bias_linear
        self.act_type = self.config.hidden_act
        self.hidden_size = self.config.hidden_size
        self.has_bias = self.config.mlp_has_bias
        mapping_output_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            mapping_output_size *= 2

        self.mapping = ColumnParallelLinear(
            self.hidden_size,
            mapping_output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            gather_output=False,
            is_expert=is_expert,
            bias_init=self.config.bias_init,
        )

        self.bias_gelu_fusion = False
        self.act_func = get_act_func(self.act_type)

        # Project back to h.
        if config.out_hidden_size is None:
            out_hidden_size = self.hidden_size
        else:
            out_hidden_size = config.out_hidden_size
        self.projection = RowParallelLinear(
            self.config.ffn_hidden_size,
            out_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            input_is_parallel=True,
            is_expert=is_expert,
            bias_init=self.config.bias_init,
        )
        use_lora = config.lora_config.use_lora
        if use_lora:
            mapping_lora = self._get_cell_lora_config(config, 'mapping')
            if mapping_lora is not None:
                self.mapping = ColumnParallelLoRA(
                    self.hidden_size,
                    mapping_output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    gather_output=False,
                    is_expert=is_expert,
                    bias_init=self.config.bias_init,
                    lora_rank=mapping_lora['rank'],
                    lora_alpha=mapping_lora['alpha'],
                    lora_dropout=mapping_lora['dropout'],
                )
            projection_lora = self._get_cell_lora_config(config, 'projection')
            if projection_lora is not None:
                self.projection = RowParallelLoRA(
                    self.config.ffn_hidden_size,
                    out_hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.has_bias,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=is_expert,
                    bias_init=self.config.bias_init,
                    lora_rank=projection_lora['rank'],
                    lora_alpha=projection_lora['alpha'],
                )

    def construct(self, hidden_states):
        """Construct function of mlp block."""
        # [B, S, H] -> [B, S, ffn_H] / [S, B, H] -> [S, B, ffn_H]
        intermediate_parallel, bias_parallel = self.mapping(hidden_states)
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        intermediate_parallel = self.act_func(intermediate_parallel)

        # [B, S, ffn_H] -> [B, S, H] / [S, B, ffn_H] -> [S, B, H]
        output, output_bias = self.projection(intermediate_parallel)
        return output, output_bias
