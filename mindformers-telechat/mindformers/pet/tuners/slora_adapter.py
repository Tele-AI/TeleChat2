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
"""Linear Layers with LoRA."""

import abc
import re
import json
import os
from typing import Tuple, Union

from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore._checkparam import args_type_check

from mindformers.modules.layers import Linear, Dropout
from mindformers.pet.pet_config import SLoraConfig
from mindformers.tools.logger import logger


class SLoraLinear(Cell):
    """
    Decorator of Linear layer for S-LoRA
    """
    def __init__(self, linear: Linear, adapter_ids: Parameter, config: SLoraConfig):
        super().__init__()
        self.adapter_ids = adapter_ids

        self.in_channels = linear.in_channels
        self.out_channels = linear.out_channels
        self.expert_num = linear.expert_num
        self.dtype = linear.dtype
        self.expert_flag = linear.expert_flag
        self.use_gmm = linear.use_gmm
        self.use_expert_group_size = linear.use_expert_group_size
        self.expert_group_size = linear.expert_group_size
        self.outer_batch = linear.outer_batch
        self.has_bias = linear.has_bias
        self.activation_flag = linear.activation_flag
        self.weight = linear.weight
        self.matmul = linear.matmul
        if self.has_bias:
            self.bias = linear.bias
            self.bias_add = linear.bias_add
        if self.activation_flag:
            self.activation = linear.activation

        self.lora_num = config.lora_num
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_scaling = self.lora_alpha / self.lora_rank

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.lora_a_gather = P.Gather()
        self.lora_b_gather = P.Gather()
        self.lora_dropout = Dropout(keep_prob=1 - config.lora_dropout)
        self.lora_a_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_a_shape = [self.lora_num, self.lora_rank, self.in_channels]
        self.lora_b_shape = [self.lora_num, self.out_channels, self.lora_rank]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        batch_size = self.adapter_ids.shape[0]
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_gather(lora_a, self.adapter_ids.reshape(-1), 0)
        lora_b = self.lora_b_gather(lora_b, self.adapter_ids.reshape(-1), 0)

        x = self.cast(x, self.dtype)
        if self.use_gmm:
            dense_result = self.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.matmul(x, weight)

        #-------- LoRA part ----------
        x = self.reshape(x, (batch_size, -1, self.in_channels))

        x = self.lora_dropout(x)
        lora_result = self.lora_a_matmul(x, lora_a)
        lora_result = self.lora_b_matmul(lora_result, lora_b)
        lora_scaling = self.cast(self.lora_scaling, self.dtype)
        lora_result = self.reshape(lora_result, dense_result.shape)
        lora_result = self.lora_mul(lora_result, lora_scaling)

        if self.has_bias:
            dense_result = self.bias_add(dense_result, self.cast(self.bias, self.dtype))
        # Result addition
        out = self.lora_add(dense_result, lora_result)
        if self.activation_flag:
            out = self.activation(out)
        out = self.cast(out, ori_dtype)
        output = self.reshape(out, out_shape)
        return output

    def shard(self):
        """
        Set the shard for the linear. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.matmul.in_strategy
        self.lora_a_gather.shard(((1, 1, strategy[1][1]), (1,)))
        self.lora_b_gather.shard(((1, strategy[1][0], 1), (1,)))
        self.lora_a_matmul.shard(((strategy[0][0], 1, strategy[0][1]), (strategy[0][0], 1, strategy[1][1])))
        self.lora_b_matmul.shard(((strategy[0][0], 1, 1), (strategy[0][0], strategy[1][0], 1)))
        self.lora_mul.shard(((strategy[0][0], strategy[1][0]), ()))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))


class SLoraAdapter(abc.ABC):
    """
    LoRA Model Manager to wrap layers
    """
    adapter_names = []
    registered_loras = {}

    def __init__(self, config: SLoraConfig, adapter_ids: Parameter):
        self.adapter_ids = adapter_ids
        self.common_lora_cell: dict = {}
        self.slora_config = config

    @classmethod
    @args_type_check(pet_config=(dict, SLoraConfig))
    def init_slora_config(cls, pet_config: Union[dict, SLoraConfig]):
        """read S-LoRA config"""
        if isinstance(pet_config, SLoraConfig):
            return pet_config
        adapter_path = pet_config.adapter_path
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"The adapter_path must be correct, but get {adapter_path}")
        with open(adapter_path, 'r') as file:
            path_dict = json.load(file)
        cls.adapter_names = list(path_dict.keys())

        num = len(path_dict) + 1
        max_rank = 0
        target_modules_set = set()
        for _, slora_path in path_dict.items():
            config_path = os.path.join(slora_path, "adapter_config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"{config_path} does not exist, "
                                 f"please pass a valid the slora path in config.adapter_path file.")
            with open(config_path, 'r') as file:
                config = json.load(file)
            max_rank = max(max_rank, int(config["r"]))
            target_modules_set.update(config["target_modules"])
        target_modules_list = [".*" + module for module in list(target_modules_set)]
        target_modules = '|'.join(target_modules_list)

        # read alpha, dropout from the first lora.
        config_path = os.path.join(path_dict[list(path_dict.keys())[0]], "adapter_config.json")
        with open(config_path, 'r') as file:
            config = json.load(file)
        alpha = int(config["lora_alpha"])
        dropout = float(config["lora_dropout"])
        return SLoraConfig(target_modules, num, max_rank, alpha, dropout)

    def register(self, name: str, cell: Cell):
        lora_a_name = name + ".lora_a"
        lora_b_name = name + ".lora_b"
        self.registered_loras[lora_a_name] = cell.lora_a_shape
        self.registered_loras[lora_b_name] = cell.lora_b_shape

    def build(self):
        self.common_lora_cell[Linear] = SLoraLinear

    def get_pet_model(self, network: Cell):
        self.build()
        self.process(network)
        network.update_parameters_name()
        return network

    def process(self, root: Cell, name_prefix: str = "model"):
        """Recursively search for target layers in the network and wrap them"""
        if root is None:
            return root
        for name, cell in root.name_cells().items():
            full_cell_name = f"{name_prefix}.{name}"
            new_cell, is_end_point = self.process_cell(full_cell_name, cell)
            if new_cell is not cell:
                root.insert_child_to_cell(name, new_cell)
                self.register(full_cell_name, new_cell)
            if not is_end_point:
                _ = self.process(new_cell, full_cell_name)
        return root

    def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
        """Match the layer with target list to get the layer with lora"""
        if not isinstance(cell, Cell):
            return cell, True
        if re.match(self.slora_config.target_modules, cell_name):
            wrap_lora = self.common_lora_cell.get(type(cell))
            if wrap_lora:
                logger.info(f"Apply LoRA to {cell_name}.")
                new_cell = wrap_lora(cell, self.adapter_ids, self.slora_config)
                new_cell.shard()
                return new_cell, True
        return cell, False
