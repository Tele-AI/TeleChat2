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
import mindspore.common.dtype as mstype

from mindformers.modules.layers import Linear
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.pet.pet_config import SLoraConfig
from mindformers.tools.logger import logger


class SLoraLinear(Cell):
    r"""
        Decorator of Linear layer for S-LoRA

        Args:
            linear (Linear): The base linear layer.
            slora_inputs (dict): The slora inputs including adapter group list.
            config (SLoraConfig): The config of SLora.

        Inputs:
            - **input_ids** (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, linear: Linear, slora_inputs: dict, config: SLoraConfig):
        super().__init__()
        self.lora_group_list = slora_inputs.get("group_list")
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

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.lora_a_transpose = P.Transpose()
        from mindspore.ops.auto_generate import GroupedMatmul
        self.lora_a_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.lora_b_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.lora_a_shape = [self.lora_num, self.lora_rank, self.in_channels]
        self.lora_b_shape = [self.lora_num, self.lora_rank, self.out_channels]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

    def _lora_linear(self, x, dense_result):
        """Forward process of LoRA branch"""
        x = self.reshape(x, (-1, self.in_channels))
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_transpose(lora_a, (0, 2, 1))
        x = self.lora_a_gmm([x], [lora_a], None, None, None, None, None, self.lora_group_list)[0]
        x = self.lora_b_gmm([x], [lora_b], None, None, None, None, None, self.lora_group_list)[0]
        x = self.reshape(x, dense_result.shape)
        out = self.lora_add(dense_result, x)
        return out

    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        if self.use_gmm:
            dense_result = self.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.matmul(x, weight)
        if self.has_bias:
            dense_result = self.bias_add(dense_result, self.cast(self.bias, self.dtype))

        out = self._lora_linear(x, dense_result)
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
        self.lora_a_transpose.shard(((1, 1, strategy[1][1]),))
        self.lora_a_gmm.shard(
            (((1, strategy[0][1]),), ((1, strategy[1][1], 1),), ((),), ((),), ((),), ((),), ((),), (1,)))
        self.lora_b_gmm.shard((((1, 1),), ((1, 1, strategy[1][0]),), ((),), ((),), ((),), ((),), ((),), (1,)))
        self.lora_add.shard(((1, strategy[1][0]), (1, strategy[1][0])))


class SLoraEmbedding(Cell):
    r"""
        Decorator of Embedding layer for SLoRA.

        Args:
            embedding (Cell): The base embedding cell.
            slora_inputs (dict): The slora inputs including adapter ids.
            config (SLoraConfig): The config of SLora.

        Inputs:
            - **input_ids** (Tensor) - Tensor of shape :math:`(batch, seq_length)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, embedding: Cell, slora_inputs: dict, config: SLoraConfig):
        super().__init__()
        self.group_list = slora_inputs.get("group_list")
        self.lora_extra_vocab_size = config.lora_extra_vocab_size
        self.lora_vocab_size = embedding.vocab_table_size + config.lora_extra_vocab_size
        self.dtype = embedding.embedding_weight.dtype
        self.embedding_weight = embedding.embedding_weight
        self.embedding_size = embedding.embedding_size
        self.gather = embedding.gather

        self.greater = P.Greater()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.eye = P.Eye()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.embedding_mul = P.Mul()
        self.embedding_gather = P.Gather()
        self.lora_gather = P.Gather()
        self.lora_a_transpose = P.Transpose()
        self.broadcast_to = P.BroadcastTo((-1, self.embedding_size))
        from mindspore.ops.auto_generate import GroupedMatmul
        self.embedding_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.lora_a_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.lora_b_gmm = GroupedMatmul(split_item=3, group_type=0)

        self.lora_a_shape = [config.lora_num, config.lora_rank, self.lora_vocab_size]
        self.lora_b_shape = [config.lora_num, config.lora_rank, embedding.embedding_size]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

        if self.lora_extra_vocab_size:
            self.lora_embedding_shape = [config.lora_num, self.lora_extra_vocab_size, self.embedding_size]
            self.lora_embedding = Parameter(initializer('zero', self.lora_embedding_shape, config.lora_dtype),
                                            requires_grad=False)

    def construct(self, input_ids):
        """Forward process"""
        x = self.reshape(input_ids, (-1,))

        added_tokens_mask = self.greater(x, self.lora_vocab_size - self.lora_extra_vocab_size - 1)
        org_tokens_mask = self.sub(1, added_tokens_mask)
        lora_mask = self.eye(self.shape(x)[0], self.shape(x)[0], mstype.float16)
        lora_mask = self.cast(lora_mask, self.dtype)

        #-------- Embedding part ----------
        embedding_weight = self.cast(self.embedding_weight, self.dtype)
        embedding_input = self.mul(x, org_tokens_mask)
        embedding = self.gather(embedding_weight, embedding_input, 0)
        org_tokens_mask = self.broadcast_to(self.reshape(org_tokens_mask, (org_tokens_mask.shape[0], 1)))
        embedding = self.embedding_mul(embedding, org_tokens_mask)

        #-------- Added Embedding part ----------
        if self.lora_extra_vocab_size:
            added_embedding = self.cast(self.lora_embedding, self.dtype)
            added_embedding_input = self.mul(self.sub(x, self.lora_vocab_size - self.lora_extra_vocab_size),
                                             added_tokens_mask)
            added_embedding = self.embedding_gmm([lora_mask],
                                                 [self.embedding_gather(added_embedding, added_embedding_input, 1)],
                                                 None, None, None, None, None, self.group_list)[0]
            added_tokens_mask = self.broadcast_to(self.reshape(added_tokens_mask, (added_tokens_mask.shape[0], 1)))
            added_embedding = self.embedding_mul(added_embedding, added_tokens_mask)
            embedding = self.add(embedding, added_embedding)

        #-------- LoRA part ----------
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_transpose(self.lora_gather(lora_a, x, 2), (0, 2, 1))
        lora_embedding = self.lora_a_gmm([lora_mask], [lora_a], None, None, None, None, None, self.group_list)[0]
        lora_embedding = self.lora_b_gmm([lora_embedding], [lora_b], None, None, None, None, None, self.group_list)[0]

        #-------- add part ---------
        out = self.add(embedding, lora_embedding)
        out = self.cast(out, self.dtype)
        output = self.reshape(out, (-1, out.shape[-1]))

        return output

    def shard(self):
        """
        Set the shard for the SLoRAEbedding. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.gather.in_strategy
        self.gather.shard(((strategy[0][0], strategy[0][1]), (1,)))


class SLoraHead(Cell):
    r"""
        Decorator of Head layer for SLoRA.

        Args:
            linear (Linear): The base head cell.
            slora_inputs (dict): The slora inputs including adapter ids.
            config (SLoraConfig): The config of SLora.

        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, lora_vocab_size)`.
    """
    def __init__(self, linear: Linear, slora_inputs: dict, config: SLoraConfig):
        super().__init__()
        self.group_list = slora_inputs.get('head_group_list')
        self.lora_num = config.lora_num
        self.lora_rank = config.lora_rank
        self.in_channels = linear.in_channels
        self.out_channels = linear.out_channels
        self.lora_extra_vocab_size = config.lora_extra_vocab_size
        self.lora_vocab_size = self.out_channels + self.lora_extra_vocab_size
        self.expert_flag = linear.expert_flag
        self.use_gmm = linear.use_gmm
        self.dtype = linear.dtype
        self.use_expert_group_size = linear.use_expert_group_size
        self.expert_group_size = linear.expert_group_size
        self.expert_num = linear.expert_num
        self.shape = linear.shape
        self.weight = linear.weight
        self.outer_batch = linear.outer_batch
        self.has_bias = linear.has_bias
        self.activation_flag = linear.activation_flag
        self.matmul = linear.matmul
        if self.has_bias:
            self.bias = linear.bias
            self.bias_add = linear.bias_add
        if self.activation_flag:
            self.activation = linear.activation

        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.gather = P.Gather()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.lora_embedding_transpose = P.Transpose()
        self.lora_a_transpose = P.Transpose()
        self.lora_concat = P.Concat(axis=-1)
        from mindspore.ops.auto_generate import GroupedMatmul
        self.lora_a_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.lora_b_gmm = GroupedMatmul(split_item=3, group_type=0)
        self.embedding_gmm = GroupedMatmul(split_item=3, group_type=0)

        if self.lora_extra_vocab_size:
            self.lora_embedding_shape = [self.lora_num, self.lora_extra_vocab_size, self.in_channels]
            self.lora_embedding = Parameter(initializer('zero', self.lora_embedding_shape, config.lora_dtype),
                                            requires_grad=False)

        self.lora_a_shape = [self.lora_num, config.lora_rank, self.in_channels]
        self.lora_b_shape = [self.lora_num, config.lora_rank, self.lora_vocab_size]
        self.lora_a = Parameter(initializer('zero', self.lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', self.lora_b_shape, config.lora_dtype), requires_grad=False)

    def _lora_head(self, x, dense_result):
        """Forward process of LoRA branch"""
        x = self.reshape(x, (-1, self.in_channels))
        lora_a = self.cast(self.lora_a, self.dtype)
        lora_b = self.cast(self.lora_b, self.dtype)
        lora_a = self.lora_a_transpose(lora_a, (0, 2, 1))
        x = self.lora_a_gmm([x], [lora_a], None, None, None, None, None, self.group_list)[0]
        x = self.lora_b_gmm([x], [lora_b], None, None, None, None, None, self.group_list)[0]
        x = self.reshape(x, self.shape(dense_result))
        out = self.lora_add(dense_result, x)
        return out

    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.lora_vocab_size,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag and not self.use_gmm:
            if self.use_expert_group_size is True:
                x = self.reshape(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)

        x = self.cast(x, self.dtype)
        if self.use_gmm:
            dense_result = self.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.matmul(x, weight)
        if self.has_bias:
            dense_result = self.bias_add(dense_result, self.cast(self.bias, self.dtype))

        if self.lora_extra_vocab_size:
            lora_embedding = self.cast(self.lora_embedding, self.dtype)
            lora_embedding = self.lora_embedding_transpose(lora_embedding, (0, 2, 1))
            lora_logit = self.embedding_gmm([x], [lora_embedding], None, None, None, None, None, self.group_list)[0]
            lora_logit = self.reshape(lora_logit, self.shape(dense_result)[:-1] + (self.lora_extra_vocab_size,))#
            dense_result = self.lora_concat([dense_result, lora_logit])

        out = self._lora_head(x, dense_result)

        if self.activation_flag:
            out = self.activation(out)
        out = self.cast(out, ori_dtype)
        output = self.reshape(out, out_shape)
        return output

    def shard(self):
        """
        Set the shard for the SLoRAHead. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
        """
        strategy = self.matmul.in_strategy
        self.lora_a_transpose.shard(((1, 1, strategy[0][1]),))
        self.lora_embedding_transpose.shard(((1, strategy[1][0], strategy[0][1]),))
        self.embedding_gmm.shard((((strategy[0][0], strategy[0][1]),), ((1, strategy[0][1], strategy[1][0]),),
                                  ((),), ((),), ((),), ((),), ((),), (1,)))
        self.lora_a_gmm.shard((((strategy[0][0], strategy[0][1]),), ((1, strategy[0][1], 1),),
                               ((),), ((),), ((),), ((),), ((),), (1,)))
        self.lora_b_gmm.shard((((strategy[0][0], 1),), ((1, 1, strategy[1][0]),),
                               ((),), ((),), ((),), ((),), ((),), (1,)))
        self.lora_concat.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))


class SLoraAdapter(abc.ABC):
    """
    LoRA Model Manager to wrap layers
    """
    adapter_names = ["base"]

    def __init__(self, config: SLoraConfig, slora_inputs: dict):
        self.slora_inputs = slora_inputs
        self.slora_config = config
        self.common_lora_cell: dict = {}
        self.registered_loras: dict = {}

    @classmethod
    @args_type_check(pet_config=(dict, SLoraConfig))
    def init_slora_config(cls, pet_config: Union[dict, SLoraConfig]):
        """read S-LoRA config"""
        if isinstance(pet_config, SLoraConfig):
            return pet_config
        if not isinstance(pet_config.adapter_path, str):
            raise TypeError(f"adapter_path should be string type, but got {type(pet_config.adapter_path)}.")
        if not os.path.exists(pet_config.adapter_path):
            raise FileNotFoundError(f"The adapter_path must be correct, but get {pet_config.adapter_path}")
        adapter_path = os.path.join(pet_config.adapter_path, "lora_adapter.json")
        with open(adapter_path, 'r') as file:
            path_dict = json.load(file)
        cls.adapter_names += list(path_dict.keys())

        num = len(cls.adapter_names)
        max_rank = 0
        target_modules_set = set()

        for _, slora_path in path_dict.items():
            config_path = os.path.join(slora_path, "adapter_config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"{config_path} does not exist, "
                                 f"please pass a valid the slora path in config.adapter_path file.")
            with open(config_path, 'r') as file:
                config = json.load(file)
            if not isinstance(config["r"], int):
                raise TypeError(f"rank should be int type, but get {type(config['r'])} type.")
            if not isinstance(config['target_modules'], list):
                raise TypeError(f"target_modules should be list type, but get {type(config['target_modules'])} type.")
            max_rank = max(max_rank, int(config["r"]))
            if not all(isinstance(module, str) for module in config["target_modules"]):
                raise TypeError(f"target_modules should be string type, but get wrong type.")
            target_modules_set.update(config["target_modules"])
        target_modules_list = [".*" + module for module in list(target_modules_set)]
        target_modules = '|'.join(target_modules_list)

        lora_extra_vocab_size = 0
        if re.match(target_modules, 'embed_tokens'):
            for _, slora_path in path_dict.items():
                added_tokens_path = os.path.join(slora_path, "added_tokens.json")
                if os.path.exists(added_tokens_path):
                    with open(added_tokens_path, 'r') as file:
                        added_tokens = json.load(file)
                    lora_extra_vocab_size = len(added_tokens)
                    break
            if lora_extra_vocab_size == 0:
                logger.warning(f"There is no added_token.json in the given path"
                               f"Model will be running with no added token")

        # read alpha, dropout from the first lora.
        config_path = os.path.join(path_dict[list(path_dict.keys())[0]], "adapter_config.json")
        with open(config_path, 'r') as file:
            config = json.load(file)
        if not isinstance(config["lora_alpha"], int):
            raise TypeError(f"lora_alpha should be int type, but get {type(config['lora_alpha'])}.")
        alpha = int(config["lora_alpha"])
        if not isinstance(config["lora_dropout"], float):
            raise TypeError(f"lora_dropout should be float type, but get {type(config['lora_dropout'])}.")
        dropout = float(config["lora_dropout"])

        return SLoraConfig(target_modules, num, max_rank, alpha, dropout, lora_extra_vocab_size)

    def register(self, name: str, cell: Cell):
        if isinstance(cell, SLoraEmbedding) and cell.lora_extra_vocab_size != 0:
            self.registered_loras['model.tok_embeddings.lora_embedding'] = cell.lora_embedding_shape
        lora_a_name = name + ".lora_a"
        lora_b_name = name + ".lora_b"
        self.registered_loras[lora_a_name] = cell.lora_a_shape
        self.registered_loras[lora_b_name] = cell.lora_b_shape

    def build(self):
        self.common_lora_cell[Linear] = SLoraLinear
        self.common_lora_cell[LlamaEmbedding] = SLoraEmbedding

    def get_pet_model(self, network: Cell):
        """Replace layers"""
        if "lm_head" in self.slora_config.target_modules:
            if hasattr(network, "lm_head"):
                new_head = SLoraHead(network.lm_head, self.slora_inputs, self.slora_config)
                new_head.shard()
                network.lm_head = new_head
                self.registered_loras["lm_head.lora_a"] = network.lm_head.lora_a_shape
                self.registered_loras["lm_head.lora_b"] = network.lm_head.lora_b_shape
                if network.lm_head.lora_extra_vocab_size != 0:
                    self.registered_loras["lm_head.lora_embedding"] = network.lm_head.lora_embedding_shape
                logger.info(f"Apply LoRA to lm_head.")
            else:
                logger.warning("The base model must has an attribute named in \'lm_head\'.")
        self.build()
        self.process(network.model)
        network.model.update_parameters_name()
        return network.model

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
        if re.match(self.slora_config.target_modules, cell_name) or \
            (re.match('.*embeddings', cell_name) and ("embed_tokens" in self.slora_config.target_modules)):
            wrap_lora = self.common_lora_cell.get(type(cell))
            if wrap_lora:
                logger.info(f"Apply LoRA to {cell_name}.")
                new_cell = wrap_lora(cell, self.slora_inputs, self.slora_config)
                new_cell.shard()
                return new_cell, True
        return cell, False
