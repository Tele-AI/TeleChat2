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
"""run parallel mlp"""

import argparse
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits, DistributedGradReducer
from mindspore.communication.management import init, get_rank, get_group_size
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, \
    get_data_parallel_world_size, get_data_parallel_group
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    OptimizerConfig,
    DatasetConfig,
    LoraConfig,
    TransformerConfig
)
from tests.st.test_distri_core.utils import TestData

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, with_attn_input=False, generate_golden=False):
    "train net"
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=0, weights=optimizer.parameters
    )
    if ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1:
        grad_reducer = DistributedGradReducer(optimizer.parameters, group=get_data_parallel_group())
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
                grads = grads[1]
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
                grads = grads[1]
            if ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1 and generate_golden:
                print("reduce gradients on group {}".format(get_data_parallel_group()))
                grads = grad_reducer(grads)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {} | Step {} | Loss {}".format(epoch, step, loss))
            step += 1
    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)

class ParallelMLP(Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, is_expert=False):
        super(ParallelMLP, self).__init__(config)
        self.config = config
        self.has_bias = self.config.mlp_has_bias
        self.hidden_size = self.config.hidden_size
        self.ffn_hidden_size = self.config.ffn_hidden_size
        # if config.model.model_config.gated_linear_unit:
        #     self.ffn_hidden_size *= 2

        self.mapping = ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            gather_output=False,
            is_expert=is_expert,
            bias_init=self.config.bias_init,
            transpose_b=False,
        )
        self.bias_gelu_fusion = False
        self.act_type = self.config.hidden_act
        # Project back to h.
        self.projection = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.has_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=is_expert,
            bias_init=self.config.bias_init,
            transpose_b=False,
        )

    def construct(self, hidden_states):
        """construct"""
        # [B, S, H] -> [B, S, ffn_H]
        intermediate_parallel, _ = self.mapping(hidden_states)
        # [B, S, ffn_H] -> [B, S, H]
        output, _ = self.projection(intermediate_parallel)
        return output


class ParallelMLPNet(nn.Cell):
    """
    define a pynative MLP net
    """
    def __init__(self, config):
        super(ParallelMLPNet, self).__init__()
        self.mlp = ParallelMLP(config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.cast = ops.Cast()
        self.dtype = config.compute_dtype

    def construct(self, x, labels):
        x = self.cast(x, self.dtype)
        output = self.mlp(x)
        output = ops.sum(output, dim=-1, keepdim=False)
        output = self.cast(output, mstype.float32)
        loss = self.loss(output, labels)
        return loss

def set_weight_seed(network, rank_id, zero3=False):
    """set init weight data"""
    rank_size = get_group_size()
    mapping_weight = np.random.normal(0, 0.02, size=(16, 64)).astype(np.float32)
    mapping_bias = np.random.normal(0, 0.02, size=(64,)).astype(np.float32)
    projection_weight = np.random.normal(0, 0.02, size=(64, 16)).astype(np.float32)
    projection_bias = np.random.normal(0, 0.02, size=(16,)).astype(np.float32)
    if zero3:
        if rank_id == 1:
            part_id = 2
        elif rank_id == 2:
            part_id = 1
        else:
            part_id = rank_id
        if rank_id == 0:
            mapping_weight = mapping_weight[:8, :32]
            projection_bias = projection_bias[:8]
        elif rank_id == 2:
            mapping_weight = mapping_weight[8:, :32]
            projection_bias = projection_bias[8:]
        elif rank_id == 1:
            mapping_weight = mapping_weight[:8, 32:]
            projection_bias = projection_bias[:8]
        elif rank_id == 3:
            mapping_weight = mapping_weight[8:, 32:]
            projection_bias = projection_bias[8:]
        mapping_bias = np.split(mapping_bias, rank_size, 0)
        projection_weight = np.split(projection_weight, rank_size, 0)
        mapping_bias = mapping_bias[part_id]
        projection_weight = projection_weight[part_id]
    else:
        if rank_id in [0, 2]:
            part_id = 0
        else:
            part_id = 1
        mapping_weight = np.split(mapping_weight, 2, -1)
        mapping_bias = np.split(mapping_bias, 2, -1)
        projection_weight = np.split(projection_weight, 2, 0)
        mapping_weight = mapping_weight[part_id]
        mapping_bias = mapping_bias[part_id]
        projection_weight = projection_weight[part_id]

    network.mlp.mapping.weight.set_data(ms.Tensor(mapping_weight))
    network.mlp.mapping.bias.set_data(ms.Tensor(mapping_bias))
    network.mlp.projection.weight.set_data(ms.Tensor(projection_weight))
    network.mlp.projection.bias.set_data(ms.Tensor(projection_bias))
    network.mlp.projection.bias.set_data(ms.Tensor(projection_bias))

def run_parallel_mlp(generate_golden):
    """
    run pynative mode mlp and load golden ckpt to generate pynative loss
    """
    config_path = "./config/mlp.yaml"
    training_config, parallel_config, optimizer_config, dataset_config, _, model_config = init_configs_from_yaml(
        config_path,
        [TrainingConfig, ModelParallelConfig, OptimizerConfig, DatasetConfig, LoraConfig, TransformerConfig],
    )

    batch_size = dataset_config.batch_size
    dataset_size = dataset_config.dataset_size
    seq_length = dataset_config.seq_length
    hidden_size = model_config.hidden_size

    init()
    initialize_model_parallel(tensor_model_parallel_size=parallel_config.tensor_model_parallel_size)

    ms.set_seed(training_config.seed)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)
    use_zero3 = True
    if generate_golden:
        use_zero3 = False
        parallel_config.zero_level = None
    network = ParallelMLPNet(config=model_config)
    rank_id = get_rank()
    set_weight_seed(network, rank_id, use_zero3)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)
    optimizer = get_optimizer(optimizer_config, training_config, network.trainable_params(), network)

    train(1, dataset, network, optimizer, None, generate_golden=generate_golden)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    args, rest_args = parser.parse_known_args()
    run_parallel_mlp(args.generate_golden)
