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
"""run transformerlayer with zero"""
import argparse
import time
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits, DistributedGradReducer
from mindspore.communication.management import init, get_rank

from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, \
    get_data_parallel_world_size, get_data_parallel_group
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformerLayer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    OptimizerConfig,
    DatasetConfig,
    LoraConfig,
    TransformerConfig
)
from mindformers.experimental.parallel_core.pynative.training.grad_handler import GradAccumulator

from tests.st.test_distri_core.utils import TestData

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
ms.set_context(pynative_synchronize=True, deterministic='ON')
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)


def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, with_attn_input=False, generate_golden=False,
          grad_accumulate=False):
    "train net"
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=0, weights=optimizer.parameters
    )
    if ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1:
        grad_reducer = DistributedGradReducer(optimizer.parameters, group=get_data_parallel_group())
    if grad_accumulate:
        micro_batch_num = 2
        accumulator = GradAccumulator(micro_batch_num)
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            t0 = time.time()
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
                grads = grads[1]
            else:
                loss, grads = grad_func(input_ids, labels)
                grads = grads[1]
            if ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1 and generate_golden:
                print("reduce gradients on group {}".format(get_data_parallel_group()))
                grads = grad_reducer(grads)
            if grad_accumulate:
                grads = accumulator(grads)
            if grads is not None:
                print("do optimizer")
                optimizer(grads)
            t1 = time.time()
            print("Epoch {} | Step {} | Time {} | Loss {} ".format(epoch, step, t1 - t0, loss))
            step += 1
    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)


class ParallelTransformerLayerNet(nn.Cell):
    """build parallel transformer layer net"""

    def __init__(self, config, with_rope=False):
        super(ParallelTransformerLayerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size // config.num_attention_heads,
                                        rotary_percent=1.0)
        self.layer = ParallelTransformerLayer(layer_number=1, config=config)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction="mean")

    def construct(self, x, attention_mask, labels):
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.layer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.layer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def set_weight_seed(network, rank_id, zero_level):
    """set init weight data"""
    for param in network.get_parameters():
        print(param.name, param.shape)
    input_norm_weight = np.random.normal(0, 0.01, size=(2048,)).astype(np.float32)
    post_attention_norm_weight = np.random.normal(0, 0.01, size=(2048,)).astype(np.float32)

    qkv_proj_weight = np.random.normal(0, 0.01, size=(6144, 2048)).astype(np.float32)
    qkv_proj_bias = np.random.normal(0, 0.01, size=(6144,)).astype(np.float32)

    out_proj_weight = np.random.normal(0, 0.01, size=(2048, 2048)).astype(np.float32)
    mapping_weight = np.random.normal(0, 0.01, size=(6144, 2048)).astype(np.float32)
    mapping_bias = np.random.normal(0, 0.01, size=(6144,)).astype(np.float32)

    projection_weight = np.random.normal(0, 0.01, size=(2048, 6144)).astype(np.float32)
    projection_bias = np.random.normal(0, 0.01, size=(2048,)).astype(np.float32)
    dp_size = get_data_parallel_world_size()

    if zero_level == "z3":
        qkv_proj_weight = np.split(qkv_proj_weight, dp_size, 0)
        qkv_proj_bias = np.split(qkv_proj_bias, dp_size, 0)
        mapping_weight = np.split(mapping_weight, dp_size, 0)
        mapping_bias = np.split(mapping_bias, dp_size, 0)

        out_proj_weight = np.split(out_proj_weight, dp_size, 0)
        projection_weight = np.split(projection_weight, dp_size, 0)
        projection_bias = np.split(projection_bias, dp_size, 0)

        part_id = rank_id
        qkv_proj_weight = qkv_proj_weight[part_id]
        qkv_proj_bias = qkv_proj_bias[part_id]
        mapping_weight = mapping_weight[part_id]
        mapping_bias = mapping_bias[part_id]
        out_proj_weight = out_proj_weight[part_id]
        projection_weight = projection_weight[part_id]
        projection_bias = projection_bias[part_id]

    network.layer.input_norm.weight.set_data(ms.Tensor(input_norm_weight))
    network.layer.post_attention_norm.weight.set_data(ms.Tensor(post_attention_norm_weight))

    network.layer.attention.qkv_proj.weight.set_data(ms.Tensor(qkv_proj_weight))
    network.layer.attention.qkv_proj.bias.set_data(ms.Tensor(qkv_proj_bias))
    network.layer.mlp.projection.weight.set_data(ms.Tensor(projection_weight))
    network.layer.mlp.projection.bias.set_data(ms.Tensor(projection_bias))

    network.layer.attention.out_proj.weight.set_data(ms.Tensor(out_proj_weight))
    network.layer.mlp.mapping.weight.set_data(ms.Tensor(mapping_weight))
    network.layer.mlp.mapping.bias.set_data(ms.Tensor(mapping_bias))


def run_parallel_transformer_layer(argument):
    """ Test ParallelTransformer. """
    # np.set_printoptions(suppress=True, threshold=np.inf)
    config_path = "./config/transformer_dp.yaml"
    if argument.param_resident:
        config_path = "./config/param_resident.yaml"
        if argument.param_resident_rate != 1.0:
            config_path = "./config/param_resident_rate.yaml"
    if argument.grad_accumulate_type == 1:
        config_path = "./config/allreduce_over_grad_accumulation.yaml"
    training_config, parallel_config, optimizer_config, dataset_config, _, model_config = init_configs_from_yaml(
        config_path,
        [TrainingConfig, ModelParallelConfig, OptimizerConfig, DatasetConfig, LoraConfig, TransformerConfig],
    )
    generate_golden = argument.generate_golden
    zero_level = argument.zero_level
    parallel_config.zero_level = zero_level
    grad_accumulate = False
    if argument.grad_accumulate_type in [1, 2]:
        grad_accumulate = True
        dataset_config.batch_size = 1

    batch_size = dataset_config.batch_size
    dataset_size = dataset_config.dataset_size
    seq_length = dataset_config.seq_length
    hidden_size = model_config.hidden_size

    init()
    initialize_model_parallel()

    ms.set_seed(training_config.seed)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    network = ParallelTransformerLayerNet(config=model_config, with_rope=True)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    rank_id = get_rank()
    set_weight_seed(network, rank_id, zero_level)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = get_optimizer(optimizer_config, training_config, network.trainable_params(), network)
    train(1, dataset, network, optimizer, None, with_attn_input=True, generate_golden=generate_golden,
          grad_accumulate=grad_accumulate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    parser.add_argument(
        '--zero_level', help="zero level"
    )
    parser.add_argument(
        '--param_resident', action='store_true', help="parameter resident"
    )
    parser.add_argument(
        '--param_resident_rate', default=1.0, type=float, help="parameter resident rate"
    )
    parser.add_argument(
        '--grad_accumulate_type', default=0, type=int, help="gradient accumulation stage"
    )
    args, rest_args = parser.parse_known_args()
    run_parallel_transformer_layer(args)
