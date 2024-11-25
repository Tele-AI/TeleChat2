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
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits, DistributedGradReducer
from mindspore.communication.management import init, get_rank, get_group_size

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, \
    get_data_parallel_world_size, get_data_parallel_group
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformerLayer
from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import VocabParallelEmbedding
from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromSequenceParallelRegion
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    OptimizerConfig,
    DatasetConfig,
    TransformerConfig
)
from tests.st.test_distri_core.utils import TestData


ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
ms.set_context(pynative_synchronize=True)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, generate_golden=False):
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
            input_ids, labels, attn_mask = data
            loss, grads = grad_func(input_ids, attn_mask, labels)
            grads = grads[1]
            if ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1 and generate_golden:
                print("reduce gradients on group {}".format(get_data_parallel_group()))
                grads = grad_reducer(grads)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {} | Step {} | Loss {}".format(epoch, step, loss))
            step += 1
    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)


class ParallelTransformerLayerNet(nn.Cell):
    """build parallel transformer layer net"""
    def __init__(self, config):
        super(ParallelTransformerLayerNet, self).__init__()
        self.config = config
        self.embedding = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            init_method=config.init_method,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel,
            config=config,
        )
        self.layer = ParallelTransformerLayer(layer_number=1, config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=config.dataset_config.data_layout == "BSH",
            tensor_parallel_output_grad=False
        )

    def construct(self, x, attention_mask, labels):
        x = self.embedding(x)
        output = self.layer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        if self.config.parallel_config.sequence_parallel:
            output = self.gather_from_sp_region(output)
        loss = self.loss(output, labels)
        return loss


def set_weight_seed(network, rank_id, zero_level):
    """set init weight data"""
    for param in network.get_parameters():
        print(param, param.shape)
    input_norm_weight = np.random.normal(0, 0.01, size=(2560,)).astype(np.float32)
    post_attention_norm_weight = np.random.normal(0, 0.01, size=(2560,)).astype(np.float32)
    qkv_proj_weight = np.random.normal(0, 0.01, size=(7680, 2560)).astype(np.float32)
    qkv_proj_bias = np.random.normal(0, 0.01, size=(7680,)).astype(np.float32)
    out_proj_weight = np.random.normal(0, 0.01, size=(2560, 2560)).astype(np.float32)
    mapping_weight = np.random.normal(0, 0.01, size=(7680, 2560)).astype(np.float32)
    mapping_bias = np.random.normal(0, 0.01, size=(7680,)).astype(np.float32)
    projection_weight = np.random.normal(0, 0.01, size=(2560, 7680)).astype(np.float32)
    projection_bias = np.random.normal(0, 0.01, size=(2560,)).astype(np.float32)
    print("zero_level", zero_level)
    if zero_level == "z3":
        rank_size = get_group_size()
        qkv_proj_weight = np.split(qkv_proj_weight, rank_size, 0)
        qkv_proj_bias = np.split(qkv_proj_bias, rank_size, 0)
        mapping_weight = np.split(mapping_weight, rank_size, 0)
        mapping_bias = np.split(mapping_bias, rank_size, 0)
        if rank_id == 1:
            part_id = 2
        elif rank_id == 2:
            part_id = 1
        else:
            part_id = rank_id
        qkv_proj_weight = qkv_proj_weight[part_id]
        qkv_proj_bias = qkv_proj_bias[part_id]
        mapping_weight = mapping_weight[part_id]
        mapping_bias = mapping_bias[part_id]

        size_0 = int(projection_weight.shape[0] / 2)
        size_1 = int(projection_weight.shape[1] / 2)
        if rank_id == 0:
            out_proj_weight = out_proj_weight[:size_0, :size_0]
            projection_weight = projection_weight[:size_0, :size_1]
            projection_bias = projection_bias[:size_0]
        elif rank_id == 2:
            out_proj_weight = out_proj_weight[size_0:, :size_0]
            projection_weight = projection_weight[size_0:, :size_1]
            projection_bias = projection_bias[size_0:]
        elif rank_id == 1:
            out_proj_weight = out_proj_weight[:size_0, size_0:]
            projection_weight = projection_weight[:size_0, size_1:]
            projection_bias = projection_bias[:size_0]
        elif rank_id == 3:
            out_proj_weight = out_proj_weight[size_0:, size_0:]
            projection_weight = projection_weight[size_0:, size_1:]
            projection_bias = projection_bias[size_0:]
    else:
        qkv_proj_weight = np.split(qkv_proj_weight, 2, 0)
        qkv_proj_bias = np.split(qkv_proj_bias, 2, 0)
        mapping_weight = np.split(mapping_weight, 2, 0)
        mapping_bias = np.split(mapping_bias, 2, 0)

        out_proj_weight = np.split(out_proj_weight, 2, 1)
        projection_weight = np.split(projection_weight, 2, 1)

        if rank_id in [0, 2]:
            part_id = 0
        else:
            part_id = 1
        qkv_proj_weight = qkv_proj_weight[part_id]
        qkv_proj_bias = qkv_proj_bias[part_id]
        mapping_weight = mapping_weight[part_id]
        mapping_bias = mapping_bias[part_id]
        out_proj_weight = out_proj_weight[part_id]
        projection_weight = projection_weight[part_id]

    network.layer.input_norm.weight.set_data(ms.Tensor(input_norm_weight))
    network.layer.post_attention_norm.weight.set_data(ms.Tensor(post_attention_norm_weight))
    network.layer.attention.qkv_proj.weight.set_data(ms.Tensor(qkv_proj_weight))
    network.layer.attention.qkv_proj.bias.set_data(ms.Tensor(qkv_proj_bias))
    network.layer.mlp.projection.weight.set_data(ms.Tensor(projection_weight))
    network.layer.mlp.projection.bias.set_data(ms.Tensor(projection_bias))

    network.layer.attention.out_proj.weight.set_data(ms.Tensor(out_proj_weight))
    network.layer.mlp.mapping.weight.set_data(ms.Tensor(mapping_weight))
    network.layer.mlp.mapping.bias.set_data(ms.Tensor(mapping_bias))


def run_parallel_transformer_layer(generate_golden, came, zero_level):
    """ Test ParallelTransformer. """
    # np.set_printoptions(suppress=True, threshold=np.inf)
    if not came:
        config_path = "./config/transformer_zero.yaml"
    else:
        config_path = "./config/transformer_came.yaml"
    training_config, parallel_config, optimizer_config, dataset_config, model_config = init_configs_from_yaml(
        config_path, [TrainingConfig, ModelParallelConfig, OptimizerConfig, DatasetConfig, TransformerConfig]
    )

    batch_size = dataset_config.batch_size
    dataset_size = dataset_config.dataset_size
    seq_length = dataset_config.seq_length
    vocab_size = model_config.vocab_size

    init()
    parallel_config.zero_level = zero_level
    tensor_parallel = parallel_config.tensor_model_parallel_size
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(training_config.seed)
    input_data = np.random.randint(0, vocab_size, (dataset_size, seq_length)).astype(np.int32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    network = ParallelTransformerLayerNet(config=model_config)
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

    print(f"optimizer: {optimizer_config.optimizer_type}, zero_level: {parallel_config.zero_level}")
    optimizer = get_optimizer(optimizer_config, training_config, network.trainable_params(), network)


    train(1, dataset, network, optimizer, None, generate_golden=generate_golden)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    parser.add_argument(
        '--came', action='store_true', help="Generate golden data for test."
    )
    parser.add_argument(
        '--zero_level', help="zero level", default=None
    )
    args, rest_args = parser.parse_known_args()
    run_parallel_transformer_layer(args.generate_golden, args.came, args.zero_level)
