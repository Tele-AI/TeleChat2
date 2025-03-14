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
"""run parallel ddp zero3"""

import os

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import CrossEntropyLoss

from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train, get_model
from mindformers.experimental.parallel_core.pynative.config import OptimizerConfig, init_configs_from_yaml, DatasetConfig
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig, TrainingConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelMLP
from tests.st.test_distri_core.utils import TestData

class ParallelMLPNet(nn.Cell):
    """
    define a pynative MLP net
    """
    def __init__(self, config):
        super(ParallelMLPNet, self).__init__()
        self.mlp0 = ParallelMLP(config=config)
        self.mlp1 = ParallelMLP(config=config)
        self.mlp2 = ParallelMLP(config=config)
        self.mlp3 = ParallelMLP(config=config)
        self.mlp4 = ParallelMLP(config=config)
        self.mlp5 = ParallelMLP(config=config)
        self.mlp6 = ParallelMLP(config=config)
        self.mlp7 = ParallelMLP(config=config)

        self.loss = CrossEntropyLoss()
        self.cast = ops.Cast()
        self.dtype = config.compute_dtype

    def construct(self, input_ids, labels):
        """ do construct and calc mean loss """
        input_id = ops.cast(input_ids, mstype.bfloat16)
        output, _ = self.mlp0(input_id)
        output, _ = self.mlp1(output)
        output, _ = self.mlp2(output)
        output, _ = self.mlp3(output)
        output, _ = self.mlp4(output)
        output, _ = self.mlp5(output)
        output, _ = self.mlp6(output)
        output, _ = self.mlp7(output)

        labels = labels
        loss = output.abs().mean()
        return loss

def run_parallel_ddp_zero3():
    """
    run pynative mode in ddp zero3
    """

    config_path = "test_zero3.yaml"
    assert os.path.exists(config_path) and config_path.endswith(('.yaml', '.yml'))
    training_config, parallel_config, dataset_config, model_config, optimizer_config = init_configs_from_yaml(
        config_path, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig, OptimizerConfig]
    )

    dataset_size = 20
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, pynative_synchronize=True, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=parallel_config.tensor_model_parallel_size)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, model_config.seq_length, model_config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, model_config.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(dataset_config.batch_size)

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        network = ParallelMLPNet(config=model_config)
        return network
    network = get_model(model_provider_func, training_config)
    print("buffer:", network[0].buffers)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)

    from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
    optimizer = get_optimizer(optimizer_config, training_config, network.trainable_params(), network)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)
    print(f"network trainable params: {network.trainable_params()}", flush=True)

    train(train_one_step_cell, dataset, training_config)



if __name__ == '__main__':
    run_parallel_ddp_zero3()
