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
""" Test ParamAndGradBuffer """
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import init
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.mint.optim import AdamW

from mindformers.experimental.parallel_core.pynative.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, get_data_parallel_world_size, get_data_parallel_rank, get_data_parallel_group
from mindformers.experimental.parallel_core.pynative.config import OptimizerConfig, ModelParallelConfig, TransformerConfig, TrainingConfig
from mindformers.experimental.parallel_core.pynative.distributed import DistributedDataParallel, DistributedDataParallelConfig
from mindformers.experimental.parallel_core.pynative.optimizer.distrib_optimizer import DistributedOptimizer
from tests.st.test_distri_core.utils import TestData, train


class TestNet2(nn.Cell):
    """ test class. """
    def __init__(self, config):
        super(TestNet2, self).__init__()
        hidden_size = config.hidden_size
        self.columnlinear = ColumnParallelLinear(input_size=hidden_size,
                                                 output_size=hidden_size,
                                                 config=config,
                                                 init_method=config.init_method,
                                                 bias=config.mlp_has_bias,
                                                 gather_output=False,
                                                 skip_bias_add=False,
                                                 bias_init=config.bias_init)
        self.rowlinear = RowParallelLinear(input_size=hidden_size,
                                           output_size=hidden_size,
                                           config=config,
                                           init_method=config.init_method,
                                           bias=config.mlp_has_bias,
                                           input_is_parallel=True,
                                           skip_bias_add=False,
                                           bias_init=config.bias_init)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, input_, label_):
        output, _ = self.columnlinear(input_)
        output, _ = self.rowlinear(output)
        output = ops.sum(output, dim=-1, keepdim=False)
        output = ops.cast(output, mstype.float32)
        loss = self.loss(output, label_)

        return loss


def run_golden_optimizer():
    """
    Feature: test DDP with DistributedOptimizer
    Description: test DDP with DistributedOptimizer
    Expectation: test success
    """
    batch_size = 1
    dataset_size = 6
    seq_length = 8
    hidden_size = 4
    tensor_parallel = 1

    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel,
                              order='tp-dp-pp')
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                  num_shards=get_data_parallel_world_size(),
                                  shard_id=get_data_parallel_rank())
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()
    model_config = TransformerConfig(vocab_size=40000,
                                     num_layers=1,
                                     num_attention_heads=1,
                                     mlp_has_bias=True,
                                     gated_linear_unit=False,
                                     hidden_size=hidden_size,
                                     ffn_hidden_size=4*hidden_size,
                                     hidden_act='gelu',
                                     parallel_config=parallel_config,
                                     params_dtype='float32',
                                     compute_dtype='float32')
    network_golden = TestNet2(config=model_config)
    optimizer_golden = AdamW(params=network_golden.get_parameters(), lr=1.0)

    train(epoch_num=1,
          dataset=dataset,
          network=network_golden,
          optimizer=optimizer_golden)


def run_distributed_optimizer():
    """
    Feature: test DDP with DistributedOptimizer
    Description: test DDP with DistributedOptimizer
    Expectation: test success
    """
    batch_size = 1
    dataset_size = 6
    seq_length = 8
    hidden_size = 4
    tensor_parallel = 1
    bucket_size = 10

    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                  num_shards=get_data_parallel_world_size(),
                                  shard_id=get_data_parallel_rank())
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()
    training_config = TrainingConfig(parallel_config=parallel_config)
    optimizer_config = OptimizerConfig(parallel_config=parallel_config)
    model_config = TransformerConfig(vocab_size=40000,
                                     num_layers=1,
                                     num_attention_heads=1,
                                     mlp_has_bias=True,
                                     gated_linear_unit=False,
                                     hidden_size=hidden_size,
                                     ffn_hidden_size=4*hidden_size,
                                     hidden_act='gelu',
                                     parallel_config=parallel_config,
                                     params_dtype='float32',
                                     compute_dtype='float32')
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        bucket_size=bucket_size,
        average_in_collective=True,
        enable_mem_align=True,
    )
    network = TestNet2(config=model_config)
    network_with_ddp = DistributedDataParallel(config=training_config,
                                               ddp_config=ddp_config,
                                               module=network)

    optimizer = AdamW(params=network_with_ddp.get_parameters(), lr=1.0)
    optimizer = DistributedOptimizer(optimizer=optimizer,
                                     config=optimizer_config,
                                     grad_scaler=None,
                                     init_state_fn=None,
                                     per_model_buffers=network_with_ddp.buffers,
                                     data_parallel_group=get_data_parallel_group(with_context_parallel=True))

    losses = train(epoch_num=1,
                   dataset=dataset,
                   network=network_with_ddp,
                   optimizer=optimizer)

    losses = list(map(lambda x: x[0], losses))
    if get_data_parallel_rank() == 0:
        golden_losses = [2.0796428, 4.841238, 0.6431562]
    elif get_data_parallel_rank() == 1:
        golden_losses = [2.079449, 2.7259526, 12.919599]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--golden', action='store_true', help='Generate golden data for test'
    )
    args, _ = parser.parse_known_args()

    if args.golden:
        run_golden_optimizer()
    else:
        run_distributed_optimizer()
