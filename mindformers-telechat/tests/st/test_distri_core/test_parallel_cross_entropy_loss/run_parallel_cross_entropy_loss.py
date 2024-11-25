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
"""run parallel cross attention"""

import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay

from mindformers.experimental.parallel_core.pynative.config import TrainingConfig, ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, get_tensor_model_parallel_world_size, get_data_parallel_world_size
from mindformers.experimental.parallel_core.pynative.tensor_parallel import ColumnParallelLinear
from mindformers.experimental.parallel_core.pynative.training.loss_func import get_loss_func

from tests.st.test_distri_core.utils import TestData, train


class ColumnParallelLinearNet(nn.Cell):
    """ColumnParallelLinear network."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_fn,
                 config=None,
                 weight_init=0.5,
                 bias_init='zeros',
                 gather_output=True):
        super(ColumnParallelLinearNet, self).__init__()
        self.linear = ColumnParallelLinear(input_size=in_channels,
                                           output_size=out_channels,
                                           config=config,
                                           init_method=weight_init,
                                           bias=True,
                                           gather_output=gather_output,
                                           skip_bias_add=False,
                                           bias_init=bias_init)
        self.loss = loss_fn

    def construct(self, x, labels):
        input_mask = ops.full((4,), 1, dtype=mstype.float32)
        output = self.linear(x)[0]
        labels = labels.reshape(-1)
        output = output.reshape(output.shape[1:])
        loss = self.loss(output, labels, input_mask)
        return loss


def run_parallel_cross_entropy_loss(loss_func_type):
    """test cross entropy loss."""
    batch_size = 1
    dataset_size = 3
    seq_length = 4
    tensor_parallel = args.tp

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.int32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(vocab_size=1,
                               num_layers=1,
                               num_attention_heads=1,
                               hidden_size=1,
                               ffn_hidden_size=1,
                               parallel_config=parallel_config)
    loss_func_kwargs = {"loss_func_type": loss_func_type}
    training_config = TrainingConfig(parallel_config=parallel_config, loss_func_kwargs=loss_func_kwargs)
    loss = get_loss_func(training_config)
    gather_output = False
    if loss_func_type == "CrossEntropyLoss":
        gather_output = True
    network = ColumnParallelLinearNet(in_channels=seq_length,
                                      out_channels=seq_length,
                                      loss_fn=loss,
                                      config=config,
                                      gather_output=gather_output)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.int32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--loss_func_type', default="CrossEntropyLoss", help="Type of loss function."
    )
    parser.add_argument(
        '--tp', type=int, default=1, help="Type of loss function."
    )

    args, rest_args = parser.parse_known_args()
    run_parallel_cross_entropy_loss(args.loss_func_type)
