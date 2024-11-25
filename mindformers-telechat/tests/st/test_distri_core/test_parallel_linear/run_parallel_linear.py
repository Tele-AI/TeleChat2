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
"""run parallel linear"""
import os
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from mindformers.modules import Linear
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, get_tensor_model_parallel_world_size, get_data_parallel_world_size
from mindformers.experimental.parallel_core.pynative.tensor_parallel import RowParallelLinear, \
    ColumnParallelLinear

from tests.st.test_distri_core.utils import TestData, train, transform_linear_params


class LinearNet(nn.Cell):
    """Golden linear network."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init=0.5,
                 bias_init='zeros'):
        super(LinearNet, self).__init__()
        self.linear = Linear(in_channels=in_channels,
                             out_channels=out_channels,
                             transpose_b=False,
                             weight_init=weight_init,
                             bias_init=bias_init,
                             param_init_type=mstype.float32,
                             compute_dtype=mstype.float32)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, labels):
        output = self.linear(x)
        loss = self.loss(output, labels)
        return loss


class RowParallelLinearNet(nn.Cell):
    """RowParallelLinear network."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_is_parallel=False,
                 config=None,
                 weight_init=0.5,
                 bias_init='zeros'):
        super(RowParallelLinearNet, self).__init__()
        self.linear = RowParallelLinear(input_size=in_channels,
                                        output_size=out_channels,
                                        config=config,
                                        init_method=weight_init,
                                        bias=True,
                                        input_is_parallel=input_is_parallel,
                                        skip_bias_add=False,
                                        bias_init=bias_init)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, labels):
        output, _ = self.linear(x)
        loss = self.loss(output, labels)
        return loss


class ColumnParallelLinearNet(nn.Cell):
    """ColumnParallelLinear network."""
    def __init__(self,
                 in_channels,
                 out_channels,
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
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, labels):
        output, _ = self.linear(x)
        loss = self.loss(output, labels)
        return loss


def generate_golden():
    """Generate golden result."""
    batch_size = 1
    dataset_size = 3
    seq_length = 16

    ms.set_context(device_id=0,
                   device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   deterministic='ON',
                   jit_config={'jit_level': 'O0'})
    init()

    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)

    network = LinearNet(in_channels=seq_length,
                        out_channels=seq_length)
    save_golden = False
    if save_golden:
        ms.save_checkpoint(network, "linear_golden.ckpt")
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None)


def run_rowparallellinear(train_args):
    """test RowParallelLinear."""
    batch_size = 1
    dataset_size = 3
    seq_length = 16
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=tensor_parallel,
        overlap_grad_reduce=train_args.grad_acc,
        gradient_accumulation_fusion=train_args.grad_acc
    )
    config = TransformerConfig(vocab_size=1,
                               num_layers=1,
                               num_attention_heads=1,
                               hidden_size=1,
                               ffn_hidden_size=1,
                               parallel_config=parallel_config)
    network = RowParallelLinearNet(in_channels=seq_length,
                                   out_channels=seq_length,
                                   input_is_parallel=False,
                                   config=config)
    if train_args.froze:
        network.linear.weight.requires_grad = False
        network.linear.bias.requires_grad = False
    save_golden = False
    if save_golden:
        golden_ckpt_path = "linear_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), \
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n" + \
            "`pytest -sv test_parallel_linear.py::TestParallelLinear::generate_golden`"
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_linear_params(golden_params, linear_type="rowparallellinear")
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."

    input_ids = Tensor(shape=(None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [2.7725887, 2.7203748, 2.6545675]
    if train_args.froze:
        golden_losses = [2.7725887, 2.7725887, 2.7725887]
    if train_args.grad_acc:
        golden_losses = [2.7725887, 2.7214084, 2.6558762]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


def run_columnparallellinear(train_args):
    """test ColumnParallelLinear."""
    batch_size = 1
    dataset_size = 3
    seq_length = 16
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=tensor_parallel,
        overlap_grad_reduce=train_args.grad_acc,
        gradient_accumulation_fusion=train_args.grad_acc
    )
    config = TransformerConfig(vocab_size=1,
                               num_layers=1,
                               num_attention_heads=1,
                               hidden_size=1,
                               ffn_hidden_size=1,
                               parallel_config=parallel_config)
    network = ColumnParallelLinearNet(in_channels=seq_length,
                                      out_channels=seq_length,
                                      config=config,
                                      gather_output=True)
    if train_args.froze:
        network.linear.weight.requires_grad = False
    save_golden = False
    if save_golden:
        golden_ckpt_path = "linear_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), \
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n" + \
            "`pytest -sv test_parallel_linear.py::TestParallelLinear::generate_golden`"
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_linear_params(golden_params, linear_type="columnparallellinear")
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."

    input_ids = Tensor(shape=(None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [2.7725887, 2.7203748, 2.6545675]
    if train_args.froze:
        golden_losses = [2.7725887, 2.7725887, 2.7725887]
    if train_args.grad_acc:
        golden_losses = [2.7725887, 2.7214084, 2.6522448]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    parser.add_argument(
        '--linear_type', default='columnparallellinear', help="Type of parallel linear."
    )
    parser.add_argument(
        '--froze', action='store_true', help="Whether to froze weight parameter or not."
    )
    parser.add_argument(
        '--grad_acc', action='store_true', help="Whether to accumulate gradients during linear backward."
    )

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        generate_golden()
    elif args.linear_type == 'columnparallellinear':
        run_columnparallellinear(args)
    elif args.linear_type == 'rowparallellinear':
        run_rowparallellinear(args)
