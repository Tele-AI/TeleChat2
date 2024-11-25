"""Utils for Test ParamAndGradBuffer"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits, DistributedGradReducer
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig
from mindformers.experimental.parallel_core.pynative.tensor_parallel import ColumnParallelLinear
from mindformers.experimental.parallel_core.pynative.distributed.distributed_data_parallel import DistributedDataParallel
from mindformers.experimental.parallel_core.pynative.optimizer.distrib_optimizer import DistributedOptimizer
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)

class TestNet(nn.Cell):
    """ Class TestNet """
    def __init__(self,
                 input_size,
                 output_size,
                 parallel_config,
                 weight_init,
                 bias_init,
                 transpose_b,
                 gather_output):
        super(TestNet, self).__init__()
        self.linear1 = ColumnParallelLinear(input_size, output_size, parallel_config, weight_init, bias_init,
                                            bias=False, transpose_b=transpose_b, gather_output=gather_output,
                                            param_init_type=mstype.float32, compute_dtype=mstype.float32)
        self.linear2 = ColumnParallelLinear(input_size, output_size, parallel_config, weight_init, bias_init,
                                            bias=False, transpose_b=transpose_b, gather_output=gather_output,
                                            param_init_type=mstype.float32, compute_dtype=mstype.float32)
        self.linear3 = ColumnParallelLinear(input_size, output_size, parallel_config, weight_init, bias_init,
                                            bias=False, transpose_b=transpose_b, gather_output=gather_output,
                                            param_init_type=mstype.float32, compute_dtype=mstype.float32)
        self.linear4 = ColumnParallelLinear(input_size, output_size, parallel_config, weight_init, bias_init,
                                            bias=False, transpose_b=transpose_b, gather_output=gather_output,
                                            param_init_type=mstype.float32, compute_dtype=mstype.float32)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, inputs, labels):
        output1 = self.linear1(inputs)
        output2 = self.linear2(output1)
        output3 = self.linear3(output2)
        output4 = self.linear4(output3)
        loss = self.loss(output4, labels)
        return loss

class TestData:
    """
    generate a test dataset
    """
    def __init__(self, data_size=None, input_data=None, label_data=None, with_attn_mask=False):
        super().__init__()
        self.with_attn_mask = with_attn_mask
        if input_data is not None:
            assert label_data is not None
            self.input_data = input_data
            self.data_size = self.input_data.shape
        else:
            self.input_data = np.random.random(data_size).astype(np.float32)
            self.data_size = self.input_data.shape
        if label_data is not None:
            assert input_data is not None
            self.label_data = label_data
        else:
            self.label_data = np.zeros(self.data_size[:2]).astype(np.float32)
        for i in range(self.data_size[0]):
            self.label_data[i][0] = 1
        seq_length = self.data_size[1]
        if self.with_attn_mask:
            self.attention_mask = np.tril(np.ones(shape=(1, seq_length, seq_length))).astype(np.uint8)

    def __getitem__(self, index):
        if self.with_attn_mask:
            return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]

def get_config_and_model(
        seq_length,
        bucket_size,
        use_distributed_optimizer=False,
    ):
    """ return specified ModelParallelConfig and TestNet """
    parallel_config = ModelParallelConfig(use_distributed_optimizer=use_distributed_optimizer,
                                          overlap_grad_reduce=True,
                                          use_zero3=False,
                                          bucket_size=bucket_size)

    model = TestNet(input_size=seq_length,
                    output_size=seq_length,
                    parallel_config=parallel_config,
                    weight_init=0.05,
                    bias_init='zeros',
                    transpose_b=True,
                    gather_output=True)

    return parallel_config, model

def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, with_attn_input=False, reduce_grad=True):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=network.trainable_params()
    )
    if reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1 \
        and not isinstance(network, DistributedDataParallel):
        grad_reducer = DistributedGradReducer(network.trainable_params(), group=get_data_parallel_group())
    all_loss = []
    for epoch in range(epoch_num):
        step = 0

        for data in dataset:
            if isinstance(network, DistributedDataParallel):
                network.zero_grad_buffer()
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
            if isinstance(network, DistributedDataParallel):
                network.final_grad_reduce()
                # grads = tuple([x.contiguous() for x in network.grad_views])
            elif reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE and get_data_parallel_world_size() > 1 \
                and not isinstance(network, DistributedDataParallel):
                print("reduce gradients on group {}".format(get_data_parallel_group()))
                grads = grad_reducer(grads)
            if isinstance(optimizer, DistributedOptimizer):
                optimizer()
            else:
                optimizer(grads)
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss)

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss
