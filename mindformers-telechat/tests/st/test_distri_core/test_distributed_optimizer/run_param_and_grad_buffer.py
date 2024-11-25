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
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel

from mindformers.experimental.parallel_core.pynative.distributed.param_and_grad_buffer import ParamAndGradBuffer
from mindformers.experimental.parallel_core.pynative.distributed.distributed_data_parallel import DistributedDataParallel
from utils import TestData, train, get_config_and_model

def run_bucket_sizes(bucket_size, use_distributed_optimizer):
    """
    Feature: boundary test for ParamAndGradBuffer
    Description: test class ParamAndGradBuffer with different bucket_size
    Expectation: test success
    """
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(order='tp-dp-pp')

    seq_length = 8
    parallel_config, network = get_config_and_model(
        seq_length=seq_length,
        bucket_size=bucket_size,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    buffer = ParamAndGradBuffer(
        parallel_config,
        param_dtype=mstype.float32,
        grad_dtype=mstype.float32,
        params=network.trainable_params(),
        bucket_size=parallel_config.bucket_size
    )

    padded_bucket_numels = [bucket.grad_data.numel() for bucket in buffer.buckets]
    unpadded_bucket_numels = [bucket.numel_unpadded for bucket in buffer.buckets]

    expected_padded_bucket_numels = []
    expected_unpadded_bucket_numels = []

    def _pad_bucket(bucket_numel):
        """ pad bucket if needed """
        pad_unit = 8
        if use_distributed_optimizer:
            return bucket_numel - (bucket_numel // pad_unit) * pad_unit
        return 0

    bucket_numel = 0
    for param in network.get_parameters():
        bucket_numel = bucket_numel + param.numel()
        if bucket_size is not None and bucket_numel >= bucket_size:
            expected_unpadded_bucket_numels.append(bucket_numel)
            expected_padded_bucket_numels.append(bucket_numel + _pad_bucket(bucket_numel))
            bucket_numel = 0
    if bucket_numel > 0:
        expected_unpadded_bucket_numels.append(bucket_numel)
        expected_padded_bucket_numels.append(bucket_numel + _pad_bucket(bucket_numel))

    assert padded_bucket_numels == expected_padded_bucket_numels
    assert unpadded_bucket_numels == expected_padded_bucket_numels

def run_ddp_loss(bucket_size):
    """
    Feature: ddp st test
    Description: test ddp loss with the baseline
    Expectation: test success
    """
    batch_size = 1
    dataset_size = 10
    seq_length = 8
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel, order='tp-dp-pp')

    parallel_config, network = get_config_and_model(
        seq_length=seq_length,
        bucket_size=bucket_size,
        use_distributed_optimizer=False
    )

    input_data = np.random.random((dataset_size, seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(batch_size)


    optimizer = AdamWeightDecay(params=network.get_parameters())
    train(epoch_num=1,
          dataset=dataset,
          network=network,
          optimizer=optimizer,
          with_attn_input=False)
    print("---------------------With DistributedDataParallel------------------------")
    ms.set_seed(2024)
    parallel_config, network_with_ddp = get_config_and_model(
        seq_length=seq_length,
        bucket_size=bucket_size,
        use_distributed_optimizer=False
    )
    network_with_ddp = DistributedDataParallel(parallel_config, network_with_ddp)

    optimizer = AdamWeightDecay(params=network_with_ddp.get_parameters())
    train(epoch_num=1,
          dataset=dataset,
          network=network_with_ddp,
          optimizer=optimizer,
          with_attn_input=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,)
    parser.add_argument('--bucket_size', default=120, required=False)
    parser.add_argument('--use_distributed_optimizer', default=False, type=bool, required=False)

    args, rest_args = parser.parse_known_args()
    bucket_limit = int(args.bucket_size) if args.bucket_size != "None" else None
    if args.mode == "bucket":
        run_bucket_sizes(bucket_limit, args.use_distributed_optimizer)
    elif args.mode == "ddp":
        run_ddp_loss(bucket_limit)
