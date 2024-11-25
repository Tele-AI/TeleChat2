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
""" Test Mixtral. """
import argparse
import os

import numpy as np
import torch
from mindformers.experimental.parallel_core.pynative.config import init_configs_from_yaml
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
    get_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_rank,
    initialize_model_parallel,
)

from mindformers.experimental.parallel_core.pynative.training import get_model, TrainOneStepCell, train, get_loss_func

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import SGD

from tests.st.test_distri_core.utils import MixtralModel, transform_mixtral_golden_params_to_pynative_params

class TestData:
    """
    generate a test dataset
    """
    def __init__(self, dataset_size=None, input_data=None, label_data=None):
        super().__init__()

        self.dataset_size = dataset_size
        self.input_data = input_data
        self.data_shape = self.input_data.shape
        self.label_data = label_data
        seq_length = self.data_shape[1]
        self.attention_mask = np.tril(np.ones(shape=(1, seq_length-1, seq_length-1))).astype(np.int32)
        self.attention_mask = self.attention_mask < 0.5

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))

    def __len__(self):
        return self.dataset_size


def run_mixtral(config):
    """ Test ParallelTransformer. """
    print(f"config is:\n{config}")
    model_config = config.model_config
    parallel_config = config.parallel_config
    training_config = config.training_config
    tp = parallel_config.tensor_model_parallel_size
    ep = parallel_config.expert_model_parallel_size
    pp = parallel_config.pipeline_model_parallel_size
    vpp = parallel_config.virtual_pipeline_model_parallel_size
    seq_length = model_config.seq_length
    micro_batch_num = config.dataset_config.micro_batch_num

    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        max_device_memory="58GB",
        deterministic='ON',
        pynative_synchronize=True
        )

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp
        )

    dp_group = get_data_parallel_group()
    ep_group = get_expert_model_parallel_group()
    tp_group = get_tensor_model_parallel_group()
    pp_group = get_pipeline_model_parallel_group()
    dp_rank = get_data_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    print(f"dp_group is {dp_group}, ep_group is {ep_group}, tp_group is {tp_group}, pp_group is {pp_group}", flush=True)
    print(f"dp_rank is {dp_rank}, ep_rank is {ep_rank}, tp_rank is {tp_rank}, pp_rank is {pp_rank}", flush=True)

    ms.set_seed(2024)

    golden_input_and_loss_path = config.dataset_config.dataset_dir

    # load golden input and loss
    assert os.path.exists(golden_input_and_loss_path), f"'{golden_input_and_loss_path}' did not exits"
    input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
    input_data = input_and_loss['input']
    assert input_data.shape == (2, seq_length+1), \
           f"expect input.shape == (2, {seq_length+1}), but got {input_data.shape}"

    # making dataset
    if ep == 1:
        dataset = TestData(dataset_size=2, input_data=input_data, label_data=input_data)
        dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
        dataset = dataset.batch(2)
    else:
        input_data = np.tile(input_data[dp_rank % 2, None], (micro_batch_num, 1))
        dataset = TestData(dataset_size=micro_batch_num, input_data=input_data, label_data=input_data)
        dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
        dataset = dataset.batch(micro_batch_num)

    # build net
    def model_provider_func(pre_process=True, post_process=True):
        """ get mixtral model """
        loss = get_loss_func(training_config)
        network = MixtralModel(
            model_config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
            )
        return network

    network = get_model(model_provider_func, training_config)

    print(f"network construct is:\n{network}")
    print("network parameters are:")
    for param in network.get_parameters():
        print(f"{param.name} {param.dtype} {param.shape}")

    # load golden ckpt
    golden_ckpt_path = config.training_config.checkpoint_dir

    assert os.path.exists(golden_ckpt_path), f"'{golden_ckpt_path}' did not exits"
    golden_params = torch.load(golden_ckpt_path, map_location=torch.device('cpu'))

    print("ckpt parameters are:")
    for name, param in golden_params.items():
        if isinstance(param, torch.Tensor):
            print(f"{name} {param.dtype} {param.shape}")

    # transform ckpt
    new_params = transform_mixtral_golden_params_to_pynative_params(
        golden_params,
        model_config
        )
    param_not_load, ckpt_not_load = ms.load_param_into_net(network, new_params)
    new_param_not_load = []
    for param_name in param_not_load:
        if "set_hidden_states" in param_name:
            continue
        new_param_not_load.append(param_name)
    param_not_load = new_param_not_load
    assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    print(f"ckpt_not_load are:\n{ckpt_not_load}", flush=True)


    optimizer = SGD(params=network.trainable_params(), learning_rate=1e-4)
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)
    # train
    train(train_one_step_cell, dataset, training_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config_mixtral_small.yaml", help="test config")
    parser.add_argument('--ep', type=int, default=2, help="expert_model_parallel_size")
    parser.add_argument('--tp', type=int, default=2, help="tensor_parallel")
    parser.add_argument('--pp', type=int, default=1, help="pipeline_parallel")
    parser.add_argument('--sp', action='store_true', help="use sequence parallel.")
    parser.add_argument('--bs', type=int, default=1, help="batch size")
    parser.add_argument('--mbn', type=int, default=1, help="micro batch num")
    parser.add_argument('--num_layers', type=int, default=2, help="micro batch num")
    parser.add_argument('--vpp', type=int, default=None, help="micro batch num")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="micro batch num")

    cli_args, rest_args = parser.parse_known_args()

    all_config = init_configs_from_yaml(cli_args.config_path)

    all_config.parallel_config.tensor_model_parallel_size = cli_args.tp
    all_config.parallel_config.expert_model_parallel_size = cli_args.ep
    all_config.parallel_config.pipeline_model_parallel_size = cli_args.pp
    all_config.parallel_config.sequence_parallel = cli_args.sp
    all_config.parallel_config.virtual_pipeline_model_parallel_size = cli_args.vpp
    all_config.dataset_config.batch_size = cli_args.bs
    all_config.dataset_config.micro_batch_num = cli_args.mbn
    all_config.model_config.num_layers = cli_args.num_layers
    if cli_args.checkpoint_dir is not None and os.path.exists(cli_args.checkpoint_dir):
        all_config.training_config.checkpoint_dir = cli_args.checkpoint_dir

    run_mixtral(all_config)
