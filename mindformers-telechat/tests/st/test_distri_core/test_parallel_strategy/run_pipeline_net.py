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
"""Pipeline parallel test"""

import os
import time
import argparse
from collections import OrderedDict
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import AdamWeightDecay
from mindspore.communication import init, get_group_size
from mindspore.communication.management import get_rank
# pylint: disable=W0611
from mindformers import MindFormerConfig
from mindformers.models.utils import convert_mstype
from mindformers.wrapper import MFTrainOneStepCell
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    initialize_model_parallel,
    is_pipeline_last_stage,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.training import PipelineTrainOneStepCell
from mindformers.experimental.parallel_core.pynative.pipeline_parallel import pipelining_1F1B_without_interleaved, \
                                                                    PipelineCell
from mindformers.experimental.parallel_core.pynative.utils import generate_state_dict, save_strategy_file
from tests.st.test_distri_core.utils import rearrange_the_ckpt_files
from test_pipeline_net import PipelineTestNet, FakeData

ms.set_seed(2024)

def train(epoch_num, dataset, network, optimizer, config, is_pipeline=True):
    """ trainning process """
    loss_list = []
    if is_pipeline:
        micro_batch_num = config.parallel_config.micro_batch_num
        train_one_step_cell = PipelineTrainOneStepCell(network,
                                                       optimizer,
                                                       config,
                                                       use_clip_grad=False,
                                                       max_grad_norm=1.0,
                                                       scale_sense=1.0,
                                                       micro_batch_num=micro_batch_num)
    else:
        train_one_step_cell = MFTrainOneStepCell(network,
                                                 optimizer,
                                                 use_clip_grad=False,
                                                 max_grad_norm=1.0,
                                                 scale_sense=1.0)
    print("PipelineTrainOneStepCell init finish!")
    for epoch in range(epoch_num):
        for step, inputs in enumerate(dataset.create_tuple_iterator()):
            start_time = time.time()
            if is_pipeline:
                loss, overflow, loss_scale, learning_rate = train_one_step_cell(pipelining_1F1B_without_interleaved,
                                                                                *inputs)
                if config.parallel_config.reduction == 'mean':
                    loss /= micro_batch_num
            else:
                loss, overflow, loss_scale, learning_rate = train_one_step_cell(*inputs)
            end_time = time.time()
            print("Epoch: {} | Step: {} | Loss: {} | overflow cond: {} | lr: {} | loss scale: {} | cost time: {}" \
                            .format(epoch, step + 1, loss, overflow,
                                    learning_rate, loss_scale, end_time - start_time), flush=True)
            loss_list.append(loss.asnumpy())
    return loss_list


def model_forward(network, input_ids, labels=None, attention_mask=None, recv_data=None):
    """ model pipeline forward process for model_customize_staged"""
    outputs = None

    # embedding layer
    if get_pipeline_model_parallel_rank() == 0:
        outputs, _ = network.embedding(input_ids)

    # if recv_data if not None, correct the first input of transformer layer
    if recv_data is not None:
        outputs = recv_data

    if network.transformer_layers is not None:
        for layer in network.transformer_layers:
            outputs = layer(outputs, attention_mask)

    # postprocess layers
    if get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_world_size() - 1:
        # final norm layer
        outputs = network.final_norm(outputs)
        # head layer
        outputs = network.head(outputs)
        # loss
        outputs = network.loss(outputs, labels)

    return outputs

def get_network_and_optimizer_and_dataset(config, pp):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()

    # init config
    np.random.seed(2024)
    config.parallel_config.pipeline_dtype = convert_mstype(config.parallel_config.pipeline_dtype)
    print(config)

    # init context
    initialize_model_parallel(pp_size=pp)
    print("pp stage num: {}".format(pp), flush=True)
    print("dp group {} | pp group {}".format(get_data_parallel_group(), \
                                             get_pipeline_model_parallel_group()), flush=True)
    print("current pp rank {}".format(get_pipeline_model_parallel_rank()), flush=True)

    # generate dataset
    dataset = FakeData(data_num=320, seq_length=config.model_config.seq_length)
    num_shards = get_data_parallel_world_size()
    shard_id = get_data_parallel_rank()
    print("dataset num shards {} | shard id {}".format(num_shards, shard_id))
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False, \
                                           num_shards=num_shards, shard_id=shard_id)
    # calculate global batch size
    global_batch_size = config.training.batch_size * config.parallel_config.micro_batch_num
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    network = PipelineTestNet(config)

    # get split model
    map_dict = OrderedDict({
        'preprocess': network.preprocess,
        'embedding': network.embedding,
        'transformer_layers': network.fake_transformer_layers,
        'final_norm': network.final_norm,
        'head': network.fake_head,
        'loss': network.total_loss,
    })

    if config.parallel_config.model_customize_staged:
        network = PipelineCell(network, map_dict, model_customize_staged=True, model_forward_func=model_forward)
    else:
        network = PipelineCell(network, map_dict)
    optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)
    network.set_train()

    return network, optimizer, dataset_parallel

src_strategy_path = "pp_strategy_ckpt_src"
dst_strategy_path = "pp_strategy_ckpt_dst"
src_network_ckpt = "pp_network_ckpt_src"
dst_network_ckpt = "pp_network_ckpt_dst"

def run_pipeline_src(config):
    """run pp mode to generate src ckpt"""
    (network, optimizer, dataset_parallel) = get_network_and_optimizer_and_dataset(config, pp=4)

    shard_info = generate_state_dict(network, optimizer)

    strategy_file = f"./{src_strategy_path}/rank_{get_rank()}/strategy.ckpt"
    save_strategy_file(shard_info, strategy_file)

    ckpt_path = f"./{src_network_ckpt}/rank_{get_rank()}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ms.save_checkpoint(optimizer, ckpt_path + "/network.ckpt")

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    loss = train(epoch_num=config.training.epochs, dataset=dataset_parallel, \
                network=network, optimizer=optimizer, config=config)

    return loss


def run_pipeline_dst(config):
    """run pp mode to transform src ckpt to dst ckpt and load the params"""
    (network, optimizer, dataset_parallel) = get_network_and_optimizer_and_dataset(config, pp=8)

    shard_info = generate_state_dict(network, optimizer)

    strategy_file = f"./{dst_strategy_path}/rank_{get_rank()}/strategy.ckpt"
    save_strategy_file(shard_info, strategy_file)

    time.sleep(5) #barrier()

    if get_rank() == 0:
        src_merged_strategy_file = "./src_merged_strategy.ckpt"
        dst_merged_strategy_file = "./dst_merged_strategy.ckpt"
        rearrange_the_ckpt_files(src_strategy_path, world_size=get_group_size())
        ms.merge_pipeline_strategys(src_strategy_path, src_merged_strategy_file)
        rearrange_the_ckpt_files(dst_strategy_path, world_size=get_group_size())
        ms.merge_pipeline_strategys(dst_strategy_path, dst_merged_strategy_file)
        ms.transform_checkpoints(src_network_ckpt, dst_network_ckpt, "dst_checkpoint",
                                 f"./src_merged_strategy.ckpt",
                                 f"./dst_merged_strategy.ckpt")
    else:
        time.sleep(15) #barrier()
    dst_params = ms.load_checkpoint(f"./{dst_network_ckpt}/rank_{get_rank()}/dst_checkpoint{get_rank()}.ckpt")
    param_not_load, _ = ms.load_param_into_net(optimizer, dst_params)
    print("param_not_load:", param_not_load)

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    loss = train(epoch_num=config.training.epochs, dataset=dataset_parallel, \
                network=network, optimizer=optimizer, config=config)

    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_src_strategy', action='store_true', help="Generate src strategy."
    )
    args = parser.parse_args()
    config_path = "test_pipeline.yaml"
    assert os.path.exists(config_path) and config_path.endswith(('.yaml', '.yml'))
    trainning_config = MindFormerConfig(os.path.realpath(config_path))

    if args.generate_src_strategy:
        pp_loss = run_pipeline_src(config=trainning_config)
        if is_pipeline_last_stage():
            np.save('./pp_src_log/pp_loss.npy', np.array(pp_loss))
    else:
        pp_loss = run_pipeline_dst(config=trainning_config)
        if is_pipeline_last_stage():
            np.save('./pp_dst_log/pp_loss.npy', np.array(pp_loss))
