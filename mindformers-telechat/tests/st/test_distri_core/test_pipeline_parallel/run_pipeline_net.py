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
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam
from mindspore.mint.optim import AdamW
from mindspore.communication import init
from mindformers.experimental.parallel_core.pynative.optimizer.distrib_optimizer import DistributedOptimizer
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    initialize_model_parallel,
    get_data_parallel_group,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size
)

from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train, get_model
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    TransformerConfig,
    DatasetConfig,
    OptimizerConfig
)
from test_pipeline_net import PipelineTestNet, FakeData

ms.set_seed(2024)

def generate_ckpt(vocab_size, seq_length, hidden_size, num_layers, share_weight=False):
    """ get ckpt dict """
    ckpt = {}
    embedding_param = ms.Parameter(ms.Tensor(np.random.random((vocab_size, hidden_size)),
                                             ms.float32),
                                   name='embedding.weight')
    ckpt['embedding.weight'] = embedding_param
    for i in range(num_layers):
        idx = i
        # first
        param_name = f'fake_transformer.fake_transformer_layers.{idx}.first_liner.weight'
        ckpt[param_name] = ms.Parameter(ms.Tensor(np.random.random((seq_length, hidden_size)), ms.float32),
                                        name=param_name)
        # second
        param_name = f'fake_transformer.fake_transformer_layers.{idx}.second_liner.weight'
        ckpt[param_name] = ms.Parameter(ms.Tensor(np.random.random((hidden_size, seq_length)), ms.float32),
                                        name=param_name)
    if not share_weight:
        ckpt['fake_head.weight'] = ms.Parameter(ms.Tensor(np.random.random((vocab_size, hidden_size)),
                                                          ms.float32),
                                                name='fake_head.weight')
    elif get_pipeline_model_parallel_world_size() > 1:
        ckpt['fake_head.weight'] = embedding_param
    ckpt['final_norm.beta'] = ms.Parameter(ms.Tensor(np.zeros((hidden_size,)),
                                                     ms.float32),
                                           name='final_norm.beta')
    ckpt['final_norm.gamma'] = ms.Parameter(ms.Tensor(np.ones((hidden_size,)),
                                                      ms.float32),
                                            name='final_norm.gamma')
    return ckpt


def run_pipeline(training_config, model_config, parallel_config, dataset_config, overlap_param_gather=False):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()

    # init context
    pp = parallel_config.pipeline_model_parallel_size
    if parallel_config.virtual_pipeline_model_parallel_size is not None and \
       parallel_config.virtual_pipeline_model_parallel_size > 1:
        vpp = parallel_config.virtual_pipeline_model_parallel_size
    else:
        vpp = None

    initialize_model_parallel(pipeline_model_parallel_size=pp,
                              virtual_pipeline_model_parallel_size=vpp)
    print("pp stage num: {}".format(pp), flush=True)
    print("vpp size: {}".format(vpp), flush=True)
    print("dp group {} | pp group {}".format(get_data_parallel_group(), \
                                             get_pipeline_model_parallel_group()), flush=True)
    print("current pp rank {}".format(get_pipeline_model_parallel_rank()), flush=True)

    # get ckpt
    ckpt_dict = generate_ckpt(model_config.vocab_size,
                              model_config.seq_length,
                              model_config.hidden_size,
                              model_config.num_layers,
                              not model_config.untie_embeddings_and_output_weights)
    # generate dataset
    dataset = FakeData(data_num=32, seq_length=model_config.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, training_config)

    if training_config.wrap_with_ddp and training_config.use_distributed_optimizer:
        optimizer_config = OptimizerConfig(parallel_config=parallel_config)
        if overlap_param_gather:
            optimizer_config.overlap_param_gather = True
        optimizer = AdamW(params=network.trainable_params(), lr=0.001)
        per_model_buffers = {}
        for model_idx, model_chunk in enumerate(network):
            per_model_buffers[model_idx] = model_chunk.buffers
        optimizer = DistributedOptimizer(optimizer=optimizer,
                                         config=optimizer_config,
                                         grad_scaler=None,
                                         init_state_fn=None,
                                         per_model_buffers=per_model_buffers,
                                         data_parallel_group=get_data_parallel_group(with_context_parallel=True))
    else:
        optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    train(train_one_step_cell, dataset_parallel, training_config)


def run_standalone(training_config, model_config, parallel_config, dataset_config):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()

    initialize_model_parallel()

    # get ckpt
    ckpt_dict = generate_ckpt(model_config.vocab_size,
                              model_config.seq_length,
                              model_config.hidden_size,
                              model_config.num_layers,
                              not model_config.untie_embeddings_and_output_weights)

    # generate dataset
    dataset = FakeData(data_num=32, seq_length=model_config.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, training_config)

    if training_config.wrap_with_ddp and training_config.use_distributed_optimizer:
        optimizer_config = OptimizerConfig(parallel_config=parallel_config)
        optimizer = AdamW(params=network.trainable_params(), lr=0.001)
        per_model_buffers = {}
        for model_idx, model_chunk in enumerate(network):
            per_model_buffers[model_idx] = model_chunk.buffers
        optimizer = DistributedOptimizer(optimizer=optimizer,
                                         config=optimizer_config,
                                         grad_scaler=None,
                                         init_state_fn=None,
                                         per_model_buffers=per_model_buffers,
                                         data_parallel_group=get_data_parallel_group(with_context_parallel=True))
    else:
        optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    train(train_one_step_cell, dataset_parallel, training_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='pp',
                        help="pp: run pp process standalone: run standalone process")
    args = parser.parse_args()
    CONFIG_PATH = "test_pipeline.yaml"
    assert os.path.exists(CONFIG_PATH) and CONFIG_PATH.endswith(('.yaml', '.yml'))
    training_config_main, parallel_config_main, dataset_config_main, model_config_main = init_configs_from_yaml(
        CONFIG_PATH, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig]
    )

    if args.run_mode == 'standalone_without_share':
        run_standalone(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'standalone_with_share':
        model_config_main.untie_embeddings_and_output_weights = False
        run_standalone(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'standalone_with_ddp':
        model_config_main.untie_embeddings_and_output_weights = False
        training_config_main.wrap_with_ddp = True
        training_config_main.use_distributed_optimizer = True
        run_standalone(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'pp_without_share':
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'pp_with_share':
        model_config_main.untie_embeddings_and_output_weights = False
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'custom_pp_with_share':
        model_config_main.untie_embeddings_and_output_weights = False
        parallel_config_main.num_layer_list = [2, 3, 1, 2]
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'pp_interleaved':
        parallel_config_main.virtual_pipeline_model_parallel_size = 2
        model_config_main.untie_embeddings_and_output_weights = False
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'pp_with_ddp':
        model_config_main.untie_embeddings_and_output_weights = False
        training_config_main.wrap_with_ddp = True
        training_config_main.use_distributed_optimizer = True
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'custom_pp_interleaved':
        parallel_config_main.virtual_pipeline_model_parallel_size = 2
        model_config_main.untie_embeddings_and_output_weights = False
        parallel_config_main.num_layer_list = [[1, 1, 1, 1], [1, 1, 1, 1]]
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main)
    elif args.run_mode == 'pp_with_ddp_overlap':
        model_config_main.untie_embeddings_and_output_weights = False
        training_config_main.wrap_with_ddp = True
        training_config_main.use_distributed_optimizer = True
        training_config_main.overlap_grad_reduce = True
        training_config_main.delay_grad_reduce = False
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main, True)
    elif args.run_mode == 'pp_with_ddp_overlap_delay':
        model_config_main.untie_embeddings_and_output_weights = False
        training_config_main.wrap_with_ddp = True
        training_config_main.use_distributed_optimizer = True
        training_config_main.overlap_grad_reduce = True
        training_config_main.delay_grad_reduce = True
        run_pipeline(training_config_main, model_config_main, parallel_config_main, dataset_config_main, True)
