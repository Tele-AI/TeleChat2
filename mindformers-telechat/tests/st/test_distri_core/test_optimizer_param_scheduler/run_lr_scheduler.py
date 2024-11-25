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
import mindspore as ms
import mindspore.dataset as ds
from mindspore.mint.optim import AdamW
from mindspore.communication import init

from mindformers.experimental.parallel_core.pynative.dist_checkpointing import load_checkpoint
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer_param_scheduler
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    TransformerConfig,
    DatasetConfig, OptimizerConfig,
)
from tests.st.test_distri_core.test_pipeline_parallel.test_pipeline_net import PipelineTestNet, FakeData

ms.set_seed(2024)

def run_lr_scheduler(training_config, model_config, dataset_config, optimizer_config, yaml):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()

    initialize_model_parallel()

    # generate dataset
    dataset = FakeData(data_num=32, seq_length=model_config.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    network = PipelineTestNet(model_config)

    optimizer = AdamW(params=network.trainable_params(), lr=0.001)

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer, optimizer_config, dataset_config, training_config)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, opt_param_scheduler, training_config, model_config)

    train(train_one_step_cell, dataset_parallel, training_config)

    if yaml == 'test_iteration_tarining.yaml':
        assert optimizer.param_groups[0]['lr'] == 2.9999999999999997e-06
        load_checkpoint(model_config, network, optimizer, opt_param_scheduler, f"./output")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_name', type=str, default='test_iteration_tarining.yaml',
                        help="test_iteration_tarining.yaml: test_lr_base_iteration_tarining")
    args = parser.parse_args()
    yaml_name = args.yaml_name
    assert os.path.exists(yaml_name) and yaml_name.endswith(('.yaml', '.yml'))
    training_config_main, dataset_config_main, model_config_main, optimizer_config_main = init_configs_from_yaml(
        yaml_name, [TrainingConfig, DatasetConfig, TransformerConfig, OptimizerConfig]
    )

    run_lr_scheduler(training_config_main, model_config_main, dataset_config_main, optimizer_config_main, yaml_name)
