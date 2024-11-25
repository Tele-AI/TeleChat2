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
"""test configs and initializations"""

import argparse

import pytest

from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_args
)

allowed_error = 1e-5


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_config_from_args():
    """
    Feature: init configs from command, as the basic test.
    Description: Test to initialize configs from command line
    Expectation: success
    """

    args = argparse.Namespace()
    args.seed = 123
    args.lr = 1e-2
    args.tensor_model_parallel_size = 2
    args.num_layers = 6
    args.train_iters = 10
    args.data_path = "./dataset"
    args.padded_vocab_size = 32000
    args.num_attention_heads = 32
    args.hidden_size = 4096
    args.ffn_hidden_size = 16384
    args.global_batch_size = 2
    args.micro_batch_size = 1
    args.adam_beta1 = 0.8
    args.adam_beta2 = 0.9
    model_type = "model_config"
    all_config = init_configs_from_args(args, model_type)
    assert all_config.training_config.seed == 123
    assert abs(all_config.optimizer_config.learning_rate - 1e-2) < allowed_error
    assert all_config.parallel_config.tensor_model_parallel_size == 2
    assert all_config.dataset_config.dataset_dir == "./dataset"
    assert all_config.dataset_config.micro_batch_num == 2
    assert all_config.dataset_config.batch_size == 1
    assert all_config.model_config.vocab_size == 32000
    assert all_config.model_config.num_attention_heads == 32
    assert all_config.model_config.hidden_size == 4096
    assert all_config.model_config.ffn_hidden_size == 16384
    assert abs(all_config.optimizer_config.betas[0] - 0.8) < allowed_error
    assert abs(all_config.optimizer_config.betas[1] - 0.9) < allowed_error


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_vpp_config_from_args():
    """
    Feature: init configs from command, as the basic test.
    Description: Test to initialize vpp related configs from command line
    Expectation: success
    """

    args = argparse.Namespace()
    args.num_layers_per_virtual_pipeline_stage = 2
    args.pipeline_model_parallel_size = 2
    args.num_layers = 8
    args.padded_vocab_size = 32000
    args.num_attention_heads = 32
    args.hidden_size = 4096
    args.ffn_hidden_size = 16384
    model_type = "model_config"
    all_config = init_configs_from_args(args, model_type)
    assert all_config.model_config.num_layers == 8
    assert all_config.parallel_config.pipeline_model_parallel_size == 2
    assert all_config.parallel_config.num_layers_per_virtual_pipeline_stage == 2
    assert all_config.parallel_config.pipeline_model_parallel_size == 2
