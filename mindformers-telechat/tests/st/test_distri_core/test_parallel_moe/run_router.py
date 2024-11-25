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
"""test router"""
import argparse
import os

import numpy as np
from mindformers import Linear
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, MoEConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    initialize_model_parallel,
)
from mindformers.experimental.parallel_core.pynative.transformer.moe.router import TopKRouter
from tests.st.test_distri_core.utils import TestData, train

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay
from mindspore.communication.comm_func import all_gather_into_tensor


class TopKRouterGolden(nn.Cell):
    """
    define a Golden TopKRouter network
    """
    def __init__(self, model_config):
        super(TopKRouterGolden, self).__init__()
        moe_configs = model_config.moe_config
        self.topk = moe_configs.moe_router_topk
        self.num_experts = moe_configs.num_experts
        self.gating = Linear(
            model_config.hidden_size,
            moe_configs.num_experts,
            has_bias=False,
            param_init_type=model_config.param_init_type,
            compute_dtype=model_config.compute_dtype
            )
        self.loss = nn.MAELoss()

    def construct(self, input_ids, labels):
        """define a forward process"""
        logits = self.gating(input_ids)
        logits = logits.reshape(-1, self.num_experts)
        top_logits, indices = ops.topk(logits, k=self.topk, dim=1)
        _ = ops.softmax(top_logits, axis=-1)

        labels = labels.reshape(-1, labels.shape[-1])
        loss = self.loss(indices.to(mstype.float32), labels)

        return loss


class TopKRouterPynative(nn.Cell):
    """
    define a pynative TopKRouter network
    """
    def __init__(self, model_config):
        super(TopKRouterPynative, self).__init__()
        moe_configs = model_config.moe_config
        self.topk = moe_configs.moe_router_topk
        self.num_experts = moe_configs.num_experts
        self.router = TopKRouter(model_config)
        self.loss = nn.MAELoss()
        self.dp_group = get_data_parallel_group()

    def construct(self, input_ids, labels):
        """define a forward process"""
        _, indices = self.router(input_ids)
        labels = labels.to(mstype.float32)
        indices = indices.to(mstype.float32)
        labels = labels.reshape(-1, labels.shape[-1])
        loss = self.loss(indices, labels)

        indices_all = all_gather_into_tensor(indices, group=self.dp_group)[0]
        labels_all = all_gather_into_tensor(labels, group=self.dp_group)[0]
        loss_all = self.loss(indices_all, labels_all)
        print(f"loss_all is {loss_all}")

        return loss


def generate_golden(model_config, args):
    """
    generate_golden
    """
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic='ON', pynative_synchronize=True)

    moe_configs = model_config.moe_config

    seq_length = model_config.seq_length
    data_input_dim = model_config.hidden_size
    en = moe_configs.num_experts
    ms.set_seed(2024)

    input_data = np.random.random((args.dataset_size, seq_length, data_input_dim)).astype(np.float32)
    label_data = np.random.randint(low=0, high=en-1, size=(args.dataset_size, seq_length, moe_configs.moe_router_topk))
    label_data = label_data.astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    dataset_parallel = dataset_parallel.batch(args.batch_size)

    model = TopKRouterGolden(model_config)
    ms.save_checkpoint(model, "./data/golden_router.ckpt")
    optimizer = AdamWeightDecay(params=model.get_parameters())

    loss_list = train(10, dataset_parallel, model, optimizer, reduce_grad=False)

    # save golden input and loss
    loss_list = np.array([x for x in loss_list])
    input_and_loss = {"input_data": input_data,
                      "label_data": label_data,
                      "loss": loss_list}
    np.save("./data/golden_router_input_and_loss.npy", input_and_loss)


def run_router_pynative(model_config, args):
    """
    run_router_pynative
    """
    parallel_config = model_config.parallel_config
    dp = parallel_config.data_parallel
    ep = parallel_config.expert_model_parallel_size

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    print("data_parallel {} | expert_model_parallel_size {}".format(dp, ep))
    initialize_model_parallel(expert_model_parallel_size=ep)

    ms.set_seed(2024)

    # load data
    golden_input_and_loss_path = "./data/golden_router_input_and_loss.npy"
    assert os.path.exists(golden_input_and_loss_path), \
           f"'{golden_input_and_loss_path}' did not exits, please run generate_golden() to generate one"

    input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
    input_data = input_and_loss['input_data']
    label_data = input_and_loss['label_data']

    dataset = TestData(input_data=input_data, label_data=label_data)
    num_shards = get_data_parallel_world_size()
    shard_id = get_data_parallel_rank()
    print("dataset num shards {} | shard id {}".format(num_shards, shard_id))
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], \
                                           num_shards=num_shards, shard_id=shard_id, shuffle=False)
    dataset_parallel = dataset_parallel.batch(args.batch_size)

    model = TopKRouterPynative(model_config)
    optimizer = AdamWeightDecay(params=model.get_parameters())

    # load golden ckpt
    golden_ckpt_path = "./data/golden_router.ckpt"
    assert os.path.exists(golden_ckpt_path), \
           "'./data/golden_router.ckpt' did not exits, please run generate_golden() to generate"
    golden_params = ms.load_checkpoint(golden_ckpt_path)
    golden_params = {"router.gating.weight": golden_params["gating.weight"]}
    param_not_load, _ = ms.load_param_into_net(model, golden_params)

    assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."

    train(10, dataset_parallel, model, optimizer, reduce_grad=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_golden', action='store_true', help="Generate golden data for test.")
    parser.add_argument('--dp', type=int, default=2, help="data_parallel")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--dataset_size', type=int, default=20, help="dataset_size")

    cli_args, rest_args = parser.parse_known_args()

    parallel_cfg = ModelParallelConfig(
        data_parallel=cli_args.dp,
        model_parallel=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        use_seq_parallel=False
        )
    moe_cfg = MoEConfig(
        num_experts=4,
        moe_router_topk=2,
        add_bias_linear=False,
        moe_token_dispatcher_type='alltoall',
        moe_z_loss_coeff=1e-3,
        moe_aux_loss_coeff=1e-2,
        moe_router_load_balancing_type='none', # ['none', 'aux_loss'],
        moe_input_noise_eps=None,
        moe_expert_capacity_factor=None,
        moe_token_drop_policy=None,
        moe_pad_expert_input_to_capacity=False,
        use_self_defined_alltoall=False,
        )
    model_cfg = TransformerConfig(
        vocab_size=1,
        num_layers=1,
        num_attention_heads=1,
        seq_length=8,
        hidden_size=16,
        ffn_hidden_size=64,
        gated_linear_unit=True,
        hidden_act="gelu",
        qkv_has_bias=True,
        mlp_has_bias=False,
        params_dtype='float32',
        param_init_type=mstype.float32,
        compute_dtype='float32',
        fp32_residual_connection=False,
        parallel_config=parallel_cfg,
        moe_config=moe_cfg
        )

    if cli_args.generate_golden:
        generate_golden(model_cfg, cli_args)
    else:
        run_router_pynative(model_cfg, cli_args)
