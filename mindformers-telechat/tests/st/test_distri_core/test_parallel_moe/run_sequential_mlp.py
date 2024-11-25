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

"""test sequential mlp"""
import argparse
import os

import numpy as np
from mindformers import MoEConfig as GoldenMoEConfig
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, MoEConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer.moe.experts import SequentialMLP
from mindformers.models.llama.llama_layer import LlamaFeedForward
from mindformers.modules.transformer.op_parallel_config import MoEParallelConfig
from tests.st.test_distri_core.utils import transform_sequential_mlp_golden_params_to_pynative_params

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.communication.management import get_rank, init
from mindspore.nn import SGD


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, data_size=None, input_data=None, label_data=None):
        super().__init__()
        _ = data_size
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]


def train(epoch_num, dataset, model, optimizer, save_ckpt_path, tokens_per_expert):
    """
    define a training procedure
    """
    def forward_fn(input_, label, tokens_per_expert):
        loss_fn = nn.MAELoss()
        output, _ = model(input_, tokens_per_expert)
        label = label.reshape(-1, label.shape[-1])
        loss = loss_fn(output, label)
        return loss

    model.set_train()
    grad_func = ms.value_and_grad(forward_fn, None, weights=optimizer.parameters)
    loss_list = []
    for epoch in range(epoch_num):
        step = 0
        for input_ids, labels in dataset:
            input_ids = input_ids.reshape(-1, input_ids.shape[-1])
            loss, grads = grad_func(input_ids, labels, tokens_per_expert)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            loss_list.append(loss)
    if save_ckpt_path is not None:
        ms.save_checkpoint(model, save_ckpt_path)
    return loss_list

class SequentialMLPGolden(nn.Cell):
    """
    define a golden SequentialMLP network
    """
    def __init__(self, num_local_experts, model_config):
        super(SequentialMLPGolden, self).__init__()
        self.local_experts = nn.SequentialCell()

        for _ in range(num_local_experts):
            expert = LlamaFeedForward(dim=model_config.hidden_size,
                                      intermediate_size=model_config.ffn_hidden_size,
                                      hidden_act=model_config.hidden_act,
                                      ffn_dim_multiplier=None,
                                      compute_dtype=model_config.compute_dtype,
                                      param_init_type=model_config.param_init_type,
                                      is_dynamic=False,
                                      parallel_config=model_config.parallel_config)
            self.local_experts.append(expert)
        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=0)

    def construct(self, hidden_states, tokens_per_expert):
        """define the forward process"""
        output_local = ops.zeros_like(hidden_states)

        start_idx = 0
        end_idx = tokens_per_expert[0]
        for expert_id, expert in enumerate(self.local_experts):
            hidden_expert = ops.index_select(hidden_states, 0, ops.arange(start_idx, end_idx))
            output = expert(hidden_expert)
            if expert_id == 0:
                output_local = output
            else:
                output_local = self.concat((output_local, output))
            if expert_id != len(self.local_experts) - 1:
                start_idx = end_idx
                end_idx += tokens_per_expert[expert_id + 1]
        return output_local, None


def generate_golden(model_config, args):
    """
    generate_golden
    """
    ms.set_context(device_id=0, device_target="Ascend", mode=ms.GRAPH_MODE)
    moe_config = model_config.moe_config

    seq_length = model_config.seq_length
    data_input_dim = model_config.hidden_size
    num_local_experts = moe_config.expert_num

    tokens_per_expert = Tensor(np.array([2, 2, 1, 3]).astype(np.int32))
    assert num_local_experts == tokens_per_expert.shape[0]

    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
    ms.set_seed(2024)

    input_data = np.random.random((args.dataset_size, seq_length, data_input_dim)).astype(np.float32)
    label_data = np.zeros((args.dataset_size, seq_length, data_input_dim)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    dataset_parallel = dataset_parallel.batch(args.batch_size)

    model = SequentialMLPGolden(num_local_experts, model_config)
    # for name, params in model.parameters_and_names():
    #     print(f"{name} {params.dtype} {params.shape}")
    ms.save_checkpoint(model, "./data/golden_sequential_mlp.ckpt")
    optimizer = SGD(params=model.get_parameters())

    loss_list = train(10, dataset_parallel, model, optimizer, None, tokens_per_expert)

    # save golden input and loss
    loss_list = np.array([x.asnumpy() for x in loss_list])
    input_and_loss = {"input": input_data,
                      "loss": loss_list}
    np.save("./data/golden_sequential_mlp_input_and_loss.npy", input_and_loss)

def run_sequential_mlp(model_config, args):
    """
    run_sequential_mlp
    """
    moe_config = model_config.moe_config
    parallel_config = model_config.parallel_config
    seq_length = model_config.seq_length
    data_input_dim = model_config.hidden_size
    num_local_experts = moe_config.num_experts

    en = moe_config.num_experts
    ep = parallel_config.expert_model_parallel_size

    tokens_per_expert = Tensor(np.array([2, 2, 1, 3]).astype(np.int32))
    assert num_local_experts == tokens_per_expert.shape[0]

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel()

    ms.set_seed(2024)

    rank_id = get_rank()
    local_expert_idx = np.arange(en).reshape(ep, -1)[rank_id].tolist()

    golden_input_and_loss_path = "./data/golden_sequential_mlp_input_and_loss.npy"
    assert os.path.exists(golden_input_and_loss_path), \
           f"'{golden_input_and_loss_path}' did not exits, please run generate_golden() to "+\
            "generate one by running below command:\n"+\
            "`pytest -sv run_sequential_mlp.py::TestSequentialMLP::run_sequential_mlp_golden`"

    input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
    input_data = input_and_loss['input']

    # input_data = np.random.random((args.dataset_size, seq_length, data_input_dim)).astype(np.float32)
    label_data = np.zeros((args.dataset_size, seq_length, data_input_dim)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    dataset_parallel = dataset_parallel.batch(args.batch_size)

    model = SequentialMLP(num_local_experts, model_config)
    optimizer = SGD(params=model.get_parameters())
    for name, params in model.parameters_and_names():
        print(f"{name} {params.dtype} {params.shape}")

    # load golden ckpt
    golden_ckpt_path = "./data/golden_sequential_mlp.ckpt"
    assert os.path.exists(golden_ckpt_path), \
           f"'{golden_ckpt_path}' did not exits, please run generate_golden() to "+\
            "generate one by running below command:\n"+\
            "`pytest -sv run_sequential_mlp.py::TestSequentialMLP::run_sequential_mlp_golden`"
    golden_params = ms.load_checkpoint(golden_ckpt_path)
    pynative_params = transform_sequential_mlp_golden_params_to_pynative_params(golden_params, local_expert_idx)
    param_not_load, _ = ms.load_param_into_net(model, pynative_params)
    assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    print("load ckpt competele.", flush=True)

    train(10, dataset_parallel, model, optimizer, None, tokens_per_expert)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="golden", help="data_parallel")
    parser.add_argument('--dp', type=int, default=2, help="data_parallel")
    parser.add_argument('--ep', type=int, default=2, help="expert_model_parallel_size")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--dataset_size', type=int, default=10, help="dataset_size")

    cli_args, rest_args = parser.parse_known_args()

    moe_parallel_cfg = MoEParallelConfig(
        data_parallel=cli_args.dp,
        model_parallel=1,
        expert_parallel=cli_args.ep,
        use_seq_parallel=False
        )
    parallel_cfg = ModelParallelConfig(
        data_parallel=cli_args.dp,
        model_parallel=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=cli_args.ep,
        use_seq_parallel=False
        )
    moe_cfg = MoEConfig(
        num_experts=4,
        moe_router_topk=1,
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
    moe_cfg_golden = GoldenMoEConfig(
        expert_num=moe_cfg.num_experts,
        capacity_factor=-1,
        aux_loss_factor=0.05,
        num_experts_chosen=1,
        routing_policy="TopkRouterV2",
        enable_sdrop=False,
        router_dense_type='float32'
        )
    model_cfg = TransformerConfig(
        vocab_size=1,
        num_layers=1,
        seq_length=8,
        hidden_size=4,
        ffn_hidden_size=16,
        num_attention_heads=8,
        gated_linear_unit=True,
        hidden_act="swiglu",
        qkv_has_bias=True,
        mlp_has_bias=False,
        params_dtype='float32',
        param_init_type=mstype.float32,
        compute_dtype='float16',
        fp32_residual_connection=False,
        parallel_config=parallel_cfg,
        moe_parallel_config=moe_parallel_cfg,
        moe_config=moe_cfg
        )

    if cli_args.mode == 'golden':
        model_cfg.moe_config = moe_cfg_golden
        generate_golden(model_cfg, cli_args)
    elif cli_args.mode == 'standalone':
        run_sequential_mlp(model_cfg, cli_args)
