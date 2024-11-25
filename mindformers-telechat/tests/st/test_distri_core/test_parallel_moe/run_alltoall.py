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
"""run parallel alltoall"""

import argparse

import numpy as np
from mindformers.experimental.parallel_core.pynative.config import LoraConfig, ModelParallelConfig, MoEConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.mappings import AllToAll
from mindformers.experimental.parallel_core.pynative.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from mindformers.modules import Linear

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, data_size=None, input_data=None):
        super().__init__()
        self.input_data = input_data
        self.data_size = data_size

    def __getitem__(self, index):
        _ = index
        return Tensor(self.input_data)

    def __len__(self):
        return self.data_size


def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=optimizer.parameters
    )

    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset.create_dict_iterator():
            loss, grads = grad_func(**data)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss)

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss


class PynativeAlltoAll(nn.Cell):
    """
    define a pynative AlltoAll2 net
    """
    def __init__(self, group, output_shape, input_shape, output_splits, input_splits, has_linear=True):
        super(PynativeAlltoAll, self).__init__()
        self.has_linear = has_linear
        self.rank_id = get_expert_model_parallel_rank()
        if self.has_linear:
            self.linear = Linear(in_channels=output_shape[1],
                                 out_channels=output_shape[1],
                                 transpose_b=False,
                                 weight_init="normal",
                                 has_bias=False,
                                 param_init_type=mstype.float32,
                                 compute_dtype=mstype.float32)
        self.alltoall = AllToAll(group, output_shape, input_shape, output_splits, input_splits)
        self.alltoall_reverse = AllToAll(group, input_shape, output_shape, input_splits, output_splits)

    def construct(self, hidden_states):
        if self.has_linear:
            output = self.linear(hidden_states)
        else:
            output = hidden_states
        output = self.alltoall(output)
        output = self.alltoall_reverse(output)
        loss = ops.dist(output, hidden_states)

        return loss, None


def run_alltoall_forward(model_config: TransformerConfig):
    """
    run pynative mode alltoall forward
    """
    ms.set_seed(1921)
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config
    seq_length = model_config.seq_length
    hidden_size = model_config.hidden_size

    dp = parallel_config.data_parallel
    tp = parallel_config.model_parallel
    ep = parallel_config.expert_model_parallel_size
    en = moe_config.num_experts
    num_local_experts = en // ep

    print("data_parallel {}, tensor_parallel {}, expert_model_parallel_size {}, num_experts {}".format(dp, tp, ep, en))
    init()
    initialize_model_parallel(expert_model_parallel_size=ep)
    print("dp group {}, tp group {}, pp group {}, ep group {}".format \
          (get_data_parallel_group(), get_tensor_model_parallel_group(), \
           get_pipeline_model_parallel_group(), get_expert_model_parallel_group()))
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    ep_group = get_expert_model_parallel_group()
    rank_id = get_expert_model_parallel_rank()

    all_hidden_states = np.arange(ep*seq_length).reshape((ep, seq_length, 1)).repeat(hidden_size, axis=-1)
    all_hidden_states = Tensor(all_hidden_states, dtype=ms.float32)
    all_indices = Tensor(np.random.randint(low=0, high=en, size=(ep, seq_length, 1), dtype=np.int32))
    hidden_states = all_hidden_states[rank_id]
    local_expert_indices = Tensor(np.arange(num_local_experts)+rank_id*num_local_experts, dtype=ms.int32)
    indices = Tensor(all_indices[rank_id])

    alltoall_token_dispatcher = MoEAlltoAllTokenDispatcher(num_local_experts, local_expert_indices, model_config)

    _ = alltoall_token_dispatcher.preprocess(indices)
    output_shape = alltoall_token_dispatcher.output_shape
    input_shape = alltoall_token_dispatcher.input_shape
    local_input_splits = alltoall_token_dispatcher.local_input_splits
    local_output_splits = alltoall_token_dispatcher.local_output_splits

    network = PynativeAlltoAll(ep_group, output_shape, input_shape,
                               local_output_splits, local_input_splits, has_linear=False)
    loss, _ = network(hidden_states)

    assert loss < 1e-8, f"AllToAllSelfDefined forward test failed, expect loss < 1e-8, but got {loss}"

def run_alltoall_bprop(model_config: TransformerConfig):
    """
    run pynative mode alltoall bprop
    """
    ms.set_seed(1921)
    parallel_config = model_config.parallel_config
    moe_config = model_config.moe_config
    seq_length = model_config.seq_length
    hidden_size = model_config.hidden_size

    dp = parallel_config.data_parallel
    tp = parallel_config.model_parallel
    ep = parallel_config.expert_model_parallel_size
    en = moe_config.num_experts
    num_local_experts = en // ep

    print("data_parallel {}, tensor_parallel {}, expert_model_parallel_size {}, num_experts {}".format(dp, tp, ep, en))
    init()
    initialize_model_parallel(expert_model_parallel_size=ep)
    print("dp group {}, tp group {}, pp group {}, ep group {}".format \
          (get_data_parallel_group(), get_tensor_model_parallel_group(), \
           get_pipeline_model_parallel_group(), get_expert_model_parallel_group()))
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    ep_group = get_expert_model_parallel_group()
    rank_id = get_expert_model_parallel_rank()

    all_hidden_states = np.arange(ep*seq_length).reshape((ep, seq_length, 1)).repeat(hidden_size, axis=-1)
    all_hidden_states = Tensor(all_hidden_states, dtype=ms.float32)
    all_indices = Tensor(np.random.randint(low=0, high=en, size=(ep, seq_length, 1), dtype=np.int32))
    hidden_states = all_hidden_states[rank_id]
    local_expert_indices = Tensor(np.arange(num_local_experts)+rank_id*num_local_experts, dtype=ms.int32)
    dataset = TestData(input_data=hidden_states, data_size=10)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states'])
    indices = Tensor(all_indices[rank_id])

    alltoall_token_dispatcher = MoEAlltoAllTokenDispatcher(num_local_experts, local_expert_indices, model_config)

    _ = alltoall_token_dispatcher.preprocess(indices)
    output_shape = alltoall_token_dispatcher.output_shape
    input_shape = alltoall_token_dispatcher.input_shape
    local_input_splits = alltoall_token_dispatcher.local_input_splits
    local_output_splits = alltoall_token_dispatcher.local_output_splits

    network = PynativeAlltoAll(ep_group, output_shape, input_shape, local_output_splits, local_input_splits)

    optimizer = AdamWeightDecay(params=network.get_parameters())
    all_loss = train(1, dataset, network, optimizer, None)

    assert all_loss[-1] < all_loss[0], \
           f"final loss {all_loss[-1]} >= first loss {all_loss[0]}, please check your code."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_forward', action='store_true', help="Generate golden data for test.")
    args, rest_args = parser.parse_known_args()
    parallel_cfg = ModelParallelConfig(
        data_parallel=2,
        model_parallel=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        use_seq_parallel=False
        )
    lora_config = LoraConfig(use_lora=False)
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
        moe_input_jitter_eps=0.001,
        moe_pad_expert_input_to_capacity=False,
        use_self_defined_alltoall=False,
        )
    model_cfg = TransformerConfig(
        num_layers=2,
        seq_length=4,
        num_attention_heads=16,
        hidden_size=32,
        ffn_hidden_size=128,
        vocab_size=64,
        group_query_attention=True,
        num_query_groups=8,
        parallel_config=parallel_cfg,
        moe_config=moe_cfg,
        lora_config=lora_config,
        attn_type='self_attn',
        qkv_has_bias=False,
        out_proj_has_bias=False,
        param_init_type="float32",
        params_dtype="float32",
        init_method='normal',
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_fill",
        mlp_has_bias=False,
        gated_linear_unit=True,
        hidden_act='silu',
        apply_residual_connection_post_norm=False,
        normalization='FusedRMSNorm',
        norm_epsilon=1.e-5,
        )

    if args.test_forward:
        run_alltoall_forward(model_cfg)
    else:
        run_alltoall_bprop(model_cfg)
