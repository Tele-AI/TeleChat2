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
"""some utility functions"""
from typing import List, Dict
import collections
import re
import os
import time
import glob
import shutil
import numpy as np

import mindspore as ms
from mindspore import mint
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.nn import DistributedGradReducer

from mindspore.communication import get_rank
from mindformers.experimental.parallel_core.pynative.dist_checkpointing import save_checkpoint

from mindformers import init_context
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_expert_model_parallel_rank,
)
from mindformers.experimental.parallel_core.pynative.distributed import DistributedDataParallel
from mindformers.experimental.parallel_core.pynative.optimizer import MixedPrecisionOptimizer
from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits, TransformerLanguageModel
from mindformers.experimental.parallel_core.pynative.transformer.module import Module

def get_res(file, pattern):
    """
    find pattern from file
    """
    with open(file, "r") as f:
        lines = f.readlines()
        for l in reversed(lines):
            if pattern in l:
                return l
    return None


def build_context(config):
    """
    build mindspore context using config
    """
    print("..........Init Context..........")
    if config.use_parallel and config.parallel_config.pipeline_model_parallel_size > 1:
        config.parallel.pipeline_stages = config.parallel_config.pipeline_model_parallel_size
    local_rank, device_num = init_context(
        use_parallel=config.use_parallel,
        context_config=config.context,
        parallel_config=config.parallel,
    )

    return local_rank, device_num


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
            return (
                Tensor(self.input_data[index]),
                Tensor(self.label_data[index]),
                Tensor(self.attention_mask),
            )
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]


def train(epoch_num,
          dataset,
          network,
          optimizer,
          save_ckpt_path=None,
          with_attn_input=False,
          reduce_grad=True,
          zero_level=-1,
          ):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=list(network.get_parameters())
    )
    if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
            and get_data_parallel_world_size(with_context_parallel=True) > 1
            and not isinstance(network, DistributedDataParallel)):
        grad_reducer = DistributedGradReducer(
            network.get_parameters(),
            group=get_data_parallel_group(with_context_parallel=True),
            mean=True,
            degree=get_data_parallel_world_size(with_context_parallel=True),
        )
    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            if isinstance(network, DistributedDataParallel):
                network.zero_grad_buffer()
            if isinstance(optimizer, MixedPrecisionOptimizer):
                optimizer.zero_grad()
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
            if isinstance(network, DistributedDataParallel):
                network.final_grad_reduce()
            if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
                    and get_data_parallel_world_size(with_context_parallel=True) > 1
                    and not isinstance(network, DistributedDataParallel)):
                if zero_level < 0:
                    print(
                        "reduce gradients on group {}".format(
                            get_data_parallel_group(with_context_parallel=True)
                        )
                    )
                    grads = grad_reducer(grads)
            if isinstance(optimizer, MixedPrecisionOptimizer):
                optimizer()
            else:
                optimizer(grads)
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss


def transform_mlp_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden mlp params to pynative params
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    pynative_params = {}
    for name, param in golden_params.items():
        new_param = param
        if "mapping.weight" in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param.transpose()[start:end, :]
        if 'projection.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param.transpose()[:, start:end]
        if 'mapping.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start:end]
        pynative_params[name] = ms.Parameter(new_param)

    return pynative_params


def transform_linear_params(params, linear_type):
    """
    transform linear parameters.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    transformed_params = {}
    for name, param in params.items():
        if linear_type == "rowparallellinear":
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            if name == "linear.weight":
                new_param = param[start:end, :]
            elif name == "linear.bias":
                new_param = param
        elif linear_type == "columnparallellinear":
            if name == "linear.weight":
                start = tp_rank * (param.shape[1] // tp_world_size)
                end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
                new_param = param[:, start:end]
            elif name == "linear.bias":
                start = tp_rank * (param.shape[0] // tp_world_size)
                end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
                new_param = param[start:end]
        transformed_params[name] = ms.Parameter(new_param)

    return transformed_params


def generate_ckpt(hidden_size,
                  module_type,
                  num_layers=2,
                  kv_hidden_size=None,
                  prefix=None,
                  vocab_size=None,
                  use_embedding=False):
    """generate graph mode module checkpoints"""
    ms.set_seed(1024)
    if not kv_hidden_size:
        kv_hidden_size = hidden_size
    has_layer_index = False
    if module_type == "transformer":
        has_layer_index = True
    if prefix is None:
        prefix = ""
        if module_type == "transformer":
            prefix = ""
        if module_type == "transformerlayer":
            prefix = "layer."
        if module_type in ["attention", "mlp"]:
            prefix = ""

    param_dict = {}
    if use_embedding:
        param_name = 'embedding.embedding_table'
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((vocab_size, hidden_size)), mstype.float32), name=param_name
        )
    for i in range(num_layers):
        # generate ffn_norm.weight
        param_name = prefix + "{}ffn_norm.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32), name=param_name
        )

        # generate attention_norm.weight
        param_name = prefix + "{}attention_norm.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32), name=param_name
        )

        # generate attention.w_qkv.weight
        param_name = prefix + "{}attention.w_qkv.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(
                np.random.random((hidden_size + 2 * kv_hidden_size, hidden_size)),
                mstype.float32,
            ),
            name=param_name,
        )

        # generate attention.w_qkv.bias
        param_name = prefix + "{}attention.w_qkv.bias".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(
                np.random.random((hidden_size + 2 * kv_hidden_size)), mstype.float32
            ),
            name=param_name,
        )

        # generate attention.wo.weight
        param_name = prefix + "{}attention.wo.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size, hidden_size)), mstype.float32),
            name=param_name,
        )

        # generate mlp.mapping.weight
        param_name = prefix + "{}mlp.mapping.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size, 4 * hidden_size)), mstype.float32),
            name=param_name,
        )

        # generate mlp.mapping.bias
        param_name = prefix + "{}mlp.mapping.bias".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((4 * hidden_size)), mstype.float32), name=param_name
        )

        # generate mlp.projection.weight
        param_name = prefix + "{}mlp.projection.weight".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((4 * hidden_size, hidden_size)), mstype.float32),
            name=param_name,
        )

        # generate mlp.projection.bias
        param_name = prefix + "{}mlp.projection.bias".format(
            str(i) + "." if has_layer_index else ""
        )
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32), name=param_name
        )

    return param_dict


def transform_transformerlayer_params(params, hidden_size, kv_hidden_size=None, prefix=""):
    """
    transform transformerlayer parameters.
    """
    if not kv_hidden_size:
        kv_hidden_size = hidden_size
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    new_params = {}
    for name, param in params.items():
        if 'embedding_table' in name:
            new_param = param
            new_params['language_model.embedding.word_embeddings.weight'] = (
                ms.Parameter(new_param)
            )
        if "ffn_norm" in name:
            new_param = param
            new_params[prefix + name.replace("ffn_norm", "post_attention_norm")] = (
                ms.Parameter(new_param)
            )
        if "attention_norm" in name:
            new_param = param
            new_params[prefix+name.replace("attention_norm", "input_norm")] = ms.Parameter(new_param)
        if 'wo.weight' in name:
            param = param.asnumpy()
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank+1) * (param.shape[0] // tp_world_size)
            new_param = param[:, start:end]
            new_params[prefix+name.replace("wo", "out_proj")] = ms.Parameter(new_param)
        if 'w_qkv.weight' in name:
            param = param.asnumpy()
            q = param[:hidden_size, :]
            k = param[hidden_size:hidden_size+kv_hidden_size, :]
            v = param[hidden_size+kv_hidden_size:, :]
            q_start = tp_rank * (q.shape[0] // tp_world_size)
            q_end = (tp_rank+1) * (q.shape[0] // tp_world_size)
            kv_start = tp_rank * (k.shape[0] // tp_world_size)
            kv_end = (tp_rank+1) * (k.shape[0] // tp_world_size)
            new_param = np.concatenate([q[q_start:q_end, :], k[kv_start:kv_end, :], v[kv_start:kv_end, :]], axis=0)
            new_params[prefix+name.replace("w_qkv.", "qkv_proj.")] = ms.Parameter(ms.Tensor(new_param))
        if 'w_qkv.bias' in name:
            param = param.asnumpy()
            q = param[:hidden_size]
            k = param[hidden_size: hidden_size + kv_hidden_size]
            v = param[hidden_size + kv_hidden_size:]
            q_start = tp_rank * (q.shape[0] // tp_world_size)
            q_end = (tp_rank + 1) * (q.shape[0] // tp_world_size)
            kv_start = tp_rank * (k.shape[0] // tp_world_size)
            kv_end = (tp_rank + 1) * (k.shape[0] // tp_world_size)
            new_param = np.concatenate(
                [q[q_start:q_end], k[kv_start:kv_end], v[kv_start:kv_end]], axis=0
            )
            new_params[prefix + name.replace("w_qkv", "qkv_proj")] = ms.Parameter(
                ms.Tensor(new_param)
            )
        if "mapping.weight" in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            # new_param = param[:, start: end]
            new_param = param.transpose()[start: end]
            new_params[prefix+name] = ms.Parameter(new_param)
        if 'mapping.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start:end]
            new_params[prefix + name] = ms.Parameter(new_param)
        if "projection.weight" in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            # new_param = param[start: end]
            new_param = param.transpose()[:, start: end]
            new_params[prefix+name] = ms.Parameter(new_param)
        if 'projection.bias' in name:
            new_param = param
            new_params[prefix + name] = ms.Parameter(new_param)

    return new_params


def transform_mlp_has_gate_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden mlp params to pynative params
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        if "w1" in name:  # w1 -> gating
            pynative_name = "mlp.gating.weight"
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
        if "w2" in name: # w2 -> projection
            pynative_name = 'mlp.projection.weight'
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end]
        if "w3" in name: # w3 -> mapping
            pynative_name = 'mlp.mapping.weight'
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
        pynative_params[pynative_name] = ms.Parameter(new_param)
        print(
            f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}",
            flush=True,
        )

    return pynative_params


def transform_vocab_embedding_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden vocab embedding params to pynative params
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    pynative_params = {}
    param = golden_params["embedding.embedding_table"]
    pynative_name = "embedding.weight"
    start = tp_rank * (param.shape[0] // tp_world_size)
    end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
    new_param = param[start:end, :]
    pynative_params[pynative_name] = ms.Parameter(new_param, name=pynative_name)

    return pynative_params


def rearrange_the_ckpt_files(parent_dir, world_size):
    """rearrage the ckpt files into its parent directory"""
    if not os.path.exists(parent_dir):
        print(f"The directory {parent_dir} does not exist.")
    else:
        for rank in range(world_size):
            rank_dir = os.path.join(parent_dir, f"rank_{rank}")
            if os.path.exists(rank_dir):
                src_strategy_files = glob.glob(os.path.join(rank_dir, "*.ckpt"))
                for src_strategy_file in src_strategy_files:
                    new_file_name = f"strategy_{rank}.ckpt"
                    new_file_path = os.path.join(parent_dir, new_file_name)
                    shutil.move(src_strategy_file, new_file_path)
                    shutil.rmtree(rank_dir)
                    print(f"Moved and renamed {src_strategy_file} to {new_file_path}")
            else:
                print(f"Rank directory {rank_dir} does not exist.")


def transform_moe_golden_params_to_pynative_params(
        golden_params: Dict[str, Tensor],
        local_expert_indices: List[int]
        ):
    """
    transform golden moe params to pynative params
    map_dict = {"w1": "gating",
                "w2": "projection",
                "w3": "mapping"}
    """

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        if "feed_forward.ffn" in name:
            for i, local_expert_id in enumerate(local_expert_indices):
                if "w1" in name:  # w1 -> gating
                    pynative_name = f"moe.experts.local_experts.{i}.gating.weight"
                if "w2" in name:  # w2 -> projection
                    pynative_name = f"moe.experts.local_experts.{i}.projection.weight"
                if "w3" in name:  # w3 -> mapping
                    pynative_name = f"moe.experts.local_experts.{i}.mapping.weight"
                if len(param.shape) == 3:
                    pynative_params[pynative_name] = ms.Parameter(param[local_expert_id])
                    print(f"{name} {param[local_expert_id].shape} -> " + \
                          f"{pynative_name} {pynative_params[pynative_name].shape}", flush=True)
                elif len(param.shape) == 2:
                    pynative_params[pynative_name] = ms.Parameter(param)
                    print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)
        if "feed_forward.router.dense.weight" in name:
            pynative_name = "moe.router.gating.weight"
            pynative_params[pynative_name] = ms.Parameter(param)
            print(
                f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}",
                flush=True,
            )

    return pynative_params


def transform_sequential_mlp_golden_params_to_pynative_params(golden_params: Dict[str, Tensor],
                                                              local_expert_indices: List[int]):
    """
    transform golden moe params to pynative params
    map_dict = {"mapping": "w1" + "w3",
                "projection": "w2"}
    """

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for i, local_expert_id in enumerate(local_expert_indices):
        w1 = golden_params[f"{local_expert_id}.w1.weight"]
        w2 = golden_params[f"{local_expert_id}.w2.weight"]
        w3 = golden_params[f"{local_expert_id}.w3.weight"]
        pynative_name = f'local_experts.{i}.mapping.weight'
        pynative_params[pynative_name] = ms.Parameter(ops.cat((w1, w3), axis=0))
        print(f"{local_expert_id}.w1.weight {w1.shape} {local_expert_id}.w3.weight {w3.shape} -> " + \
              f"{pynative_name} {pynative_params[pynative_name].shape}", flush=True)

        pynative_name = f'local_experts.{i}.projection.weight'
        pynative_params[pynative_name] = ms.Parameter(w2)
        print(f"{local_expert_id}.w2.weight {w2.shape} -> " + \
              f"{pynative_name} {pynative_params[pynative_name].shape}", flush=True)
    return pynative_params


def transform_mixtral_golden_params_to_pynative_params(
        golden_params: Dict[str, Tensor],
        model_config
    ):
    """transform mixtral moe params to pynative params"""

    tp_rank = get_tensor_model_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()

    ep = model_config.parallel_config.expert_model_parallel_size
    en = model_config.moe_config.num_experts
    expert_per_rank = en // ep
    global_expert_id_on_this_rank = [i + ep_rank * expert_per_rank for i in range(expert_per_rank)]

    hidden_size = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    num_query_groups = num_heads
    if model_config.group_query_attention:
        num_query_groups = model_config.num_query_groups
    qkv_num_heads = num_heads + num_query_groups * 2
    kv_head_multiplier = num_heads // num_query_groups

    new_params = collections.OrderedDict()
    print("↓↓↓↓↓↓↓↓↓↓↓ pt -> ckpt map ↓↓↓↓↓↓↓↓↓↓↓")
    for k, v in golden_params.items():
        if "_extra_state" in k:
            continue
        golden_prefix = "decoder."
        pynative_prefix = "language_model.encoder."
        # "decoder." -> "network.model.transformer."
        new_name = k.replace(golden_prefix, pynative_prefix)
        # embedding.word_embeddings.weight -> language_model.embedding.word_embeddings.weight
        if "embedding.word_embeddings.weight" in k:
            new_name = new_name.replace(
                "embedding.word_embeddings.weight",
                "language_model.embedding.word_embeddings.weight"
                )
            v = v.numpy()
            if tp_world_size > 1:
                v = np.split(v, tp_world_size, axis=0)[tp_rank]
            new_params[new_name] = Parameter(v, name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # self_attention.linear_qkv.layer_norm_weight -> input_norm.weight
        if "input_layernorm.weight" in k:
            new_name = new_name.replace("input_layernorm.weight", "input_norm.weight")
            new_params[new_name] = Parameter(v.numpy(), name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
         # self_attention.linear_qkv.layer_norm_weight -> input_norm.weight
        if "self_attention.linear_qkv.layer_norm_weight" in k:
            new_name = new_name.replace("self_attention.linear_qkv.layer_norm_weight", "input_norm.weight")
            new_params[new_name] = Parameter(v.numpy(), name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # self_attention.linear_qkv.weight -> attention.qkv_proj.weight
        if "self_attention.linear_qkv.weight" in k:
            new_name = new_name.replace("self_attention.linear_qkv.weight", "attention.qkv_proj.weight")
            v = v.numpy()
            if not model_config.group_query_attention:
                v = np.array(np.split(v, num_heads * 3))
                querry_layer, key_layer, value_layer = v[0::3], v[1::3], v[2::3]
                if tp_world_size > 1:
                    querry_layer = np.split(querry_layer, tp_world_size, axis=0)[tp_rank]
                    key_layer = np.split(key_layer, tp_world_size, axis=0)[tp_rank]
                    value_layer = np.split(value_layer, tp_world_size, axis=0)[tp_rank]
                v = np.concatenate((querry_layer, key_layer, value_layer)).reshape(-1, hidden_size)
            else: # using gqa, only suitale for q_heads:k_heads:v_heads = kv_head_multiplier:1:1
                v = np.array(np.split(v, qkv_num_heads))
                idx = 0
                querry_layer, key_layer, value_layer = [], [], []
                while idx < v.shape[0]:
                    querry_layer.append(v[idx:idx+kv_head_multiplier])
                    key_layer.append(v[idx+kv_head_multiplier, None])
                    value_layer.append(v[idx+kv_head_multiplier+1, None])
                    idx += (kv_head_multiplier + 2)
                querry_layer = np.array(querry_layer).reshape(-1, hidden_size)
                key_layer = np.array(key_layer).reshape(-1, hidden_size)
                value_layer = np.array(value_layer).reshape(-1, hidden_size)
                if tp_world_size > 1:
                    querry_layer = np.split(querry_layer, tp_world_size, axis=0)[tp_rank]
                    key_layer = np.split(key_layer, tp_world_size, axis=0)[tp_rank]
                    value_layer = np.split(value_layer, tp_world_size, axis=0)[tp_rank]
                qkv_layer = (querry_layer, key_layer, value_layer)

                v = np.concatenate(qkv_layer)
            new_params[new_name] = Parameter(v, name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # self_attention.linear_proj.weight -> out_proj.weight
        if "self_attention.linear_proj.weight" in k:
            new_name = new_name.replace("self_attention.linear_proj.weight", "attention.out_proj.weight")
            v = v.numpy()
            if tp_world_size > 1:
                v = np.split(v, tp_world_size, axis=1)[tp_rank]
            new_params[new_name] = Parameter(v, name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # pre_mlp_layernorm.weight -> post_attention_norm.weight
        if "pre_mlp_layernorm.weight" in k:
            new_name = new_name.replace("pre_mlp_layernorm.weight", "post_attention_norm.weight")
            new_params[new_name] = Parameter(v.numpy(), name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # mlp.router.weight -> mlp.router.gating.weight
        if "router.weight" in k:
            new_name = new_name.replace("router.weight", "router.gating.weight")
            new_params[new_name] = Parameter(v.numpy(), name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # linear_fc1.weight[:ffn_hidden_size, :] -> gating.weight
        # linear_fc1.weight[ffn_hidden_size:, :] -> mapping.weight
        if "local_experts" in k:
            param_expert_id = int(new_name.split(".")[7])
            if param_expert_id in global_expert_id_on_this_rank:
                new_name = new_name.replace(f"local_experts.{param_expert_id}",
                                            f"local_experts.{param_expert_id - ep_rank * expert_per_rank}")
                if "linear_fc1.weight" in k:

                    v_mapping = v.numpy()
                    if tp_world_size > 1:
                        v_mapping = np.split(v_mapping, tp_world_size, axis=0)[tp_rank]
                    mapping_name = new_name.replace("linear_fc1.weight", "mapping.weight")
                    new_params[mapping_name] = Parameter(v_mapping, name=new_name)
                    print(f"{k} {v.shape} {v.dtype} -> "+\
                          f"{mapping_name} {new_params[mapping_name].shape} {new_params[mapping_name].dtype}")
                    continue
                # linear_fc2.weight.T -> projection.weight
                if "linear_fc2.weight" in k:
                    new_name = new_name.replace("linear_fc2.weight", "projection.weight")
                    v = v.numpy()
                    if tp_world_size > 1:
                        v = np.split(v, tp_world_size, axis=1)[tp_rank]
                    new_params[new_name] = Parameter(v, name=new_name)
                    print(f"{k} {v.shape} {v.dtype} -> "+\
                          f"{new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
                    continue
        # final_layernorm.weight -> language_model.encoder.final_norm.weight
        if "final_layernorm.weight" in k:
            new_name = "language_model.encoder.final_norm.weight"
            new_params[new_name] = Parameter(v.numpy(), name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
        # output_layer.weight -> network.lm_head.weight
        if "output_layer.weight" in k:
            new_name = new_name.replace("output_layer.weight", "language_model.output_layer.weight")
            v = v.numpy()
            if tp_world_size > 1:
                v = np.split(v, tp_world_size, axis=0)[tp_rank]
            new_params[new_name] = Parameter(v, name=new_name)
            print(f"{k} {v.shape} {v.dtype} -> {new_name} {new_params[new_name].shape} {new_params[new_name].dtype}")
            continue
    print("↑↑↑↑↑↑↑↑↑↑↑ pt -> ckpt map ↑↑↑↑↑↑↑↑↑↑↑", flush=True)

    return new_params


def _transform_ckpt_helper(config, model, optimizer, src_ckpt_path, dst_ckpt_path, ckpt_prefix="network", timeout=15):
    """ helper function for transform ckpt """
    save_checkpoint(config, model, optimizer, None, dst_ckpt_path, only_save_strategy=True)
    time.sleep(5)
    if get_rank() == 0:
        src_merged_strategy_file = dst_ckpt_path + "/src_merged_strategy.ckpt"
        dst_merged_strategy_file = dst_ckpt_path + "/dst_merged_strategy.ckpt"
        ms.merge_pipeline_strategys(os.path.join(src_ckpt_path, "strategy"), src_merged_strategy_file)
        ms.merge_pipeline_strategys(os.path.join(dst_ckpt_path, "strategy"), dst_merged_strategy_file)
        ms.transform_checkpoints(src_ckpt_path, dst_ckpt_path, ckpt_prefix,
                                 src_merged_strategy_file,
                                 dst_merged_strategy_file,
                                 output_format='safetensors')
    else:
        time.sleep(timeout)


def read_loss_from_log(file_path):
    """ reading loss from log """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            match_str = re.search(r'Loss: (\d+\.\d+)', line)
            if match_str:
                loss_value = float(match_str.group(1))
                losses.append(loss_value)
    return losses


# pylint: disable=W0613
class MixtralModel(Module):
    r"""
    Mixtral Model
    Args:
        config (TransformerConfig): the config of network;
        num_tokentypes (int): if > 0, using tokentypes embedding. Default: 0;
        parallel_output: (bool), Specifies whether return paralleled output on each tensor parallel rank. Default: True;
        pre_process (bool) when using pipeline parallel, indicate whether it's the first stage. Default: True,
        post_process (bool) when using pipeline parallel, indicate whether it's the last stage. Default: True,
        loss_func: loss function

    Returns:
        output (Tensor): mixtral loss or hidden states

    Examples:
    ```python
    def model_provider_func(pre_process=True, post_process=True):
        ''' get mixtral model '''
        loss = get_loss_func(config.training_config)
        network = MixtralModel(
            model_config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
            )
        return network

    network = get_model(model_provider_func, parallel_config)
    ```
    """

    def __init__(
            self,
            config: TransformerConfig,
            num_tokentypes: int = 0,
            parallel_output: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
            loss_func=None,
            **kwargs):
        super(MixtralModel, self).__init__()
        self.config = config
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.pad_token = config.dataset_config.pad_token
        self.compute_dtype = config.compute_dtype
        self.share_embeddings_and_output_weights = not config.untie_embeddings_and_output_weights

        self.language_model = TransformerLanguageModel(
            config,
            encoder_attn_mask_type=None,
            num_tokentypes=num_tokentypes,
            pre_process=self.pre_process,
            post_process=self.post_process
            )
        if self.post_process:
            self.head = ParallelLMLogits(
                config=config,
                bias=False,
                compute_dtype=config.compute_dtype
                )

            self.loss = loss_func

        if self.share_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(
            self, input_ids: ms.Tensor,
            labels: ms.Tensor = None,
            attention_mask: ms.Tensor = None,
            tokentype_ids: ms.Tensor = None,
            inference_params: ms.Tensor = None,
            loss_mask: ms.Tensor = None):
        """
        Forward of mixtral model.

        Args:
            input_ids (Tensor): the tokenized inputs with datatype int32
            attention_mask (Tensor):
        Returns:
            output (Tensor): the output of mixtral decoderlayer
        """
                # ensure `input_ids` and `labels` shape are [bs, seq]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(dim=0)
        elif labels is None:
            labels = input_ids

        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        position_ids = None

        if loss_mask is None:
            if self.pad_token is None:
                raise RuntimeError("If 'pad_token' is not pass into model, the 'loss_mask' must be not None.")
            loss_mask = mint.ne(input_ids, self.pad_token).astype(self.compute_dtype)

        hidden_states = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            )
        # pylint: disable=R1705
        if self.post_process:
            if not self.share_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            # pylint: disable=E1121
            loss = self.post_language_model_processing(
                hidden_states,
                labels,
                logit_weights,
                loss_mask
                )
            return loss
        else:
            return hidden_states

    def post_language_model_processing(
            self,
            lm_output,
            labels,
            logit_weights,
            loss_mask):
        """define post language model process"""
        logits = self.head(lm_output, logit_weights, self.parallel_output)

        logits = logits.reshape(-1, logits.shape[-1]).to(mstype.float32)
        labels = labels.reshape(-1,).to(mstype.int32)

        loss = self.loss(logits, labels, loss_mask)

        return loss
