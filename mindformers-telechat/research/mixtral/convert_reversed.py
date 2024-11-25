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
"""
Convert mixtral weight.
Support mindspore format.
"""

import os
import argparse
from pathlib import Path
import torch
import mindspore as ms

from mindspore.ops.operations import Cast
from safetensors.torch import save_file
from mindformers.trainer.utils import get_last_checkpoint

ms.set_context(device_target='CPU')
cpu_cast = Cast().set_device('CPU')


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('tok_embeddings.embedding_weight', 'embed_tokens.weight')
    name = name.replace('.feed_forward.router.dense.', '.block_sparse_moe.gate.')
    name = name.replace('.feed_forward.ffn.', '.block_sparse_moe.experts.')
    name = name.replace('.attention.wk.', '.self_attn.k_proj.')
    name = name.replace('.attention.wo.', '.self_attn.o_proj.')
    name = name.replace('.attention.wq.', '.self_attn.q_proj.')
    name = name.replace('.attention.wv.', '.self_attn.v_proj.')
    name = name.replace('.attention_norm.', '.input_layernorm.')
    name = name.replace('.ffn_norm.', '.post_attention_layernorm.')
    return name


def merge_ms_ckpt(ckpt_dir, strategy_dir, rank_id=0):
    """merge ms weight with strategy files."""
    print("Now merging strategy files...")
    if not os.path.exists(strategy_dir):
        raise Exception("Trying convert mindspore distributed ckpt, but strategy ckpt path is not exit. \
                        Please make sure your path is correct.")
    merge_strategy_name = os.path.join(strategy_dir, 'merge_strategy.ckpt')
    ms.merge_pipeline_strategys(strategy_dir, merge_strategy_name)
    print("Now merging distributed ckpt...")
    merge_ckpt_save_path = os.path.join(ckpt_dir, "merge_ckpt")
    if not os.path.exists(merge_ckpt_save_path):
        os.makedirs(merge_ckpt_save_path)
    merge_ckpt_name = 'ckpt_merge_'
    ms.transform_checkpoints(src_checkpoints_dir=ckpt_dir,
                             dst_checkpoints_dir=merge_ckpt_save_path,
                             ckpt_prefix=merge_ckpt_name,
                             src_strategy_file=merge_strategy_name,
                             dst_strategy_file=None)
    print("Merge mindspore distributed ckpt completed")
    return os.path.join(merge_ckpt_save_path, \
                        f'rank_{rank_id}/' + merge_ckpt_name + f'{rank_id}.ckpt')


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, strategy_dir=None, **kwargs):
    """convert ms weight to torch."""
    is_single_ckpt_path = input_path.endswith(".ckpt")
    single_ckpt_dir = sorted(Path(input_path).glob("*.ckpt"))
    distributed_ckpt_dir = sorted(Path(input_path).glob("rank_*"))
    if is_single_ckpt_path or len(single_ckpt_dir) == 1:
        ckpt_path = input_path if is_single_ckpt_path else single_ckpt_dir[0]
        ms_ckpt_convertor(ckpt_path, output_path, dtype)
    elif distributed_ckpt_dir:
        if len(distributed_ckpt_dir) == 1:
            ckpt_path = get_last_checkpoint(distributed_ckpt_dir[0])
            ms_ckpt_convertor(ckpt_path, output_path, dtype)
        else:
            ckpt_path = merge_ms_ckpt(input_path, strategy_dir)
            ms_ckpt_convertor(ckpt_path, output_path, dtype)
    else:
        raise Exception("Invalid mindspore ckpt path, the path format can be:\n \
                        1.Specific path with ckpt name \n \
                        2.File dir of only one ckpt file \n \
                        3.File dir of containing distributed ckpt folder, \
                          multiple ckpt folder format should be: \
                          rank_{0..n}/*.ckpt")


def ms_ckpt_convertor(ckpt_path, output_name, dtype):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{ckpt_path}'", flush=True)
    param_dict = ms.load_checkpoint(ckpt_path)
    output_dict = {}
    for name, value in param_dict.items():
        name = name_replace(name)
        value = cpu_cast(value, ms.float32).asnumpy()
        if name == "model.norm_out.weight":
            name = "model.norm.weight"
            value = torch.from_numpy(value).to(dtype)
            print(f"\rprocessing parameter: {name} {value.shape}   ", flush=True)
            output_dict[name] = value
        else:
            if ".block_sparse_moe.experts." not in name:
                value = torch.from_numpy(value).to(dtype)
                print(f"\rprocessing parameter: {name} {value.shape}   ", flush=True)
                output_dict[name] = value
            else:
                for index in range(8):
                    new_name = name.split('experts.')[0] + "experts." + str(index) + name.split("experts")[1]
                    print(f"\rprocessing parameter: {new_name} {value[index].shape}   ", flush=True)
                    output_dict[new_name] = torch.from_numpy(value[index]).to(dtype)
    if output_name.endswith(".safetensors"):
        save_file(output_dict, output_name)
    else:
        if not os.path.exists(output_name):
            os.makedirs(output_name)
        save_file(output_dict, output_name + "mixtral8_7b.safetensors")
    print(f"\rConvert ms checkpoint finished, the huggingface checkpoint is saved in '{output_name}'.", flush=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', type=str, default='./mixtral/ms_ckpt/')
    parser.add_argument('--torch_ckpt_dir', type=str, default='./mixtral/torch_ckpt/')
    parser.add_argument('--strategy_dir', type=str, default='None')
    parser.add_argument('--dtype', default=None, choices=['fp16', 'fp32', 'bf16'])
    args = parser.parse_args()
    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_ckpt_dir,
                     strategy_dir=args.strategy_dir, dtype=args.dtype, **extra_kwargs)
