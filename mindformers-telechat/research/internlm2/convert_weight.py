# Copyright 2023 Huawei Technologies Co., Ltd
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
Convert internlm2 weight.
Support huggingface format.
"""

import os
import json
import argparse

import mindspore as ms
import torch

from mindformers.utils.convert_utils import pt2ms


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('tok_embeddings.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('wqkv', 'w')
    name = name.replace('.norm.', '.norm_out.')
    name = name.replace('output', 'lm_head')
    return name


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import AutoModelForCausalLM
        model_hf = AutoModelForCausalLM.from_pretrained(input_path, trust_remote_code=True)
        args_hf = read_json(os.path.join(input_path, "config.json"))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{input_path}', Error {e}.", flush=True)
        return False

    num_key_value_heads = args_hf["num_key_value_heads"]
    hidden_size = args_hf["hidden_size"]
    num_attention_heads = args_hf["num_attention_heads"]
    head_dim = hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads

    qkv_concat = kwargs.get("qkv_concat", True)

    ckpt_list = []
    for name, value in model_hf.named_parameters():
        name = name_replace(name)
        if not qkv_concat and '.w.' in name:
            slices = torch.split(value, head_dim * (num_key_value_groups + 2))
            q_name = name.replace('.w.', '.wq.')
            q_value = torch.cat([slice[:-2 * head_dim, :] for slice in slices], dim=0)
            k_name = name.replace('.w.', '.wk.')
            k_value = torch.cat([slice[-2 * head_dim:-head_dim, :] for slice in slices], dim=0)
            v_name = name.replace('.w.', '.wv.')
            v_value = torch.cat([slice[-head_dim:, :] for slice in slices], dim=0)
            print(f'\rprocessing parameter: {q_name} {q_value.shape}')
            ckpt_list.append({'name': q_name, 'data': pt2ms(q_value, dtype)})
            print(f'\rprocessing parameter: {k_name} {k_value.shape}')
            ckpt_list.append({'name': k_name, 'data': pt2ms(k_value, dtype)})
            print(f'\rprocessing parameter: {v_name} {v_value.shape}')
            ckpt_list.append({'name': v_name, 'data': pt2ms(v_value, dtype)})
        else:
            print(f'\rprocessing parameter: {name} {value.shape}')
            ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, os.path.join(output_path))
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./internlm2-7b/')
    parser.add_argument('--mindspore_ckpt_path', default='./internlm2_7b.ckpt')
    parser.add_argument('--qkv_concat', default=True, type=bool)
    parser.add_argument('dtype', default='float16', type=str, choices=['float16', 'float32', 'bfloat16'])
    args = parser.parse_args()
    dtype_map = {'float16': ms.float16, 'float32': ms.float32, 'bfloat16': ms.bfloat16}
    convert_pt_to_ms(input_path=args.torch_ckpt_dir,
                     output_path=args.mindspore_ckpt_path,
                     qkv_concat=args.qkv_concat,
                     dtype=dtype_map.get(args.dtype))
