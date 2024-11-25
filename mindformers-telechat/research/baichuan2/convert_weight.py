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
Convert Baichuan weight.
Support huggingface format.
"""

import os
import json
import argparse

import mindspore as ms

from mindformers.utils.convert_utils import pt2ms


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert baichuan hf weight to ms."""
    ckpt_dir = os.path.dirname(input_path)
    print(f"Trying to convert huggingface checkpoint in '{ckpt_dir}'.", flush=True)
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model_hf = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True)
        args_hf = read_json(os.path.join(ckpt_dir, "config.json"))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Error {e}.", flush=True)
        return False

    dim = args_hf["hidden_size"]

    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if 'W_pack' in name:
            values = torch.split(value, dim)
            wq = name.replace('.self_attn.W_pack', '.attention.wq')  # '.self_attn.q_proj.', '.attention.wq.'
            q_value = values[0]
            wk = name.replace('.self_attn.W_pack', '.attention.wk')
            k_value = values[1]
            wv = name.replace('.self_attn.W_pack', '.attention.wv')
            v_value = values[2]
            print(f'\rprocessing parameter: {wq} {q_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wq, 'data': pt2ms(q_value, dtype)})
            print(f'\rprocessing parameter: {wk} {k_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wk, 'data': pt2ms(k_value, dtype)})
            print(f'\rprocessing parameter: {wv} {v_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wv, 'data': pt2ms(v_value, dtype)})
            continue
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert baichuan checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_path', default='./hf.bin')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    args = parser.parse_args()
    convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path)
