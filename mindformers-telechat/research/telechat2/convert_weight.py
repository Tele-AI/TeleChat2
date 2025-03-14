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
Convert Telechat weight.
Support huggingface format.
"""

import os
import argparse
from glob import glob

import numpy as np
import torch

import mindspore as ms
from mindformers.tools.utils import str2bool
from mindformers.tools import logger
from mindformers.utils.convert_utils import pt2ms

dtype_map = {
    'float32': ms.float32,
    'bfloat16': ms.bfloat16,
    'float16': ms.float16
}


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('transformer.word_embeddings.weight', 'model.tok_embeddings.embedding_weight')
    name = name.replace('.input_layernorm', '.attention_norm')
    name = name.replace('.self_attention.dense.', '.attention.wo.')
    name = name.replace('.self_attention.dense.bias.', '.attention.wo.bias.')
    name = name.replace('.self_attention.query.', '.attention.wq.')
    name = name.replace('.self_attention.key_value.', '.attention.wk_v.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.down_proj.bias.', '.feed_forward.w2.bias.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('lm_head.', 'lm_head.')
    name = name.replace('transformer.ln_f.', 'model.norm_out.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert telechat hf weight to ms."""
    files = list(glob(os.path.join(input_path, "pytorch_model*.bin")))
    files.sort()
    pt_states_list = []
    for per_file in files:
        pt_states = torch.load(per_file, map_location='cpu')
        pt_states_list.append(pt_states)

    ckpt_list = []
    for pt_states in pt_states_list:
        for name, value in pt_states.items():
            name = name_replace(name)
            if name.startswith('transformer.h.'):
                name = name.replace('transformer.h.', 'model.layers.')
            logger.info(f'\rprocessing parameter: {name} {value.shape}')
            ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    logger.info(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.")
    return True


def transformer_new_kv_param(n_kv_heads, w_kv_weight, mp):
    """transform new kv param."""
    head_dim = 128
    size = n_kv_heads // mp * 2
    split_size_or_sections = size * [head_dim]
    print("w_kv_weight split_size_or_sections size {}, {}:".format(len(split_size_or_sections),
                                                                   split_size_or_sections))
    new_kv_weight = ms.ops.split(w_kv_weight, split_size_or_sections, 0)

    new_k_weight = []
    new_v_weight = []
    for i in range(0, size):
        if i % 2 == 0:
            new_k_weight.append(new_kv_weight[i].asnumpy())
        else:
            new_v_weight.append(new_kv_weight[i].asnumpy())
    new_k_weight = np.concatenate(new_k_weight, 0)
    new_v_weight = np.concatenate(new_v_weight, 0)
    return new_k_weight, new_v_weight


def convert_qkv_concat_weight(n_kv_heads, num_layers, param_dict, mp=1):
    """convert qkv concat weight"""
    for i in range(num_layers):
        # qkv weight concat
        wq_weight_name = f"model.layers.{i}.attention.wq.weight"
        w_kv_weight_name = f"model.layers.{i}.attention.wk_v.weight"
        qkv_concat_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
        if wq_weight_name not in param_dict:
            break
        wq_weight = param_dict[wq_weight_name]
        w_kv_weight = param_dict[w_kv_weight_name]

        new_k_weight, new_v_weight = transformer_new_kv_param(n_kv_heads, w_kv_weight, mp)
        qkv_weight = np.concatenate((wq_weight.asnumpy(), new_k_weight, new_v_weight), 0)
        param_dict[qkv_concat_weight_name] = ms.Parameter(qkv_weight, name=qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"

        ffn_gate_weight = param_dict[ffn_gate_weight_name]
        ffn_hidden_weight = param_dict[ffn_hidden_weight_name]
        gate_hidden_weight = np.concatenate((ffn_gate_weight.asnumpy(), ffn_hidden_weight.asnumpy()), 0)
        param_dict[gate_hidden_concat_weight_name] = ms.Parameter(gate_hidden_weight,
                                                                  name=gate_hidden_concat_weight_name)

        param_dict.pop(wq_weight_name)
        param_dict.pop(w_kv_weight_name)
        param_dict.pop(ffn_gate_weight_name)
        param_dict.pop(ffn_hidden_weight_name)
        print("transform: {}".format(qkv_concat_weight_name))
        print("transform: {}".format(gate_hidden_concat_weight_name))

    return param_dict


def convert_to_qkv_concat(model_name, pre_ckpt_path, mindspore_ckpt_path):
    """convert previous ckpt to qkv concat ckpt"""
    if model_name == "telechat_7B":
        n_kv_heads = 32
        num_layers = 30
    elif model_name == "telechat_35B":
        n_kv_heads = 48
        num_layers = 64
    elif model_name == "telechat_115B":
        n_kv_heads = 8
        num_layers = 96
    else:
        raise ValueError("model_name:{} is not supported.".format(model_name))

    if os.path.isdir(pre_ckpt_path):
        rank_dir_list = os.listdir(pre_ckpt_path)
        mp = len(rank_dir_list)
        for rank_dir in rank_dir_list:
            rank_dir_name = str(rank_dir)
            rank_id = rank_dir_name.split("_")[1]
            checkpoint_path = os.path.join(pre_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            print("checkpoint_path: {}".format(checkpoint_path))
            params = ms.load_checkpoint(checkpoint_path)
            params = convert_qkv_concat_weight(n_kv_heads, num_layers, params, mp)

            save_dir = os.path.join(mindspore_ckpt_path, rank_dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(mindspore_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            ms.save_checkpoint(params, save_path)
    else:
        params = ms.load_checkpoint(pre_ckpt_path)
        params = convert_qkv_concat_weight(n_kv_heads, num_layers, params)
        ms.save_checkpoint(params, mindspore_ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Telechat convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="",
                        help="The input torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--dtype", default='float32', choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")
    parser.add_argument('--qkv_concat', default=False, type=str2bool)
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--model_name', default="telechat_7B", type=str)
    args = parser.parse_args()
    ms_dtype = dtype_map.get(args.dtype)

    if args.qkv_concat:
        # for example
        # python convert_weight.py --model_name=telechat_115B --qkv_concat=True --pre_ckpt_path=/path/to/llama_dist_ckpt_dir  --mindspore_ckpt_path=/path/to/new_llama_dist_ckpt_dir
        # python convert_weight.py --model_name=telechat_7B --qkv_concat=True --pre_ckpt_path=/path/to/llama.ckpt  --mindspore_ckpt_path=/path/to/new_llama.ckpt
        convert_to_qkv_concat(args.model_name, args.pre_ckpt_path, args.mindspore_ckpt_path)
    else:
        # convert hf ckpt to ms
        convert_pt_to_ms(input_path=args.torch_path, output_path=args.mindspore_path, dtype=ms_dtype)
