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

"""Convert MindSpore checkpoint to Torch"""
import os
import re
import argparse
import torch
from tqdm import tqdm

import mindspore
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


def layer_name_mapping(model_name, key):
    """Convert huggingface PP weights mapping in MindSpore.

    return: new_name
    """
    prefix = ''
    # Handle first and last layers
    if model_name == "telechat_52b":
        layer_rename_map = {
            "transformer.wte.weight": "model.tok_embeddings.embedding_weight",
            "ln_1.weight": "attention_norm.weight",
            "attn.c_attn.weight": "attention.w_qkv.weight",
            "attn.c_proj.weight": "attention.wo.weight",
            "ln_2.weight": "ffn_norm.weight",
            "mlp.c_fc.weight": "feed_forward.w_gate_hidden.weight",
            "mlp.c_proj.weight": "feed_forward.w2.weight",
            "transformer.ln_f.weight": "model.norm_out.weight",
            "lm_head.weight": "lm_head.weight"
        }
    else:
        layer_rename_map = {
            "word_embeddings.weight": "model.tok_embeddings.embedding_weight",
            "input_layernorm.weight": "attention_norm.weight",
            "self_attention.dense.weight": "attention.wo.weight",
            "self_attention.dense.bias": "attention.wo.bias",
            "self_attention.query.weight": "attention.wq.weight",
            "self_attention.key_value.weight": "attention.wk_v.weight",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "mlp.down_proj.bias": "feed_forward.w2.bias",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "ln_f.weight": "model.norm_out.weight"
        }
        if model_name == "telechat_12b":
            del layer_rename_map["word_embeddings.weight"]
            del layer_rename_map["ln_f.weight"]
            layer_rename_map["lm_head.weight"] = "lm_head.weight"
            layer_rename_map["transformer.word_embeddings.weight"] = "model.tok_embeddings.embedding_weight"
            layer_rename_map["transformer.ln_f.weight"] = "model.norm_out.weight"
    if key in layer_rename_map:
        return prefix + layer_rename_map[key]

    # Handle transformer blocks
    if model_name == "telechat_7b":
        match = re.match(r'^\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    else:
        match = re.match(r'^\w+\.\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    layer_number = int(match.group(1))
    text = match.group(2)
    return f"{prefix}model.layers.{layer_number}." + layer_rename_map.get(text)


def hf_to_ms(hf_weights, model_name, ms_dtype=mindspore.float16, for_save=False):
    """Convert hf layers to ms."""
    ms_params = {}
    transpose = P.Transpose()
    split = P.Split(axis=0, output_num=2)
    reshape = P.Reshape()
    concat = P.Concat(axis=1)
    for k, v in hf_weights.items():
        if model_name == "telechat_52b" and (k.endswith("attn.masked_bias") or k.endswith("attn.bias")):
            continue
        new_name = layer_name_mapping(model_name, k)
        print(f"process: {new_name}")
        new_tensor = Tensor(v.float().detach().numpy(), ms_dtype)
        if model_name == "telechat_52b":
            if new_name.endswith("attention.wo.weight"):
                new_tensor = transpose(new_tensor, (1, 0))
            if new_name.endswith("attention.w_qkv.weight"):
                new_tensor = transpose(new_tensor, (1, 0))
                new_tensor = reshape(new_tensor, \
                    (3, args.num_heads, args.hidden_size // args.num_heads, args.hidden_size))
                new_tensor = transpose(new_tensor, (1, 0, 2, 3))
                if args.mp > 1:
                    new_tensor = reshape(new_tensor, \
                        (args.mp, args.num_heads // args.mp, 3, args.hidden_size // args.num_heads, args.hidden_size))
                    new_tensor = transpose(new_tensor, (0, 2, 1, 3, 4))
                new_tensor = reshape(new_tensor, (-1, args.hidden_size))
            if new_name.endswith("w_gate_hidden.weight"):
                new_tensor = transpose(new_tensor, (1, 0))
                if args.mp > 1:
                    ori_h, ori_w = new_tensor.shape
                    gate_weight, hidden_weight = split(new_tensor)
                    gate_weight = reshape(gate_weight, (args.mp, gate_weight.shape[0] // args.mp, gate_weight.shape[1]))
                    hidden_weight = reshape(hidden_weight, \
                        (args.mp, hidden_weight.shape[0] // args.mp, hidden_weight.shape[1]))
                    weight = concat((gate_weight, hidden_weight))
                    new_tensor = reshape(weight, (ori_h, ori_w))
            if new_name.endswith("w2.weight"):
                new_tensor = transpose(new_tensor, (1, 0))
        ms_params[new_name] = Parameter(new_tensor, name=new_name)
    if for_save:
        return [{'name': k, 'data': v} for k, v in ms_params.items()]
    return ms_params


def process_shard_files(files, config, ms_dtype=mindspore.float16):
    ''' torch ckpt files loop'''
    if not config.mindspore_path.endswith(".ckpt"):
        if not config.mindspore_path:
            config.mindspore_path = "./convert_torch_to_ms_output"
        os.makedirs(config.mindspore_path, exist_ok=True)
        ms_file_name = "mindspore_" + args.model_name + ".ckpt"
        save_file = os.path.join(config.mindspore_path, ms_file_name)
    else:
        save_file = config.mindspore_path

    combine_params = []
    for per_file in tqdm(files):
        pt_states = torch.load(per_file, map_location='cpu')
        ms_params = hf_to_ms(pt_states, config.model_name, ms_dtype, True)
        combine_params.extend(ms_params)
        del ms_params
    mindspore.save_checkpoint(combine_params, save_file)
    print(f"*** finish torch convert ms model, ms_ckpt save in {save_file} ***")


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
    parser.add_argument("--model_name",
                        type=str,
                        default="telechat_52b",
                        help="The name of model, supports name in {'telechat_7b', 'telechat_12b', 'telechat_52b'}")
    parser.add_argument("--mp",
                        type=str,
                        default=4,
                        help="The name of model, supports name in {'telechat_7b', 'telechat_12b', 'telechat_52b'}")
    parser.add_argument("--num_heads",
                        type=int,
                        default=64,
                        help="The num_heads telechat 52B.")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=8192,
                        help="The hidden_size telechat 52B.")
    args = parser.parse_args()

    # convert hf ckpt to ms
    files_list = []
    for file_name in os.listdir(args.torch_path):
        if file_name.startswith("pytorch_model") and file_name.endswith(".bin"):
            files_list.append(os.path.join(args.torch_path, file_name))

    process_shard_files(files=files_list, config=args, ms_dtype=mindspore.float32)
