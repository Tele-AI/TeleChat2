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

def layer_name_mapping(key):
    """Convert huggingface PP weights mapping in MindSpore.

    return: new_name
    """
    prefix = ''
    # Handle first and last layers
    layer_rename_map = {
        "transformer.word_embeddings.weight": "model.tok_embeddings.embedding_weight",
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
        "lm_head.weight": "lm_head.weight",
        "transformer.ln_f.weight": "model.norm_out.weight"
    }
    if key in layer_rename_map:
        return prefix + layer_rename_map[key]

    # Handle transformer blocks
    match = re.match(r'^\w+\.\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    layer_number = int(match.group(1))
    text = match.group(2)
    return f"{prefix}model.layers.{layer_number}." + layer_rename_map[text]

def hf_to_ms(hf_weights, ms_dtype=mindspore.float16, for_save=False):
    """Convert hf layers to ms."""
    ms_params = {}
    for k, v in hf_weights.items():
        new_name = layer_name_mapping(k)
        new_tensor = Tensor(v.float().detach().numpy(), ms_dtype)
        ms_params[new_name] = Parameter(new_tensor, name=new_name)
    if for_save:
        return [{'name': k, 'data': v} for k, v in ms_params.items()]
    return ms_params

def process_shard_files(files, config, ms_dtype=mindspore.float16):
    ''' torch ckpt files loop'''
    if config.mindspore_path and not os.path.exists(args.mindspore_path):
        os.makedirs(config.mindspore_path, exist_ok=True)

    ms_file_name = "mindspore"
    combine_params = []
    for per_file in tqdm(files):
        pt_states = torch.load(per_file, map_location='cpu')
        ms_params = hf_to_ms(pt_states, ms_dtype, True)
        combine_params.extend(ms_params)
        del ms_params
    save_file = config.mindspore_path + '/' + ms_file_name + '.ckpt'
    mindspore.save_checkpoint(combine_params, save_file)


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
    args = parser.parse_args()

    # convert hf ckpt to ms
    files_list = []
    for file_name in os.listdir(args.torch_path):
        if file_name.startswith("pytorch_model") and file_name.endswith(".bin"):
            files_list.append(os.path.join(args.torch_path, file_name))
    process_shard_files(files=files_list, config=args)
    current_path = os.getcwd()
    mindspore_ckpt_path = os.path.join(current_path, args.mindspore_path)
    print("*** finish torch convert ms model, ms_ckpt save in {} ***".format(mindspore_ckpt_path))
