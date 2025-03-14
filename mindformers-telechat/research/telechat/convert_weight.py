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
"""Convert checkpoint from huggingface"""
import os.path
import re
import argparse
import torch
import mindspore as ms

from mindformers.utils.convert_utils import pt2ms


def layer_name_mapping(telechat_type, key):
    """Convert huggingface PP weights mapping in MindSpore.

    return: new_name
    """
    # Handle first and last layers
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
    if telechat_type == "telechat_12b":
        del layer_rename_map["word_embeddings.weight"]
        del layer_rename_map["ln_f.weight"]
        layer_rename_map["lm_head.weight"] = "lm_head.weight"
        layer_rename_map["transformer.word_embeddings.weight"] = "model.tok_embeddings.embedding_weight"
        layer_rename_map["transformer.ln_f.weight"] = "model.norm_out.weight"
    if key in layer_rename_map:
        return layer_rename_map[key]

    # Handle transformer blocks
    match = re.match(r'^\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    layer_number = int(match.group(1))
    text = match.group(2)
    return f"model.layers.{layer_number}." + layer_rename_map.get(text)


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms"""
    telechat_type = kwargs.pop("telechat_type", "telechat_12b")
    state_dict = {}
    torch_dir = os.path.dirname(input_path)
    for file_name in os.listdir(torch_dir):
        if file_name.startswith("pytorch_model") and file_name.endswith(".bin"):
            file_name = os.path.join(torch_dir, file_name)
            state_dict.update(torch.load(file_name, map_location='cpu'))

    ms_params = []
    for k, v in state_dict.items():
        ms_params.append({'name': layer_name_mapping(telechat_type, k), 'data': pt2ms(v, dtype)})

    ms.save_checkpoint(ms_params, output_path)
    print(f"*** finish torch convert ms model, ms_ckpt save in {output_path} ***")


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
    parser.add_argument("--telechat_type",
                        type=str,
                        default="telechat_12b",
                        help="Telechat version.")
    args = parser.parse_args()

    # convert hf ckpt to ms
    convert_pt_to_ms(args.torch_path, args.mindspore_path, telechat_type=args.telechat_type)
