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
import re
import argparse
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt

layer_rename_map = {
    'model.tok_embeddings.embedding_weight': 'word_embeddings.weight',
    'attention_norm.weight': 'input_layernorm.weight',
    'attention.wo.weight': 'self_attention.dense.weight',
    'attention.wo.bias': 'self_attention.dense.bias',
    'attention.wq.weight': 'self_attention.query.weight',
    'attention.wk_v.weight': 'self_attention.key_value.weight',
    'feed_forward.w1.weight': 'mlp.gate_proj.weight',
    'feed_forward.w2.weight': 'mlp.down_proj.weight',
    'feed_forward.w2.bias': 'mlp.down_proj.bias',
    'feed_forward.w3.weight': 'mlp.up_proj.weight',
    'ffn_norm.weight': 'post_attention_layernorm.weight',
    'model.norm_out.weight': 'ln_f.weight'
}


def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    telechat_type = kwargs.pop("telechat_type", "telechat_12b")
    if telechat_type == "telechat_12b":
        layer_rename_map["lm_head.weight"] = "lm_head.weight"
        layer_rename_map["model.tok_embeddings.embedding_weight"] = "transformer.word_embeddings.weight"
        layer_rename_map["model.norm_out.weight"] = "transformer.ln_f.weight"
    param_dict = ms.load_checkpoint(input_path)
    state_dict = {}
    for name, value in param_dict.items():
        value = ms2pt(value, dtype)
        if name in layer_rename_map:
            name = layer_rename_map[name]
        else:
            match = re.match(r"model\.layers\.(\d+).(\w+\.\w+\.\w+|\w+\.\w+)$", name)
            layer_number = int(match.group(1))
            text = match.group(2)
            value = layer_rename_map.get(text)
            if value:
                name = f"h.{layer_number}.{value}"
            else:
                raise ValueError(f"text:{text} is not in layer_number:{layer_number}.")
        state_dict[name] = value

    torch.save(state_dict, output_path)
    print(f"*** finish ms convert torch model, torch_ckpt save in {output_path} ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Telechat convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        default="",
                        help="The input torch checkpoint path.")
    parser.add_argument("--telechat_type",
                        type=str,
                        default="telechat_12b",
                        help="Telechat version.")
    args = parser.parse_args()

    # convert hf ckpt to ms
    convert_ms_to_pt(args.mindspore_path, args.torch_path, telechat_type=args.telechat_type)
