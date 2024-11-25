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

""" Convert checkpoint from mindspore."""

import argparse
import collections
from collections import OrderedDict
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt

name_pt2ms = {
    "cls_token": "cls_tokens",
    "attn.proj": "attention.projection",
    "attn.q_bias": "attention.dense1.bias",
    "attn.v_bias": "attention.dense3.bias",
    "norm1.weight": "layernorm1.gamma",
    "norm1.bias": "layernorm1.beta",
    "norm2.weight": "layernorm2.gamma",
    "norm2.bias": "layernorm2.beta",
    "fc_norm.weight": "fc_norm.gamma",
    "fc_norm.bias": "fc_norm.beta",
    "ln_vision.weight": "ln_vision.gamma",
    "ln_vision.bias": "ln_vision.beta",
    "mlp.fc2.weight": "output.mapping.weight",
    "mlp.fc1.bias": "output.mapping.bias",
    "mlp.fc1.weight": "output.projection.weight",
    "mlp.fc2.bias": "output.projection.bias",
    "LayerNorm.": "layernorm.",
    "layernorm.weight": "layernorm.gamma",
    "layernorm.bias": "layernorm.beta",
    "embeddings.weight": "embeddings.embedding_table",
    "self": "self_att",
}


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert ms to pt
    """
    state_dict = {}
    model_ms = ms.load_checkpoint(input_path)
    attention_dict = collections.defaultdict(lambda: {})
    for name, value in model_ms.items():
        value = ms2pt(value, dtype)
        if "qkv.weight" in name:
            if "attention.dense1" in name:
                name = name.replace("attention.dense1", "attn.qkv")
                attention_dict[name]['dense1'] = value
                continue
            if "attention.dense2" in name:
                name = name.replace("attention.dense2", "attn.qkv")
                attention_dict[name]['dense2'] = value
                continue
            if "attention.dense3" in name:
                name = name.replace("attention.dense3", "attn.qkv")
                attention_dict[name]['dense3'] = value
                continue
        else:
            if name.endswith('attention.dense2.bias'):
                continue
            if name.endswith("output.mapping.weight") or \
                    name.endswith("output.projection.weight") or \
                    name.endswith("attention.projection.weight"):
                value = value.T
            for replace_from, replace_to in reversed(name_pt2ms.items()):
                name = name.replace("qformer.", "Qformer.")
                name = name.replace(replace_to, replace_from)
            state_dict[name] = value
    for name, value_dict in attention_dict.items():
        state_dict[name] = torch.cat((value_dict['dense1'], value_dict['dense2'], value_dict['dense3']))

    pth_dict = OrderedDict()
    pth_dict['model'] = state_dict

    torch.save(pth_dict, output_path)
    print(f"\n----------------- convert {input_path} to {output_path} Finished! -----------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="blip2 weight convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="blip2_pretrain.ckpt",
                        help="The mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        required=True,
                        default="blip2_pretrain.pth",
                        help="The output torch checkpoint path.")

    opt = parser.parse_args()

    convert_ms_to_pt(opt.mindspore_path, opt.torch_path)
