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
"""Convert checkpoint from mindspore"""
import argparse
import collections

import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt

ms_name = [
    "backbone.blocks.{}.layernorm1.gamma",
    "backbone.blocks.{}.layernorm1.beta",
    "backbone.blocks.{}.layernorm2.gamma",
    "backbone.blocks.{}.layernorm2.beta",
    "backbone.blocks.{}.attention.beta",
    "backbone.blocks.{}.attention.masked_beta",
    "backbone.blocks.{}.attention.projection.weight",
    "backbone.blocks.{}.attention.projection.bias",
    "backbone.blocks.{}.attention.dense1.weight",
    "backbone.blocks.{}.attention.dense1.bias",
    "backbone.blocks.{}.attention.dense2.weight",
    "backbone.blocks.{}.attention.dense2.bias",
    "backbone.blocks.{}.attention.dense3.weight",
    "backbone.blocks.{}.attention.dense3.bias",
    "backbone.blocks.{}.output.mapping.weight",
    "backbone.blocks.{}.output.mapping.bias",
    "backbone.blocks.{}.output.projection.weight",
    "backbone.blocks.{}.output.projection.bias",
]

torch_name = [
    "h.{}.ln_1.weight",
    "h.{}.ln_1.bias",
    "h.{}.ln_2.weight",
    "h.{}.ln_2.bias",
    "h.{}.attn.bias",
    "h.{}.attn.masked_bias",
    "h.{}.attn.c_proj.weight",
    "h.{}.attn.c_proj.bias",
    "h.{}.attn.c_attn.weight.q",
    "h.{}.attn.c_attn.bias.q",
    "h.{}.attn.c_attn.weight.k",
    "h.{}.attn.c_attn.bias.k",
    "h.{}.attn.c_attn.weight.v",
    "h.{}.attn.c_attn.bias.v",
    "h.{}.mlp.c_fc.weight",
    "h.{}.mlp.c_fc.bias",
    "h.{}.mlp.c_proj.weight",
    "h.{}.mlp.c_proj.bias"
]

addition_mindspore = [
    "backbone.layernorm.gamma",
    "backbone.layernorm.beta",
    "backbone.embedding.word_embedding.embedding_table",
    "backbone.embedding.position_embedding.embedding_table",
]

addition_torch = [
    "ln_f.weight",
    "ln_f.bias",
    "wte.weight",
    "wpe.weight",
]


def generate_weight_map(total_layers,
                        mindspore_params_per_layer,
                        torch_params_per_layer,
                        mindspore_additional_params,
                        torch_additional_params):
    """
    generate weight map
    """
    map_dict = dict(zip(mindspore_additional_params, torch_additional_params))
    for i in range(total_layers):
        for idx, ms_para in enumerate(mindspore_params_per_layer):
            map_dict[ms_para.format(i)] = torch_params_per_layer[idx].format(i)

    return map_dict


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert ms to pt
    """
    state_dict = {}
    print(f"Trying to convert mindspore checkpoint in {input_path}.")
    model_ms = ms.load_checkpoint(input_path)

    # to compatible with three types of gpt,calculate missed param nums
    count = 0
    for n in ms_name:
        if n.format(0) not in model_ms:
            count += 1

    if len(ms_name) != len(torch_name):
        raise ValueError(f"len(ms_name) should equal len(torch_name), but len(ms_name) got {len(ms_name)},"
                         f"len(torch_name) got {len(torch_name)}.")
    if len(addition_mindspore) != len(addition_mindspore):
        raise ValueError(f"len(addition_mindspore) should equal to len(addition_mindspore), "
                         f"but len(addition_mindspore) got {len(addition_mindspore)},"
                         f"len(addition_mindspore) got {len(addition_mindspore)}.")
    total_layers, flag = divmod(len(model_ms) - len(addition_mindspore), len(ms_name) - count)
    if flag:
        raise Exception("The weight names don't match.")
    weight_map = generate_weight_map(total_layers, ms_name, torch_name, addition_mindspore, addition_torch)

    attention_dict = collections.defaultdict(lambda: {})
    for name, value in model_ms.items():
        value = ms2pt(value, dtype)
        if "attention.dense" in name and name.endswith('weight'):
            value = value.transpose(1, 0)
        name = weight_map[name]

        if name.endswith('.q'):
            name = name.rstrip('.q')
            attention_dict[name]['q'] = value
            continue
        if name.endswith('.k'):
            name = name.rstrip('.k')
            attention_dict[name]['k'] = value
            continue
        if name.endswith('.v'):
            name = name.rstrip('.v')
            attention_dict[name]['v'] = value
            continue

        state_dict[name] = value

    for name, value_dict in attention_dict.items():
        state_dict[name] = torch.cat((value_dict['q'], value_dict['k'], value_dict['v']), -1)

    torch.save(state_dict, output_path)
    print(f"Convert finished, the output is saved to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        required=True,
                        default=None,
                        help="The output torch checkpoint path.")

    opt = parser.parse_args()
    convert_ms_to_pt(opt.mindspore_path, opt.torch_path)
