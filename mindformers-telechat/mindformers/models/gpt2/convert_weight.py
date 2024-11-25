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
"""Convert checkpoint from torch/huggingface"""
import argparse
import torch
import mindspore as ms

from mindspore import save_checkpoint

from mindformers.utils.convert_utils import pt2ms

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


def generate_params_dict(total_layers,
                         mindspore_params_per_layer,
                         torch_params_per_layer,
                         mindspore_additional_params,
                         torch_additional_params):
    """
    Generate the total parameter mapping of mindspore and pytorch.

    Args:
        total_layers(int): The total layers of the net.
        mindspore_params_per_layer(list): The list of params per layer for the net of mindspore.
        torch_params_per_layer(list): The list of params per layer for the net of pytorch.
        mindspore_additional_params(list): The list of params outside the layer for the net of mindspore
        torch_additional_params(list): The list  of params outside the layer for the net of pytorch.

    Returns:
        A list of tuple. The first element is the parameter name of mindspore,
        the another is the parameter name of pytorch.
    """
    mapped_params = list(zip(mindspore_params_per_layer, torch_params_per_layer))
    ms_extend_param_list = []
    torch_extend_param_list = []
    for i in range(total_layers):
        for ms_para, torch_para in mapped_params:
            src = ms_para.format(i)
            tgt = torch_para.format(i)

            ms_extend_param_list.append(src)
            torch_extend_param_list.append(tgt)

    mapped_params = list(zip(mindspore_additional_params, torch_additional_params))
    for ms_para, torch_para in mapped_params:
        ms_extend_param_list.append(ms_para)
        torch_extend_param_list.append(torch_para)

    return list(zip(ms_extend_param_list, torch_extend_param_list))


def print_dict(input_dict):
    """
    Print the keys and values of input dict

    Args:
        input_dict(dict): input dict with key and value.

    Returns:
        None
    """
    for k, v in input_dict.items():
        print(f"Param: {k} with shape {v.shape}")


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """
    convert pt to ms
    """
    layers = kwargs.pop('layers', 12)
    weight_dict = torch.load(input_path, map_location='cpu')
    print_dict(weight_dict)
    mapped_params = generate_params_dict(total_layers=layers,
                                         mindspore_params_per_layer=ms_name,
                                         torch_params_per_layer=torch_name,
                                         mindspore_additional_params=addition_mindspore,
                                         torch_additional_params=addition_torch)
    split_torch_attention(weight_dict, dtype=dtype)

    new_ckpt_list = []
    # Currently, the ms_extend_param the torch_extend_param is the full parameters.
    for src, tgt in mapped_params:
        if tgt in weight_dict:
            value = weight_dict[tgt]
            # split the attention layer for q, k, v

            if 'c_attn.weight' in tgt:
                print("tgt:", tgt)
                value = ms.Tensor(value.transpose([1, 0]))

            print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")

            new_ckpt_list.append({"data": value, "name": src})

    save_checkpoint(new_ckpt_list, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


def split_torch_attention(state, dtype):
    """
    split torch attention
    """
    s = list(state.keys())
    for name in s:
        if name.endswith('attn.c_attn.weight') or name.endswith('attn.c_attn.bias'):
            value = pt2ms(state.pop(name), dtype)
            q, k, v = ms.numpy.split(value, 3, -1)
            state[name + '.q'] = ms.Tensor(q)
            state[name + '.k'] = ms.Tensor(k)
            state[name + '.v'] = ms.Tensor(v)
        else:
            state[name] = pt2ms(state[name], dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument('--layers',
                        type=int,
                        default=12,
                        help="The number of layers of the model to be converted.")
    parser.add_argument("--torch_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="The output mindspore checkpoint path.",
                        help="Use device nums, default is 128.")

    opt = parser.parse_args()
    convert_pt_to_ms(input_path=opt.torch_path, output_path=opt.mindspore_path, layers=opt.layers)
