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

"""Convert checkpoint from torch"""
import argparse
import torch
from mindspore import save_checkpoint, Tensor


def generate_params_dict(total_layers,
                         mindspore_params_per_layer,
                         torch_params_per_layer,
                         mindspore_top_layer,
                         torch_top_layer):
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
    map_params = list(
        zip(mindspore_params_per_layer, torch_params_per_layer))
    output_dict = {}
    for i in range(total_layers):
        for ms_para, torch_para in map_params:
            src = ms_para.format(i)
            tgt = torch_para.format(i)
            output_dict[tgt] = src
    for ms_para, torch_para in zip(mindspore_top_layer, torch_top_layer):
        output_dict[torch_para] = ms_para
    return output_dict


def print_dict(input_dict):
    """
    Print the keys and values of input dict

    Args:
        input_dict(dict): input dict with key and value.

    Returns:
        None
    """
    for k, v in input_dict.items():
        print(f"Param: {k} with shape {v}")


def walk_dict(state_dict, mapped_param: dict):
    """Transfer params"""
    new_ckpt_list = []
    print("Converting Embedding layers...")
    word_embeddings = state_dict['module']['language_model']['embedding']['word_embeddings']['weight']
    new_ckpt_list.append({"data": Tensor(word_embeddings.cpu().numpy(
    )), "name": "backbone.embedding.word_embedding.embedding_table"})
    position_embeddings = state_dict['module']['language_model']['embedding']['position_embeddings']['weight']
    new_ckpt_list.append({"data": Tensor(position_embeddings.cpu().numpy(
    )), "name": "backbone.embedding.position_embedding.embedding_table"})

    print("Converting QueryEmbedding layers...")
    query_embeddings = state_dict['module']['language_model']['topQueryEmbedding']['top_query_embeddings']['weight']
    new_ckpt_list.append({"data": Tensor(query_embeddings.cpu().numpy(
    )), "name": "backbone.top_query_embedding.embedding_table"})

    print("Converting FinalLayerNorm layers...")
    final_layernorm_weight = state_dict['module']['language_model']['transformer']['final_layernorm.weight']
    new_ckpt_list.append({"data": Tensor(
        final_layernorm_weight.cpu().numpy()), "name": "backbone.layernorm.gamma"})
    final_layernorm_bias = state_dict['module']['language_model']['transformer']['final_layernorm.bias']
    new_ckpt_list.append({"data": Tensor(
        final_layernorm_bias.cpu().numpy()), "name": "backbone.layernorm.beta"})

    print("Converting Transformer layers...")
    for layer_name in state_dict['module']['language_model']['transformer'].keys():
        params = state_dict['module']['language_model']['transformer'][layer_name]
        if layer_name in mapped_param.keys():
            if "h_to_4h.weight" in layer_name or "4h_to_h.weight" in layer_name \
                or "attention.dense.weight" in layer_name:
                new_ckpt_list.append(
                    {"data": Tensor(params.cpu().numpy().T), "name": mapped_param[layer_name]})
            else:
                new_ckpt_list.append(
                    {"data": Tensor(params.cpu().numpy()), "name": mapped_param[layer_name]})
    return new_ckpt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument('--layers',
                        type=int,
                        default=39,
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
    para_dict = torch.load(opt.torch_path, map_location='cpu')

    ms_name = [
        "backbone.blocks.{}.layernorm1.gamma",
        "backbone.blocks.{}.layernorm1.beta",
        "backbone.blocks.{}.layernorm2.gamma",
        "backbone.blocks.{}.layernorm2.beta",
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
        'layers.{}.input_layernorm.weight',
        'layers.{}.input_layernorm.bias',
        'layers.{}.post_attention_layernorm.weight',
        'layers.{}.post_attention_layernorm.bias',
        'layers.{}.attention.dense.weight',
        'layers.{}.attention.dense.bias',
        'layers.{}.attention.query.weight',
        'layers.{}.attention.query.bias',
        'layers.{}.attention.key.weight',
        'layers.{}.attention.key.bias',
        'layers.{}.attention.value.weight',
        'layers.{}.attention.value.bias',
        'layers.{}.mlp.dense_h_to_4h.weight',
        'layers.{}.mlp.dense_h_to_4h.bias',
        'layers.{}.mlp.dense_4h_to_h.weight',
        'layers.{}.mlp.dense_4h_to_h.bias'
    ]

    ms_top_layer_name = [
        "backbone.top_query_layer.layernorm1.gamma",
        "backbone.top_query_layer.layernorm1.beta",
        "backbone.top_query_layer.layernorm2.gamma",
        "backbone.top_query_layer.layernorm2.beta",
        "backbone.top_query_layer.attention.projection.weight",
        "backbone.top_query_layer.attention.projection.bias",
        "backbone.top_query_layer.attention.dense1.weight",
        "backbone.top_query_layer.attention.dense1.bias",
        "backbone.top_query_layer.attention.dense2.weight",
        "backbone.top_query_layer.attention.dense2.bias",
        "backbone.top_query_layer.attention.dense3.weight",
        "backbone.top_query_layer.attention.dense3.bias",
        "backbone.top_query_layer.output.mapping.weight",
        "backbone.top_query_layer.output.mapping.bias",
        "backbone.top_query_layer.output.projection.weight",
        "backbone.top_query_layer.output.projection.bias",
    ]

    torch_top_layer_name = [
        'topQueryLayer.input_layernorm.weight',
        'topQueryLayer.input_layernorm.bias',
        'topQueryLayer.post_attention_layernorm.weight',
        'topQueryLayer.post_attention_layernorm.bias',
        'topQueryLayer.attention.dense.weight',
        'topQueryLayer.attention.dense.bias',
        'topQueryLayer.attention.query.weight',
        'topQueryLayer.attention.query.bias',
        'topQueryLayer.attention.key.weight',
        'topQueryLayer.attention.key.bias',
        'topQueryLayer.attention.value.weight',
        'topQueryLayer.attention.value.bias',
        'topQueryLayer.mlp.dense_h_to_4h.weight',
        'topQueryLayer.mlp.dense_h_to_4h.bias',
        'topQueryLayer.mlp.dense_4h_to_h.weight',
        'topQueryLayer.mlp.dense_4h_to_h.bias'
    ]

    mapped_params = generate_params_dict(opt.layers,
                                         ms_name,
                                         torch_name,
                                         ms_top_layer_name, torch_top_layer_name)

    # new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    new_ckpt = walk_dict(para_dict, mapped_params)
    for item in new_ckpt:
        print(f"para_name:{item['name']}, shape:{item['data'].shape}")
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
