# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import torch
from mindspore import save_checkpoint, Tensor


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


def get_converted_ckpt(mapped_params, weight_dict):
    """
    Print the keys of the loaded checkpoint

    Args:
        mapped_params(dict): The loaded checkpoint. The key is parameter name and value is the numpy array.
        weight_dict(dict): The loaded pytorch checkpoint.

    Returns:
        None
    """
    new_ckpt_list = []
    # Currently, the ms_extend_param the torch_extend_param is the full parameters.
    for src, tgt in mapped_params:
        value = weight_dict[tgt].numpy()
        is_transpose = ""
        if '.o.' in tgt or '.wi.' in tgt or '.wo.' in tgt:
            value = np.transpose(value, [1, 0])
            is_transpose = " transposed"
        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}" + f"---{is_transpose}")
        new_ckpt_list.append({"data": Tensor(value), "name": src})
    return new_ckpt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument('--layers',
                        type=int,
                        default=1,
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
    state_dict = torch.load(opt.torch_path, map_location='cpu')

    ms_name = [
        "t5_model.tfm_encoder.blocks.{}.layernorm1.gamma",
        "t5_model.tfm_encoder.blocks.{}.layernorm2.gamma",
        "t5_model.tfm_encoder.blocks.{}.attention.dense1.weight",
        "t5_model.tfm_encoder.blocks.{}.attention.dense2.weight",
        "t5_model.tfm_encoder.blocks.{}.attention.dense3.weight",
        "t5_model.tfm_encoder.blocks.{}.attention.projection.weight",
        "t5_model.tfm_encoder.blocks.{}.output.mapping.weight",
        "t5_model.tfm_encoder.blocks.{}.output.projection.weight",

        "t5_model.tfm_decoder.blocks.{}.layernorm1.gamma",
        "t5_model.tfm_decoder.blocks.{}.cross_attention_layernorm.gamma",
        "t5_model.tfm_decoder.blocks.{}.layernorm2.gamma",
        "t5_model.tfm_decoder.blocks.{}.attention.dense1.weight",
        "t5_model.tfm_decoder.blocks.{}.attention.dense2.weight",
        "t5_model.tfm_decoder.blocks.{}.attention.dense3.weight",
        "t5_model.tfm_decoder.blocks.{}.attention.projection.weight",

        "t5_model.tfm_decoder.blocks.{}.cross_attention.dense1.weight",
        "t5_model.tfm_decoder.blocks.{}.cross_attention.dense2.weight",
        "t5_model.tfm_decoder.blocks.{}.cross_attention.dense3.weight",
        "t5_model.tfm_decoder.blocks.{}.cross_attention.projection.weight",
        "t5_model.tfm_decoder.blocks.{}.output.mapping.weight",
        "t5_model.tfm_decoder.blocks.{}.output.projection.weight",
    ]

    torch_name = [
        "encoder.block.{}.layer.0.layer_norm.weight",
        "encoder.block.{}.layer.1.layer_norm.weight",
        "encoder.block.{}.layer.0.SelfAttention.q.weight",
        "encoder.block.{}.layer.0.SelfAttention.k.weight",
        "encoder.block.{}.layer.0.SelfAttention.v.weight",
        "encoder.block.{}.layer.0.SelfAttention.o.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wi.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wo.weight",

        "decoder.block.{}.layer.0.layer_norm.weight",
        "decoder.block.{}.layer.1.layer_norm.weight",
        "decoder.block.{}.layer.2.layer_norm.weight",
        "decoder.block.{}.layer.0.SelfAttention.q.weight",
        "decoder.block.{}.layer.0.SelfAttention.k.weight",
        "decoder.block.{}.layer.0.SelfAttention.v.weight",
        "decoder.block.{}.layer.0.SelfAttention.o.weight",

        "decoder.block.{}.layer.1.EncDecAttention.q.weight",
        "decoder.block.{}.layer.1.EncDecAttention.k.weight",
        "decoder.block.{}.layer.1.EncDecAttention.v.weight",
        "decoder.block.{}.layer.1.EncDecAttention.o.weight",

        "decoder.block.{}.layer.2.DenseReluDense.wi.weight",
        "decoder.block.{}.layer.2.DenseReluDense.wo.weight",
    ]

    addition_mindspore = [
        "t5_model.encoder_layernorm.gamma",
        "t5_model.decoder_layernorm.gamma",
        "t5_model.tfm_embedding_lookup.embedding_table",
        "t5_model.tfm_encoder.blocks.0.attention.bias_generator.embeddings_table",
        "t5_model.tfm_decoder.blocks.0.attention.bias_generator.embeddings_table",
    ]

    addition_torch = [
        "encoder.final_layer_norm.weight",
        "decoder.final_layer_norm.weight",
        "shared.weight",
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    ]


    mapped_param = generate_params_dict(total_layers=opt.layers,
                                        mindspore_params_per_layer=ms_name,
                                        torch_params_per_layer=torch_name,
                                        mindspore_additional_params=addition_mindspore,
                                        torch_additional_params=addition_torch)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
