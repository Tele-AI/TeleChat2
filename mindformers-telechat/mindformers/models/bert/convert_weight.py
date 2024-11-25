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
        if tgt not in weight_dict:
            if "LayerNorm.gamma" in tgt:
                tgt = tgt.replace("gamma", "weight")
            if "LayerNorm.beta" in tgt:
                tgt = tgt.replace("beta", "bias")
        value = weight_dict[tgt].numpy()
        if 'output.dense.weight' in tgt or 'intermediate.dense.weight' in tgt:
            value = np.transpose(value, [1, 0])

        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")

        new_ckpt_list.append({"data": Tensor(value), "name": src})
    return new_ckpt_list


def split_torch_attention(state):
    s = list(state.keys())
    for name in s:
        if name.endswith('attn.c_attn.weight') or name.endswith('attn.c_attn.bias'):
            value = state.pop(name)
            q, k, v = np.split(value.numpy(), 3, -1)
            state[name + '.q'] = torch.tensor(q, dtype=value.dtype)
            state[name + '.k'] = torch.tensor(k)
            state[name + '.v'] = torch.tensor(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BERT convert script")
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
        "bert.bert_encoder.encoder.blocks.{}.attention.dense1.weight",
        "bert.bert_encoder.encoder.blocks.{}.attention.dense1.bias",
        "bert.bert_encoder.encoder.blocks.{}.attention.dense2.weight",
        "bert.bert_encoder.encoder.blocks.{}.attention.dense2.bias",
        "bert.bert_encoder.encoder.blocks.{}.attention.dense3.weight",
        "bert.bert_encoder.encoder.blocks.{}.attention.dense3.bias",
        "bert.bert_encoder.encoder.blocks.{}.attention.projection.weight",
        "bert.bert_encoder.encoder.blocks.{}.attention.projection.bias",
        "bert.bert_encoder.encoder.blocks.{}.layernorm2.gamma",
        "bert.bert_encoder.encoder.blocks.{}.layernorm2.beta",
        "bert.bert_encoder.encoder.blocks.{}.output.mapping.weight",
        "bert.bert_encoder.encoder.blocks.{}.output.mapping.bias",
        "bert.bert_encoder.encoder.blocks.{}.output.projection.weight",
        "bert.bert_encoder.encoder.blocks.{}.output.projection.bias",
        "bert.bert_encoder.encoder.blocks.{}.layernorm1.gamma",
        "bert.bert_encoder.encoder.blocks.{}.layernorm1.beta",
    ]

    torch_name = [
        "bert.encoder.layer.{}.attention.self.query.weight",
        "bert.encoder.layer.{}.attention.self.query.bias",
        "bert.encoder.layer.{}.attention.self.key.weight",
        "bert.encoder.layer.{}.attention.self.key.bias",
        "bert.encoder.layer.{}.attention.self.value.weight",
        "bert.encoder.layer.{}.attention.self.value.bias",
        "bert.encoder.layer.{}.attention.output.dense.weight",
        "bert.encoder.layer.{}.attention.output.dense.bias",
        "bert.encoder.layer.{}.attention.output.LayerNorm.gamma",
        "bert.encoder.layer.{}.attention.output.LayerNorm.beta",
        "bert.encoder.layer.{}.intermediate.dense.weight",
        "bert.encoder.layer.{}.intermediate.dense.bias",
        "bert.encoder.layer.{}.output.dense.weight",
        "bert.encoder.layer.{}.output.dense.bias",
        "bert.encoder.layer.{}.output.LayerNorm.gamma",
        "bert.encoder.layer.{}.output.LayerNorm.beta",
    ]

    addition_mindspore = [
        "bert.word_embedding.embedding_table",
        "bert.embedding_postprocessor.full_position_embedding.embedding_table",
        "bert.embedding_postprocessor.token_type_embedding.embedding_table",
        "bert.embedding_postprocessor.layernorm.gamma",
        "bert.embedding_postprocessor.layernorm.beta",
        "bert.dense.weight",
        "bert.dense.bias",
        "bert.mlmloss.dense.weight",
        "bert.mlmloss.dense.bias",
        "bert.mlmloss.layernorm.gamma",
        "bert.mlmloss.layernorm.beta",
        "bert.mlmloss.vocab_dense.weight",
    ]

    addition_torch = [
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.gamma",
        "bert.embeddings.LayerNorm.beta",
        "bert.pooler.dense.weight",
        "bert.pooler.dense.bias",
        "cls.predictions.transform.dense.weight",
        "cls.predictions.transform.dense.bias",
        "cls.predictions.transform.LayerNorm.gamma",
        "cls.predictions.transform.LayerNorm.beta",
        "cls.predictions.decoder.weight"

    ]

    mapped_param = generate_params_dict(total_layers=opt.layers,
                                        mindspore_params_per_layer=ms_name,
                                        torch_params_per_layer=torch_name,
                                        mindspore_additional_params=addition_mindspore,
                                        torch_additional_params=addition_torch)
    split_torch_attention(state_dict)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
