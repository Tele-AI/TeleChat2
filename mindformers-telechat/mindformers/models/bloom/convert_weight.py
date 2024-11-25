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
import mindspore
from mindspore import Parameter

from mindformers.utils.convert_utils import pt2ms


def layer_name_mapping(key):
    """Convert huggingface PP weights mapping in MindSpore.

    return: split, new_name
    """
    prefix = ''
    if 'transformer' in key:
        prefix = 'transformer.'
        key = key.replace('transformer.', '')
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "embedding.word_embedding.embedding_table",
        "word_embeddings_layernorm.weight": "embedding.norm.gamma",
        "word_embeddings_layernorm.bias": "embedding.norm.beta",
        "ln_f.weight": "ln_f.gamma",
        "ln_f.bias": "ln_f.beta",
        "input_layernorm.weight": "layernorm1.gamma",
        "input_layernorm.bias": "layernorm1.beta",
        "self_attention.query_key_value.weight": "attention.dense{}.weight",
        "self_attention.query_key_value.bias": "attention.dense{}.bias",
        "self_attention.dense.weight": "attention.projection.weight",
        "self_attention.dense.bias": "attention.projection.bias",
        "post_attention_layernorm.weight": "layernorm2.gamma",
        "post_attention_layernorm.bias": "layernorm2.beta",
        "mlp.dense_h_to_4h.weight": "output.mapping.weight",
        "mlp.dense_h_to_4h.bias": "output.mapping.bias",
        "mlp.dense_4h_to_h.weight": "output.projection.weight",
        "mlp.dense_4h_to_h.bias": "output.projection.bias",
        "lm_head.weight": "head.weight",
        "lm_head.bias": "head.bias",
    }

    split = False
    if key in layer_rename_map:
        return split, prefix + layer_rename_map[key]

    # Handle transformer blocks
    match = re.match(r'^\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    layer_number = int(match.group(1))
    text = match.group(2)
    if "self_attention.query_key_value" in key:
        split = True
    return split, f"{prefix}blocks.{layer_number}." + layer_rename_map.get(text)


def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """Convert hf layers to ms."""
    n_head = kwargs.pop('n_head', 32)
    hidden_size = kwargs.pop('hidden_size', 4096)
    hf_weights = torch.load(input_path, map_location='cpu')
    ms_params = {}
    for k, v in hf_weights.items():
        print(k, v.shape, v.dtype)
        split, new_name = layer_name_mapping(k)
        if split:
            if 'weight' in new_name:
                v = v.reshape(n_head, 3, hidden_size // n_head, v.shape[-1])
                v_list = v.tensor_split(3, dim=1)
                for i in range(1, 4):
                    tmp_name = new_name.format(i)
                    print(v_list[i - 1].shape)
                    tmp_tensor = pt2ms(v_list[i - 1].reshape(-1, v_list[i - 1].shape[-1]), dtype)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
            else:
                v = v.reshape(n_head, 3, hidden_size // n_head)
                v_list = v.tensor_split(3, dim=1)
                for i in range(1, 4):
                    tmp_name = new_name.format(i)
                    print(v_list[i - 1].shape)
                    tmp_tensor = pt2ms(v_list[i - 1].reshape(-1), dtype)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
        else:
            if ('projection' in new_name or 'mapping' in new_name) and 'weight' in new_name:
                new_tensor = pt2ms(v.transpose(0, 1), dtype)
            else:
                new_tensor = pt2ms(v, dtype)
            ms_params[new_name] = Parameter(new_tensor, name=new_name)

    params_list = [{'name': k, 'data': v} for k, v in ms_params.items()]
    mindspore.save_checkpoint(params_list, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bloom convert script")
    parser.add_argument('--n_head',
                        type=int,
                        default=32,
                        required=True,
                        help="The number of head of the model to be converted.")
    parser.add_argument('--hidden_size',
                        type=int,
                        default=4096,
                        required=True,
                        help="The number of hidden size of the model to be converted.")
    parser.add_argument("--torch_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The input torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default=None,
                        help="The output mindspore checkpoint path.")
    config = parser.parse_args()

    # convert hf ckpt to ms
    convert_pt_to_ms(input_path=config.torch_path, output_path=config.mindspore_path, n_head=config.n_head,
                     hidden_size=config.hidden_size)
