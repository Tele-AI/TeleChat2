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
import collections
import re
import argparse
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt

layer_rename_map = {
    'embedding.word_embedding.embedding_table': 'word_embeddings.weight',
    'embedding.norm.gamma': 'word_embeddings_layernorm.weight',
    'embedding.norm.beta': 'word_embeddings_layernorm.bias',
    'ln_f.gamma': 'ln_f.weight',
    'ln_f.beta': 'ln_f.bias',
    'layernorm1.gamma': 'input_layernorm.weight',
    'layernorm1.beta': 'input_layernorm.bias',
    'attention.projection.weight': 'self_attention.dense.weight',
    'attention.projection.bias': 'self_attention.dense.bias',
    'layernorm2.gamma': 'post_attention_layernorm.weight',
    'layernorm2.beta': 'post_attention_layernorm.bias',
    'output.mapping.weight': 'mlp.dense_h_to_4h.weight',
    'output.mapping.bias': 'mlp.dense_h_to_4h.bias',
    'output.projection.weight': 'mlp.dense_4h_to_h.weight',
    'output.projection.bias': 'mlp.dense_4h_to_h.bias',
    'head.weight': 'lm_head.weight',
    'head.bias': 'lm_head.bias'
}


def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """Convert ms layers to hf."""
    print(f"Trying to convert huggingface checkpoint in {input_path}.")
    n_head = kwargs.pop('n_head', 32)
    hidden_size = kwargs.pop('hidden_size', 4096)
    model_ms = ms.load_checkpoint(input_path)
    state_dict = {}

    attention_dict = collections.defaultdict(lambda: {})
    for name, value in model_ms.items():
        value = ms2pt(value, dtype)

        if name in layer_rename_map:
            if ('projection' in name or 'mapping' in name) and 'weight' in name:
                value = value.transpose(0, 1)

            new_name = layer_rename_map[name]
            state_dict[new_name] = value
            continue
        match = re.match(r'^\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', name)
        layer_number = int(match.group(1))
        text = match.group(2)

        attention_match = re.match(r'^\w*\.\d+\.attention\.dense([1-3])\.(weight|bias)$', name)
        if attention_match:
            dense_n = int(attention_match.group(1))
            new_name = name.replace(f'attention.dense{dense_n}', 'self_attention.query_key_value')
            new_name = new_name.replace('blocks', 'h')
            if 'weight' in new_name:
                attention_dict[new_name][dense_n] = value.reshape(n_head, 1, hidden_size // n_head, value.shape[-1])
            else:
                attention_dict[new_name][dense_n] = value.reshape(n_head, 1, hidden_size // n_head)
        else:
            if ('projection' in name or 'mapping' in name) and 'weight' in name:
                value = value.transpose(0, 1)
            name = f"h.{layer_number}." + layer_rename_map.get(text)
            state_dict[name] = value
    for name, value_dict in attention_dict.items():
        merge_value = torch.cat((value_dict[1], value_dict[2], value_dict[3]), dim=1)
        if 'weight' in name:
            merge_value = merge_value.reshape(-1, merge_value.shape[-1])
        else:
            merge_value = merge_value.reshape(-1)
        state_dict[name] = merge_value

    torch.save(state_dict, output_path)
    print(f"Convert finished, the output is saved to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bloom convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default=None,
                        help="The input mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The output torch checkpoint path.")
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

    args = parser.parse_args()
    convert_ms_to_pt(input_path=args.mindspore_path, output_path=args.torch_path, n_head=args.n_head,
                     hidden_size=args.hidden_size)
