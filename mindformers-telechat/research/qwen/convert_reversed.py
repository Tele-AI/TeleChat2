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
"""
Convert Qwen weight.
Support mindspore format.
"""

import argparse
import collections
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt, is_lora_param


def _name_replace(name: str):
    """replace mindformers parameter name to huggingface."""
    name = name.replace('.layers.', '.h.')

    name = name.replace('.wte.embedding_weight', '.wte.weight')

    name = name.replace("attention.wo.", 'attn.c_proj.')

    name = name.replace('attention_norm.', 'ln_1.')
    name = name.replace('ffn_norm.', 'ln_2.')

    name = name.replace('feed_forward.w1.', 'mlp.w1.')
    name = name.replace('feed_forward.w3.', 'mlp.w2.')
    name = name.replace('feed_forward.w2.', 'mlp.c_proj.')
    return name


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert mindspore weights files to huggingface."""
    model_hf = ms.load_checkpoint(input_path)

    state_dict = {}
    attention_dict = collections.defaultdict(lambda: {})
    for name, value in model_hf.items():
        print('Parameter (name=%s, shape=%s, dtype=%s, requires_grad=%s)' % (
            name, value.data.shape, value.data.dtype, value.data.requires_grad))
        value = ms2pt(value, dtype)
        if is_lora_param(name):
            name = name.replace('mindpet_delta_lora_a', 'lora_A.weight')
            name = name.replace('mindpet_delta_lora_b', 'lora_B.weight')
        if 'attention.wq.bias' in name:
            name = name.replace('attention.wq', 'attn.c_attn')
            attention_dict[name]['wq'] = value
            continue
        if 'attention.wk.bias' in name:
            name = name.replace('attention.wk', 'attn.c_attn')
            attention_dict[name]['wk'] = value
            continue
        if 'attention.wv.bias' in name:
            name = name.replace('attention.wv', 'attn.c_attn')
            attention_dict[name]['wv'] = value
            continue

        hfname = _name_replace(name)
        if hfname != name:
            print('name:  %s->%s' % (name, hfname))
        state_dict[hfname] = value

    for name, value in attention_dict.items():
        hfname = _name_replace(name)
        if hfname != name:
            print('name:  %s->%s' % (name, hfname))
        state_dict[hfname] = torch.cat((value['wq'], value['wk'], value['wv']))

    print('Saving converted weights to %s...' % output_path)
    torch.save(state_dict, output_path)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen convert script")
    parser.add_argument("--mindspore_ckpt_path",
                        default="./run/qwen_7b_ms.ckpt",
                        help="The ms checkpoint path.")
    parser.add_argument("--torch_ckpt_path",
                        required=True,
                        help="The output checkpoint path.")

    args = parser.parse_args()
    convert_ms_to_pt(args.mindspore_ckpt_path, args.torch_ckpt_path)
