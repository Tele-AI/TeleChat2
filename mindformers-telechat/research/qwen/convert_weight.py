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
Support huggingface format.
"""

import argparse
import os

import mindspore as ms
from transformers import AutoModelForCausalLM
from mindformers.utils.convert_utils import pt2ms

ATTENTION_WEIGHT_NAME = 'attn.c_attn.weight'
ATTENTION_BIAS_NAME = 'attn.c_attn.bias'


def _name_replace(name: str):
    """replace huggingface parameter name to mindformers."""
    name = name.replace('.h.', '.layers.')

    name = name.replace('.wte.weight', '.wte.embedding_weight')

    name = name.replace('attn.c_proj.', "attention.wo.")

    name = name.replace('ln_1.', 'attention_norm.')
    name = name.replace('ln_2.', 'ffn_norm.')

    name = name.replace('mlp.w1.', 'feed_forward.w1.')
    name = name.replace('mlp.w2.', 'feed_forward.w3.')
    name = name.replace('mlp.c_proj.', 'feed_forward.w2.')
    return name


def convert_attention_weight(name, value, ckpt_weights):
    split_value = ms.numpy.array_split(value, 3)
    attention_weight_names = ['attention.wq.weight', 'attention.wk.weight', 'attention.wv.weight']

    for index, _ in enumerate(split_value):
        cur_name = name.replace(ATTENTION_WEIGHT_NAME, attention_weight_names[index])
        ckpt_weights.append({'name': cur_name, 'data': ms.Tensor(split_value[index])})


def convert_attention_bias(name, value, ckpt_weights):
    split_value = ms.numpy.array_split(value, 3)
    attention_bias_names = ['attention.wq.bias', 'attention.wk.bias', 'attention.wv.bias']

    for index, _ in enumerate(split_value):
        cur_name = name.replace(ATTENTION_BIAS_NAME, attention_bias_names[index])
        ckpt_weights.append({'name': cur_name, 'data': ms.Tensor(split_value[index])})


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert huggingface weights files to mindspore."""
    ckpt_dir = os.path.dirname(input_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="cpu", trust_remote_code=True)

    ckpt_weights = []
    for k, v in model.state_dict().items():
        print('Parameter (name=%s, shape=%s, dtype=%s, requires_grad=%s)' % (k, v.shape, v.dtype, v.requires_grad))
        value = pt2ms(v, dtype)

        msname = _name_replace(k)
        if msname != k:
            print('name:  %s->%s' % (k, msname))

        if ATTENTION_WEIGHT_NAME in msname:
            convert_attention_weight(msname, value, ckpt_weights)
            continue

        if ATTENTION_BIAS_NAME in msname:
            convert_attention_bias(msname, value, ckpt_weights)
            continue

        ckpt_weights.append({'name': msname, 'data': value})

    print('Saving converted weights to %s...' % output_path)
    ms.save_checkpoint(ckpt_weights, output_path)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen convert script")
    parser.add_argument("--torch_ckpt_path",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        required=True,
                        help="The output checkpoint path.")
    parser.add_argument("--dtype", default=None, choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")

    _args = parser.parse_args()

    from mindformers.models.utils import convert_mstype
    _dtype = convert_mstype(_args.dtype)

    convert_pt_to_ms(_args.torch_ckpt_path, _args.mindspore_ckpt_path, dtype=_dtype)
