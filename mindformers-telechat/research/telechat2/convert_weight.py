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
Convert Telechat weight.
Support huggingface format.
"""

import os
import argparse
from glob import glob
import torch

import mindspore as ms

from mindformers.tools import logger
from mindformers.utils.convert_utils import pt2ms

dtype_map = {
    'float32': ms.float32,
    'bfloat16': ms.bfloat16,
    'float16': ms.float16
}


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('transformer.word_embeddings.weight', 'model.tok_embeddings.embedding_weight')
    name = name.replace('.input_layernorm', '.attention_norm')
    name = name.replace('.self_attention.dense.', '.attention.wo.')
    name = name.replace('.self_attention.dense.bias.', '.attention.wo.bias.')
    name = name.replace('.self_attention.query.', '.attention.wq.')
    name = name.replace('.self_attention.key_value.', '.attention.wk_v.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.down_proj.bias.', '.feed_forward.w2.bias.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('lm_head.', 'lm_head.')
    name = name.replace('transformer.ln_f.', 'model.norm_out.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert telechat hf weight to ms."""
    files = list(glob(os.path.join(input_path, "pytorch_model*.bin")))
    files.sort()
    pt_states_list = []
    for per_file in files:
        pt_states = torch.load(per_file, map_location='cpu')
        pt_states_list.append(pt_states)

    ckpt_list = []
    for pt_states in pt_states_list:
        for name, value in pt_states.items():
            name = name_replace(name)
            if name.startswith('transformer.h.'):
                name = name.replace('transformer.h.', 'model.layers.')
            logger.info(f'\rprocessing parameter: {name} {value.shape}')
            ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    logger.info(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Telechat convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="",
                        help="The input torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--dtype", default='float32', choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")
    args = parser.parse_args()
    ms_dtype = dtype_map.get(args.dtype)

    # convert hf ckpt to ms
    convert_pt_to_ms(input_path=args.torch_path, output_path=args.mindspore_path, dtype=ms_dtype)
