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
Support mindformers format.
"""

import argparse
import torch

import mindspore as ms

from mindformers.tools import logger
from mindformers.utils.convert_utils import ms2pt

dtype_map = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}


def name_replace(name: str):
    """replace ms param name to hf."""
    name = name.replace("model.tok_embeddings.embedding_weight", "transformer.word_embeddings.weight")
    name = name.replace("attention_norm.weight", "input_layernorm.weight")
    name = name.replace("attention.wo.weight", "self_attention.dense.weight")
    name = name.replace("attention.wo.bias", "self_attention.dense.bias")
    name = name.replace("attention.wq.weight", "self_attention.query.weight")
    name = name.replace("attention.wk_v.weight", "self_attention.key_value.weight")
    name = name.replace("feed_forward.w1.weight", "mlp.gate_proj.weight")
    name = name.replace("feed_forward.w2.weight", "mlp.down_proj.weight")
    name = name.replace("feed_forward.w2.bias", "mlp.down_proj.bias")
    name = name.replace("feed_forward.w3.weight", "mlp.up_proj.weight")
    name = name.replace("ffn_norm.weight", "post_attention_layernorm.weight")
    name = name.replace("model.norm_out.weight", "transformer.ln_f.weight")
    name = name.replace("lm_head.weight", "lm_head.weight")
    return name


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert telechat ms weight to hf."""
    logger.info(f"Trying to convert mindspore checkpoint in '{input_path}'.")
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        value = ms2pt(value, dtype)
        name = name_replace(name)
        if name.startswith("model.layers."):
            name = name.replace("model.layers.", "transformer.h.")

        state_dict[name] = value
        logger.info(f'\rprocessing parameter: {name} {value.shape}')

    torch.save(state_dict, output_path)
    logger.info(f"\rConvert telechat checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_path', default='transform.ckpt')
    parser.add_argument('--torch_path', default='torch.bin')
    parser.add_argument("--dtype", default='float32', choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")
    args = parser.parse_args()
    torch_dtype = dtype_map.get(args.dtype)

    convert_ms_to_pt(input_path=args.mindspore_path, output_path=args.torch_path, dtype=torch_dtype)
