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
import argparse
import torch
import mindspore as ms

from mindformers.utils.convert_utils import ms2pt


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('.ffn_norm.', '.post_attention_layernorm.')
    weight_name = weight_name.replace('.attention_norm.', '.input_layernorm.')
    weight_name = weight_name.replace('.feed_forward.w3.', '.mlp.up_proj.')
    weight_name = weight_name.replace('.feed_forward.w2.', '.mlp.down_proj.')
    weight_name = weight_name.replace('.feed_forward.w1.', '.mlp.gate_proj.')
    weight_name = weight_name.replace('.attention.wo.', '.self_attn.o_proj.')
    weight_name = weight_name.replace('.attention.wv.', '.self_attn.v_proj.')
    weight_name = weight_name.replace('.attention.wk.', '.self_attn.k_proj.')
    weight_name = weight_name.replace('.attention.wq.', '.self_attn.q_proj.')
    weight_name = weight_name.replace('output.', 'lm_head.')
    weight_name = weight_name.replace('tok_embeddings.', 'embed_tokens.')

    return weight_name


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """
    convert ms to pt
    """
    print(f"Trying to convert mindspore checkpoint in {input_path}.")
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        value = ms2pt(value, dtype)
        if name == 'model.norm_out.weight':
            name = 'model.norm.weight'
        if name == 'lm_head.weight':
            name = 'output.weight'
        if name == 'model.tok_embeddings.embedding_weight':
            name = 'model.tok_embeddings.weight'
        name = name_replace(name)
        state_dict[name] = value
        print(name, value.shape)

    torch.save(state_dict, output_path)
    print(f"Convert finished, the output is saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--torch_ckpt_path', default='./output.bin')

    args = parser.parse_args()
    convert_ms_to_pt(args.mindspore_ckpt_path, args.torch_ckpt_path)
