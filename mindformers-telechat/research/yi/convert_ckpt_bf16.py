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
transform huggingface model to mindspore ckpt.
"""

import argparse
import mindspore as ms
from transformers import LlamaForCausalLM
import torch


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('lm_head.', 'output.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
    weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
    weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    args = parser.parse_args()
    model_hf = LlamaForCausalLM.from_pretrained(args.torch_ckpt_dir)
    ckpt_list = []
    for name, value in model_hf.named_parameters():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'
        value = value.detach().to(torch.float32).numpy()
        print(name, value.shape)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=ms.bfloat16)})

    ms.save_checkpoint(ckpt_list, args.mindspore_ckpt_path)
