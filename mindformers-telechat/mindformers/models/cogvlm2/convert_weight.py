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
"""Convert cogvlm2 weight."""
import os
import argparse
from glob import glob
import torch
from safetensors import safe_open

import mindspore as ms
from mindformers.utils.convert_utils import pt2ms


def name_replace_visual(key, value):
    """replace variable name in visual model."""
    if key.endswith('cls_embedding'):
        value = torch.unsqueeze(value, 0)
        return 'vision_encoder.cls_token', value
    if key.endswith('position_embedding.weight'):
        value = torch.unsqueeze(value, 0)
        return 'vision_encoder.pos_embed', value
    if key.startswith('model.vision.patch_embedding.proj'):
        name = key.replace('model.vision.patch_embedding', 'vision_encoder.patch_embed')
        return name, value

    name = key.replace('model.vision.transformer.layers', 'vision_encoder.blocks')
    name = name.replace('attention.dense', 'attn.proj')
    name = name.replace('attention.query_key_value', 'attn.qkv')
    name = name.replace('input_layernorm.weight', 'norm1.gamma')
    name = name.replace('input_layernorm.bias', 'norm1.beta')
    name = name.replace('post_attention_layernorm.weight', 'norm2.gamma')
    name = name.replace('post_attention_layernorm.bias', 'norm2.beta')
    return name, value


def name_replace_adapter(key, value):
    """replace variable name in adapter model."""
    name = key.replace('model.vision', 'mlp_adapter')
    name = name.replace('linear_proj.norm1.weight', 'linear_proj.norm1.gamma')
    name = name.replace('linear_proj.norm1.bias', 'linear_proj.norm1.beta')
    return name, value


def name_replace_llama3(key, value, lora_name=''):
    """replace variable name in llama3 model."""
    if key == 'model.norm.weight':
        return f'llm_model.{lora_name}model.norm_out.weight', value
    if key == 'lm_head.weight':
        return f'llm_model.{lora_name}lm_head.weight', value

    if key.startswith('model.embed_tokens'):
        return f'llm_model.{lora_name}model.tok_embeddings.embedding_weight', value

    if key.endswith('self_attn.language_expert_query_key_value.weight'):
        cur_key = f"llm_model.{lora_name}{key}"
        q_value = value[:4096, :]
        k_value = value[4096:5120, :]
        v_value = value[5120:, :]
        name = [
            cur_key.replace('self_attn.language_expert_query_key_value', 'attention.wq'),
            cur_key.replace('self_attn.language_expert_query_key_value', 'attention.wk'),
            cur_key.replace('self_attn.language_expert_query_key_value', 'attention.wv'),
        ]
        return name, [q_value, k_value, v_value]

    name = f"llm_model.{lora_name}{key}"
    name = name.replace('input_layernorm', 'attention_norm')
    name = name.replace('self_attn.language_expert_dense', 'attention.wo')
    name = name.replace('mlp.language_mlp.gate_proj', 'feed_forward.w1')
    name = name.replace('mlp.language_mlp.up_proj', 'feed_forward.w3')
    name = name.replace('mlp.language_mlp.down_proj', 'feed_forward.w2')
    name = name.replace('post_attention_layernorm', 'ffn_norm')
    name = name.replace('model.norm.weight', 'model.norm_out.weight')
    return name, value


def name_replace_llama3_for_image(key, value):
    """replace variable name in llama3 model."""
    if key.startswith('model.embed_tokens'):
        return 'llm_model.model.tok_embeddings.embedding_weight', value

    name = f"llm_model.{key}"
    name = name.replace('model.norm.weight', 'model.norm_out.weight')
    name = name.replace('mlp.language_mlp.gate_proj', 'mlp.language_mlp.w1')
    name = name.replace('mlp.language_mlp.up_proj', 'mlp.language_mlp.w3')
    name = name.replace('mlp.language_mlp.down_proj', 'mlp.language_mlp.w2')
    name = name.replace('mlp.vision_mlp.gate_proj', 'mlp.vision_mlp.w1')
    name = name.replace('mlp.vision_mlp.up_proj', 'mlp.vision_mlp.w3')
    name = name.replace('mlp.vision_mlp.down_proj', 'mlp.vision_mlp.w2')
    return name, value


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=ms.float32, modal="video", **kwargs):
    """convert hf weight to ms."""
    ckpts = glob(os.path.join(input_path, '*.safetensors'))
    ckpts.sort()
    model_params = dict()
    for ckpt in ckpts:
        with safe_open(ckpt, framework='pt', device='cpu') as f:
            for k in f.keys():
                model_params[k] = f.get_tensor(k)

    if kwargs.get('sft') == 'lora':
        lora_name = 'pet_model.lora_model.'
    else:
        lora_name = ''
    checkpoints = []
    saved_freq = False
    for k, v in model_params.items():
        if k.endswith('self_attn.rotary_emb.inv_freq'):
            if not saved_freq:
                k = f'llm_model.{lora_name}model.freqs_mgr.freqs'
                saved_freq = True
            else:
                continue

        elif k.startswith('model.vision.patch_embedding') or k.startswith('model.vision.transformer'):
            k, v = name_replace_visual(k, v)
        elif k.startswith('model.vision'):
            k, v = name_replace_adapter(k, v)
        else:
            if modal == "video":
                k, v = name_replace_llama3(k, v, lora_name=lora_name)
            else:
                k, v = name_replace_llama3_for_image(k, v)

        if isinstance(k, list) and isinstance(v, list):
            for idx, _ in enumerate(k):
                print(f'\rprocessing parameter: {k[idx]} {v[idx].shape}')
                checkpoints.append({'name': k[idx], 'data': pt2ms(v[idx], dtype)})
        else:
            print(f'\rprocessing parameter: {k} {v.shape}')
            checkpoints.append({'name': k, 'data': pt2ms(v, dtype)})

    ms.save_checkpoint(checkpoints, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVlm2 convert script")
    parser.add_argument('--modal', default="video", choices=["video", "image"])
    parser.add_argument('--torch_ckpt_dir', default='./cogvlm2-video-llama3-chat/')
    parser.add_argument('--mindspore_ckpt_path', default='./cogvlm2-video-llama3-chat.ckpt')
    parser.add_argument('--dtype', default='float32', type=str, choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--sft', default='full', type=str, choices=['full', 'lora'])
    args = parser.parse_args()

    dtype_map = {'float16': ms.float16, 'float32': ms.float32, 'bfloat16': ms.bfloat16}

    convert_pt_to_ms(
        modal=args.modal,
        input_path=args.torch_ckpt_dir, output_path=args.mindspore_ckpt_path,
        dtype=dtype_map.get(args.dtype, None),
        sft=args.sft
    )
