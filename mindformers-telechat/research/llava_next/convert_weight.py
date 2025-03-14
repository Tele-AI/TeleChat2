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
Convert Llava-Next-Video weight.
Support huggingface format.
"""

import argparse
import importlib

import mindspore as ms
import torch
from transformers import AutoConfig

from mindformers.utils.convert_utils import pt2ms


def get_cls_from_transformers(model_dir, class_name):
    module_ = importlib.import_module("transformers")
    model_class = getattr(module_, class_name)
    return getattr(model_class, "from_pretrained")(model_dir)


def _llm_name_replace(name: str):
    """replace hf param name to ms."""
    return name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight') \
        .replace('.self_attn.q_proj.', '.attention.wq.') \
        .replace('.self_attn.k_proj.', '.attention.wk.') \
        .replace('.self_attn.v_proj.', '.attention.wv.') \
        .replace('.self_attn.o_proj.', '.attention.wo.') \
        .replace('.mlp.gate_proj.', '.feed_forward.w1.') \
        .replace('.mlp.down_proj.', '.feed_forward.w2.') \
        .replace('.mlp.up_proj.', '.feed_forward.w3.') \
        .replace('.input_layernorm.', '.attention_norm.') \
        .replace('.post_attention_layernorm.', '.ffn_norm.') \
        .replace('.norm.', '.norm_out.')


def collect_vit_transformer_attn(name, value, ckpt_weights, dtype=ms.float16, qkv_dict=dict):
    """convert attention name of vit transformer"""
    name = name.replace("self_attn", "attn")
    if "out_proj" in name:
        ms_name = name
        ckpt_weights.append({"name": name, "data": pt2ms(value, dtype=dtype)})
        print(f"name:  {name}->{ms_name}")
    else:
        qkv_dict[name] = value


def _vit_name_replace(name: str):
    """replace vit module name"""
    name = name.replace("embeddings.", "") \
        .replace("position_embedding.weight", "positional_embedding") \
        .replace("multi_modal_projector.linear_1", "adapter.adapter") \
        .replace("multi_modal_projector.linear_2", "adapter.adapter_2") \
        .replace("patch_embedding", "conv1")
    if "layernorm" in name or "layer_norm" in name or "layrnorm" in name:
        name = name.replace("weight", "gamma") \
            .replace("bias", "beta")
    return name.replace(".layer_norm1.", ".ln_1.") \
        .replace(".layer_norm2.", ".ln_2.") \
        .replace(".pre_layrnorm.", ".ln_pre.") \
        .replace(".post_layernorm.", ".ln_post.") \
        .replace(".fc1.", ".c_fc.") \
        .replace(".fc2.", ".c_proj.")


def convert_vit_qkv_concat(qkv_params_dict, ckpt_weights, dtype):
    """convert split qkv matrix into one matrix"""
    assmue_layer_num = len(qkv_params_dict) // 6
    pub_text = "vision_encoder.transformer.resblocks."
    for i in range(assmue_layer_num):
        wq_weight_name = f"{pub_text}{i}.attn.q_proj.weight"
        wk_weight_name = f"{pub_text}{i}.attn.k_proj.weight"
        wv_weight_name = f"{pub_text}{i}.attn.v_proj.weight"
        qkv_concat_weight_name = f"{pub_text}{i}.attn.in_proj.weight"

        wq_bias_name = f"{pub_text}{i}.attn.q_proj.bias"
        wk_bias_name = f"{pub_text}{i}.attn.k_proj.bias"
        wv_bias_name = f"{pub_text}{i}.attn.v_proj.bias"
        qkv_concat_bias_name = f"{pub_text}{i}.attn.in_proj.bias"

        wq_weight = qkv_params_dict[wq_weight_name]
        wk_weight = qkv_params_dict[wk_weight_name]
        wv_weight = qkv_params_dict[wv_weight_name]
        qkv_weight = torch.cat((wq_weight, wk_weight, wv_weight), 0)

        wq_bias = qkv_params_dict[wq_bias_name]
        wk_bias = qkv_params_dict[wk_bias_name]
        wv_bias = qkv_params_dict[wv_bias_name]
        qkv_bias = torch.cat((wq_bias, wk_bias, wv_bias), 0)

        ckpt_weights.append({"name": qkv_concat_weight_name, "data": pt2ms(qkv_weight, dtype=dtype)})
        ckpt_weights.append({"name": qkv_concat_bias_name, "data": pt2ms(qkv_bias, dtype=dtype)})
        print(f"convert {wq_weight_name} {wk_weight_name} {wv_weight_name} to {qkv_concat_weight_name}")
        print(f"convert {wq_bias_name} {wk_bias_name} {wv_bias_name} to {qkv_concat_bias_name}\n")
        print(f'\rprocessing parameter: {qkv_concat_weight_name} {qkv_weight.shape}\n', end='', flush=True)
        print(f'\rprocessing parameter: {qkv_concat_bias_name} {qkv_bias.shape}\n', end='', flush=True)


def convert_vit_weight(name, value, ckpt_weights, dtype, qkv_dict):
    """convert vit weights"""
    name = name.replace("vision_tower.vision_model.", "vision_encoder.")
    name = name.replace("encoder.layers", "transformer.resblocks")
    if "self_attn" in name:  # transformer attn in ViT
        collect_vit_transformer_attn(name, value, ckpt_weights, dtype, qkv_dict)
    else:
        ms_name = _vit_name_replace(name)
        if ms_name != name:
            print(f"name:  {name}->{ms_name}")
        ckpt_weights.append({"name": ms_name, "data": pt2ms(value, dtype=dtype)})


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """Convert huggingface weights files to mindspore ckpt format."""

    config = AutoConfig.from_pretrained(input_path)
    print(config.architectures[0])
    model = get_cls_from_transformers(input_path, config.architectures[0])

    from_language = kwargs.get("from_language", None)
    from_vision = kwargs.get("from_vision", None)
    if from_vision is not None:
        model = model.vision_model

    ckpt_weights = []

    qkv_dict = {}
    for name, param in model.named_parameters():
        print(f"Parameter (name={name}, shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad})")
        if from_language is not None:
            name = "language_model." + name
        elif from_vision is not None:
            name = "vision_encoder." + name
        if "language_model" not in name:
            convert_vit_weight(name, param, ckpt_weights, dtype, qkv_dict)
        else:
            ms_name = _llm_name_replace(name)
            if ms_name != name:
                print(f"name:  {name}->{ms_name}")
            print(f'\rprocessing parameter: {name} {param.shape}', end='', flush=True)
            ckpt_weights.append({"name": ms_name, "data": pt2ms(param, dtype=dtype)})
    convert_vit_qkv_concat(qkv_dict, ckpt_weights, dtype)
    print(f"Saving converted weights to {output_path}...")
    ms.save_checkpoint(ckpt_weights, output_path)
    print("Transform Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_path', default='')
    parser.add_argument('--mindspore_ckpt_path', default='llava_next.ckpt')
    args = parser.parse_args()
    convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path)
