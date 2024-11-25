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
Convert QwenVL weight.
Support huggingface format.
"""

import argparse
import os

import mindspore as ms
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from mindformers.tools import str2bool

QWEN_ATTENTION_WEIGHT_NAME = "attn.c_attn.weight"
QWEN_ATTENTION_BIAS_NAME = "attn.c_attn.bias"
QWEN_EMBEDDING_WEIGHT_NAME = "wte.embedding_weight"
QWEN_LM_HEAD_WEIGHT_NAME = "lm_head.weight"


def _qwen_name_replace(name: str):
    """replace huggingface parameter name to mindformers."""
    name = name.replace(".h.", ".layers.")

    name = name.replace(".wte.weight", ".wte.embedding_weight")

    name = name.replace("attn.c_proj.", "attention.wo.")

    name = name.replace("ln_1.", "attention_norm.")
    name = name.replace("ln_2.", "ffn_norm.")

    name = name.replace("mlp.w1.", "feed_forward.w1.")
    name = name.replace("mlp.w2.", "feed_forward.w3.")
    name = name.replace("mlp.c_proj.", "feed_forward.w2.")
    name = "llm_model." + name
    return name


def convert_qwen_attention_weight(name, value, ckpt_weights, dtype=ms.float16, use_qkv_concat=False):
    """convert attention weight of qwen"""
    if not use_qkv_concat:
        split_arr = np.array_split(value, 3)
        attention_weight_names = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight"]

        for index, split_arr_item in enumerate(split_arr):
            cur_name = name.replace(QWEN_ATTENTION_WEIGHT_NAME, attention_weight_names[index])
            ckpt_weights.append({"name": cur_name, "data": ms.Tensor(split_arr_item, dtype=dtype)})
    else:
        cur_name = name.replace(QWEN_ATTENTION_WEIGHT_NAME, "attention.w.weight")
        ckpt_weights.append({"name": cur_name, "data": ms.Tensor(value, dtype=dtype)})


def convert_qwen_attention_bias(name, value, ckpt_weights, dtype=ms.float16, use_qkv_concat=False):
    """convert attention bias of qwen"""
    if not use_qkv_concat:
        split_arr = np.array_split(value, 3)
        attention_bias_names = ["attention.wq.bias", "attention.wk.bias", "attention.wv.bias"]

        for index, split_arr_item in enumerate(split_arr):
            cur_name = name.replace(QWEN_ATTENTION_BIAS_NAME, attention_bias_names[index])
            ckpt_weights.append({"name": cur_name, "data": ms.Tensor(split_arr_item, dtype=dtype)})
    else:
        cur_name = name.replace(QWEN_ATTENTION_BIAS_NAME, "attention.w.bias")
        ckpt_weights.append({"name": cur_name, "data": ms.Tensor(value, dtype=dtype)})


def convert_vit_resampler_attention(name, value, ckpt_weights, dtype=ms.float16):
    """convert attention bias of vit resampler"""
    if "ln" in name:
        ms_name = name.replace("weight", "gamma")
        ms_name = ms_name.replace("bias", "beta")
    elif "in_proj_weight" in name:
        value = np.array_split(value, 3)
        ms_name = [name.replace("in_proj_weight", f"dense{i}.weight") for i in (1, 2, 3)]
    elif "in_proj_bias" in name:
        value = np.array_split(value, 3)
        ms_name = [name.replace("in_proj_bias", f"dense{i}.bias") for i in (1, 2, 3)]
    elif "out_proj" in name:
        if "weight" in name:
            value = np.transpose(value, (1, 0))
        ms_name = name.replace("out_proj", "projection")
    else:
        ms_name = name
    if not isinstance(ms_name, (tuple, list)):
        cur_ms_name = (ms_name,)
    else:
        cur_ms_name = ms_name
    if not isinstance(value, (tuple, list)):
        cur_value = (value,)
    else:
        cur_value = value
    for n, p in zip(cur_ms_name, cur_value):
        if n != name:
            print(f"name:  {name}->{n}")
        ckpt_weights.append({"name": n, "data": ms.Tensor(p, dtype=dtype)})


def convert_vit_transformer_attn(name, value, ckpt_weights, dtype=ms.float16, vit_num_head=16):
    """convert attention bias of vit transformer"""
    if "in_proj" in name:
        if value.shape[0] % (3 * vit_num_head) != 0:
            raise ValueError(f"The 3 * vit_num_head({3 * vit_num_head}) must be divisible "
                             f"by value.shape[0]({value.shape[0]}).")
        value = np.array_split(value, 3 * vit_num_head)
        if "weight" in name:
            value = [np.vstack(value[i::3]) for i in range(3)]
        else:  # bias
            value = [np.concatenate(value[i::3]) for i in range(3)]
        ms_name = [name.replace("in_proj", f"dense{i}") for i in (1, 2, 3)]
    elif "out_proj" in name:
        if "weight" in name:
            value = np.transpose(value, (1, 0))
        ms_name = name.replace("out_proj", "projection")
    else:
        ms_name = name
    if not isinstance(ms_name, (tuple, list)):
        cur_ms_name = (ms_name,)
    else:
        cur_ms_name = ms_name
    if not isinstance(value, (tuple, list)):
        cur_value = (value,)
    else:
        cur_value = value
    for n, p in zip(cur_ms_name, cur_value):
        if n != name:
            print(f"name:  {name}->{n}")
        ckpt_weights.append({"name": n, "data": ms.Tensor(p, dtype=dtype)})


def _vit_name_replace(name: str):
    """replace vit module name"""
    if "ln" in name:  # layer norm
        name = name.replace("weight", "gamma")
        name = name.replace("bias", "beta")
    elif "in_proj_weight" in name:
        name = name.replace("in_proj_weight", "in_proj.weight")
    elif "in_proj_bias" in name:
        name = name.replace("in_proj_bias", "in_proj.bias")
    elif "token_embedding.weight" in name:
        name = name.replace("token_embedding.weight", "token_embedding.embedding_table")
    return name


def convert_vit_weight(name, value, ckpt_weights, dtype, vit_num_head):
    """convert vit weights"""
    name = name.replace("transformer.visual.", "")
    name = "vision_encoder." + name
    if "attn_pool" in name:
        convert_vit_resampler_attention(name, value, ckpt_weights, dtype)
    elif "attn" in name:  # transformer in ViT
        convert_vit_transformer_attn(name, value, ckpt_weights, dtype, vit_num_head)
    else:
        ms_name = _vit_name_replace(name)
        if ms_name != name:
            print(f"name:  {name}->{ms_name}")
        ckpt_weights.append({"name": ms_name, "data": ms.Tensor(value, dtype=dtype)})


def convert_pt_to_ms(input_path, output_path, torch_dtype=torch.float16, dtype=ms.float16, **kwargs):
    """Convert huggingface weights files to mindspore ckpt format."""
    vit_num_head = kwargs.get("vit_num_head", 16)
    enable_emb_opt = kwargs.get("enable_emb_opt", False)
    emb_strategy = kwargs.get("emb_strategy", None)
    use_qkv_concat = kwargs.get("use_qkv_concat", False)

    model = AutoModelForCausalLM.from_pretrained(input_path, device_map="cpu", trust_remote_code=True)

    ckpt_weights = []
    for name, param in model.named_parameters():
        print(f"Parameter (name={name}, shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad})")
        try:
            value = param.detach().numpy()
        except TypeError:
            value = param.detach().to(torch_dtype).cpu().numpy()
            print(f"dtype:  {param.dtype}->{value.dtype}")

        if "visual" in name:
            convert_vit_weight(name, value, ckpt_weights, dtype, vit_num_head)
        else:
            ms_name = _qwen_name_replace(name)
            if ms_name != name:
                print(f"name:  {name}->{ms_name}")
            if enable_emb_opt:
                if emb_strategy is None:
                    raise ValueError("num_cards should be set when enable_emb_opt is True")
                if QWEN_EMBEDDING_WEIGHT_NAME in ms_name or QWEN_LM_HEAD_WEIGHT_NAME in ms_name:
                    vocab_size = value.shape[0]
                    mult, remainder = divmod(vocab_size, 512 * emb_strategy)
                    if remainder > 0:
                        mult += 1
                    new_vocab_size = mult * 512 * emb_strategy
                    pad = np.zeros((new_vocab_size - vocab_size, value.shape[1]), dtype=value.dtype)
                    value = np.concatenate([value, pad], axis=0)
                    print(f"{ms_name} shape from {vocab_size} to {new_vocab_size}")
                    ckpt_weights.append({"name": ms_name, "data": ms.Tensor(value, dtype=dtype)})
                    continue

            if QWEN_ATTENTION_WEIGHT_NAME in ms_name:
                convert_qwen_attention_weight(ms_name, value, ckpt_weights, dtype, use_qkv_concat)
                continue

            if QWEN_ATTENTION_BIAS_NAME in ms_name:
                convert_qwen_attention_bias(ms_name, value, ckpt_weights, dtype, use_qkv_concat)
                continue
            ckpt_weights.append({"name": ms_name, "data": ms.Tensor(value, dtype=dtype)})

    print(f"Saving converted weights to {output_path}...")
    ms.save_checkpoint(ckpt_weights, output_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen convert script")
    parser.add_argument("--torch_ckpt_dir",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        default="./run/qwen_7b_ms.ckpt",
                        help="The output checkpoint path.")
    parser.add_argument("--dtype",
                        default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="The data type of the converted weight.")
    parser.add_argument("--enable_emb_opt",
                        default=None,
                        type=str2bool,
                        help="Enable embedding optimization.")
    parser.add_argument("--emb_strategy",
                        default=None,
                        type=int,
                        help="parallel strategy for embedding")
    parser.add_argument("--vit_num_head",
                        default=16,
                        type=int,
                        help="The number of head in ViT.")
    parser.add_argument("--use_qkv_concat",
                        default=None,
                        type=str2bool,
                        help="Whether to use qkv concat in attention weight.")

    args = parser.parse_args()

    mindspore_ckpt_path = os.path.expanduser(args.mindspore_ckpt_path)
    output_dir = os.path.dirname(mindspore_ckpt_path)
    if output_dir and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mapping = {
        "float16": ms.float16,
        "float32": ms.float32,
        "bfloat16": ms.bfloat16
    }

    convert_pt_to_ms(input_path=args.torch_ckpt_dir, output_path=mindspore_ckpt_path, torch_dtype=torch.float32,
                     dtype=mapping.get(args.dtype), vit_num_head=args.vit_num_head, enable_emb_opt=args.enable_emb_opt,
                     emb_strategy=args.emb_strategy, use_qkv_concat=args.use_qkv_concat)
