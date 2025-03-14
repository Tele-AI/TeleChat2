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
import re
from collections import defaultdict
from typing import Dict

import mindspore as ms
import numpy as np
import torch
from tqdm import tqdm

from mindformers import MindFormerConfig
from mindformers.models.glm2.glm2_config import ChatGLM2Config
from mindformers.utils.convert_utils import ms2pt, is_lora_param

digit_pattern = re.compile(r"([0-9]+)")
qkv_pattern = re.compile(r"(wq|wk|wv)")
wb_pattern = re.compile(r"(weight|bias)")
mlp_pattern = re.compile(r"(dense_left|dense_right)")


def extract_attn_info(name: str):
    """extract attn info"""
    layer = digit_pattern.findall(name)[0]
    qkv = qkv_pattern.findall(name)
    qkv = qkv[0] if qkv else None
    wb = wb_pattern.findall(name)[0]
    return layer, qkv, wb


def extract_mlp_info(name: str):
    """extract mlp info"""
    layer = digit_pattern.findall(name)[0]
    mlp = mlp_pattern.findall(name)[0]
    wb = wb_pattern.findall(name)[0]
    return layer, mlp, wb


def rearrange_w(w: np.ndarray, head_dim: int):
    """rearrange w"""
    w = w.T
    hidden_size, projection_size = w.shape
    w = w.reshape(hidden_size, projection_size // head_dim, head_dim)
    index = np.arange(head_dim) // 2
    index[1::2] += head_dim // 2
    w = w[..., index]
    w = w.reshape(hidden_size, projection_size)
    return w.T


def rearrange_b(b: np.ndarray, head_dim: int):
    """rearrange b"""
    projection_size = b.shape[0]
    b = b.reshape(projection_size // head_dim, head_dim)
    index = np.arange(head_dim) // 2
    index[1::2] += head_dim // 2
    b = b[..., index]
    b = b.reshape(-1,)
    return b


def pt2npy(param: torch.Tensor):
    """pt2npy"""
    return param.to(torch.float32).cpu().numpy()


def npy2pt(array: np.ndarray, dtype):
    """npy2pt"""
    return torch.Tensor(array).to(dtype)


def attn_rearange(pt_param: Dict, config: ChatGLM2Config):
    """attention rearrange"""
    is_mqa = config.multi_query_attention  # Multi query attention
    kv_channels = config.kv_channels
    num_attention_heads = config.num_attention_heads
    projection_size = kv_channels * num_attention_heads
    n_kv_head = config.multi_query_group_num
    head_dim = kv_channels
    q_hidden_size = projection_size
    kv_hidden_size = head_dim * n_kv_head if is_mqa else projection_size

    qkv_weights = defaultdict(dict)
    qkv_bias = defaultdict(dict)

    for name, param in pt_param.items():
        if "query_key_value" not in name:
            continue
        layer, _, wb = extract_attn_info(name)
        if wb == "weight":
            qkv_weights[layer]['name'] = name
            qkv_weights[layer]['dtype'] = param.dtype
            qkv_weights[layer]['param'] = pt2npy(param)
        else:
            qkv_bias[layer]['name'] = name
            qkv_bias[layer]['dtype'] = param.dtype
            qkv_bias[layer]['param'] = pt2npy(param)

    for layer, params in qkv_weights.items():
        name = params['name']
        wq, wk, wv = np.split(params['param'], [q_hidden_size, q_hidden_size + kv_hidden_size], axis=0)
        wq = rearrange_w(wq, head_dim)
        wk = rearrange_w(wk, head_dim)
        concat_qkv = np.concatenate(
            [wq, wk, wv],
            axis=0
        )
        pt_param[name] = npy2pt(concat_qkv, dtype=params['dtype'])

    for layer, params in qkv_bias.items():
        name = params['name']
        bq, bk, bv = np.split(params['param'], [q_hidden_size, q_hidden_size + kv_hidden_size], axis=0)
        bq = rearrange_b(bq, head_dim)
        bk = rearrange_b(bk, head_dim)
        concat_qkv = np.concatenate(
            [bq, bk, bv],
            axis=0
        )
        pt_param[name] = npy2pt(concat_qkv, dtype=params['dtype'])


def attn_merge(pt_param: Dict, config: ChatGLM2Config):
    """attention merge"""
    kv_channels = config.kv_channels
    head_dim = kv_channels

    # {'0': {'wq': {'param': param, 'name': ..., 'dtype': ...}}}
    qkv_weights = defaultdict(lambda: defaultdict(dict))
    qkv_bias = defaultdict(lambda: defaultdict(dict))

    for name, param in pt_param.items():
        if "wq" not in name and "wk" not in name and "wv" not in name:
            continue
        layer, qkv, wb = extract_attn_info(name)
        if wb == "weight":
            qkv_weights[layer][qkv]['name'] = name
            qkv_weights[layer][qkv]['dtype'] = param.dtype
            qkv_weights[layer][qkv]['param'] = pt2npy(param)
        else:
            qkv_bias[layer][qkv]['name'] = name
            qkv_bias[layer][qkv]['dtype'] = param.dtype
            qkv_bias[layer][qkv]['param'] = pt2npy(param)

    for layer, params in qkv_weights.items():
        params["wq"]['param'] = rearrange_w(params["wq"]['param'], head_dim)
        params["wk"]['param'] = rearrange_w(params["wk"]['param'], head_dim)
        merged_name = params['wq']['name'].replace("wq", "query_key_value")
        merged_param = np.concatenate(
            [
                params["wq"]['param'],
                params["wk"]['param'],
                params["wv"]['param']
            ],
            axis=0
        )
        for qkv in ('wq', 'wk', 'wv'):
            pt_param.pop(params[qkv]['name'])
        pt_param[merged_name] = npy2pt(merged_param, dtype=params["wq"]['dtype'])

    for layer, params in qkv_bias.items():
        params["wq"]['param'] = rearrange_b(params["wq"]['param'], head_dim)
        params["wk"]['param'] = rearrange_b(params["wk"]['param'], head_dim)
        merged_name = params['wq']['name'].replace("wq", "query_key_value")
        merged_param = np.concatenate(
            [
                params["wq"]['param'],
                params["wk"]['param'],
                params["wv"]['param']
            ],
            axis=0
        )
        for qkv in ('wq', 'wk', 'wv'):
            pt_param.pop(params[qkv]['name'])
        pt_param[merged_name] = npy2pt(merged_param, dtype=params["wq"]['dtype'])


def mlp_merge(pt_param: Dict):
    """mlp merge"""
    dense_weights = defaultdict(lambda: defaultdict(dict))
    dense_bias = defaultdict(lambda: defaultdict(dict))
    for name, param in pt_param.items():
        if "dense_left" not in name and "dense_right" not in name:
            continue
        layer, mlp, wb = extract_mlp_info(name)
        if wb == "weight":
            dense_weights[layer][mlp]['name'] = name
            dense_weights[layer][mlp]['dtype'] = param.dtype
            dense_weights[layer][mlp]['param'] = pt2npy(param)
        else:
            dense_bias[layer][mlp]['name'] = name
            dense_bias[layer][mlp]['dtype'] = param.dtype
            dense_bias[layer][mlp]['param'] = pt2npy(param)

    for layer, params in dense_weights.items():
        merged_name = params['dense_left']['name'].replace("dense_left", "dense_h_to_4h")
        merged_param = np.concatenate(
            [
                params["dense_left"]['param'],
                params["dense_right"]['param'],
            ],
            axis=0
        )
        for qkv in ('dense_left', 'dense_right'):
            pt_param.pop(params[qkv]['name'])
        pt_param[merged_name] = npy2pt(merged_param, dtype=params["dense_left"]['dtype'])

    for layer, params in dense_bias.items():
        merged_name = params['dense_left']['name'].replace("dense_left", "dense_h_to_4h")
        merged_param = np.concatenate(
            [
                params["dense_left"]['param'],
                params["dense_right"]['param'],
            ],
            axis=0
        )
        for qkv in ('dense_left', 'dense_right'):
            pt_param.pop(params[qkv]['name'])
        pt_param[merged_name] = npy2pt(merged_param, dtype=params["dense_left"]['dtype'])


# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, config, dtype=torch.float32, **kwargs):
    """ Convert MindSpore model file to pytorch model file. """
    ckpt_dict = ms.load_checkpoint(input_path)
    print('parameter convert....')
    pt_param = {}
    for k, v in tqdm(ckpt_dict.items()):
        v = ms2pt(v, dtype)
        if "embedding_weight" in k:
            k = k.replace("embedding_weight", "word_embeddings.weight")
        if is_lora_param(k):
            k = k.replace(".tk_delta_lora_a", ".lora_A.weight")
            k = k.replace(".tk_delta_lora_b", ".lora_B.weight")
        pt_param[k] = v

    # Convert pytorch model file to MindSpore model file.
    config: ChatGLM2Config = MindFormerConfig(config)['model']['model_config']
    config = ChatGLM2Config(**config)

    # qkv weight split
    if not config.qkv_concat:
        attn_merge(pt_param, config)
    else:
        attn_rearange(pt_param, config)

    # mlp weight split
    if not config.mlp_concat:
        mlp_merge(pt_param)

    print('saving pt ckpt....')
    torch.save(pt_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLM2/3 weight convert script")
    parser.add_argument("--mindspore_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help='The mindspore checkpoint path.')
    parser.add_argument("--torch_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The output torch checkpoint path.")
    parser.add_argument("--dtype",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="The dtype of transformed mindspore weight.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to model config yaml")

    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    opt = parser.parse_args()
    convert_ms_to_pt(opt.mindspore_ckpt_path, opt.torch_ckpt_path, config=opt.config,
                     dtype=mapping.get(opt.dtype, ms.bfloat16),)
