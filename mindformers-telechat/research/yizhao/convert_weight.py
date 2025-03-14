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
"""Convert checkpoint from torch/huggingface"""
import argparse
from typing import List

import mindspore as ms
import numpy as np
from tqdm import tqdm
from transformers import AutoModel

from mindformers import MindFormerConfig
from mindformers.utils.convert_utils import pt2ms
from research.yizhao.yizhao_model.yizhao_config import YiZhaoConfig

dtype_mapping = {
    ms.float16: np.float16,
    ms.float32: np.float32,
    ms.bfloat16: np.float32
}


def npy2ms(arr: np.array, dtype):
    return ms.Tensor(arr, dtype=dtype)


def rearange_w(w: np.ndarray, head_dim: int):
    """rearange_w"""
    w = w.T
    hidden_size, projection_size = w.shape
    w = w.reshape(hidden_size, projection_size // head_dim, head_dim)
    w = np.concatenate(
        [
            w[..., 0::2],
            w[..., 1::2]
        ],
        axis=-1
    )
    w = w.reshape(hidden_size, projection_size)
    return w.T


def rearange_b(b: np.ndarray, head_dim: int):
    """rearange_b"""
    projection_size = b.shape[0]
    b = b.reshape(projection_size // head_dim, head_dim)
    b = np.concatenate(
        [
            b[..., 0::2],
            b[..., 1::2],
        ],
        axis=-1
    )
    b = b.reshape(-1,)
    return b


def attn_split(param_list: List, config: YiZhaoConfig):
    """attn_split"""
    is_mqa = config.multi_query_attention  # Multi query attention
    kv_channels = config.kv_channels
    num_attention_heads = config.num_attention_heads
    projection_size = kv_channels * num_attention_heads
    n_kv_head = config.multi_query_group_num
    head_dim = kv_channels
    q_hidden_size = projection_size
    kv_hidden_size = head_dim * n_kv_head if is_mqa else projection_size

    param_idx_to_del = []
    param_split = []

    for idx, item in enumerate(param_list):
        name: str = item['name']
        data: ms.Tensor = item['data']
        if "query_key_value" not in name:
            continue
        param_idx_to_del.append(idx)
        org_dtype = data.dtype
        data = data.to(ms.float32).asnumpy() if org_dtype == ms.bfloat16 else data.asnumpy()
        # Split on axis 0 for enabling transpose_b
        wq, wk, wv = np.split(data, [q_hidden_size, q_hidden_size + kv_hidden_size], axis=0)
        w_name = name.replace("query_key_value", "wq")
        k_name = name.replace("query_key_value", "wk")
        v_name = name.replace("query_key_value", "wv")
        if config.use_llama_rope:
            # rearange wq
            if len(wq.shape) == 2:
                wq = rearange_w(wq, head_dim)
                wk = rearange_w(wk, head_dim)
            else:
                wq = rearange_b(wq, head_dim)
                wk = rearange_b(wk, head_dim)
        if not config.qkv_concat:
            param_split.append({"name": w_name, "data": npy2ms(wq, org_dtype)})
            param_split.append({"name": k_name, "data": npy2ms(wk, org_dtype)})
            param_split.append({"name": v_name, "data": npy2ms(wv, org_dtype)})
        else:
            concat_qkv = np.concatenate(
                [wq, wk, wv],
                axis=0
            )
            param_split.append({"name": name, "data": npy2ms(concat_qkv, org_dtype)})

    for idx in reversed(param_idx_to_del):  # delete element in list by index
        param_list.pop(idx)

    param_list += param_split


def mlp_split(param_list: List, config: YiZhaoConfig):
    """mlp_split"""
    ffn_hidden_size = config.ffn_hidden_size
    param_idx_to_del = []
    param_split = []

    for idx, item in enumerate(param_list):
        name: str = item['name']
        data: ms.Tensor = item['data']
        if "dense_h_to_4h" not in name:
            continue
        param_idx_to_del.append(idx)
        org_dtype = data.dtype
        data = data.to(ms.float32).asnumpy() if org_dtype == ms.bfloat16 else data.asnumpy()
        # Split on axis 0 for enabling transpose_b
        w_left, w_right = np.split(data, [ffn_hidden_size], axis=0)
        left_name = name.replace("dense_h_to_4h", "dense_left")
        right_name = name.replace("dense_h_to_4h", "dense_right")
        param_split.append({"name": left_name, "data": npy2ms(w_left, org_dtype)})
        param_split.append({"name": right_name, "data": npy2ms(w_right, org_dtype)})

    for idx in reversed(param_idx_to_del):  # delete element in list by index
        param_list.pop(idx)

    param_list += param_split


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, config, dtype=ms.float32, **kwargs):
    """ Convert pytorch model file to MindSpore model file. """
    config: YiZhaoConfig = MindFormerConfig(config)['model']['model_config']
    config = YiZhaoConfig(**config)
    model = AutoModel.from_pretrained(input_path)

    print('parameter convert....')
    ms_param = []
    for k, v in tqdm(model.state_dict().items()):
        if "word_embeddings.weight" in k:
            k = k.replace("word_embeddings.weight", "embedding_weight")
        ms_param.append({"name": k, "data": pt2ms(v, dtype)})

    # qkv weight split
    if not config.qkv_concat or config.use_llama_rope:
        attn_split(ms_param, config)

    # mlp weight split
    if not config.mlp_concat:
        mlp_split(ms_param, config)

    ms.save_checkpoint(ms_param, output_path)
    print(f"Convert finished, the output is saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLM2/3 weight convert script")
    parser.add_argument("--torch_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help='The output mindspore checkpoint path.')
    parser.add_argument("--dtype",
                        type=str,
                        default="bf16",
                        help="The dtype of transformed mindspore weight.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to model config yaml")
    mapping = {
        "fp32": ms.float32,
        "fp16": ms.float16,
        "bf16": ms.bfloat16
    }

    opt = parser.parse_args()
    convert_pt_to_ms(opt.torch_ckpt_path, opt.mindspore_ckpt_path, dtype=mapping[opt.dtype], config=opt.config)
