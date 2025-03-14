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
Convert utils.
"""
import numpy as np
import torch
import mindspore as ms
from mindspore.ops.operations import Cast

cpu_cast = Cast().set_device("CPU")


def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    if value.dtype == torch.bfloat16:
        np_value = value.detach().cpu().to(torch.float32).numpy()
    else:
        np_value = value.detach().numpy()

    if dtype:
        return ms.Tensor(np_value, dtype=dtype)
    return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)


def ms2pt(value: ms.Tensor, dtype) -> torch.Tensor:
    """
    convert ms.Tensor to torch.Tensor with specified dtype
    """
    if value.dtype == ms.bfloat16:
        np_value = cpu_cast(value, ms.float32).asnumpy()
    else:
        np_value = value.asnumpy()

    if dtype:
        return torch.from_numpy(np_value).cpu().to(dtype)
    return torch.from_numpy(np_value).cpu().to(torch.bfloat16) if value.dtype == ms.bfloat16 else torch.from_numpy(
        np_value)


def is_lora_param(key: str) -> bool:
    """
    is lora parameter of model weight
    """
    if 'lora' in key.lower():
        return True
    return False


def qkv_concat_hf2mg(qkv_weights: np.ndarray, num_heads, n_kv_heads, hidden_size):
    """
    convert qkv_concat weight with huggingface format to megatron format.
    """
    qkv_dim = len(qkv_weights.shape)
    if qkv_dim == 2:
        w, h = qkv_weights.shape
    elif qkv_dim == 1:
        # cur qkv_weight is bias
        w = qkv_weights.shape[0]
        qkv_weights = qkv_weights.reshape(w, -1)
        h = 1
    else:
        raise ValueError("qkv_weights shape is not supported.")
    n_rep = num_heads // n_kv_heads
    q_channel = hidden_size
    kv_channel = hidden_size // n_rep
    q_weight = qkv_weights[: q_channel, :]
    k_weight = qkv_weights[q_channel: q_channel + kv_channel, :]
    v_weight = qkv_weights[q_channel + kv_channel: q_channel + 2 * kv_channel, :]
    q_w_reshape = q_weight.reshape(n_kv_heads, hidden_size // n_kv_heads, -1)
    k_w_reshape = k_weight.reshape(n_kv_heads, hidden_size // num_heads, -1)
    v_w_reshape = v_weight.reshape(n_kv_heads, hidden_size // num_heads, -1)
    cat_qkv_weight = np.concatenate((q_w_reshape, k_w_reshape, v_w_reshape), axis=1)
    out_qkv_weight = cat_qkv_weight.reshape(w, h)
    if qkv_dim == 1:
        out_qkv_weight = out_qkv_weight.reshape(w,)
    return out_qkv_weight


def ffn_concat_hf2mg(ffn_weights: np.ndarray, ffn_hidden_size):
    """
        convert ffn_concat weight with huggingface format to megatron format.
    """
    w, h = ffn_weights.shape
    gate_weight = ffn_weights[: w // 2, :]
    hidden_weight = ffn_weights[w // 2: w // 2 * 2, :]
    gate_w_reshape = gate_weight.reshape(ffn_hidden_size, 1, -1)
    hidden_w_reshape = hidden_weight.reshape(ffn_hidden_size, 1, -1)
    cat_ffn_weight = np.concatenate((gate_w_reshape, hidden_w_reshape), axis=1)
    out_ffn_weight = cat_ffn_weight.reshape(w, h)
    return out_ffn_weight
