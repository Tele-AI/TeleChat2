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
""" utils """

from mindspore import Tensor, ops

from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_size
from mindformers.experimental.infer.core.layers import _update_sharded_state_dict

__all__ = ["get_attn_mask_func", "generate_state_dict"]


def attn_mask_fill(attention_scores: Tensor, attention_mask, fill_value=-10000.0):
    """mask attention scores with the mask value"""
    attention_scores = ops.masked_fill(
        attention_scores,
        attention_mask,
        Tensor(fill_value, attention_scores.dtype),
    )
    return attention_scores


def attn_mask_add(attention_scores: Tensor, attention_mask):
    """Llama attention mask function"""
    score_dtype = attention_scores.dtype
    attention_scores = ops.add(
        attention_scores, ops.Cast()(attention_mask, score_dtype)
    )
    return attention_scores


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": attn_mask_fill,
    "attn_mask_add": attn_mask_add,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill', 'attn_mask_add'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]


def generate_state_dict(network):
    """Generate the sharded state dict for network"""

    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, dict_=model_state_dict)
    state_dict['model'] = model_state_dict
    state_dict['optimizer'] = {}
    return state_dict
