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

from contextlib import contextmanager

from mindspore import Tensor, ops, Parameter, mint
from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_size, get_tensor_model_parallel_world_size

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


def _update_sharded_state_dict(network, sharded_state_dict):
    """Update shared state dict with network"""
    cells = network.name_cells()
    for _, subcell in cells.items():
        if subcell == network:
            continue
        if hasattr(subcell, "sharded_state_dict"):
            sharded_state_dict.update(subcell.sharded_state_dict())
        else:
            _update_sharded_state_dict(subcell, sharded_state_dict)


def generate_state_dict(network):
    """Generate the sharded state dict for network"""

    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, sharded_state_dict=model_state_dict)
    model_param_dict = network.parameters_dict()

    for name in model_param_dict:
        if name not in model_state_dict:
            model_state_dict[name] = {'shape': model_param_dict[name].shape,
                                      'shard': tuple([1] * model_param_dict[name].ndim)}

    state_dict['model'] = model_state_dict
    state_dict['optimizer'] = {}
    return state_dict


def get_tp_world_size():
    tp_size = get_tensor_model_parallel_world_size()
    return tp_size if tp_size else 1


def create_empty_parameter(shape, *, dtype=None, device=None, **kwargs):
    """Create an empty parameter."""
    def get_param(*args):
        return [Tensor, args[0]]

    @contextmanager
    def replace_class_method(cls, name, new_method):
        old_method = getattr(cls, name)

        setattr(cls, name, new_method)
        yield
        setattr(cls, name, old_method)

    data = mint.empty(shape, dtype=dtype, device=device)

    with replace_class_method(Parameter, "_get_parameter_new_args", get_param):
        param = Parameter(data, **kwargs)
    return param
