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
"""Test module for qkv_concat as megatron"""
import argparse
import os
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication import init
from mindformers.experimental.graph.transformer.transformer import ParallelAttention
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.models.llama.llama_transformer import LLamaAttention
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.layers import FreqsMgr

ms.set_context(mode=ms.GRAPH_MODE)
rank_id = os.environ.get('RANK_ID')
if rank_id is not None:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
    init()
seed = 22
ms.set_seed(seed)
np.random.seed(seed)


def convert_qkv_weight_hf2meg(q_weight, k_weight, v_weight, num_query_groups, num_attention_heads, head_dim):
    """convert weight of qkv from hf format to megatron format."""
    if isinstance(q_weight, ms.Tensor):
        q_weight = q_weight.asnumpy()
    if isinstance(k_weight, ms.Tensor):
        k_weight = k_weight.asnumpy()
    if isinstance(v_weight, ms.Tensor):
        v_weight = v_weight.asnumpy()
    _, h = q_weight.shape
    w = num_attention_heads * head_dim + 2 * num_query_groups * head_dim
    n_rep = num_attention_heads // num_query_groups
    q_w_reshape = q_weight.reshape(num_query_groups, n_rep * head_dim, -1)
    k_w_reshape = k_weight.reshape(num_query_groups, head_dim, -1)
    v_w_reshape = v_weight.reshape(num_query_groups, head_dim, -1)
    cat_qkv_weight = np.concatenate((q_w_reshape, k_w_reshape, v_w_reshape), axis=1)
    out_qkv_weight = cat_qkv_weight.reshape(w, h)
    return out_qkv_weight


def convert_qkv_weight_meg2hf(meg_weight, num_query_groups, num_attention_heads, head_dim, concat_out=False):
    """convert weight of megatron qkv_conact format to hf format."""
    if isinstance(meg_weight, ms.Tensor):
        meg_weight = meg_weight.asnumpy()
    w, _ = meg_weight.shape
    assert w == num_attention_heads * head_dim + 2 * num_query_groups * head_dim
    n_rep = num_attention_heads // num_query_groups
    q_channel = num_attention_heads * head_dim
    kv_channel = num_query_groups * head_dim
    qkv_weight = meg_weight.reshape(num_query_groups, (n_rep + 2) * head_dim, -1)
    q_weight = qkv_weight[:, :n_rep * head_dim, :]
    q_weight = q_weight.reshape(q_channel, -1)
    k_weight = qkv_weight[:, n_rep * head_dim:n_rep * head_dim + head_dim, :]
    k_weight = k_weight.reshape(kv_channel, -1)
    v_weight = qkv_weight[:, n_rep * head_dim + head_dim:n_rep * head_dim + 2 * head_dim, :]
    v_weight = v_weight.reshape(kv_channel, -1)
    if concat_out:
        out_weight = np.concatenate((q_weight, k_weight, v_weight), 0)
    else:
        out_weight = [q_weight, k_weight, v_weight]
    return out_weight


class NewNet(nn.Cell):
    """A model class of new interface."""

    def __init__(self, config, q_weight, k_weight, v_weight, qkv_weight, out_weight=None):
        super(NewNet, self).__init__()
        self.parallel_attn = ParallelAttention(config, layer_number=1)
        if config.qkv_concat:
            self.parallel_attn.qkv_proj.weight = qkv_weight
        else:
            self.parallel_attn.q_proj.weight = q_weight
            self.parallel_attn.k_proj.weight = k_weight
            self.parallel_attn.v_proj.weight = v_weight

        if out_weight is not None:
            self.parallel_attn.out_proj.weight = out_weight

    def construct(self, x, mask, rotary_embedding):
        out, _ = self.parallel_attn(x, mask, rotary_pos_emb=rotary_embedding)
        return out


class OldNet(nn.Cell):
    """A model class of old interface."""

    def __init__(self, config, q_weight, k_weight, v_weight, out_weight):
        super(OldNet, self).__init__()
        parallel_config = TransformerOpParallelConfig()
        parallel_config.data_parallel = config.data_parallel
        parallel_config.model_parallel = config.tensor_parallel
        self.llama_attn = LLamaAttention(dim=config.hidden_size, n_heads=config.num_attention_heads,
                                         n_kv_heads=config.num_query_groups, qkv_concat=False,
                                         compute_dtype=config.compute_dtype,
                                         softmax_compute_dtype=config.softmax_compute_dtype,
                                         rotary_dtype=mstype.float16, qkv_has_bias=False,
                                         use_past=False, is_dynamic=False, use_rope_slice=False,
                                         use_flash_attention=config.use_flash_attn, use_ring_attention=False,
                                         use_attn_mask_compression=config.use_attn_mask_compression,
                                         parallel_config=parallel_config)
        self.llama_attn.wq.weight = ms.Parameter(q_weight)
        self.llama_attn.wk.weight = ms.Parameter(k_weight)
        self.llama_attn.wv.weight = ms.Parameter(v_weight)
        self.llama_attn.wo.weight = out_weight

    def construct(self, x, mask, rotary_embedding):
        out = self.llama_attn(x, rotary_embedding, mask=mask)
        return out


def get_config(hidden_size, dp, cp, tp):
    """A get_config function."""
    config = TransformerConfig()
    config.param_init_dtype = mstype.float32
    config.compute_dtype = mstype.float16
    config.group_query_attention = True
    config.num_attention_heads = 8
    config.num_query_groups = 4
    config.hidden_size = hidden_size
    config.use_flash_attn = True
    config.data_parallel = dp
    config.tensor_parallel = tp
    config.context_parallel = cp
    config.qkv_concat = True
    config.use_attn_mask_compression = False
    config.add_qkv_bias = False
    config.add_bias_linear = False
    config.softmax_compute_dtype = mstype.float32
    config.attention_dropout = 0.0
    config.apply_query_key_layer_scaling = False
    config.mask_func_type = 'attn_mask_add'
    return config


def run_test():
    """A run_qkv_test function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=1, required=True, type=int, help='data_parallel')
    parser.add_argument('--cp', default=1, required=True, type=int, help='context_parallel')
    parser.add_argument('--tp', default=1, required=True, type=int, help='tensor_parallel')
    args_, _ = parser.parse_known_args()

    batch, seq_length, hidden_size = (4, 256, 64)
    config = get_config(hidden_size, args_.dp, args_.cp, args_.tp)

    input_shape = (batch, seq_length, hidden_size)
    input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float16))
    mask = Tensor(np.triu(np.ones((seq_length, seq_length)), 1), mstype.uint8)
    if config.use_flash_attn and config.use_attn_mask_compression is False:
        mask = mask.reshape((1, 1, seq_length, seq_length))
        mask = mask.tile((batch, 1, 1, 1))
    head_dim = config.hidden_size // config.num_attention_heads

    # initialize weights
    q_w = ms.Parameter(np.random.standard_normal((hidden_size, hidden_size)).astype(np.float32))
    k_w = ms.Parameter(np.random.standard_normal((head_dim * config.num_query_groups, hidden_size)).astype(np.float32))
    v_w = ms.Parameter(np.random.standard_normal((head_dim * config.num_query_groups, hidden_size)).astype(np.float32))

    meg_weight = convert_qkv_weight_hf2meg(q_w, k_w, v_w, config.num_query_groups, config.num_attention_heads, head_dim)
    meg_weight = ms.Parameter(meg_weight)
    hf_weight = convert_qkv_weight_meg2hf(meg_weight, config.num_query_groups, config.num_attention_heads, head_dim)
    hf_weight = [ms.Parameter(x) for x in hf_weight]

    # 1. qkv_concat=False model init
    config.qkv_concat = False
    model_split = NewNet(config, q_w, k_w, v_w, None)
    out_w = model_split.parallel_attn.out_proj.weight

    # 2. qkv_concat=True model init
    config.qkv_concat = True
    model_concat = NewNet(config, None, None, None, meg_weight, out_weight=out_w)

    # 3. qkv_concat=False model init of old interface
    model_split_old = OldNet(config, hf_weight[0], hf_weight[1], hf_weight[2], out_w)

    # rope
    rotary_embedding = RotaryEmbedding(head_dim)
    rotary_embedding_old = FreqsMgr(head_dim=head_dim, seq_length=seq_length, max_position_embedding=seq_length,
                                    rotary_dtype=mstype.float16)

    # construct
    freqs = rotary_embedding(seq_length)
    freqs_old = rotary_embedding_old(seq_length)
    out_split = model_split(input_, mask, freqs)
    out_concat = model_concat(input_, mask, freqs)
    out_split_old = model_split_old(input_, mask, freqs_old)
    ret1 = np.allclose(out_split.asnumpy(), out_concat.asnumpy(), rtol=1e-4, atol=1e-4)
    ret2 = np.allclose(out_split.asnumpy(), out_split_old.asnumpy(), rtol=5e-4, atol=5e-4)

    assert ret1 and ret2


run_test()
