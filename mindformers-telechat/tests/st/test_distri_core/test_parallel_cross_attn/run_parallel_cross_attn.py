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
"""run parallel cross attention"""

import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import get_rank, init
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelAttention
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.transformer.enums import AttnType
from mindformers.modules.layers import FreqsMgr
from mindformers.modules.transformer.transformer import default_transformer_recompute_config
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config

from tests.st.test_distri_core.utils import TestData, train
from llama_attention import LLamaCrossAttention


class LlamaAttentionNet(nn.Cell):
    """ LlamaAttentionNet. """
    def __init__(
                self,
                seq_length,
                hidden_size,
                num_attention_heads,
                qkv_concat=False,
                compute_dtype=mstype.float16,
                softmax_compute_dtype=mstype.float32,
                rotary_type=mstype.float32,
                param_init_type=mstype.float32,
                qkv_has_bias=True,
                parallel_config=default_dpmp_config,
                with_rope=False
    ):
        super(LlamaAttentionNet, self).__init__()
        self.freqs_mgr = FreqsMgr(head_dim=hidden_size // num_attention_heads,
                                  seq_length=seq_length,
                                  rotary_dtype=mstype.float32)
        self.attention = LLamaCrossAttention(
            dim=hidden_size,
            n_heads=num_attention_heads,
            qkv_concat=qkv_concat,
            compute_dtype=compute_dtype,
            softmax_compute_dtype=softmax_compute_dtype,
            rotary_dtype=rotary_type,
            param_init_type=param_init_type,
            qkv_has_bias=qkv_has_bias,
            parallel_config=parallel_config,
            with_rope=with_rope)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.seq_length = seq_length

    def construct(self, x, attention_mask, labels):
        """ construct. """
        encoder_output = x
        freqs_cis = self.freqs_mgr(self.seq_length)
        output = self.attention(x, freqs_cis, attention_mask, encoder_output=encoder_output)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


class ParallelCrossAttentionNet(nn.Cell):
    """ ParallelAttentionNet. """
    def __init__(self, config, with_rope=False, attn_type=AttnType.cross_attn):
        super(ParallelCrossAttentionNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size // config.num_attention_heads,
                                        rotary_percent=1.0)
        self.attention = ParallelAttention(layer_number=1, config=config, attention_type=attn_type)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct. """
        encoder_output = x
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output, _ = self.attention(x, attention_mask, encoder_output=encoder_output, rotary_pos_emb=emb)
        else:
            output, _ = self.attention(x, attention_mask, encoder_output=encoder_output)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def generate_golden():
    """
    run graph mode cross attention to generate golden ckpt and loss
    """
    batch_size = 1
    dataset_size = 10
    seq_length = 8
    hidden_size = 16
    num_attention_heads = 2

    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    init()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', 'attention_mask'])
    dataset = dataset.batch(batch_size)

    default_dpmp_config.recompute = default_transformer_recompute_config
    network = LlamaAttentionNet(seq_length=seq_length,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                qkv_concat=False,
                                qkv_has_bias=True,
                                with_rope=False,
                                softmax_compute_dtype=mstype.float16)
    network.attention.wq.weight.set_data(ms.Tensor(np.ones((hidden_size, hidden_size)).astype(np.float32) * 0.1))
    network.attention.wq.bias.set_data(ms.Tensor(np.zeros((hidden_size,)).astype(np.float32)))
    network.attention.wk.weight.set_data(ms.Tensor(np.ones((hidden_size, hidden_size)).astype(np.float32) * 0.1))
    network.attention.wk.bias.set_data(ms.Tensor(np.zeros((hidden_size,)).astype(np.float32)))
    network.attention.wv.weight.set_data(ms.Tensor(np.ones((hidden_size, hidden_size)).astype(np.float32) * 0.1))
    network.attention.wv.bias.set_data(ms.Tensor(np.zeros((hidden_size,)).astype(np.float32)))
    network.attention.wo.weight.set_data(ms.Tensor(np.ones((hidden_size, hidden_size)).astype(np.float32) * 0.1))

    ms.save_checkpoint(network, "golden.ckpt")
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)


def run_parallel_cross_attn():
    """ Test ParallelAttention. """
    batch_size = 1
    dataset_size = 10
    seq_length = 8
    hidden_size = 16
    num_attention_heads = 2
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
    rank_id = get_rank()

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', 'attention_mask'])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=1,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=hidden_size,
                               attention_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               params_dtype='float32',
                               compute_dtype='float16',
                               softmax_compute_dtype='float16',
                               attention_dropout=0.0,
                               mask_func_type="attn_mask_add",
                               share_embedding_weight=False,
                               attention_softmax_in_fp32=False)

    network = ParallelCrossAttentionNet(config=config, with_rope=False)

    network.attention.q_proj.weight.set_data(
        ms.Tensor(np.ones((hidden_size // 2, hidden_size)).astype(np.float32) * 0.1))
    network.attention.q_proj.bias.set_data(ms.Tensor(np.zeros((hidden_size // 2,)).astype(np.float32)))
    network.attention.kv_proj.weight.set_data(ms.Tensor(np.ones((hidden_size, hidden_size)).astype(np.float32) * 0.1))
    network.attention.kv_proj.bias.set_data(ms.Tensor(np.zeros((hidden_size,)).astype(np.float32)))
    network.attention.out_proj.weight.set_data(
        ms.Tensor(np.ones((hidden_size, hidden_size // 2)).astype(np.float32) * 0.1))

    ms.save_checkpoint(network, f"pynative_{rank_id}.ckpt")
    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attention_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attention_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        generate_golden()
    else:
        run_parallel_cross_attn()
