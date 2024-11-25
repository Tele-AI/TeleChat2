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
""" Test ParallelAttention. """
import os
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import AdamWeightDecay
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init

from mindformers.models.llama.llama_transformer import LLamaAttention
from mindformers.modules.layers import FreqsMgr
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.transformer import default_transformer_recompute_config
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelAttention
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding

from tests.st.test_distri_core.utils import TestData, train, transform_transformerlayer_params, generate_ckpt


class LlamaAttentionNet(nn.Cell):
    """ LlamaAttentionNet. """
    def __init__(
            self,
            seq_length,
            hidden_size,
            num_attention_heads,
            n_kv_heads=None,
            qkv_concat=True,
            compute_dtype=mstype.float32,
            softmax_compute_dtype=mstype.float32,
            rotary_type=mstype.float32,
            param_init_type=mstype.float32,
            qkv_has_bias=True,
            parallel_config=default_dpmp_config,
            use_flash_attention=False,
        ):
        super(LlamaAttentionNet, self).__init__()
        self.freqs_mgr = FreqsMgr(head_dim=hidden_size//num_attention_heads,
                                  seq_length=seq_length,
                                  rotary_dtype=mstype.float32)
        self.use_fa = use_flash_attention
        self.attention = LLamaAttention(dim=hidden_size,
                                        n_heads=num_attention_heads,
                                        n_kv_heads=n_kv_heads,
                                        qkv_concat=qkv_concat,
                                        compute_dtype=compute_dtype,
                                        softmax_compute_dtype=softmax_compute_dtype,
                                        rotary_dtype=rotary_type,
                                        param_init_type=param_init_type,
                                        qkv_has_bias=qkv_has_bias,
                                        parallel_config=parallel_config,
                                        use_flash_attention=use_flash_attention)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct. """
        if self.use_fa and attention_mask.ndim == 3:
            attention_mask = ops.expand_dims(attention_mask, axis=1)
        freqs_cis = self.freqs_mgr(x.shape[1])
        output = self.attention(x, freqs_cis, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


class ParallelAttentionNet(nn.Cell):
    """ ParallelAttentionNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelAttentionNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_attention_heads,
                                        rotary_percent=1.0)
        self.attention = ParallelAttention(layer_number=1, config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct."""
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output, _ = self.attention(x, attention_mask, rotary_pos_emb=emb)
        else:
            output, _ = self.attention(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def generate_golden_with_rope(use_fa=False, group_query_attention=False):
    """ Generate golden. """
    batch_size = 1
    dataset_size = 3
    seq_length = 8
    num_attention_heads = 4
    num_query_groups = 2
    hidden_size = 16

    ms.set_context(device_id=0,
                   device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   deterministic='ON',
                   jit_config={'jit_level': 'O0'})
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
                                n_kv_heads=num_query_groups if group_query_attention else None,
                                qkv_concat=True,
                                qkv_has_bias=True,
                                use_flash_attention=use_fa)
    kv_hidden_size = hidden_size // num_attention_heads * num_query_groups if group_query_attention else None
    param_dict = generate_ckpt(hidden_size=hidden_size,
                               module_type='attention',
                               kv_hidden_size=kv_hidden_size)
    ms.load_param_into_net(network, param_dict)
    save_golden = False
    if save_golden:
        suffix = "_with_fa" if use_fa else ""
        ms.save_checkpoint(network, "attention_golden{}.ckpt".format(suffix))
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)


def run_parallel_attention_with_rope(use_fa=False, group_query_attention=False):
    """ Test ParallelAttention. """
    batch_size = 1
    dataset_size = 3
    seq_length = 8
    num_attention_heads = 4
    num_query_groups = 2
    hidden_size = 16
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=1,
                               num_attention_heads=num_attention_heads,
                               group_query_attention=group_query_attention,
                               num_query_groups=num_query_groups,
                               hidden_size=hidden_size,
                               ffn_hidden_size=hidden_size,
                               attn_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               params_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               attention_dropout=0.0,
                               mask_func_type="attn_mask_add",
                               use_flash_attention=use_fa,
                               fa_config={
                                   'pre_tokens': 65536,
                                   'next_tokens': 0,
                                   'sparse_mode': 0,
                                   'input_layout': 'BNSD',
                               })
    network = ParallelAttentionNet(config=config, with_rope=True)
    kv_hidden_size = hidden_size // num_attention_heads * num_query_groups if group_query_attention else None
    save_golden = False
    if save_golden:
        suffix = "_with_fa" if use_fa else ""
        golden_ckpt_path = "attention_golden{}.ckpt".format(suffix)
        assert os.path.exists(golden_ckpt_path), \
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n" + \
            "`pytest -sv test_attention.py::TestParallelAttention::generate_golden`"
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(golden_params,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=kv_hidden_size)
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(hidden_size=hidden_size, module_type='attention', kv_hidden_size=kv_hidden_size)
        pynative_params = transform_transformerlayer_params(param_dict,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=kv_hidden_size)
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [4.3615346, 137.4123, 6.082371]
    if group_query_attention:
        golden_losses = [12.902611, 101.94232, 1.8575361]
    if use_fa:
        golden_losses = [20.140625, 137.71875, 7.1989370]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    parser.add_argument(
        '--use_fa', action='store_true', help="Whether use flashattention."
    )
    parser.add_argument(
        '--group_query_attention', action='store_true', help="Whether use group attention."
    )

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        generate_golden_with_rope(args.use_fa, args.group_query_attention)
    else:
        run_parallel_attention_with_rope(args.use_fa, args.group_query_attention)
