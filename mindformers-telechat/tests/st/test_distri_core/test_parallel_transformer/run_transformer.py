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
""" Test ParallelTransformer. """
import os
import argparse
from typing import Optional
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

from mindformers.modules import FeedForward
from mindformers.models.llama.llama_transformer import LLamaAttention
from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.modules.layers import FreqsMgr
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.transformer import default_transformer_recompute_config
from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding

from tests.st.test_distri_core.utils import TestData, train, transform_transformerlayer_params, generate_ckpt


class LLamaDecodeLayer(nn.Cell):
    """ LLamaDecodeLayer. """
    def __init__(self,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 ffn_dim: int = 2048,
                 hidden_act: str = 'gelu',
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 qkv_concat=False,
                 compute_dtype=mstype.float32,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config):
        super().__init__()
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.shape = ops.Shape()
        self.reshape = ops.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = ops.Add()
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = LLamaAttention(dim=dim,
                                        n_heads=n_heads,
                                        n_kv_heads=n_kv_heads,
                                        qkv_concat=qkv_concat,
                                        compute_dtype=compute_dtype,
                                        softmax_compute_dtype=softmax_compute_dtype,
                                        rotary_dtype=rotary_dtype,
                                        param_init_type=param_init_type,
                                        qkv_has_bias=True,
                                        use_past=use_past,
                                        parallel_config=parallel_config)

        self.mlp = FeedForward(hidden_size=dim,
                               ffn_hidden_size=ffn_dim,
                               dropout_rate=0.,
                               hidden_act=hidden_act,
                               param_init_type=param_init_type,
                               parallel_config=parallel_config,
                               compute_dtype=compute_dtype)

    def construct(self, x, freqs_cis, attention_mask):
        """ Forward of transformer block. """
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, attention_mask)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.mlp(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)

        return out


class LlamaTransformerNet(nn.Cell):
    """ LlamaTransformerNet. """
    def __init__(
            self,
            num_layers,
            seq_length,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            qkv_concat=True,
            parallel_config=default_dpmp_config,
            reduction=None
        ):
        super(LlamaTransformerNet, self).__init__()
        self.freqs_mgr = FreqsMgr(head_dim=hidden_size//num_attention_heads,
                                  seq_length=seq_length,
                                  rotary_dtype=mstype.float32)
        self.transformer = nn.CellList()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            layer = LLamaDecodeLayer(seq_length=seq_length,
                                     layer_id=i,
                                     dim=hidden_size,
                                     ffn_dim=ffn_hidden_size,
                                     n_heads=num_attention_heads,
                                     qkv_concat=qkv_concat,
                                     parallel_config=parallel_config)
            self.transformer.append(layer)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, x, attention_mask, label):
        """ construct. """
        freqs_cis = self.freqs_mgr(x.shape[1])
        h = x
        for i in range(self.num_layers):
            h = self.transformer[i](h, freqs_cis, attention_mask)
        output = ops.sum(h, dim=-1, keepdim=False)
        loss = self.loss(output, label)
        return loss

class ParallelTransformerNet(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_attention_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, model_type=None, post_norm=False)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct. """
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def generate_golden():
    """ Generate golden. """
    batch_size = 1
    dataset_size = 3
    num_layers = 2
    seq_length = 16
    num_attention_heads = 8
    hidden_size = 64

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
    network = LlamaTransformerNet(num_layers=num_layers,
                                  seq_length=seq_length,
                                  hidden_size=hidden_size,
                                  ffn_hidden_size=4*hidden_size,
                                  num_attention_heads=num_attention_heads,
                                  qkv_concat=True,
                                  parallel_config=default_dpmp_config)
    param_dict = generate_ckpt(hidden_size=hidden_size,
                               module_type='transformer',
                               num_layers=num_layers)
    ms.load_param_into_net(network, param_dict)
    save_golden = False
    if save_golden:
        ms.save_checkpoint(network, "transformer_golden.ckpt")
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)


def run_parallel_transformer():
    """ Test ParallelTransformer. """
    batch_size = 1
    dataset_size = 3
    num_layers = 2
    seq_length = 16
    num_attention_heads = 8
    hidden_size = 64
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
                               num_layers=num_layers,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               attention_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               params_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               hidden_dropout=0.0,
                               attention_dropout=0.0,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               ffn_hidden_size=4*hidden_size,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               norm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = False
    if save_golden:
        golden_ckpt_path = "transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), \
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n" + \
            "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(golden_params,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=None,
                                                            prefix="transformer.layers.")
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert not param_not_load, f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(hidden_size=hidden_size,
                                   module_type='transformer',
                                   num_layers=num_layers)
        pynative_params = transform_transformerlayer_params(param_dict,
                                                            hidden_size=hidden_size,
                                                            kv_hidden_size=None,
                                                            prefix='transformer.layers.')
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    losses = list(map(lambda x: x[0], losses))
    golden_losses = [1487.9048, 2019.5625, 931.9822]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        generate_golden()
    else:
        run_parallel_transformer()
