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

import argparse
import copy
import numpy as np

import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import AdamWeightDecay
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.communication import get_rank

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.config import LoraConfig, TransformerConfig, ModelParallelConfig
from mindformers.experimental.parallel_core.pynative.utils import valid_lora_config
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.enums import ModelType

from tests.st.test_distri_core.utils import TestData, train


def mark_only_lora_as_trainable(network):
    """mark only lora parameters as trainable"""
    for param in network.get_parameters():
        if 'lora' in param.name:
            param.requires_grad = True
        else:
            param.requires_grad = False


class ParallelTransformerNet(Module):
    """ ParallelTransformerNet. """

    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                        rotary_percent=1.0)
        use_lora = config.lora_config.use_lora
        transformer_config = copy.deepcopy(config)
        if use_lora:
            transformer_config.update_lora_config('transformer')
        self.transformer = ParallelTransformer(config=transformer_config, post_norm=False,
                                               model_type=ModelType.encoder_or_decoder)
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


def run_parallel_transformer_pretrain():
    """ Test ParallelTransformer pretrain. """
    batch_size = 1
    dataset_size = 10
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

    lora_config = LoraConfig(use_lora=False)
    parallel_config = ModelParallelConfig(expert_model_parallel_size=1, sequence_parallel=False)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               group_query_attention=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout=0.0,
                               attention_dropout=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               norm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=True)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)
    rank_id = get_rank()
    ms.save_checkpoint(network, f'pretrain_rank_{rank_id}.ckpt')


def run_parallel_transformer_lora():
    """ Test ParallelTransformer lora. """
    batch_size = 1
    dataset_size = 10
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

    target_cells = [
        {'target_cells': [
            '.*.out_proj',
            '.*.mapping',
            '.*.projection',
            'transformer.layers.0.attention.qkv_proj',
            ]
        },
        {'cell': 'transformer.layers.0.attention.qkv_proj',
         'rank': 4,
         'alpha': 16
        },
    ]
    lora_config = LoraConfig(use_lora=True, target_cells=target_cells)
    parallel_config = ModelParallelConfig(expert_model_parallel_size=1, sequence_parallel=False)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               group_query_attention=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout=0.0,
                               attention_dropout=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               norm_epsilon=1.e-5)
    rank_id = get_rank()
    pretrain_params = ms.load_checkpoint(f'pretrain_rank_{rank_id}.ckpt')
    config = valid_lora_config(config, pretrain_params)

    network = ParallelTransformerNet(config=config, with_rope=True)

    not_load_network_params = ms.load_param_into_net(network, pretrain_params)
    print('params not load in network:', not_load_network_params)
    mark_only_lora_as_trainable(network)
    ms.save_checkpoint(network, f'lora_rank{rank_id}_init.ckpt')

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    losses = train(10, dataset, network, optimizer, None, with_attn_input=True)
    for loss in losses[-10:]:
        assert loss < 1.0e-3
    rank_id = get_rank()
    ms.save_checkpoint(network, f'lora_rank{rank_id}.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain', action='store_true', help="model pretrain."
    )

    args, rest_args = parser.parse_known_args()
    if args.pretrain:
        run_parallel_transformer_pretrain()
    else:
        run_parallel_transformer_lora()
