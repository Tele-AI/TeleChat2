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
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, grad
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, get_rank
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding

from tests.st.test_distri_core.utils import TestData


class ParallelTransformerNet(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_attention_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, post_norm=False, model_type=None)
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


def run_parallel_transformer(recompute_method):
    """ Test ParallelTransformer. """
    seed = 2024
    batch_size = 1
    dataset_size = 3
    num_layers = 2
    seq_length = 16
    num_attention_heads = 8
    hidden_size = 32
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON', pynative_synchronize=True)
    use_sequence_parallel = True

    full_recompute_list = None
    select_recompute_list = None
    select_comm_recompute_list = None
    recompute_config = None
    if recompute_method == 0:
        full_recompute_list = [num_layers]
    elif recompute_method == 1:
        select_recompute_list = [num_layers]
    elif recompute_method == 2:
        select_comm_recompute_list = [num_layers]
    if recompute_method != -1:
        recompute_config = {"recompute": full_recompute_list,
                            "select_recompute": select_recompute_list,
                            "select_comm_recompute": select_comm_recompute_list}
    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel, use_sequence_parallel=use_sequence_parallel)

    ms.set_seed(seed)
    ms.manual_seed(seed)
    np.random.seed(seed)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)
    parallel_config = ModelParallelConfig(tensor_parallel=tensor_parallel,
                                          recompute_config=recompute_config,
                                          sequence_parallel=use_sequence_parallel)
    fa_config = {"input_layout": 'BNSD'}
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=num_layers,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               attention_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               param_init_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               ffn_hidden_size=4*hidden_size,
                               fa_config=fa_config,
                               use_flash_attention=True,
                               hidden_act="gelu",
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=False)
    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)
    grad_fn = grad(network.construct, grad_position=(0), weights=None)

    input_ids = Tensor(np.random.random((batch_size, seq_length, hidden_size)).astype(np.float32), ms.float32)
    attn_mask = Tensor(np.tril(np.ones(shape=(1, 1, 2 * seq_length, 2 * seq_length))).astype(np.uint8), ms.float32)
    labels = Tensor(np.zeros((batch_size, seq_length)).astype(np.float32), ms.float32)

    grad_value = grad_fn(input_ids, attn_mask, labels)
    if get_rank() == 0:
        if recompute_config is None:
            # without recompute
            np.save('grad_without_recompute.npy', grad_value.asnumpy())
            return
        grad_without_recompute = np.load('./grad_without_recompute.npy')
        print('grad_value:', grad_value.asnumpy())
        print('grad_without_recompute:', grad_without_recompute)
        assert np.allclose(
            grad_without_recompute, grad_value.asnumpy(), atol=1e-3
        ), "Gradient checkpointed recompute failed."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute_method', type=int, default=-1,
                        help="0 full, 1 select, 2 select comm, -1 not recompute")
    args, rest_args = parser.parse_known_args()
    run_parallel_transformer(args.recompute_method)
