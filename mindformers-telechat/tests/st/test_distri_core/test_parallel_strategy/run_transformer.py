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
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, get_rng_state
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig, OptimizerConfig, \
    TrainingConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.dist_checkpointing import load_checkpoint, save_checkpoint
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from tests.st.test_distri_core.utils import TestData, train, _transform_ckpt_helper
class ParallelTransformerNet(Module):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_attention_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, post_norm=False)
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

def run_parallel_transformer(test_mode, tp_size, rng_mode):
    """ Test ParallelTransformer. """
    batch_size = 1
    dataset_size = 3
    num_layers = 2
    seq_length = 16
    num_attention_heads = 4
    hidden_size = 16
    tensor_parallel = tp_size
    if test_mode in ["transform_src", "transform_dst"]:
        dropout_rate = 0.0
    else:
        dropout_rate = 0.1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    training_config = TrainingConfig(parallel_config=parallel_config)
    if test_mode == "transform_src":
        parallel_config.zero_level = "z3"
    elif test_mode == "transform_dst":
        parallel_config.zero_level = "z2"
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=num_layers,
                               num_attention_heads=num_attention_heads,
                               hidden_size=hidden_size,
                               attention_type='cross_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               parallel_config=parallel_config,
                               params_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               hidden_dropout=dropout_rate,
                               attention_dropout=dropout_rate,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               ffn_hidden_size=4*hidden_size,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               norm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=True)
    zero_config = {}
    zero_config['grad_allreduce_op'] = "mean"
    optimizer_config = OptimizerConfig(parallel_config=parallel_config,
                                       optimizer_type="AdamW",
                                       beta1=0.9,
                                       beta2=0.999,
                                       eps=1.e-6,
                                       learning_rate=1.e-3,
                                       zero_config=zero_config,
                                       weight_decay=0.0)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = get_optimizer(optimizer_config, training_config, network.trainable_params(), network)

    if test_mode == "transform_src":
        save_checkpoint(config, network, optimizer, None, f"./output/transformer_0")
        losses = train(1, dataset, network, optimizer, None, with_attn_input=True, zero_level=3)
        np.save(f"./loss_transform_src.npy", np.array(losses))
    elif test_mode == "transform_dst":
        _transform_ckpt_helper(config, network, optimizer,
                               f"./output/transformer_0", f"./output/transformer_0_1",
                               ckpt_prefix="transformed")
        load_checkpoint(config, network, optimizer, None, f"./output/transformer_0_1")
        losses = train(1, dataset, network, optimizer, None, with_attn_input=True, zero_level=2)
        np.save(f"./loss_transform_dst.npy", np.array(losses))
    else:
        if rng_mode == "save":
            losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
            save_checkpoint(config, network, optimizer, None, f"./output/transformer_0")
            losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
        else:
            load_checkpoint(config, network, optimizer, None, f"./output/transformer_0")
            losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
        losses = list(map(lambda x: x[0], losses))
        np.save(f"./rng_state_{rng_mode}.npy", np.array(get_rng_state()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, required=True, choices=["transform_src", "transform_dst", "rng_check"])
    parser.add_argument('--tp_size', type=int, required=False, default=1, help="TP size")
    parser.add_argument('--rng_mode', type=str, required=False, default="save", choices=["save", "load"],
                        help="'save' saves rng state while 'load' restores the saving state")

    args, rest_args = parser.parse_known_args()
    run_parallel_transformer(args.test_mode, args.tp_size, args.rng_mode)
