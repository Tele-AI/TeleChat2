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
""" Test Global Norm. """

import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication.management import init
from mindspore.mint.optim import AdamW
from mindspore.nn import CrossEntropyLoss

from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    GatherFromSequenceParallelRegion,
    ScatterToSequenceParallelRegion
)
from mindformers.experimental.parallel_core.pynative.transformer.enums import (
    ModelType
)

from utils import TestData, train


class ParallelTransformerNet(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False, use_sequence_parallel=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size // config.num_attention_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, post_norm=True, model_type=ModelType.encoder_or_decoder)
        self.loss = CrossEntropyLoss()
        self.use_sequence_parallel = use_sequence_parallel
        self.scatter_to_sp_region = ScatterToSequenceParallelRegion(need_to_swapaxes=False)
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            tensor_parallel_output_grad=False,
            need_to_swapaxes=False
        )

    def construct(self, x, attention_mask, labels):
        """ construct. """
        emb = self.rope(max_seq_len=x.shape[1])
        if self.use_sequence_parallel:
            x = x.swapaxes(0, 1).contiguous()
            x = self.scatter_to_sp_region(x)
            x = x.swapaxes(0, 1).contiguous()
        if self.with_rope:
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        if self.use_sequence_parallel:
            output = output.swapaxes(0, 1).contiguous()
            output = self.gather_from_sp_region(output)
            output = output.swapaxes(0, 1).contiguous()
        output = output.transpose(0, 2, 1)
        loss = self.loss(output, labels)
        return loss


def run_parallel_transformer(use_sequence_parallel=True):
    """ Test ParallelTransformer. """
    batch_size = 1
    dataset_size = 10
    num_layers = 4
    seq_length = 4096
    num_heads = 32
    hidden_size = 4096
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tp_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.int32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_parallel=tensor_parallel, sequence_parallel=use_sequence_parallel)
    config = TransformerConfig(seq_length=seq_length,
                               vocab_size=1,
                               num_layers=num_layers,
                               num_attention_heads=num_heads,
                               hidden_size=hidden_size,
                               attention_type='self_attn',
                               qkv_has_bias=True,
                               out_proj_has_bias=True,
                               parallel_config=parallel_config,
                               param_init_dtype='float32',
                               compute_dtype='float32',
                               softmax_compute_dtype='float32',
                               hidden_dropout=0.0,
                               attention_dropout=0.0,
                               mask_func_type="attn_mask_fill",
                               mlp_has_bias=True,
                               ffn_hidden_size=4 * hidden_size,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               norm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=True, use_sequence_parallel=use_sequence_parallel)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.int32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(params=network.get_parameters())

    _, all_norm = train(1, dataset, network, optimizer, None, with_attn_input=True,
                        use_sequence_parallel=use_sequence_parallel)
    golden_norm = [226.186662, 0.1741715, 1016.479]
    for i in range(3):
        assert np.allclose(golden_norm[i], all_norm[i], rtol=1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_sequence_parallel', action='store_true', help="use sequence parallel."
    )

    args, rest_args = parser.parse_known_args()
    run_parallel_transformer(args.use_sequence_parallel)
