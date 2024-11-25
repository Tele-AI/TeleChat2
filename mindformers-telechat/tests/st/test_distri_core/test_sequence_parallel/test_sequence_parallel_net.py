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
"""run sequence parallel."""
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore.nn import AdamWeightDecay
from mindspore.nn import CrossEntropyLoss
from mindspore.communication.management import init

from mindformers.tools.register.config import MindFormerConfig
from mindformers.modules import VocabEmbedding, Linear
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    initialize_model_parallel,
    get_data_parallel_world_size,
    get_tensor_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    GatherFromSequenceParallelRegion
)

from tests.st.test_distri_core.utils import TestData


def train(
        epoch_num,
        dataset,
        network,
        optimizer):
    """Base train function."""
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=optimizer.parameters
    )

    loss_list = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            input_ids, labels = data
            loss, grads = grad_func(input_ids, labels)
            loss = ops.depend(loss, optimizer(grads))
            print(f"Epoch {epoch} | Step {step} | Loss {loss}")
            step += 1
            loss_list.append(loss.asnumpy())
    return loss_list


class VocabEmbeddingNet(nn.Cell):
    """Golden embedding and linear with sequence parallel network."""

    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            embedding_size: int,
            config: MindFormerConfig,
            param_init=0.1,
            param_init_type=mstype.float32,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_size = self.config.model_config.hidden_size
        self.batch_size = self.config.training.batch_size

        self.embedding1 = VocabEmbedding(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            param_init=param_init,
            param_init_type=param_init_type,
        )
        self.embedding2 = VocabEmbedding(
            vocab_size=self.seq_length,
            embedding_size=embedding_size,
            param_init=param_init
        )

        self.embedding_dropout = nn.Dropout(
            keep_prob=1.0 - self.config.model_config.embedding_dropout_prob
        )

        self.linear1 = Linear(
            in_channels=self.config.model_config.hidden_size,
            out_channels=self.config.model_config.ffn_hidden_size,
            weight_init=param_init,
            bias_init='zeros',
            has_bias=False
        )

        self.linear2 = Linear(
            in_channels=self.config.model_config.ffn_hidden_size,
            out_channels=self.config.model_config.hidden_size,
            weight_init=param_init,
            bias_init='zeros',
            has_bias=False
        )

        self.loss = CrossEntropyLoss()

    def construct(self, x, labels):
        """Construct of golden case."""
        pos = ms.Tensor(np.arange(
            int(self.batch_size * self.seq_length)
        ).reshape((self.batch_size, self.seq_length))).astype(mstype.int32)
        w_emb, _ = self.embedding1(x)
        p_emb, _ = self.embedding2(pos)
        emb = w_emb + p_emb
        intermediate = self.linear1(emb)
        output = self.linear2(intermediate)
        result = self.loss(output, labels)
        return result


class VocabParallelEmbeddingNet(nn.Cell):
    """VocabparallelEmbedding, ColumnParallelLinear and
    RowParallelLinear with sequence parallel network."""

    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            config,
            param_init=0.1
    ):
        super().__init__()
        config.parallel_config.overlap_grad_reduce = args.overlap_grad_reduce
        config.parallel_config.gradient_accumulation_fusion = args.gradient_accumulation_fusion
        self.config = config
        self.seq_length = seq_length
        self.batch_size = config.training.batch_size

        self.embedding1 = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.model_config.hidden_size,
            init_method=param_init,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel,
            config=config,
            param_init_dtype=mstype.float32,
        )
        self.embedding2 = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.model_config.hidden_size,
            init_method=param_init,
            reduce_scatter_embeddings=config.parallel_config.sequence_parallel,
            config=config,
            param_init_dtype=mstype.float32,
        )

        self.linear1 = ColumnParallelLinear(
            input_size=config.model_config.hidden_size,
            output_size=config.model_config.ffn_hidden_size,
            config=config,
            init_method=param_init,
            bias_init='zeros',
            gather_output=False,
            skip_bias_add=False,
            bias=False,
            param_init_dtype=mstype.float32,
            compute_dtype=mstype.float16)

        self.linear2 = RowParallelLinear(
            input_size=config.model_config.ffn_hidden_size,
            output_size=config.model_config.hidden_size,
            input_is_parallel=True,
            config=config,
            init_method=param_init,
            bias_init='zeros',
            bias=False,
            skip_bias_add=False,
            param_init_dtype=mstype.float32,
            compute_dtype=mstype.float16)
        self.loss = CrossEntropyLoss()
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=config.dataset_config.data_layout == "BSH",
            tensor_parallel_output_grad=False
        )

    def construct(self, x, labels):
        """Construct of sequence parallel case."""
        pos = ms.Tensor(np.arange(int(self.seq_length)).reshape(
            (self.batch_size, self.seq_length)))
        w_emb = self.embedding1(x)
        p_emb = self.embedding2(pos)
        emb = w_emb + p_emb
        intermediate, _ = self.linear1(emb)
        output, _ = self.linear2(intermediate)
        if self.config.parallel_config.sequence_parallel:
            output1 = self.gather_from_sp_region(output)
        else:
            output1 = output
        result = self.loss(output1, labels)
        return result


def generate_golden_net():
    """Generate golden result."""
    batch_size = 1
    dataset_size = 2
    seq_length = 8
    vocab_size = 8
    embedding_size = 4
    tensor_parallel = 2

    ms.set_context(device_id=0, device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()

    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(2024)
    input_data = np.random.randint(
        0, vocab_size, (dataset_size, seq_length)
    ).astype(np.int32)
    label_data = np.zeros(
        (dataset_size, seq_length, embedding_size)).astype(
            np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(
        dataset, column_names=[
            "input_ids", "labels"])
    dataset = dataset.batch(batch_size)

    network = VocabEmbeddingNet(
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_size=embedding_size,
        param_init=0.1,
        config=MindFormerConfig('./test_sequence_parallel.yaml')
    )

    param_init = ops.arange(
        0, vocab_size // tensor_parallel * embedding_size, dtype=mstype.float32
    ).reshape(vocab_size // tensor_parallel, embedding_size)
    param_init = ops.cat((param_init, param_init), axis=0)
    param_init = ms.Tensor(param_init)
    network.embedding1.embedding_table.set_data(param_init)

    input_data = ms.Tensor(shape=(None, None), dtype=mstype.int32)
    labels = ms.Tensor(shape=(None, None, None), dtype=mstype.float32)
    network.set_inputs(input_data, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    loss_list = train(1, dataset, network, optimizer)
    return loss_list


def sequence_parallel_net():
    """test parallel network"""
    batch_size = 1
    dataset_size = 2
    seq_length = 8
    vocab_size = 8
    embedding_size = 4
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    ms.set_seed(2024)
    input_data = np.random.randint(
        0, vocab_size, (dataset_size, seq_length)
    ).astype(np.int32)
    label_data = np.zeros(
        (dataset_size, seq_length, embedding_size)).astype(
            np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(
        dataset, column_names=[
            "input_ids", "labels"])
    dataset = dataset.batch(batch_size)

    network = VocabParallelEmbeddingNet(
        vocab_size=vocab_size,
        seq_length=seq_length,
        param_init=0.1,
        config=MindFormerConfig('./test_sequence_parallel.yaml')
    )

    param_init = ops.arange(
        0, vocab_size // tensor_parallel * embedding_size, dtype=mstype.float32
    ).reshape(vocab_size // tensor_parallel, embedding_size)
    param_init = ms.Tensor(param_init)
    network.embedding1.weight.set_data(param_init)

    input_data = ms.Tensor(shape=(None, None), dtype=mstype.int32)
    labels = ms.Tensor(shape=(None, None, None), dtype=mstype.float32)
    network.set_inputs(input_data, labels)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    loss_list = train(1, dataset, network, optimizer)
    return loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_golden",
        action="store_true",
        help="Generate golden data for test.")
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        help="Generate golden data for test.")
    parser.add_argument(
        "--gradient_accumulation_fusion",
        action="store_true",
        help="Generate golden data for test.")

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        golden_loss = generate_golden_net()
        np.save('./golden_loss.npy', golden_loss)
    elif args.gradient_accumulation_fusion:
        sp_loss = sequence_parallel_net()
        np.save('./sp_overlap_grad_scc.npy', sp_loss)
    elif args.overlap_grad_reduce:
        sp_loss = sequence_parallel_net()
        np.save('./sp_overlap.npy', sp_loss)
    else:
        sp_loss = sequence_parallel_net()
        np.save('./use_sequence_parallel_loss.npy', sp_loss)
