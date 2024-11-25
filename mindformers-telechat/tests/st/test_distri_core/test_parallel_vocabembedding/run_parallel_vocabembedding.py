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
"""run parallel vocab embedding"""

import argparse
import os

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import Tensor, nn
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay, CrossEntropyLoss

from mindformers import MindFormerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.tensor_parallel.layers import (
    VocabParallelEmbedding,
)
from mindformers.modules import VocabEmbedding
from mindformers.modules.transformer import EmbeddingOpParallelConfig
from tests.st.test_distri_core.utils import (
    TestData,
    train,
    transform_vocab_embedding_golden_params_to_pynative_params,
)

default_embedding_parallel_config = EmbeddingOpParallelConfig()


class VocabEmbeddingNet(nn.Cell):
    """
    define a graph VocabEmbedding net
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 parallel_config=default_embedding_parallel_config,
                 param_init_type=mstype.float32):
        super().__init__()
        self.embedding = VocabEmbedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            parallel_config=parallel_config,
            param_init_type=param_init_type,
        )
        self.loss = CrossEntropyLoss()

    def construct(self, input_ids, labels):
        output, _ = self.embedding(input_ids)
        loss = self.loss(output.transpose(0, 2, 1), labels)
        return loss


class ParallelVocabEmbeddingNet(nn.Cell):
    """
    define a pynative VocabEmbedding net
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 parallel_config=None,
                 param_init_type=mstype.float32):
        super().__init__()
        self.embedding = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            init_method='normal',
            reduce_scatter_embeddings=parallel_config.sequence_parallel,
            config=parallel_config,
            param_init_dtype=param_init_type,
        )
        self.loss = CrossEntropyLoss()

    def construct(self, input_ids, labels):
        output, _ = self.embedding(input_ids)
        loss = self.loss(output.transpose(0, 2, 1), labels)
        return loss


def generate_golden():
    """
    run graph mode vocab embedding to generate golden ckpt and loss
    """
    batch_size = 1
    dataset_size = 10
    seq_length = 8
    vocab_size = 32
    embeding_size = 64
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic="ON")
    init()

    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length)).astype(np.int32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.int32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=["input_ids", "labels"])
    dataset = dataset.batch(batch_size)

    network = VocabEmbeddingNet(
        vocab_size=vocab_size,
        embedding_size=embeding_size,
        param_init_type=mstype.float32,
    )
    ms.save_checkpoint(network, "vocab_embeding_golden.ckpt")
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None)


def run_parallel_vocabembedding():
    """
    run pynative mode vocab embedding and load golden ckpt to generate pynative loss
    """
    batch_size = 1
    dataset_size = 10
    seq_length = 8
    vocab_size = 32
    embeding_size = 64
    tensor_parallel = 2
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length)).astype(np.int32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.int32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=["input_ids", "labels"])
    dataset = dataset.batch(batch_size)

    parallel_config = MindFormerConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    network = ParallelVocabEmbeddingNet(
        vocab_size=vocab_size,
        embedding_size=embeding_size,
        parallel_config=parallel_config,
        param_init_type=mstype.float32,
    )
    golden_ckpt_path = "vocab_embeding_golden.ckpt"
    assert os.path.exists(golden_ckpt_path), (
        "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
        + "`pytest -sv test_parallel_vocabembedding.py::TestParallelVocabEmbedding::generate_golden`"
    )
    golden_params = ms.load_checkpoint(golden_ckpt_path)
    pynative_params = transform_vocab_embedding_golden_params_to_pynative_params(
        golden_params
    )
    param_not_load, _ = ms.load_param_into_net(network, pynative_params)
    assert (
        not param_not_load
    ), f"{param_not_load} was not loaded in this net, test failed."

    input_ids = Tensor(shape=(None, None), dtype=mstype.int32)
    labels = Tensor(shape=(None, None), dtype=mstype.int32)
    network.set_inputs(input_ids, labels)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_golden", action="store_true", help="Generate golden data for test."
    )

    args, rest_args = parser.parse_known_args()
    if args.generate_golden:
        generate_golden()
    else:
        run_parallel_vocabembedding()
