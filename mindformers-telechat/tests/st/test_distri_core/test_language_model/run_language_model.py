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
"""Language model test"""
import os
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam, SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindformers.modules import VocabEmbedding
from mindformers.modules.transformer import EmbeddingOpParallelConfig
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.transformer import default_transformer_recompute_config
from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train
from mindformers.experimental.parallel_core.pynative.transformer import TransformerLanguageModel
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    TransformerConfig,
    DatasetConfig,
)
from tests.st.test_distri_core.utils import generate_ckpt, transform_transformerlayer_params
from tests.st.test_distri_core.utils import train as train_graph
from tests.st.test_distri_core.test_parallel_transformer.run_transformer import LlamaTransformerNet

ms.set_seed(1024)
ds.set_seed(1024)

default_embedding_parallel_config = EmbeddingOpParallelConfig()

class FakeData():
    """ generate fake data for language model test """
    def __init__(self, data_num, seq_length, input_data=None):
        super().__init__()
        if input_data is not None:
            self.input_data = input_data
            self.data_num = self.input_data.shape[0]
            self.seq_length = self.input_data[0].shape[0]
        else:
            input_data = np.random.randint(0, 100, (data_num, seq_length + 1))
            self.input_data = input_data[:, :-1]
            self.labels = input_data[:, 1:]
        ones = np.ones((data_num, 1, seq_length, seq_length), dtype=np.int32)
        self.attention_mask = np.tril(ones)

    def __getitem__(self, index):
        return ms.Tensor(self.input_data[index], dtype=ms.int32), \
               ms.Tensor(self.labels[index], dtype=ms.float32), \
               ms.Tensor(self.attention_mask[index], dtype=ms.int32)

    def __len__(self):
        return self.input_data.shape[0]

class GraphLanguageModel(ms.nn.Cell):
    """ Test language model """
    def __init__(self, config):
        super().__init__()
        self.embedding = VocabEmbedding(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            parallel_config=default_embedding_parallel_config,
            param_init_type=config.params_dtype,
        )
        self.llama_transformer = LlamaTransformerNet(config.num_layers,
                                                     config.seq_length,
                                                     config.hidden_size,
                                                     config.hidden_size * 4,
                                                     config.num_attention_heads,
                                                     True,
                                                     default_dpmp_config,
                                                     reduction='mean')

    def construct(self, input_ids, attention_mask, labels):
        """ Graph language model test forward """
        output, _ = self.embedding(input_ids)
        loss = self.llama_transformer(output, attention_mask, labels)
        return loss


class ParallelLanguageModel(ms.nn.Cell):
    """ Test language model """
    def __init__(self, config):
        super().__init__()
        self.language_model = TransformerLanguageModel(config, encoder_attn_mask_type=None)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction='mean')

    def construct(self, input_ids, attention_mask, labels):
        """ Language model test forward """
        hidden_states = self.language_model(input_ids, None, attention_mask)
        output = ms.ops.sum(hidden_states, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_parallel_language_model(training_config, model_config, dataset_config):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    # init
    init()
    initialize_model_parallel()

    # generate dataset
    dataset = FakeData(data_num=16, seq_length=model_config.seq_length)
    fake_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['input_ids', 'labels', 'attention_mask'],
                                       shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    fake_dataset = fake_dataset.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init ckpt
    param_dict = generate_ckpt(hidden_size=model_config.hidden_size,
                               module_type='transformer',
                               num_layers=model_config.num_layers,
                               prefix=None,
                               vocab_size=model_config.vocab_size,
                               use_embedding=True)
    param_dict = transform_transformerlayer_params(param_dict,
                                                   hidden_size=model_config.hidden_size,
                                                   kv_hidden_size=None,
                                                   prefix='language_model.encoder.layers.')
    # init model
    network = ParallelLanguageModel(model_config)

    # load ckpt
    ms.load_param_into_net(network, param_dict)

    optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)

    # train
    train(train_one_step_cell, fake_dataset, training_config)


def generate_golden(model_config, dataset_config):
    """main function."""
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   deterministic='ON',
                   jit_config={'jit_level': 'O0'})

    # init
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    # generate dataset
    dataset = FakeData(data_num=16, seq_length=model_config.seq_length)
    fake_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['input_ids', 'labels', 'attention_mask'],
                                       shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    fake_dataset = fake_dataset.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # add config
    default_dpmp_config.recompute = default_transformer_recompute_config

    # init ckpt
    param_dict = generate_ckpt(hidden_size=model_config.hidden_size,
                               module_type='transformer',
                               num_layers=model_config.num_layers,
                               prefix='llama_transformer.transformer.',
                               vocab_size=model_config.vocab_size,
                               use_embedding=True)

    # init model
    network = GraphLanguageModel(model_config)

    ms.load_param_into_net(network, param_dict)

    # init opt
    optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # train
    _ = train_graph(1, fake_dataset, network, optimizer, None, with_attn_input=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    args, _ = parser.parse_known_args()
    CONFIG_PATH = "test_language_model.yaml"
    assert os.path.exists(CONFIG_PATH) and CONFIG_PATH.endswith(('.yaml', '.yml'))
    training_config_main, parallel_config_main, dataset_config_main, model_config_main = init_configs_from_yaml(
        CONFIG_PATH, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig]
    )
    if args.generate_golden:
        generate_golden(model_config_main, dataset_config_main)
    else:
        run_parallel_language_model(training_config_main, model_config_main, dataset_config_main)
