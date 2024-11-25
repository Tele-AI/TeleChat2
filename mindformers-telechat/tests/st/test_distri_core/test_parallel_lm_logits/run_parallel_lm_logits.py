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
import argparse
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam
from mindspore.communication import init
from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train
from mindformers.experimental.parallel_core.pynative.transformer import TransformerLanguageModel, ParallelLMLogits
from mindformers.experimental.parallel_core.pynative.create_comm import initialize_model_parallel, get_tensor_model_parallel_world_size, get_data_parallel_world_size
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    TransformerConfig,
    DatasetConfig,
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromModelParallelRegion

ms.set_seed(2024)
ds.set_seed(2024)

class FakeData():
    """ generate fake data for language model test """
    def __init__(self, data_num, seq_length, input_data=None):
        super().__init__()
        if input_data is not None:
            self.input_data = input_data
            self.data_num = self.input_data.shape[0]
            self.seq_length = self.input_data[0].shape[0]
        else:
            self.input_data = np.random.randint(0, 100, (data_num, seq_length + 1))
        ones = np.ones((data_num, 1, seq_length, seq_length), dtype=np.int32)
        self.attention_mask = np.tril(ones)
        self.position_ids = np.random.randint(0, 20, (data_num, seq_length))

    def __getitem__(self, index):
        return ms.Tensor(self.input_data[index], dtype=ms.int32), \
               ms.Tensor(self.position_ids[index], dtype=ms.int32), \
               ms.Tensor(self.attention_mask[index], dtype=ms.int32)

    def __len__(self):
        return self.input_data.shape[0]

class TestLanguageModel(ms.nn.Cell):
    """ Test language model """
    def __init__(self, config, parallel_output):
        super().__init__()
        self.transformer_language_model = TransformerLanguageModel(config, encoder_attn_mask_type=None)
        self.head = ParallelLMLogits(config=config.parallel_config,
                                     bias=False,
                                     compute_dtype=config.compute_dtype)
        self.loss = ms.nn.CrossEntropyLoss(reduction='mean')
        self.parallel_output = parallel_output
        self.gather_from_mp_region = GatherFromModelParallelRegion()

    def construct(self, input_ids, position_ids, attention_mask):
        """ Language model test forward """
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]
        hidden_states = self.transformer_language_model(input_ids, position_ids, attention_mask)
        logits_weights = self.transformer_language_model.embedding.word_embeddings.weight
        logits_weights = logits_weights.transpose(1, 0)
        logits = self.head(hidden_states, logits_weights, parallel_output=self.parallel_output)
        if self.parallel_output:
            logits = self.gather_from_mp_region(logits)
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.contiguous().reshape(-1).astype(ms.int32)
        loss = self.loss(logits.contiguous(), labels)
        return loss


def run_parallel_lm_logits(training_config, model_config, dataset_config):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    # init
    init()
    initialize_model_parallel(tensor_model_parallel_size=args.tp)

    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    # generate dataset
    dataset = FakeData(data_num=16, seq_length=model_config.seq_length)
    fake_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['input_ids', 'position_ids', 'attention_mask'],
                                       shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    fake_dataset = fake_dataset.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    network = TestLanguageModel(model_config, parallel_output=args.parallel_output)
    optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)

    # train
    train(train_one_step_cell, fake_dataset, training_config)

if __name__ == '__main__':
    CONFIG_PATH = "test_parallel_lm_logits.yaml"
    assert os.path.exists(CONFIG_PATH) and CONFIG_PATH.endswith(('.yaml', '.yml'))
    training_config_main, parallel_config_main, dataset_config_main, model_config_main = init_configs_from_yaml(
        CONFIG_PATH, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1, help="tensor_parallel")
    parser.add_argument('--parallel_output', action="store_true", help="parallel output")
    args, rest_args = parser.parse_known_args()
    run_parallel_lm_logits(training_config_main, model_config_main, dataset_config_main)
