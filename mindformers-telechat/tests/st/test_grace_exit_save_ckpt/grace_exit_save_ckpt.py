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
"""
Test module for testing resume training from specified checkpoint.
How to run this:
pytest tests/st/test_grace_exit_save_ckpt/test_parallel_grace_exit_save_ckpt.py
"""
import os
from glob import glob
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindformers import build_context
from mindformers.tools.utils import (
    get_epoch_and_step_from_ckpt_name
)
from mindformers.trainer import Trainer
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.tools.register import MindFormerConfig


SEED = 42
NUM_LAYERS = 2
NUM_HEADS = 4
HIDDEN_SIZE = 512
SEQ_LENGTH = 1024
DATA_SIZE = 1024


def generator():
    """dataset generator"""
    for i in range(DATA_SIZE):
        np.random.seed(SEED + i)
        input_ids = np.random.randint(low=0, high=DATA_SIZE, size=(SEQ_LENGTH + 1,)).astype(np.int32)
        yield input_ids


def get_checkpoints_path(checkpoint_dir):
    """get checkpoints path"""
    checkpoints_path = glob(os.path.join(checkpoint_dir, "*.ckpt"))
    checkpoints_path.sort(key=get_epoch_and_step_from_ckpt_name)
    return checkpoints_path


def llama_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition

    config = MindFormerConfig("./test_grace_exit_save_ckpt.yaml")
    build_context(config)

    model_config = LlamaConfig(num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
                               num_heads=NUM_HEADS, hidden_size=HIDDEN_SIZE,
                               parallel_config=config.parallel_config)
    model = LlamaForCausalLM(model_config)

    # Training using first dataset.
    dataset = GeneratorDataset(generator, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model, args=config, train_dataset=dataset)

    trainer.train(train_checkpoint=False)


llama_trainer_train_from_instance()
