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
pytest tests/st/test_resume/test_parallel_resume.py
"""
import os
import json
import time
import random
from glob import glob
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindformers import build_context
from mindformers.tools.utils import (
    LOCAL_DEFAULT_PATH,
    get_real_rank,
    get_real_group_size,
    get_epoch_and_step_from_ckpt_name
)
from mindformers.trainer import Trainer, TrainingArguments
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig
from mindformers.tools.logger import logger

SEED = 42
NUM_LAYERS = 2
NUM_HEADS = 4
HIDDEN_SIZE = 512
SEQ_LENGTH = 1024
DATA_SIZE = 64


def generator_1():
    """dataset generator"""
    for i in range(DATA_SIZE):
        np.random.seed(SEED + i)
        input_ids = np.random.randint(low=0, high=DATA_SIZE, size=(SEQ_LENGTH + 1,)).astype(np.int32)
        yield input_ids


def generator_2():
    """dataset generator"""
    for i in range(DATA_SIZE // 2):
        np.random.seed(SEED + DATA_SIZE // 2 + i)
        input_ids = np.random.randint(low=0, high=DATA_SIZE, size=(SEQ_LENGTH + 1,)).astype(np.int32)
        yield input_ids


def get_checkpoints_path(checkpoint_dir):
    """get checkpoints path"""
    checkpoints_path = glob(os.path.join(checkpoint_dir, "*.ckpt"))
    checkpoints_path.sort(key=get_epoch_and_step_from_ckpt_name)
    return checkpoints_path


def wait_training_over():
    """wait current training task saving checkpoint over"""
    meta_json = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                             "rank_{}".format(get_real_rank()), "meta.json")
    with open(meta_json, "r") as json_file:
        meta_data = json.load(json_file)
    last_epoch = meta_data["last_epoch"]
    last_step = meta_data["last_step"]
    logger.info(f"Rank_{get_real_rank()} get last_epoch={last_epoch}, last_step={last_step}")
    start_time = time.time()
    while True:
        save_over = True
        for rank_id in range(get_real_group_size()):
            meta_json = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                                     "rank_{}".format(rank_id), "meta.json")
            with open(meta_json, "r") as json_file:
                meta_data = json.load(json_file)
            compare_epoch = meta_data["last_epoch"]
            compare_step = meta_data["last_step"]
            if last_epoch != compare_epoch or last_step != compare_step:
                logger.info(f"Rank_{rank_id}'s last_epoch or last_step is not equal to Rank_{get_real_rank()}"
                            f"expect last_epoch={last_epoch}, last_step={last_step},"
                            f"but get last_epoch={compare_epoch}, last_step={compare_step}")
                save_over = False
                time.sleep(0.1 + random.uniform(0, 0.1))
        if save_over:
            break
        if time.time() - start_time > 60:
            raise TimeoutError("Wait current training task saving checkpoint over timeout!")


def llama_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    config = TrainingArguments(
        num_train_epochs=1,
        batch_size=2,
        use_parallel=True,
        data_parallel=2,
        model_parallel=2,
        pipeline_stage=2,
        micro_batch_num=2,
        save_steps=1,
        save_total_limit=8,
        sink_size=1,
        total_steps=8,
        save_directory=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel"),
    )

    build_context(config)

    # Model
    model_config = LlamaConfig(num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
                               num_heads=NUM_HEADS, hidden_size=HIDDEN_SIZE,
                               parallel_config=config.get_parallel_config())
    model = LlamaForCausalLM(model_config)

    # Training using first dataset.
    dataset = GeneratorDataset(generator_1, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model, args=config, train_dataset=dataset)
    trainer.train(train_checkpoint=False)

    wait_training_over()

    checkpoint_dir = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                                  "rank_{}".format(get_real_rank()))
    checkpoints_path = get_checkpoints_path(checkpoint_dir)
    for _ in range(len(checkpoints_path) // 2):
        os.remove(checkpoints_path.pop())

    # Resume training using the new second dataset.
    config.num_train_epochs = 2

    model = LlamaForCausalLM(model_config)

    dataset = GeneratorDataset(generator_2, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model, args=config, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint"),
                  resume_training=True, ignore_data_skip=True)

    wait_training_over()

    # Resume training after interruption in the new dataset.
    checkpoints_path = get_checkpoints_path(checkpoint_dir)

    model = LlamaForCausalLM(model_config)

    dataset = GeneratorDataset(generator_2, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model, args=config, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint"),
                  resume_training=os.path.basename(checkpoints_path[-3]), data_skip_steps=2)

    # Resume training after interruption in the new dataset with double batch_size.
    config.per_device_train_batch_size = 4
    config.total_steps = 4

    model = LlamaForCausalLM(model_config)

    dataset = GeneratorDataset(generator_2, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=16)

    trainer = Trainer(model=model, args=config, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint"),
                  resume_training=os.path.basename(checkpoints_path[-3]), data_skip_steps=1)

llama_trainer_train_from_instance()
