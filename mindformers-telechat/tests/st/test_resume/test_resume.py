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
Test module for testing resume training.
How to run this:
    pytest tests/st/test_resume/test_resume.py
"""
import os
from functools import partial
from glob import glob
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer, TrainingArguments
from mindformers.tools.utils import get_real_rank

from tests.st.test_model.test_llama2_model.base_model import get_model, get_config
from tests.utils.resume_train import extract_loss_values, get_file_mtime

ms.set_context(mode=0)
cur_dir = os.path.dirname(os.path.abspath(__file__))

set_seed(0)
np.random.seed(0)


def generate_data(seq_len, vocab_size, batch_size=8, step_num=4):
    """generate data for testing model."""
    input_ids = np.random.randint(
        low=0, high=vocab_size, size=(step_num * batch_size, seq_len + 1,)).astype(np.int32)
    for input_data in input_ids:
        yield input_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_resume_train_from_instance():
    """
    Feature: Trainer.train(resume_training=True)
    Description: Test resume training from saved ckpt with Trainer.
    Expectation: ValueError
    """
    batch_size = 8
    args = TrainingArguments(
        batch_size=batch_size,
        num_train_epochs=2,
        save_steps=4,
        save_directory=f"{cur_dir}/resume_ckpt"
    )

    config = get_config()
    model = get_model(config)

    prepare_data = partial(generate_data,
                           seq_len=config.seq_length,
                           vocab_size=config.vocab_size,
                           batch_size=batch_size)
    dataset = GeneratorDataset(prepare_data, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=batch_size)

    # first train process
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    trainer.train(train_checkpoint=False)

    checkpoint_dir = os.path.join(f"{cur_dir}/resume_ckpt", "checkpoint", "rank_{}".format(get_real_rank()))
    ckpt_path = glob(os.path.join(checkpoint_dir, "*.ckpt"))
    ckpt_path = sorted(ckpt_path, key=get_file_mtime)
    for _ in range(1):
        os.remove(ckpt_path.pop())

    # resume train process
    dataset = GeneratorDataset(prepare_data, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=batch_size)

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    trainer.train(resume_from_checkpoint=os.path.join(f"{cur_dir}/resume_ckpt", "checkpoint"),
                  resume_training=True)
    loss = extract_loss_values(f"output/log/rank_0/info.log")
    assert abs(loss[-4] - loss[-2]) < 0.005
    assert abs(loss[-3] - loss[-1]) < 0.005
