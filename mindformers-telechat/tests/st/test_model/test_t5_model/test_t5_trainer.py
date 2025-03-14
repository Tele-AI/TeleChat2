# Copyright 2022 Huawei Technologies Co., Ltd
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
Test module for testing the t5 trainer used for mindformers.
How to run this:
pytest tests/st/test_model/test_t5_model/test_t5_trainer.py
"""
import os
import shutil

import numpy as np
import pytest
import mindspore as ms
from mindspore.dataset import MindDataset, GeneratorDataset
from mindspore.mindrecord import FileWriter

from mindformers.trainer import Trainer, \
    TranslationTrainer, TrainingArguments
from mindformers import T5Config, T5ForConditionalGeneration

ms.set_context(mode=0)


def generator(src_length=16, target_length=8):
    """dataset generator"""
    input_ids = np.random.randint(low=0, high=15, size=(src_length,)).astype(np.int32)
    attention_mask = np.ones((src_length,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(target_length,)).astype(np.int32)

    for _ in range(2):
        yield input_ids, attention_mask, labels


def write_mindrecord(ds_generator, data_record_path):
    """Using the generator to get mindrecords"""
    ds = GeneratorDataset(ds_generator, column_names=["input_ids", "attention_mask", "labels"])

    writer = FileWriter(file_name=data_record_path, shard_num=1, overwrite=True)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "test_schema")
    for item in ds.create_dict_iterator():
        for k in item.keys():
            item[k] = item[k].asnumpy()
        writer.write_raw_data([item])
    writer.commit()


def modify_attrs(net, key, value):
    if hasattr(net, key):
        setattr(net, key, value)
        print(f"Set the net {net.__class__.__name__} with {key}:{value}")
    for cell in net.cells():
        modify_attrs(cell, key, value)


def write_raw_text_data(stage, data_record_path):
    """writes the fake translation data"""
    source = ["We went through the whole range of emotions during this period.",
              "The positive reaction of pilots and Federation officials makes me hope that this year we will "
              "be organizing champions again"
              " said rally manager, Dan Codreanu."]
    target = ['Am trecut prin toată gama de trăiri în această perioadă.',
              "Reacția pozitivă a piloților și oficialilor Federației mă face să sper că vom fi și în acest an "
              "campion la organizare a spus managerul raliului, Dan Codreanu."]

    src_path = os.path.join(data_record_path, f'{stage}.source')
    tgt_path = os.path.join(data_record_path, f'{stage}.target')
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(src_path, flags_, 0o750), 'w') as sfp:
        with os.fdopen(os.open(tgt_path, flags_, 0o750), 'w') as tfp:
            for x, y in zip(source, target):
                sfp.write(x + '\n')
                tfp.write(y + '\n')


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestTranslationTrainer:
    """Test Translation Trainer"""
    def setup_class(self):
        self.dir_path = os.path.join(os.path.dirname(__file__), 'fake_dataset')
        os.makedirs(self.dir_path, exist_ok=True)
        self.abs_path = os.path.join(self.dir_path, 't5_dataset')
        write_mindrecord(generator(src_length=16, target_length=8), self.abs_path)

        self.raw_text_path = os.path.join(os.path.dirname(__file__), 'raw_text_dataset')
        os.makedirs(self.raw_text_path, exist_ok=True)
        write_raw_text_data(stage='train', data_record_path=self.raw_text_path)

    def teardown_class(self):
        shutil.rmtree(self.dir_path, ignore_errors=True)
        shutil.rmtree(self.raw_text_path, ignore_errors=True)

    def get_mindfiles_from_path(self, dir_path):
        dataset_files = []
        for r, _, f in os.walk(dir_path):
            for file in f:
                if not file.endswith("db"):
                    dataset_files.append(os.path.join(r, file))
        return dataset_files

    @pytest.mark.run(order=1)
    def test_trainer_with_translation_args_train(self):
        """
        Feature: Create Trainer From Config
        Description: Test Trainer API to train from config
        Expectation: TypeError
        """
        batch_size = 1
        config = TrainingArguments(num_train_epochs=1, batch_size=batch_size, seed=2022,
                                   optim="adamw", adam_beta1=0.9, learning_rate=0.001)

        dataset = MindDataset(dataset_files=self.get_mindfiles_from_path(self.dir_path),
                              columns_list=["input_ids", "attention_mask", "labels"])
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(1)

        model_config = T5Config(batch_size=batch_size, num_heads=8, num_layers=1, hidden_size=32,
                                seq_length=16, max_decode_length=8)
        # Model
        model = T5ForConditionalGeneration(model_config)
        mim_trainer = Trainer(task='translation',
                              model=model,
                              args=config,
                              train_dataset=dataset)
        mim_trainer.train()

    @pytest.mark.run(order=2)
    def test_trainer_predict(self):
        """
        Feature: Test Predict of the Trainer
        Description: Test Predict
        Expectation: TypeError
        """
        # change the length for quick training
        model_config = T5Config(seq_length=32, max_decode_length=32)
        model = T5ForConditionalGeneration(model_config)
        mim_trainer = TranslationTrainer(model_name="t5_small")
        mim_trainer.predict(input_data="hello world", network=model)
