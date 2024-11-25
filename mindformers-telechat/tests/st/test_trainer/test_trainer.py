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
Test module for testing the trainer interface used for mindformers.
How to run this:
    pytest tests/st/test_trainer/test_trainer.py
"""
import copy
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from mindformers import GPT2LMHeadModel, GPT2Config
from mindformers import Trainer, TrainingArguments
from mindformers.core.callback import MFLossMonitor

ms.set_context(mode=0)

EPOCHS = 1
NUM_LAYERS = 1
HIDDEN_SIZE = 16
NUM_HEADS = 2
SEQ_LENGTH = 32
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
EVAL_STEPS = 4
DATA_SIZE = 8


def generator_train():
    """train dataset generator"""
    seq_len = SEQ_LENGTH + 1
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(DATA_SIZE):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = SEQ_LENGTH
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(DATA_SIZE):
        yield train_data


class IterableTrain:
    """train iterable dataset"""

    def __init__(self):
        self._index = 0
        seq_len = SEQ_LENGTH + 1
        self.input_ids = np.random.randint(low=0, high=15, size=(DATA_SIZE, seq_len)).astype(np.int32)
        self.input_mask = np.ones_like(self.input_ids)

    def __next__(self):
        if self._index >= len(self.input_ids):
            raise StopIteration
        item = (self.input_ids[self._index], self.input_mask[self._index])
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.input_ids)


class IterableEval:
    """eval iterable dataset"""

    def __init__(self):
        self._index = 0
        seq_len = SEQ_LENGTH
        self.input_ids = np.random.randint(low=0, high=15, size=(DATA_SIZE, seq_len)).astype(np.int32)
        self.input_mask = np.ones_like(self.input_ids)

    def __next__(self):
        if self._index >= len(self.input_ids):
            raise StopIteration
        item = (self.input_ids[self._index], self.input_mask[self._index])
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.input_ids)


class AccessibleTrain:
    """train accessible dataset"""

    def __init__(self):
        seq_len = SEQ_LENGTH + 1
        self.input_ids = np.random.randint(low=0, high=15, size=(16, seq_len)).astype(np.int32)
        self.input_mask = np.ones_like(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.input_mask[index]

    def __len__(self):
        return len(self.input_ids)


class AccessibleEval:
    """eval accessible dataset"""

    def __init__(self):
        seq_len = SEQ_LENGTH
        self.input_ids = np.random.randint(low=0, high=15, size=(16, seq_len)).astype(np.int32)
        self.input_mask = np.ones_like(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.input_mask[index]

    def __len__(self):
        return len(self.input_ids)


MODEL_CONFIG = GPT2Config(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                          num_heads=NUM_HEADS, seq_length=SEQ_LENGTH)
MODEL = GPT2LMHeadModel(MODEL_CONFIG)

MINDFOREMR_CONFIG = Trainer.get_task_config(task="text_generation", model_name="gpt2")
MINDFOREMR_CONFIG.model.model_config.num_layers = NUM_LAYERS
MINDFOREMR_CONFIG.model.model_config.hidden_size = HIDDEN_SIZE
MINDFOREMR_CONFIG.model.model_config.num_heads = NUM_HEADS
MINDFOREMR_CONFIG.model.model_config.seq_length = SEQ_LENGTH
MINDFOREMR_CONFIG.model.model_config.checkpoint_name_or_path = ""
MINDFOREMR_CONFIG.eval_step_interval = EVAL_STEPS

TRAINING_ARGUMENTS = TrainingArguments(do_eval=True,
                                       metric_type="PerplexityMetric",
                                       eval_steps=EVAL_STEPS,
                                       per_device_train_batch_size=TRAIN_BATCH_SIZE,
                                       per_device_eval_batch_size=EVAL_BATCH_SIZE,
                                       train_dataset_in_columns=["input_ids", "input_mask"],
                                       eval_dataset_in_columns=["input_ids", "input_mask"],
                                       num_train_epochs=EPOCHS)

TRAIN_DATASET = GeneratorDataset(generator_train, column_names=["input_ids", "input_mask"])
TRAIN_DATASET = TRAIN_DATASET.batch(batch_size=TRAIN_BATCH_SIZE)
EVAL_DATASET = GeneratorDataset(generator_eval, column_names=["input_ids", "input_mask"])
EVAL_DATASET = EVAL_DATASET.batch(batch_size=EVAL_BATCH_SIZE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
class TestTrainerInit:
    """A test class for testing Trainer module."""

    def test_init_missing_arguments(self):
        """
        Feature: Trainer.__init__()
        Description: Test trainer init missing arguments.
        Expectation: ValueError
        """
        test_cases = [
            (None, "general", None, None, "Neither `task`, `model`, `model_name`, nor `args` are configured."),
            (None, "general", "gpt2", None, "The `task` is needed"),
            (None, "general", None, "gpt2", "The `task` is needed"),
            (None, "general", "gpt2", "gpt2", "The `task` is needed"),
            (None, "general", MODEL, None, "The `args` is needed"),
            (None, "general", MODEL, "gpt2", "The `task` is needed"),
            (None, "text_generation", None, None, "A model name is needed"),
            (TRAINING_ARGUMENTS, "general", None, None, "A model instance is needed"),
            (TRAINING_ARGUMENTS, "general", "gpt2", None, "The `task` is needed"),
            (TRAINING_ARGUMENTS, "general", None, "gpt2", "The `task` is needed"),
            (TRAINING_ARGUMENTS, "general", "gpt2", "gpt2", "The `task` is needed"),
            (TRAINING_ARGUMENTS, "text_generation", None, None, "A model name is needed")]
        for test_case in test_cases:
            args, task, model, model_name, expected = test_case
            with pytest.raises(ValueError) as excinfo:
                Trainer(args=args, task=task, model=model, model_name=model_name)
            assert expected in str(excinfo.value)

    def test_init_with_valid_arguments(self):
        """
        Feature: Trainer.__init__()
        Description: Test trainer init with valid arguments.
        Expectation: No exception
        """
        test_cases = [
            (None, "text_generation", "gpt2", None),
            (None, "text_generation", None, "gpt2"),
            (None, "text_generation", "gpt2", "gpt2"),
            (None, "text_generation", MODEL, None),
            (None, "text_generation", MODEL, "gpt2"),
            (MINDFOREMR_CONFIG, "general", None, None),
            (MINDFOREMR_CONFIG, "general", "gpt2", None),
            (MINDFOREMR_CONFIG, "general", None, "gpt2"),
            (MINDFOREMR_CONFIG, "general", "gpt2", "gpt2"),
            (MINDFOREMR_CONFIG, "general", MODEL, None),
            (MINDFOREMR_CONFIG, "general", MODEL, "gpt2"),
            (MINDFOREMR_CONFIG, "text_generation", None, None),
            (MINDFOREMR_CONFIG, "text_generation", "gpt2", None),
            (MINDFOREMR_CONFIG, "text_generation", None, "gpt2"),
            (MINDFOREMR_CONFIG, "text_generation", "gpt2", "gpt2"),
            (MINDFOREMR_CONFIG, "text_generation", MODEL, None),
            (MINDFOREMR_CONFIG, "text_generation", MODEL, "gpt2"),
            (TRAINING_ARGUMENTS, "general", MODEL, None),
            (TRAINING_ARGUMENTS, "general", MODEL, "gpt2"),
            (TRAINING_ARGUMENTS, "text_generation", "gpt2", None),
            (TRAINING_ARGUMENTS, "text_generation", None, "gpt2"),
            (TRAINING_ARGUMENTS, "text_generation", "gpt2", "gpt2"),
            (TRAINING_ARGUMENTS, "text_generation", MODEL, None),
            (TRAINING_ARGUMENTS, "text_generation", MODEL, "gpt2")]
        for test_case in test_cases:
            args, task, model, model_name = test_case
            Trainer(args=args, task=task, model=model, model_name=model_name)

    def test_add_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer add callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        assert trainer.callback_list == "MFLossMonitor"

        trainer.add_callback(MFLossMonitor)
        assert trainer.callback_list == "MFLossMonitor\nMFLossMonitor"

    def test_remove_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer remove callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        trainer.remove_callback(callback)
        assert trainer.callback_list == ""

        trainer.add_callback(callback)
        trainer.remove_callback(MFLossMonitor)
        assert trainer.callback_list == ""

    def test_pop_callback(self):
        """
        Feature: Trainer with callbacks
        Description: Test trainer pop callback.
        Expectation: No exception
        """
        trainer = Trainer(args=TRAINING_ARGUMENTS, model=MODEL)
        callback = MFLossMonitor()

        trainer.add_callback(callback)
        callback1 = trainer.pop_callback(callback)
        assert trainer.callback_list == ""

        trainer.add_callback(callback)
        callback2 = trainer.pop_callback(MFLossMonitor)
        assert callback1 == callback2


def run_trainer(args, task, model_name, train_dataset, eval_dataset):
    """static method of running trainer."""
    model_config = GPT2Config(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                              num_heads=NUM_HEADS, seq_length=SEQ_LENGTH)
    model = GPT2LMHeadModel(model_config)
    trainer = Trainer(args=args, task=task, model=model, model_name=model_name,
                      train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.config.runner_config.epochs = EPOCHS
    trainer.config.eval_step_interval = EVAL_STEPS

    trainer.train(do_eval=True)
    trainer.evaluate(eval_dataset=eval_dataset)

    model_config = GPT2Config(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                              num_heads=NUM_HEADS, seq_length=SEQ_LENGTH, use_past=True)
    model = GPT2LMHeadModel(model_config)
    trainer = Trainer(args=args, task="text_generation", model=model, model_name=model_name)
    trainer.predict(input_data="hello")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_without_args():
    """
    Feature: Trainer
    Description: Test trainer without args.
    Expectation: No exception
    """
    run_trainer(None, "text_generation", "gpt2", TRAIN_DATASET, EVAL_DATASET)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_generator():
    """
    Feature: Trainer
    Description: Test trainer with generator.
    Expectation: No exception
    """
    run_trainer(MINDFOREMR_CONFIG, "general", None, generator_train, generator_eval)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_check_for_nan_in_loss_and_grad():
    """
    Feature: Trainer
    Description: Test trainer with check_for_nan_in_loss_and_grad.
    Expectation: ValueError
    """
    args = copy.deepcopy(MINDFOREMR_CONFIG)
    args.check_for_nan_in_loss_and_grad = True
    with pytest.raises(ValueError) as excinfo:
        run_trainer(args, "general", None, generator_train, generator_eval)
        assert "global_norm is [nan], terminate training." in str(excinfo.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_iterable_dataset():
    """
    Feature: Trainer
    Description: Test trainer with iterable dataset.
    Expectation: No exception
    """
    run_trainer(TRAINING_ARGUMENTS, "general", None, IterableTrain(), IterableEval())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer_with_accessible_dataset():
    """
    Feature: Trainer
    Description: Test trainer with accessible dataset.
    Expectation: No exception
    """
    run_trainer(TRAINING_ARGUMENTS, "text_generation", "gpt2", AccessibleTrain(), AccessibleEval())
