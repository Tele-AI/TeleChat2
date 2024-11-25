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
Test module for testing the paralleled mixtral interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_mixtral_model/test_parallel_train.py
pytest tests/st/test_model/test_mixtral_model/test_parallel_predict.py
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import set_seed
from mindspore.communication import init
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.modules.transformer.moe import MoEConfig
from tests.st.training_checker import TrainingChecker

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                             gradients_mean=True,
                             full_batch=True,
                             enable_parallel_optimizer=True)
init()


def generator_train():
    """train dataset generator"""
    seq_len = 513
    step_num = 20
    batch_size = 8
    vocab_size = 32000
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]


def build_model(test_mode):
    """init task trainer."""
    set_seed(0)
    np.random.seed(0)

    args = TrainingArguments(batch_size=8, num_train_epochs=1, use_parallel=True)
    moe_config = MoEConfig(expert_num=8,
                           capacity_factor=1.1,
                           aux_loss_factor=0.05,
                           routing_policy="TopkRouterV2",
                           enable_sdrop=True)
    model_config = LlamaConfig(num_layers=2,
                               hidden_size=5120,
                               num_heads=16,
                               seq_length=512,
                               batch_size=8,
                               use_flash_attention=True,
                               moe_config=moe_config)
    model = LlamaForCausalLM(model_config)

    if test_mode == 'test_train':
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=8)

        loss_list_std = [10.627518, 10.672284, 10.732680, 10.806048, 10.792619,
                         10.727634, 10.822031, 10.857632, 10.890938, 10.882463,
                         10.841049, 10.801310, 10.761316, 10.732450, 10.725170,
                         10.708043, 10.682218, 10.681205, 10.686296, 10.660928]
        avg_step_time_std = 125
        callback = TrainingChecker(loss_list_std=loss_list_std,
                                   avg_step_time_std=avg_step_time_std,
                                   micro_batch_num=2,
                                   micro_batch_interleave_num=2)

        task_trainer = Trainer(task='text_generation',
                               model=model,
                               args=args,
                               train_dataset=train_dataset,
                               callbacks=callback)
    else:
        task_trainer = Trainer(task='text_generation', model=model, args=args)
    return task_trainer


def msrun_mixtral_8p_train():
    """test msrun launch mixtral on 8p for Trainer.train()."""
    task_trainer = build_model('test_train')
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    task_trainer.set_parallel_config(data_parallel=2,
                                     model_parallel=2,
                                     expert_parallel=2,
                                     pipeline_stage=2,
                                     micro_batch_num=2,
                                     micro_batch_interleave_num=2)
    task_trainer.train()


def msrun_mixtral_8p_predict():
    """test msrun launch mixtral on 8p for Trainer.predict()."""
    task_trainer = build_model('test_predict')
    predict_data = ["hello world!"] * 8
    task_trainer.set_parallel_config(data_parallel=1, model_parallel=8, pipeline_stage=1, micro_batch_num=1)
    task_trainer.predict(input_data=predict_data, max_length=20, repetition_penalty=1, top_k=3, top_p=1, batch_size=8)


def msrun_launch_mixtral_8p():
    """
    Feature: Trainer.train() Trainer.predict() Trainer.export()
    Description: Test trainer for train/predict/export on parallel mode.
    Expectation: TypeError, ValueError, RuntimeError
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_train', default=False, type=bool,
        help='test msrun launch llama on train mode.')
    parser.add_argument(
        '--test_predict', default=False, type=bool,
        help='test msrun launch llama on predict mode.')

    args = parser.parse_args()
    if args.test_train:
        msrun_mixtral_8p_train()
    elif args.test_predict:
        msrun_mixtral_8p_predict()


msrun_launch_mixtral_8p()
