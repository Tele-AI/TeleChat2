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
Test module for testing tensorboard used for mindformers.
How to run this:
    pytest tests/st/test_ut/test_tensorboard.py
"""
import os
import pytest
import numpy as np

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters
from mindformers.core import MFLossMonitor
from mindformers.tools.register.config import MindFormerConfig
from mindformers.utils.tensorboard import get_text_mapping
from mindformers.models.llama import LlamaForCausalLM, LlamaConfig

cur_dir = os.path.dirname(os.path.abspath(__file__))
_GLOBAL_TENSORBOARD_WRITER = None

ms.set_context(mode=0)

_CHECK_TEXT_MAPPING = {
    'seed', 'output_dir', 'run_mode', 'use_parallel', 'resume_training', 'ignore_data_skip', 'data_skip_steps',
    'load_checkpoint', 'load_ckpt_format', 'auto_trans_ckpt', 'transform_process_num', 'src_strategy_path_or_dir',
    'load_ckpt_async', 'only_save_strategy', 'profile', 'profile_communication', 'profile_level', 'profile_memory',
    'profile_start_step', 'profile_stop_step', 'profile_rank_ids', 'profile_pipeline', 'init_start_profile',
    'layer_decay', 'layer_scale', 'lr_scale', 'lr_scale_factor', 'micro_batch_interleave_num', 'remote_save_url',
    'callbacks', 'context', 'data_size', 'device_num', 'do_eval', 'eval_callbacks', 'eval_step_interval',
    'eval_epoch_interval', 'eval_dataset', 'eval_dataset_task', 'lr_schedule', 'metric', 'model', 'moe_config',
    'optimizer', 'parallel_config', 'parallel', 'recompute_config', 'remove_redundancy', 'runner_config',
    'runner_wrapper', 'tensorboard', 'train_dataset_task', 'train_dataset', 'trainer', 'swap_config'
}

def generator_train():
    """train dataset generator"""
    seq_len = 4097
    step_num = 1
    batch_size = 1
    vocab_size = 32000
    input_ids = np.random.randint(low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)
    for idx, _ in enumerate(input_ids):
        yield input_ids[idx]

class TestTensorBoard:
    """A test class for testing pipeline."""

    def setup_method(self):
        """set _GLOBAL_TENSORBOARD_WRITER"""
        set_seed(0)
        np.random.seed(0)

        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids"])
        self.train_dataset = train_dataset.batch(batch_size=1)
        config_path = os.path.join(cur_dir, 'test_tensorboard.yaml')
        self.config = MindFormerConfig(config_path)
        self.tensorboard_dir = os.path.join(cur_dir, 'tensorboard')
        self.config.tensorboard = MindFormerConfig()
        self.config.tensorboard.tensorboard_dir = self.tensorboard_dir
        model_config = LlamaConfig(**self.config.model.model_config)
        self.model = LlamaForCausalLM(model_config)
        self.lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_ratio=0.01, warmup_steps=0, total_steps=10)
        group_params = get_optimizer_grouped_parameters(model=self.model)
        self.optimizer = FP32StateAdamWeightDecay(params=group_params,
                                                  beta1=0.9,
                                                  beta2=0.95,
                                                  eps=1.e-6,
                                                  weight_decay=0.1,
                                                  learning_rate=self.lr_schedule)
        self.callback = MFLossMonitor(learning_rate=self.lr_schedule, origin_epochs=1, dataset_size=10)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_set_tensorboard(self):
        """
        Feature: Tensorboard.
        Description: Test Tensorboard functional
        Expectation: AssertionError
        """
        task_trainer = Trainer(task='text_generation',
                               model=self.model,
                               args=self.config,
                               train_dataset=self.train_dataset,
                               callbacks=self.callback,
                               optimizers=self.optimizer
                               )
        task_trainer.train()
        self.tensorboard_dir = os.path.join(self.tensorboard_dir, 'rank_0')
        assert os.path.exists(self.tensorboard_dir),\
            f"TensorBoard directory {self.tensorboard_dir} does not exist."
        event_files = [f for f in os.listdir(self.tensorboard_dir) if f.startswith('events')]
        assert event_files, "No event files found in the TensorBoard directory."
        event_file_path = os.path.join(self.tensorboard_dir, event_files[0])
        file_size = os.path.getsize(event_file_path)
        assert file_size > 40, f"Expected event file size greater than 40 bytes, but got {file_size} bytes."
        text_map = get_text_mapping()
        assert text_map.issubset(_CHECK_TEXT_MAPPING), \
            (f"Text information written is incorrect. The following keys are not found in _CHECK_TEXT_MAPPING: "
             f"{text_map - _CHECK_TEXT_MAPPING}. Please check and modify _CHECK_TEXT_MAPPING, and also review and "
             f"update the documentation: https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/"
             f"source_zh_cn/function/tensorboard.md.")
