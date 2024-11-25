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
Test module for testing the qwenvl interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_qwenvl_model/test_training_precision.py
"""
import os
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset

MFPATH = os.path.abspath(__file__)
MFPATH = os.path.abspath(MFPATH + '/../../../../../')
sys.path.append(MFPATH)
sys.path.append(MFPATH + '/research/qwenvl')

# pylint: disable=C0413
import mindformers as mf
from mindformers import CosineWithWarmUpLR, FP32StateAdamWeightDecay
from mindformers import Trainer, MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers.trainer.optimizer_grouped_parameters import get_optimizer_grouped_parameters

from research.qwenvl.qwenvl import QwenVL
from research.qwenvl.qwenvl_config import QwenVLConfig
from research.qwenvl.qwenvl_processor import QwenVLImageProcessor, QwenVLProcessor
from research.qwenvl.qwenvl_tokenizer import QwenVLTokenizer
from research.qwenvl.qwenvl_transform import QwenVLTransform
from research.qwenvl.qwen.optim import AdamWeightDecayX
from research.qwenvl.qwen.qwen_model import QwenForCausalLM
from research.qwenvl.qwen.qwen_config import QwenConfig
from tests.st.training_checker import TrainingChecker

ms.set_context(mode=0)

def register_modules():
    """register modules"""
    MindFormerRegister.register_cls(AdamWeightDecayX, MindFormerModuleType.OPTIMIZER)
    MindFormerRegister.register_cls(QwenVL, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenForCausalLM, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLTokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(QwenVLTransform, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(QwenVLProcessor, MindFormerModuleType.PROCESSOR)
    MindFormerRegister.register_cls(QwenVLImageProcessor, MindFormerModuleType.PROCESSOR)

def generator_train():
    """train dataset generator"""
    seq_len = 513
    img_size = 448
    data_size = 160
    vocab_size = 151936
    input_ids = np.random.randint(low=0, high=vocab_size, size=(data_size, seq_len)).astype(np.int32)
    images = np.random.uniform(low=-2, high=2, size=(data_size, 1, 3, img_size, img_size))
    image_context_pos = np.array([[0, i] for i in range(20, 276)]).reshape((1, -1, 2))
    labels = np.ones_like(input_ids)
    for idx in range(data_size):
        yield input_ids[idx], images[idx], image_context_pos, labels[idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestQwenVLTrainingPrecision:
    """A test class for testing training precision."""

    def setup_method(self):
        """init task trainer."""
        set_seed(0)
        np.random.seed(0)

        register_modules()
        train_dataset = GeneratorDataset(generator_train,
                                         column_names=["input_ids", "images", "image_context_pos", "labels"])
        train_dataset = train_dataset.batch(batch_size=4)

        vision_model_config = {
            'arch': {'type': 'QwenVLVisionModel'},
            'model_config': {
                'type': 'QwenVLVisionConfig',
                'num_hidden_layers': 2,
            },
        }
        vision_model = MindFormerConfig(**vision_model_config)
        llm_model_config = {
            'arch': {'type': 'QwenForCausalLM'},
            'model_config': {
                'type': 'QwenConfig',
                'num_layers': 2,
                'seq_length': 512,
                'vocab_size': 151936,
                'intermediate_size': 11008,
                'enable_slice_dp': False,
                'embedding_parallel_optimizer': False,
                'rms_norm_eps': 1.0e-6,
                'emb_dropout_prob': 0.0,
                'eos_token_id': 151643,
                'pad_token_id': 151643,
                'ignore_token_id': -100,
                'rotary_dtype': "float16",
                'use_flash_attention': True,
                'is_dynamic': True,
                'num_blocks': 128,
                'top_k': 0,
                'top_p': 0.8,
                'do_sample': False,
                'enable_emb_opt': True,
                'rotary_pct': 1.0,
                'rotary_emb_base': 10000,
                'kv_channels': 128,
            }
        }
        llm_model = MindFormerConfig(**llm_model_config)

        model_config = dict(
            vision_model=vision_model,
            llm_model=llm_model,
            freeze_vision=True,
            freeze_resampler=False,
            freeze_llm=False,
            compute_dtype="bfloat16",
            param_init_type="float16",
            softmax_compute_type="float32",
            is_dynamic=True,
            block_size=32,
            num_blocks=128,
        )
        model = QwenVL(QwenVLConfig(**model_config))

        config = self.get_config(model_config)
        lr_schedule = CosineWithWarmUpLR(learning_rate=1.e-5, warmup_ratio=0.01, warmup_steps=0, total_steps=20)
        group_params = get_optimizer_grouped_parameters(model=model)
        optimizer = FP32StateAdamWeightDecay(params=group_params,
                                             beta1=0.9,
                                             beta2=0.95,
                                             eps=1.e-8,
                                             weight_decay=0.1,
                                             learning_rate=lr_schedule)

        loss_list_std = [11.313704, 11.313704, 10.204470, 10.204470,
                         9.254555, 9.254555, 9.307570, 9.307570,
                         8.937981, 8.937981, 9.355089, 9.355089,
                         8.983348, 8.983348, 9.248279, 9.248279,
                         9.021189, 9.021189, 9.221209, 9.221209,
                         9.226854, 9.226854, 8.972058, 8.972058,
                         9.148109, 9.148109, 8.856930, 8.856930,
                         8.892754, 8.892754, 9.076948, 9.076948,
                         9.212030, 9.212030, 8.934267, 8.934267,
                         9.510830, 9.510830, 9.303289, 9.303289]
        callback = TrainingChecker(loss_list_std=loss_list_std)

        self.task_trainer = Trainer(task='multi_modal_to_text_generation',
                                    model=model,
                                    args=config,
                                    train_dataset=train_dataset,
                                    callbacks=callback,
                                    optimizers=optimizer)

    def get_config(self, model_config):
        """init config for training"""
        config = {
            'trainer': {
                'type': 'MultiModalToTextGenerationTrainer',
                'model_name': 'qwenvl'
            },
            'train_dataset': {
                'tokenizer': {
                    'type': 'QwenVLTokenizer',
                    'vocab_file': "./checkpoint_download/qwenvl/qwen.tiktoken"
                }
            },
            'train_dataset_task': {},
            'runner_config': {
                'epochs': 1,
                'batch_size': 4,
                'sink_mode': True,
                'sink_size': 2
            },
            'runner_wrapper': {
                'type': 'MFTrainOneStepCell',
                'scale_sense': {
                    'type': 'DynamicLossScaleUpdateCell',
                    'loss_scale_value': 64,
                    'scale_factor': 2,
                    'scale_window': 1000
                },
                'use_clip_grad': True,
            },
            'parallel': {},
            'model': {'model_config': model_config},
            'micro_batch_interleave_num': 1,
            'callbacks': [{'type': 'MFLossMonitor'}]

        }
        config = MindFormerConfig(**config)
        if not os.path.exists("./checkpoint_download/qwenvl/qwen.tiktoken"):
            mf.tools.download_tools.download_with_progress_bar(
                'https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwen.tiktoken',
                "./checkpoint_download/qwenvl/qwen.tiktoken"
            )
        return config

    @pytest.mark.run(order=1)
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: AssertionError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.config.callbacks = self.task_trainer.config.callbacks[:1]
        self.task_trainer.train()
