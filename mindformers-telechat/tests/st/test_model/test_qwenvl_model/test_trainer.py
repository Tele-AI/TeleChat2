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
pytest tests/st/test_model/test_qwenvl_model/test_trainer.py
"""
import os
import sys
import pytest

import mindspore as ms

MFPATH = os.path.abspath(__file__)
MFPATH = os.path.abspath(MFPATH + '/../../../../../')
sys.path.append(MFPATH)
sys.path.append(MFPATH + '/research/qwenvl')
# pylint: disable=C0413
import mindformers as mf
from mindformers import Trainer, MindFormerConfig, MindFormerRegister, MindFormerModuleType

from research.qwenvl.qwenvl import QwenVL
from research.qwenvl.qwenvl_config import QwenVLConfig
from research.qwenvl.qwenvl_processor import QwenVLImageProcessor, QwenVLProcessor
from research.qwenvl.qwenvl_tokenizer import QwenVLTokenizer
from research.qwenvl.qwenvl_transform import QwenVLTransform
from research.qwenvl.qwen.optim import AdamWeightDecayX
from research.qwenvl.qwen.qwen_model import QwenForCausalLM
from research.qwenvl.qwen.qwen_config import QwenConfig

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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestQwenVLTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        register_modules()

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
                'max_decode_length': 512
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

        self.task_trainer = Trainer(task='image_to_text_generation', model=model, args=config)

    def get_config(self, model_config):
        """init config for prediction"""
        config = {
            'trainer': {
                'type': 'ImageToTextGenerationTrainer',
                'model_name': 'qwenvl'
            },
            'parallel': {},
            'model': {'model_config': model_config},
            'processor': {
                'type': 'QwenVLProcessor',
                'image_processor': {
                    'type': 'QwenVLImageProcessor',
                    'image_size': 448
                },
                'tokenizer': {
                    'type': 'QwenVLTokenizer',
                    'max_length': 32,
                    'vocab_file': "./checkpoint_download/qwenvl/qwen.tiktoken"
                },
                'max_length': 512
            }
        }
        if not os.path.exists("./checkpoint_download/qwenvl/qwen.tiktoken"):
            mf.tools.download_tools.download_with_progress_bar(
                'https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwen.tiktoken',
                "./checkpoint_download/qwenvl/qwen.tiktoken"
            )
        config = MindFormerConfig(**config)
        return config

    @pytest.mark.run(order=1)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        img_url = 'https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg'
        prompt = 'Describe the image in English:'
        query_item = [{"image": img_url}, {"text": prompt}]
        self.task_trainer.predict(input_data=[query_item], max_length=20)
