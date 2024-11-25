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
Test module for testing the interface used for mindformers.
How to run this:
pytest tests/st/test_auto_register
"""
import os
import pytest

from mindformers.tools import MindFormerConfig
from mindformers.core import build_lr, build_optim, build_metric, build_callback
from mindformers.models import build_model, build_processor
from mindformers.trainer import build_trainer
from mindformers.pipeline import build_pipeline
from mindformers.wrapper import build_wrapper


path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

yaml_path = os.path.join(path, 'st', 'test_auto_register', 'test_auto_register.yaml')
register_path = os.path.join(path, 'st', 'test_auto_register', 'register_path')
os.environ["REGISTER_PATH"] = register_path
test_config = MindFormerConfig(yaml_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_from_config_for_auto_register():
    """
    Feature: Build API from config in using auto_register.
    Description: Test build function to instance API from config.
    Expectation: TypeError.
    """
    lr_schedule = build_lr(test_config.lr_schedule)
    assert isinstance(lr_schedule, object), "instance lr_schedule failed"

    optim = build_optim(test_config.optimizer)
    assert isinstance(optim, object), "instance optimizer failed"

    metric = build_metric(test_config.metric)
    assert isinstance(metric, object), "instance metric failed"

    model = build_model(test_config.model)
    assert isinstance(model, object), "instance model failed"

    runner_wrapper = build_wrapper(test_config.runner_wrapper)
    assert isinstance(runner_wrapper, object), "instance runner_wrapper failed"

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    test_config.processor.tokenizer.vocab_file = f"{root_path}/utils/llama2_tokenizer/tokenizer.model"
    processor = build_processor(test_config.processor)
    assert isinstance(processor, object), "instance processor failed"

    callbacks = build_callback(test_config.callbacks)
    assert isinstance(callbacks, list), "instance callbacks failed"

    pipeline = build_pipeline(test_config.pipeline)
    assert isinstance(pipeline, object), "instance pipeline failed"

    trainer = build_trainer(test_config.trainer)
    assert isinstance(trainer, object), "instance trainer failed"
