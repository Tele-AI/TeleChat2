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
Test baichuan2 evaluate.
How to run this:
    pytest tests/st/test_model/test_baichuan2_model/test_eval.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestBaichuan2Eval:
    """A test class for testing model evaluate."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base model evaluate
        Description: Test base model evaluate.
        Expectation: AssertionError
        """
        ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
        runner = ModelTester(run_mode='eval', batch_size=2, use_label=True)

        model_config = get_config()
        model_config.batch_size = runner.batch_size  # set batch size for prediction
        # if set 4096, cause Memory pool not enough by large alibi tensor
        model_config.seq_length = 512
        model_config.vocab_size = 32000  # default to use llama2 tokenizer

        model = get_model(model_config)

        runner.set_eval(model, model_config)
