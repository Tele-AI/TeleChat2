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
Test module for testing the chat interface used for mindformers.
How to run this:
pytest tests/st/test_text_generator/test_chat.py
"""
import os
import pytest

from mindspore import context

from mindformers import AutoModel

from tests.utils.model_tester import create_tokenizer


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestChat:
    """A test class for testing chat interface."""

    def setup_method(self):
        """setup method."""
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        context.set_context(mode=0, jit_config={"jit_level": "O0", "infer_boost": "on"})
        os.environ['RUN_MODE'] = 'predict'

        self.model = AutoModel.from_pretrained("llama2_7b",
                                               num_layers=2,
                                               seq_length=100,
                                               use_past=True,
                                               download_checkpoint=False)
        self.tokenizer = create_tokenizer()

    @pytest.mark.run(order=1)
    def test_chat(self):
        """
        Feature: chat.
        Description: Test chat.
        """
        _, _ = self.model.chat(tokenizer=self.tokenizer, query="hello")
