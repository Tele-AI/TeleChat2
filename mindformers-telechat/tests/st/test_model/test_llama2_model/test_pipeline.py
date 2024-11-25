# Copyright 2023 Huawei Technologies Co., Ltd
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
Test module for testing the llama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llama_model/test_pipeline.py
"""
import mindspore as ms

from mindformers import pipeline

ms.set_context(mode=0)


class TestLlamaPipelineMethod:
    """A test class for testing pipeline."""
    def setup_method(self):
        """setup method."""
        self.test_llm_list = ['llama_7b']

    def test_pipeline(self):
        """
        Feature: pipeline.
        Description: Test pipeline by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        for model_type in self.test_llm_list:
            task_pipeline = pipeline(task='text_generation', model=model_type, max_length=20)
            task_pipeline("hello!", top_k=3)
