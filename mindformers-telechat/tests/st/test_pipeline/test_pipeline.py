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
Test module for testing pipeline function.
How to run this:
pytest tests/st/test_pipeline/test_pipeline.py
"""
import mindspore as ms

from mindformers import AutoModel, AutoTokenizer
from mindformers import TextGenerationPipeline, pipeline

ms.set_context(mode=0)


class TestPipeline:
    """A test class for testing pipeline features."""
    def setup_method(self):
        """setup method."""
        self.task_name = "text_generation"
        self.model_name = "gpt2"
        self.lora_model_name = "gpt2_lora"
        self.batch_size = 1
        self.use_past = True

    def test_text_generation(self):
        """
        Feature: text_generation pipeline.
        Description: Test basic function of text_generation pipeline.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question = "An increasing sequence: one,"
        model = AutoModel.from_pretrained(self.model_name, use_past=self.use_past)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        task_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        output = task_pipeline(question,
                               max_new_tokens=32,
                               do_sample=False)
        print(output)
        assert "An increasing sequence: one, two," in output[0]['text_generation_text'][0]

    def test_pipeline(self):
        """
        Feature: pipeline interface.
        Description: Test basic function of pipeline api.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question = "An increasing sequence: one,"
        task_pipeline = pipeline(self.task_name,
                                 self.model_name,
                                 use_past=self.use_past)
        output = task_pipeline(question,
                               max_new_tokens=32,
                               do_sample=False)
        print(output)
        assert "An increasing sequence: one, two," in output[0]['text_generation_text'][0]

    def test_lora_pipeline(self):
        """
        Feature: pipeline interface.
        Description: Test basic function of pipeline api.
        Expectation: TypeError, ValueError, RuntimeError
        """
        question = "An increasing sequence: one,"
        task_pipeline = pipeline(self.task_name,
                                 self.lora_model_name,
                                 use_past=self.use_past)
        output = task_pipeline(question,
                               max_new_tokens=32,
                               do_sample=False)
        print(output)
        assert "An increasing sequence: one, two," in output[0]['text_generation_text'][0]


class TestPipelineExperiment:
    """A test class for testing experiment pipeline features."""
    def setup_method(self):
        """setup method."""
        self.task_name = "text_generation"
        self.model = "MindSpore-Lab/baichuan2_7b_chat"

    def test_pipeline(self):
        pipeline_task = pipeline(task=self.task_name,
                                 model=self.model,
                                 model_kwargs={"use_past": True},
                                 framework='ms',
                                 trust_remote_code=True)
        pipeline_result = pipeline_task("<reserved_106>你是谁？<reserved_107>", do_sample=False)
        assert "百川" in pipeline_result[0]['text_generation_text'][0]
