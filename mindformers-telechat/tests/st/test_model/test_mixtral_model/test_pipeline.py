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
Test module for testing the mixtral interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_mixtral_model/test_pipeline.py
"""
import mindspore as ms

from mindformers import pipeline
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.modules.transformer.moe import MoEConfig

ms.set_context(mode=0)


class TestMixtralPipelineMethod:
    """A test class for testing pipeline."""
    def setup_method(self):
        """setup method."""
        self.test_llm_list = ['mixtral_8x7b']

    def test_pipeline(self):
        """
        Feature: pipeline.
        Description: Test pipeline by input model type.
        Expectation: TypeError, ValueError, RuntimeError
        """
        moe_config = MoEConfig(expert_num=8,
                               capacity_factor=1.1,
                               aux_loss_factor=0.05,
                               routing_policy="TopkRouterV2",
                               enable_sdrop=True)
        model_config = LlamaConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512, moe_config=moe_config)
        model = LlamaForCausalLM(model_config)
        task_pipeline = pipeline(task='text_generation', model=model, max_length=20)
        task_pipeline("hello!", top_k=3)
