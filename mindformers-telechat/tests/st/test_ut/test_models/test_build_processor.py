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
"""test build processor."""
from mindformers import MindFormerConfig
from mindformers.models.llama import LlamaTokenizer
from mindformers.models.gpt2 import GPT2Processor, GPT2Tokenizer
from mindformers.models.blip2 import Blip2Processor, Blip2ImageProcessor
from mindformers.models.build_processor import build_processor


class TestBuildProcessor:
    """A test class for testing build_processor() method."""

    def test_build_blip2_processor(self):
        """test build blip2 processor."""
        config = MindFormerConfig("configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml")
        processor = build_processor(config.processor)
        assert isinstance(processor, Blip2Processor)
        assert isinstance(processor.image_processor, Blip2ImageProcessor)
        assert isinstance(processor.tokenizer, LlamaTokenizer)

    def test_build_gpt2_processor(self):
        """test build gpt2 processor."""
        config = MindFormerConfig("configs/gpt2/pretrain_gpt2_small_fp16.yaml")
        processor = build_processor(config.processor)
        assert isinstance(processor, GPT2Processor)
        assert isinstance(processor.tokenizer, GPT2Tokenizer)
