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
Test module for testing the transformer_config and convert_to used for mindformers.
How to run this:
    pytest tests/st/test_static_distri_core/test_transformer_config/test_transformer_config.py
"""
import pytest

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config


class ParallelConfig:
    """A test config class for testing TransformerConfig"""
    def __init__(self,
                 data_parallel: int = 1,
                 model_parallel: int = 1,
                 context_parallel: int = 1,
                 vocab_emb_dp: bool = True):
        super(ParallelConfig, self).__init__()
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.context_parallel = context_parallel
        self.vocab_emb_dp = vocab_emb_dp


class TestConfig(PretrainedConfig):
    """A test config class for testing TransformerConfig"""
    def __init__(self,
                 vocab_size: int = 1,
                 hidden_size: int = 1,
                 a: int = 0,
                 parallel_config: ParallelConfig = ParallelConfig(2, 2, 2),
                 ):
        super(TestConfig, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.a = a
        self.parallel_config = parallel_config


class TestTransformerConfig:
    """A test class for testing TransformerConfig"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_transformer_config(self):
        """
        Feature: TransformerConfig
        Description: Test TransformerConfig
        Exception: AssertionError
        """
        config = TestConfig(vocab_size=2, hidden_size=768)
        transformer_config = TransformerConfig()
        assert transformer_config.hidden_size != config.hidden_size
        assert transformer_config.ffn_hidden_size == 4 * transformer_config.hidden_size
        assert not hasattr(transformer_config, "vocab_size")
        assert transformer_config.padded_vocab_size != config.vocab_size
        assert not hasattr(transformer_config, "a")
        assert transformer_config.tensor_parallel != config.parallel_config.model_parallel
        transformer_config = convert_to_transformer_config(config, transformer_config)
        assert transformer_config.hidden_size == config.hidden_size
        assert transformer_config.ffn_hidden_size == 4 * transformer_config.hidden_size
        assert not hasattr(transformer_config, "vocab_size")
        assert transformer_config.padded_vocab_size == config.vocab_size
        assert transformer_config.a == config.a
        assert transformer_config.tensor_parallel == config.parallel_config.model_parallel
