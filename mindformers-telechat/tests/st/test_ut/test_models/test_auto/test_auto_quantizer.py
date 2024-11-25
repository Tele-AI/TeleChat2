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
"""test AutoQuantizer."""

import pytest

from mindformers.utils.quantization_config import PtqConfig
from mindformers.modules.quantizers import AutoQuantizationConfig, AutoQuantizer
from mindformers.modules.quantizers.rtn_quantizer import RtnQuantizer
from mindformers.utils.quantization_config import QuantizationConfigMixin

MOCK_PTQ_QUANT_W8A16_DICT = {'quant_method': 'ptq', 'weight_dtype': 'int8', 'activation_dtype': 'None',
                             'kvcache_dtype': 'None', 'modules_to_not_convert': ['lm_head'], 'algorithm_args': {},
                             'outliers_suppression': 'None'}
PTQ_QUANT_W8A16_MODEL = "llama2_13b_w8a16"
NOT_QUANT_MODEL = "llama2_13b"


class TestAutoQuantizer:
    """A test class for testing AutoQuantizer/AutoQuantizationConfig."""

    def test_auto_quantizer_config_from_dict(self):
        """test init AutoQuantizationConfig from yaml."""
        config = AutoQuantizationConfig.from_dict(MOCK_PTQ_QUANT_W8A16_DICT)
        assert isinstance(config, QuantizationConfigMixin)

    def test_auto_quantizer_config_from_dict_with_none_quant_method(self):
        """test exception init AutoQuantizationConfig from yaml."""
        MOCK_PTQ_QUANT_W8A16_DICT.pop('quant_method')
        with pytest.raises(ValueError):
            AutoQuantizationConfig.from_dict(MOCK_PTQ_QUANT_W8A16_DICT)

    def test_auto_quantizer_config_from_dict_with_unsupported_quant_method(self):
        """test exception init AutoQuantizationConfig from yaml."""
        MOCK_PTQ_QUANT_W8A16_DICT['quant_method'] = 'unsupported method'
        with pytest.raises(ValueError):
            AutoQuantizationConfig.from_dict(MOCK_PTQ_QUANT_W8A16_DICT)

    def test_auto_quantizer_config_from_pretrained(self):
        """test init AutoQuantizationConfig from model name."""
        config = AutoQuantizationConfig.from_pretrained(PTQ_QUANT_W8A16_MODEL)
        assert isinstance(config, PtqConfig)

    def test_auto_quantizer_config_from_pretrained_with_unsupported_model(self):
        """test init AutoQuantizationConfig from model name."""
        with pytest.raises(ValueError):
            AutoQuantizationConfig.from_pretrained(NOT_QUANT_MODEL)

    def test_auto_quantizer_from_config(self):
        """test init AutoQuantizer from config."""
        config = AutoQuantizationConfig.from_pretrained(PTQ_QUANT_W8A16_MODEL)
        quantizer = AutoQuantizer.from_config(config)
        assert isinstance(quantizer, RtnQuantizer)

    def test_auto_quantizer_from_config_with_unsupported_quant_method(self):
        """test exception init AutoQuantizer from config."""
        config = AutoQuantizationConfig.from_pretrained(PTQ_QUANT_W8A16_MODEL)
        config.quant_method = 'unsupport method'
        with pytest.raises(ValueError):
            AutoQuantizer.from_config(config)

    def test_auto_quantizer_from_from_pretrained(self):
        """test init AutoQuantizer from model name."""
        quantizer = AutoQuantizer.from_pretrained(PTQ_QUANT_W8A16_MODEL)
        assert isinstance(quantizer, RtnQuantizer)
