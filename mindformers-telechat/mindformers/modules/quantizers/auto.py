#  Copyright 2024 HuggingFace Inc. team.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Auto Quantizers Class."""
from typing import Dict, Union

from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.utils.quantization_config import (
    QuantizationConfigMixin,
    PtqConfig,
    RtnConfig,
    SmoothQuantConfig,
)
from mindformers.modules.quantizers.ptq_quantizer import PtqQuantizer
from mindformers.modules.quantizers.rtn_quantizer import RtnQuantizer
from mindformers.modules.quantizers.smooth_quant_quantizer import SmoothQuantQuantizer


AUTO_QUANTIZER_MAPPING = {
    "ptq": PtqQuantizer,
    "rtn": RtnQuantizer,
    "smooth_quant": SmoothQuantQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "ptq": PtqConfig,
    "rtn": RtnConfig,
    "smooth_quant": SmoothQuantConfig,
}

__all__ = [
    "AutoQuantizationConfig",
    "AutoQuantizer"]


class AutoQuantizationConfig:
    """
    The Auto quantization config class that takes care of automatically dispatching to the correct
    gold stick quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: Dict):
        """instantiate quantization config from given dict automatically"""
        quant_method = quantization_config_dict.get("quant_method", None)

        if quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. "
                "Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )
        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """instantiate quantization config from given model name or path to model config automatically"""
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. "
                f"Make sure that the model is correctly quantized."
            )
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)

        return quantization_config


class AutoQuantizer:
    """
     The Auto quantizer class that takes care of automatically instantiating to the correct
    `Quantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        """instantiate quantizer from given quantization config automatically"""
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )
        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """instantiate quantizer from given model name or path to model config automatically"""
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)
