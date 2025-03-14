# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Qwen2 models' APIs."""
import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.experimental.models.qwen2.modeling_qwen2_infer import InferenceQwen2ForCausalLM

__all__ = ['Qwen2ForCausalLM']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Qwen2ForCausalLM:
    r"""
    Provide Qwen2 Model for training and inference.
    Args:
        config (Qwen2Config): The config of Qwen2 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get run mode to init different model.

        Args:
            config (Qwen2Config): The config of qwen2 model.

        Raises:
            NotImplementedError: Train mode is not supported for Qwen2 model.

        Returns:
            Tensor, the loss or logits of the network

        """
        if os.environ.get("RUN_MODE", "predict") == "predict":
            return InferenceQwen2ForCausalLM(config=config)
        raise NotImplementedError("Train mode is not supported for Qwen2 model.")
