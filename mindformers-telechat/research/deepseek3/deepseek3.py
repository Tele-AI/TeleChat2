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
"""DeepseekV3 models' APIs."""
import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from deepseek3_model_train import TrainingDeepseekV3ForCausalLM
from deepseek3_model_infer import InferenceDeepseekV3ForCausalLM

__all__ = ['DeepseekV3ForCausalLM']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class DeepseekV3ForCausalLM:
    r"""
    Provide DeepseekV3 Model for training and inference.
    Args:
        config (DeepseekV3Config): The config of DeepseekV3 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config, *args, **kwargs):
        # get run mode to init different model.
        # predict mode used to deploy.
        # when predict mode not supported, we can use online_predict mode to do inference task.
        if os.environ.get("RUN_MODE") == "predict":
            return InferenceDeepseekV3ForCausalLM(config=config)
        return TrainingDeepseekV3ForCausalLM(config=config, *args, **kwargs)
