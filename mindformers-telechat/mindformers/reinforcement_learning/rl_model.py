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
"""Rl model for llm model."""
from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.reinforcement_learning.constants import RlType
from mindformers.reinforcement_learning.dpo_model import DPOModel
from mindformers.reinforcement_learning.rl_config import DPOConfig, RlConfig
from mindformers.tools.logger import logger

# Mapping of rl_preprocess models.
RL_TYPE_TO_MODEL_MAPPING = {
    RlType.DPO.value: DPOModel
}

# Mapping of rl_preprocess configs.
RL_TYPE_TO_CONFIG_MAPPING = {
    RlType.DPO.value: DPOConfig
}


@args_type_check(config=(dict, RlConfig))
def get_rl_model(base_model: PreTrainedModel, ref_model: PreTrainedModel, config: Union[dict, RlConfig]):
    """
    Get model with rl_preprocess model.

    Args:
        base_model (PreTrainedModel): The pretrained model for rl_preprocess.
        config (RlConfig): The config of rl_preprocess algrithm.

    Return:
        model(PreTrainedModel)
    """
    rl_type = config.get("rl_type")

    if not RL_TYPE_TO_MODEL_MAPPING.get(rl_type):
        logger.warning("%s doesn't have rl_preprocess model currently.", rl_type)
        return base_model

    config = RL_TYPE_TO_CONFIG_MAPPING[rl_type](**config)
    return RL_TYPE_TO_MODEL_MAPPING[rl_type](config, base_model, ref_model)
