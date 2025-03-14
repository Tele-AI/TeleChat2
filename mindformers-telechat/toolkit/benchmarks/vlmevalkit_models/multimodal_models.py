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
"""MindFormers model abstract instance."""
from dataclasses import dataclass
import numpy as np

import mindspore as ms

from mindformers import build_context, logger, GenerationConfig
from mindformers import AutoModel, AutoConfig, AutoTokenizer, AutoProcessor
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.model_runner import register_auto_class


@dataclass
class ModelOutput:
    """Return some parameters of the initialized model."""
    config: any
    model_config: any
    generation_config: any
    processor: any
    tokenizer: any
    model: any
    batch_size: int


def init_model(model_path, config_path):
    """Init models."""
    config = MindFormerConfig(config_path)
    # register to Auto Class
    register_auto_class(config, model_path, class_type="AutoConfig")
    register_auto_class(config, model_path, class_type="AutoTokenizer")
    register_auto_class(config, model_path, class_type="AutoModel")
    register_auto_class(config, model_path, class_type="AutoProcessor")

    build_context(config)
    logger.info(f"Build context finished.")

    model_config = AutoConfig.from_pretrained(config_path)
    if not hasattr(model_config, "max_position_embedding") or not model_config.max_position_embedding:
        model_config.max_position_embedding = model_config.seq_length

    generation_config = GenerationConfig.from_model_config(model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    logger.info(f"Build tokenizer finished.")

    model = AutoModel.from_config(model_config)
    logger.info(f"Build model finished.")

    processor = AutoProcessor.from_pretrained(config_path, trust_remote_code=True, use_fast=True)
    logger.info(f"Build processor finished.")

    batch_size = 1
    ms_model = ms.Model(model)
    seq_length = model_config.seq_length
    input_ids = np.ones(shape=tuple([batch_size, seq_length]))
    inputs = model.prepare_inputs_for_predict_layout(input_ids)
    transform_and_load_checkpoint(config, ms_model, model, inputs, do_predict=True)
    logger.info(f"Load checkpoints finished.")
    return ModelOutput(config=config, model_config=model_config, generation_config=generation_config,
                       processor=processor, tokenizer=tokenizer, model=model, batch_size=batch_size)
