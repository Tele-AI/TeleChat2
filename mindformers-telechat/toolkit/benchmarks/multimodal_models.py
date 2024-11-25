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
from functools import partial
import numpy as np

from vlmeval.vlm.base import BaseModel

import mindspore as ms
from mindformers import build_context, logger, GenerationConfig
from mindformers import AutoModel, AutoConfig, AutoTokenizer, AutoProcessor
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.model_runner import register_auto_class


def get_model(args):
    mindformers_series = {}
    for index, model_name in enumerate(args.model):
        model_path = args.model_path[index]
        config_path = args.config_path[index]
        mindformers_series[model_name] = partial(MFModel, model_path, config_path)
    return mindformers_series


class MFModel(BaseModel):
    """A base class for MindFormers multimodal large models used for evaluation,
    and other classes can inherit this class to complete the evaluation.

    Args:
        model_path(str): The path containing the model configuration file as well as other model-related files.
        config_path(str): The path of the model configuration file ending with ".yaml".
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, config_path):
        self.config = MindFormerConfig(config_path)
        # register to Auto Class
        register_auto_class(self.config, model_path, class_type="AutoConfig")
        register_auto_class(self.config, model_path, class_type="AutoTokenizer")
        register_auto_class(self.config, model_path, class_type="AutoModel")
        register_auto_class(self.config, model_path, class_type="AutoProcessor")

        build_context(self.config)
        logger.info(f"Build context finished.")

        self.model_config = AutoConfig.from_pretrained(config_path)
        if not hasattr(self.model_config, "max_position_embedding") or not self.model_config.max_position_embedding:
            self.model_config.max_position_embedding = self.model_config.seq_length

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        logger.info(f"Build tokenizer finished.")

        self.model = AutoModel.from_config(self.model_config)
        logger.info(f"Build model finished.")

        self.processor = AutoProcessor.from_pretrained(config_path, trust_remote_code=True, use_fast=True)
        logger.info(f"Build processor finished.")

        self.batch_size = 1
        ms_model = ms.Model(self.model)
        seq_length = self.model_config.seq_length
        input_ids = np.ones(shape=tuple([self.batch_size, seq_length]))
        inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(self.config, ms_model, self.model, inputs, do_predict=True)
        logger.info(f"Load checkpoints finished.")

    def update_config(self):
        if self.config.trainer.model_name == "cogvlm2-image-llama3-chat":
            self.generation_config.max_new_tokens = 2048
            self.generation_config.max_length = 4096
            self.generation_config.do_sample = False
        else:
            raise ValueError(f'This model {self.config.trainer.model_name} is currently not supported. ')

    # pylint: disable=W0613
    def generate_inner(self, message, dataset=None):
        """Perform model predict and return predict results.

        Args:
            message (List[Dict]): Chat text.
            dataset (str): Dataset name.

        Returns:
            Predict result.
        """
        self.update_config()

        tmp_inputs = []
        for s in message:
            input_data = {}
            if s.get('type') == 'image':
                input_data["image"] = s.get("value")
            elif s.get('type') == 'text':
                input_data["text"] = s.get("value")
            tmp_inputs.append(input_data)
        inputs = tmp_inputs * self.batch_size

        # Perform model predict and process the data for predict.
        data = self.processor(inputs)
        res = self.model.generate(
            **data,
            generation_config=self.generation_config,
        )
        input_id_length = np.max(np.argwhere(data.get("input_ids")[0] != self.tokenizer.pad_token_id)) + 1
        result = self.tokenizer.decode(res[0][input_id_length:], skip_special_tokens=True)
        return result
