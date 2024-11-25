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
"""SLora model for all llm model"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.pet.pet_config import SLoraConfig
from mindformers.pet.tuners.slora_adapter import SLoraAdapter
from mindformers.tools.logger import logger


class SLoraModel(PreTrainedModel):
    """
    SLoRA model for LLM. Provide a flexible and efficient way to adjust and
    optimize pre-trained models by adding SLoRA structures to the base pre-trained models.

    Args:
        config (LoraConfig): slora config, defines SLoRA algorithm.
        base_model (PreTrainedModel): Pre-trained base model for prediction.

    Returns:
        An instance of SLoraModel.
    """

    def __init__(self, config: SLoraConfig, base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.config.pet_config = config
        self._check_config()
        self.lora_list = []
        self.adapter_ids = Parameter(initializer('zero', [self.config.batch_size], mstype.int32), requires_grad=False)
        # add slora layer.
        self.lora_model = self.add_adapter(base_model)

    def add_adapter(self, base_model: PreTrainedModel):
        """Add adapter for layers."""
        slora_adapter = SLoraAdapter(self.config.pet_config, self.adapter_ids)
        if hasattr(base_model, "backbone"):
            base_model.backbone = slora_adapter.get_pet_model(base_model.backbone)
        elif hasattr(base_model, "model"):
            base_model.model = slora_adapter.get_pet_model(base_model.model)
        elif hasattr(base_model, "transformer"):
            base_model.transformer = slora_adapter.get_pet_model(base_model.transformer)
        elif hasattr(base_model, "llm_model"):
            base_model.llm_model = slora_adapter.get_pet_model(base_model.llm_model)
        else:
            logger.warning("The base model must has an attribute named in \'backbone\',"
                           "\'model\', or \'transformer\', which define transformer blocks.")
        self.lora_list = slora_adapter.registered_loras
        return base_model

    def _check_config(self):
        if self.config.pet_config.target_modules is None:
            raise ValueError(f"No target modules for lora layer.")

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.lora_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        adapter_ids = kwargs.get("adapter_ids")
        adapter_ids_np = [0] * batch_size
        if adapter_ids is not None:
            if len(adapter_ids) != batch_size:
                raise ValueError("adapter_ids has different length with inputs.")
            slora_names = SLoraAdapter.adapter_names
            for batch in range(batch_size):
                adapter = adapter_ids[batch]
                if adapter in SLoraAdapter.adapter_names:
                    adapter_ids_np[batch] = slora_names.index(adapter) + 1
                else:
                    logger.warning(f"Can not find {adapter} in registered adapter names, "
                                   f"supported adapter list:{slora_names}")
        else:
            logger.warning(f"SLoRA adapter ids got none.")
        self.adapter_ids.set_data(Tensor.from_numpy(np.array(adapter_ids_np, dtype=np.int32)), slice_shape=True)
        return self.lora_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return self.lora_model.prepare_inputs_for_predict_layout(input_ids, **kwargs)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.lora_model.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        self.adapter_ids.set_data(Tensor(shape=[None], dtype=mstype.int32), slice_shape=True)
        return self.lora_model.set_dynamic_inputs(**kwargs)

    def to_embeddings(self, tokens):
        return self.lora_model.to_embeddings(tokens)

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                  q_seq_lens=None):
        return self.lora_model(input_ids=input_ids,
                               labels=labels,
                               input_position=input_position,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               input_embeds=input_embeds,
                               init_reset=init_reset,
                               batch_valid_length=batch_valid_length,
                               batch_index=batch_index,
                               zactivate_len=zactivate_len,
                               block_tables=block_tables,
                               slot_mapping=slot_mapping,
                               prefix_keys_values=prefix_keys_values,
                               llm_boost_inputs=llm_boost_inputs,
                               q_seq_lens=q_seq_lens)
