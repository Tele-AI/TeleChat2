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
from mindspore import Tensor, Parameter, mint
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, Constant

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
        self.gather = P.Gather()
        bs = self.config.batch_size
        self.group_list_buffer = np.empty((config.lora_num,))
        self.group_list = Parameter(initializer(Constant(bs), [config.lora_num], mstype.int64), requires_grad=False)
        self.head_group_list = Parameter(initializer(Constant(bs), [config.lora_num], mstype.int64),
                                         requires_grad=False)
        self.lora_inputs = {"head_group_list": self.head_group_list,
                            "group_list": self.group_list}
        # add slora layer.
        self.lora_adapter = SLoraAdapter(self.config.pet_config, self.lora_inputs)
        self.lora_model = self.add_adapter(base_model)

    def add_adapter(self, base_model: PreTrainedModel):
        """Add adapter for layers."""
        base_model.model = self.lora_adapter.get_pet_model(base_model)
        return base_model

    def _check_config(self):
        if self.config.pet_config.target_modules is None:
            raise ValueError(f"No target modules for lora layer.")

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.lora_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_slora(self, adapter_ids_np):
        """prepare lora inputs for slora layers"""
        sort_map = np.argsort(adapter_ids_np[0])
        unsort_map = Tensor.from_numpy(np.argsort(sort_map).astype(np.int32))
        sort_map_flatten = Tensor.from_numpy(np.argsort(adapter_ids_np[1]).astype(np.int32))
        adapter_mask = np.eye(self.config.pet_config.lora_num)[adapter_ids_np[1]]
        adapter_cnt = np.sum(adapter_mask, axis=0)
        adapter_group_list = np.cumsum(adapter_cnt, dtype=np.int64)
        mask = mint.index_select(Tensor.from_numpy(np.array(adapter_ids_np[0], dtype=np.int64)), 0,
                                 Tensor.from_numpy(sort_map.astype(np.int32)))
        head_adapter_mask = np.eye(self.config.pet_config.lora_num)[mask.asnumpy()]
        head_adapter_cnt = np.sum(head_adapter_mask, axis=0)
        head_group_list = np.cumsum(head_adapter_cnt, dtype=np.int64)

        if (adapter_group_list != self.group_list_buffer).any():
            P.Assign()(self.head_group_list, Tensor.from_numpy(np.array(head_group_list, dtype=np.int64)))
            P.Assign()(self.group_list, Tensor.from_numpy(np.array(adapter_group_list, dtype=np.int64)))
            self.group_list_buffer = adapter_group_list
        return sort_map, unsort_map, sort_map_flatten

    def prepare_adapter_ids_for_slora(self, adapter_ids, batch_valid_length, prefill):
        """prepare adapter ids for slora layers"""
        batch_size = batch_valid_length.shape[0]
        adapter_ids_list = [0] * batch_size
        adapter_ids_flatten = []
        slora_names = self.lora_adapter.adapter_names
        for batch in range(batch_size):
            if adapter_ids is not None:
                if len(adapter_ids) != batch_size:
                    raise ValueError("adapter_ids has different length with inputs.")
                adapter = adapter_ids[batch]
                if adapter in slora_names:
                    adapter_ids_list[batch] = slora_names.index(adapter)
                elif adapter is None:
                    logger.warning(f"SLoRA adapter id got none for batch {batch}, use base model without SLoRA.")
                else:
                    logger.warning(f"Can not find {adapter} in registered adapter names for batch {batch}, "
                                   f"use base model without SLoRA, supported adapter list:{slora_names}")
            else:
                logger.warning(f"SLoRA adapter ids got none, use base model without SLoRA.")
            seq_len = batch_valid_length[batch] if prefill else 1
            adapter_ids_flatten += [adapter_ids_list[batch]] * seq_len
        return adapter_ids_list, adapter_ids_flatten

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        adapter_ids = kwargs.get("adapter_ids")
        batch_valid_length = kwargs.get("valid_length_each_example")
        prefill = kwargs.get("prefill")
        adapter_ids_np = self.prepare_adapter_ids_for_slora(adapter_ids, batch_valid_length, prefill)
        sort_map, unsort_map, sort_map_flatten = self.prepare_inputs_for_slora(adapter_ids_np)
        model_inputs = self.lora_model.prepare_inputs_for_generation(input_ids, **kwargs)
        model_inputs["revert_ids"] = unsort_map
        if (sort_map != np.arange(batch_valid_length.shape[0])).any():
            sort_dim = 1 if prefill else 0
            model_inputs["input_ids"] = mint.index_select(model_inputs["input_ids"], sort_dim, sort_map_flatten)
            if "block_tables" in model_inputs:
                sort_map = Tensor.from_numpy(sort_map.astype(np.int32))
                model_inputs["block_tables"] = mint.index_select(model_inputs["block_tables"], 0, sort_map)
            if "slot_mapping" in model_inputs:
                model_inputs["slot_mapping"] = mint.index_select(model_inputs["slot_mapping"], 0, sort_map_flatten)
        return model_inputs

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return self.lora_model.prepare_inputs_for_predict_layout(input_ids, **kwargs)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.lora_model.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        parallel_decoding = self.lora_model.parallel_decoding
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        dynamic_position_ids = Tensor(shape=[None, None], dtype=mstype.int32) if parallel_decoding else None
        dynamic_mask = Tensor(shape=[None, None], dtype=mstype.float16) if parallel_decoding else None
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32) if parallel_decoding else None
        dynamic_revert_ids = Tensor(shape=[None], dtype=mstype.int32)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables, dynamic_slot_mapping,
                            dynamic_prefix_keys_values, None, dynamic_q_seq_lens)
        elif self.lora_model.use_past:
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables, dynamic_slot_mapping,
                            None, None, dynamic_q_seq_lens, dynamic_revert_ids)
        elif kwargs.get("pre_gather", False):
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, None, None, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            None, None, None, None, None, None)
        logger.info("Set dynamic input for slora.")

    def to_embeddings(self, tokens):
        return self.lora_model.to_embeddings(tokens)

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                  q_seq_lens=None, revert_ids=None):
        r"""
        LlamaForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor, optional): the tokenized labels with datatype int32,
                Tensor of shape :math:`(batch, seq\_length)`. Default: ``None`` .
            input_position(Tensor, optional): current position, used by model.predict. Default: ``None`` .
            position_ids(Tensor, optional): Reserved param, not used. Default: ``None`` .
            attention_mask(Tensor, optional): Reserved param, not used. Default: ``None`` .
            input_embeds(Tensor, optional): the input embedding Tensor of shape
                :math:`(batch, seq\_length, hidden_size)`. Default: ``None`` .
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction.  Default: ``Tensor([True])`` .
            batch_valid_length(Tensor, optional): the past calculated the index with datatype int32,
                used for incremental prediction. Tensor of shape :math:`(batch_size,)`.  Default: ``None`` .
            block_tables (Tensor[int64], optional): Store mapping tables for each sequence. Default: ``None`` .
            slot_mapping (Tensor[int32], optional): Store token cache physical slot index. Default: ``None`` .
            q_seq_lens (Tensor[int32], optional): In parallel decoding, the query may be flattened.
                The Paged Attention operator need `q_seq_lens` to obtain the length information. Default: ``None`` .
            loss_mask (Tensor, optional): Used to determine whether the corresponding token position participates
                in the loss calculation. If the value is :math:`(1)`, the loss of the position is calculated,
                and :math:`(0)` is not calculated. Default: ``None`` .
            revert_ids (Tensor, optional): Used to recover the order of prediction logits in SLoRA. Default: ``None`` .

        Returns:
            Tensor, The loss or (logits, tokens, input_mask) of the network.
        """
        logits = self.lora_model(input_ids=input_ids,
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
        if revert_ids is not None:
            logits = self.gather(logits, revert_ids, 0)
        return logits
