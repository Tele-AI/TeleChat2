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
"""LLaMA models' APIs."""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import mint
from mindspore import Tensor, ops
from mindspore.communication import get_group_size

from mindformers.experimental.infer.core.layers import ColumnParallelLinear
from mindformers.experimental.infer.core.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_info, initialize_model_parallel
from mindformers.models.llama.llama import LlamaPreTrainedModel
from mindformers.modules import Linear
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode, get_disable_custom_fa

from .utils import convert_model_config

__all__ = ["ParallelLlamaForCausalLM"]


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ParallelLlamaForCausalLM(LlamaPreTrainedModel):
    r"""
    Provide llama training loss or logits through network.

    Args:
        config (LlamaConfig): The config of llama model.

    Returns:
        output: Tensor, the output of llama decoderlayer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        if get_group_info('tp').group is None:
            initialize_model_parallel(get_group_size(), order='tp')
        self.config = convert_model_config(config)
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.disable_custom_fa = get_disable_custom_fa()

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.slice = ops.StridedSlice()
        self.not_equal = ops.NotEqual()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.ones = ops.Ones()
        self.gather = ops.Gather(1)
        self.prefill_gather_flatten = ops.Gather()
        self.sub_batch_valid_len = ops.Sub()
        self.model = ParallelTransformer(config=config)
        if config.parallel_config.vocab_emb_dp:
            self.lm_head = Linear(
                in_channels=config.hidden_size,
                out_channels=config.vocab_size,
                weight_init="normal",
                has_bias=False,
                param_init_type=config.param_init_type,
                compute_dtype=config.compute_dtype
            )
        else:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                config=config.parallel_config,
                bias=False,
                gather_output=True,
                param_init_type=config.param_init_dtype,
                compute_dtype=config.compute_dtype,
            )

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

        self.use_past = config.use_past

    def prepare_inputs_for_prefill_flatten(self, input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]
        input_ids_bs = input_ids.shape[0]
        if batch_valid_length_bs == input_ids_bs and batch_valid_length_bs > 1:
            input_ids_list = []
            for i in range(batch_valid_length_bs):
                context_len = batch_valid_length[i]
                input_ids_list.append(input_ids[i][:context_len])
            input_ids = np.concatenate(input_ids_list, 0)
            input_ids = input_ids.reshape((1, -1))
            slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        return model_inputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {}
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        model_inputs["input_ids"] = Tensor.from_numpy(
            input_ids.astype(np.int32))

        prefill = kwargs.get("prefill")
        if self.disable_custom_fa and prefill:
            batch_valid_length = kwargs.get("valid_length_each_example")
            slot_mapping = kwargs.get("slot_mapping")
            model_inputs = self.prepare_inputs_for_prefill_flatten(input_ids, batch_valid_length, slot_mapping,
                                                                   model_inputs)
        return model_inputs

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None, None)

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None):
        """
        Forward of llama model.
        """
        bsz, _ = self.shape(input_ids)
        if batch_valid_length is not None:
            batch_valid_length = batch_valid_length.reshape(-1,)
        else:
            batch_valid_length = self.ones((bsz,), mstype.int32)
        output = self.model(input_ids, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping, prefix_keys_values)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            if self.disable_custom_fa:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = self.prefill_gather_flatten(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
            else:
                output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        logits = self.cast(logits, mstype.float32)
        if self.predict_run_mode:
            return self.reshape(logits, (-1, logits.shape[-1]))
        input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
        return logits, input_ids, input_mask

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
