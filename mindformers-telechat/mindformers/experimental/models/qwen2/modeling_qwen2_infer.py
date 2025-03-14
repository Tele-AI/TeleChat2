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
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Condition

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, ops
from mindspore.communication import get_group_size
from mindspore.communication._comm_helper import _is_initialized

from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_info, initialize_model_parallel
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.infer.core.norm import get_norm
from mindformers.experimental.infer.core.mlp import MLP, MLPSubmodules
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec
from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.core.self_attention import (
    CoreAttention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.experimental.infer.core.flash_attention import FlashAttention
from mindformers.experimental.infer.core.gpt_model import GPTModel
from mindformers.experimental.models.qwen2.configuration_qwen2 import Qwen2Config

__all__ = ["InferenceQwen2ForCausalLM"]


def get_gpt_layer_spec(config) -> ModuleSpec:
    r"""
    build gpt layer.

    Args:
        config (Qwen2Config): The config of qwen2 model.

    Returns:
        ModuleSpec: gpt layer

    """
    from mindformers.experimental.infer.core.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    self_attn = ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            core_attention=FlashAttention if config.use_flash_attention else CoreAttention,
            linear_proj=RowParallelLinear,
            linear_qkv=ColumnParallelLinear if config.qkv_concat else None,
            linear_q=ColumnParallelLinear if not config.qkv_concat else None,
            linear_k=ColumnParallelLinear if not config.qkv_concat else None,
            linear_v=ColumnParallelLinear if not config.qkv_concat else None
        )
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm(config),
            self_attention=self_attn,
            pre_mlp_layernorm=get_norm(config),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear
                )
            )
        )
    )


class Qwen2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen2Config
    base_model_prefix = "Qwen2"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceQwen2ForCausalLM(Qwen2PreTrainedModel):
    r"""
    Provide qwen2 model infer through network.

    Args:
        config (Qwen2Config): The config of qwen2 model.

    Returns:
        output: Tensor, the output of qwen2 decoderlayer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        if get_group_info('tp').group is None and _is_initialized():
            initialize_model_parallel(get_group_size(), order='tp')
        transformer_config = TransformerConfig()
        self.config = convert_to_transformer_config(config, transformer_config)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.max_position_embeddings = self.config.max_position_embeddings
        self.compute_dtype = self.config.compute_dtype
        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.sub = ops.Sub()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.tp_group_size = get_tp_world_size()
        self.is_prefill = True
        self.model = GPTModel(config=self.config,
                              transformer_layer_spec=get_gpt_layer_spec(self.config),
                              vocab_size=self.vocab_size,
                              rotary_base=self.config.rotary_base)

    def set_dynamic_inputs(self):
        """ dynamic shape"""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_positions = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_context_lens_tensor = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_positions, dynamic_batch_valid_length,
                        dynamic_context_lens_tensor, dynamic_block_tables,
                        dynamic_slot_mapping, None, None, None)
        logger.info("Set dynamic input for qwen2.")

    def add_flags_custom_mcore(self, is_prefill):
        r"""
        Add flag to distinguish fa and pa.

        Args:
            is_prefill: flag to distinguish fa and pa.

        Returns:

        """
        self.add_flags(is_prefill=is_prefill)
        self.model.add_flags(is_prefill=is_prefill)
        self.model.decoder.add_flags(is_prefill=is_prefill)
        self.model.casual_mask.add_flags(is_prefill=is_prefill)
        for layer in self.model.decoder.layers:
            layer.self_attention.flash_attention.add_flags(is_prefill=is_prefill)

    # pylint: disable=W0613
    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None,
                  block_tables=None, slot_mapping=None, kv_cache=None, attention_mask=None, attn_metadata=None):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            kv_cache: key cache and value cache.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata

        Returns:
            logits: the output logits.

        """
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        return logits


    @classmethod
    def convert_name(cls, weight_name):
        r"""
        convert HuggingFace weight name to MindFormers weight name.

        Args:
            weight_name: huggingface weight names.

        Returns:
            weight_name: converted weight names.

        """
        origin_name = weight_name
        weight_name = weight_name.replace('embed_tokens.', 'embedding.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.self_attention.linear_q.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.self_attention.linear_k.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.self_attention.linear_v.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.self_attention.linear_proj.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.mlp.gating.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.mlp.linear_fc2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.mlp.linear_fc1.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.pre_mlp_layernorm.')
        weight_name = weight_name.replace('.norm.', '.decoder.final_norm.')
        weight_name = weight_name.replace('lm_head.', 'model.output_layer.')
        weight_name = weight_name.replace('.embedding.weight', '.embedding.embedding_weight')
        weight_name = weight_name.replace('.layers.', '.decoder.layers.')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        print("weight name: ", weight_name, flush=True)
        return weight_name

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        r"""
        convert HuggingFace weight dict to MindFormers weight dict.

        Args:
            source_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Raises:
            ValueError: value error

        Returns:
            target_dict: converted weight dict.

        """
        model_config = kwargs.get("model_config")
        qkv_concat = model_config.qkv_concat
        target_dict = {}
        wq_keys = []
        wk_keys = []
        wv_keys = []
        w1_keys = []
        w3_keys = []

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'linear_q':
                    wq_keys.append(k)
                if part[-2] == 'linear_k':
                    wk_keys.append(k)
                if part[-2] == 'linear_v':
                    wv_keys.append(k)
                if part[-2] == 'gating':
                    w1_keys.append(k)
                if part[-2] == 'linear_fc1':
                    w3_keys.append(k)

        if qkv_concat:
            qkv_dict = kwargs.get('qkv_dict', None)
            if not isinstance(qkv_dict, DictProxy):
                raise ValueError(f'qkv_queue must be a queue, when qkv_concat is True, but got {qkv_dict}.')
            condition = kwargs.get('condition', None)
            if not isinstance(condition, Condition):
                raise ValueError(f'condition must be a Condition, when qkv_concat is True, but got {condition}.')
            _concat_qkv_weight(wq_keys, wk_keys, wv_keys, model_config, qkv_dict, condition, target_dict)
            _concat_ffn_weight(w1_keys, w3_keys, model_config, qkv_dict, condition, target_dict)

        return target_dict

    @classmethod
    def convert_map_dict(cls, source_dict, **kwargs):
        r"""
        convert HuggingFace map dict to MindFormers map dict.

        Args:
            source_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Returns:
            target_dict: converted weight dict.

        """
        qkv_concat = kwargs.pop("qkv_concat", False)
        target_dict = {}
        wq_keys = []
        w1_keys = []

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'linear_q':
                    wq_keys.append(k)
                if part[-2] == 'gating':
                    w1_keys.append(k)

        if qkv_concat:
            for wq_key in wq_keys:
                wk_key = wq_key.replace('linear_q', 'linear_k')
                wv_key = wq_key.replace('linear_q', 'linear_v')
                wq_value = target_dict.pop(wq_key)
                target_dict.pop(wk_key)
                target_dict.pop(wv_key)

                w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
                w_qkv_value = wq_value
                target_dict.update({w_qkv_key: w_qkv_value})
            for w1_key in w1_keys:
                w3_key = w1_key.replace('gating', 'linear_fc1')
                w1_value = target_dict.pop(w1_key)
                target_dict.pop(w3_key)

                w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
                w_gate_hidden_value = w1_value
                target_dict.update({w_gate_hidden_key: w_gate_hidden_value})

        return target_dict

    @classmethod
    def obtain_qkv_ffn_concat_keys(cls):
        qkv_key = "linear_qkv"
        concat_keys = [qkv_key]
        logger.info(f"{cls.__name__} qkv/ffn concat keys are {concat_keys}")
        return concat_keys

    def clear_kv_cache(self):
        return self.model.clear_kv_cache()


def _concat_qkv_weight(wq_keys, wk_keys, wv_keys, model_config, qkv_dict, condition, target_dict):
    r"""
    concat qkv weight from dicts.

    Args:
        wq_keys: query weight name.
        wk_keys: key weight name.
        wv_keys: value weight name.
        model_config: model config.
        qkv_dict: query, key, value weight dict.
        condition: condition to manager context.
        target_dict: converted weight dict.

    Returns:

    """
    from mindformers.utils.convert_utils import qkv_concat_hf2mg

    num_heads = model_config.num_heads
    n_kv_heads = model_config.n_kv_heads or num_heads
    hidden_size = model_config.hidden_size

    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for wk_key in wk_keys:
        wq_key = wk_key.replace('linear_k', 'linear_q')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wk_key] = target_dict.pop(wk_key)  # add extra weight to shared dict
                condition.notify_all()
    for wv_key in wv_keys:
        wq_key = wv_key.replace('linear_v', 'linear_q')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wv_key] = target_dict.pop(wv_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat qkv
    for wq_key in wq_keys:
        wk_key = wq_key.replace('linear_q', 'linear_k')
        wv_key = wq_key.replace('linear_q', 'linear_v')
        wq_value = target_dict.pop(wq_key)
        wk_value = target_dict.pop(wk_key, None)
        wv_value = target_dict.pop(wv_key, None)

        # get missing weight from shared dict
        if wk_value is None:
            with condition:
                condition.wait_for(lambda: wk_key in qkv_dict.keys())
                wk_value = qkv_dict.pop(wk_key)
        if wv_value is None:
            with condition:
                condition.wait_for(lambda: wv_key in qkv_dict.keys())
                wv_value = qkv_dict.pop(wv_key)

        w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
        w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
        # qkv weight format: hf -> mg
        w_qkv_value_mg = qkv_concat_hf2mg(w_qkv_value, num_heads, n_kv_heads, hidden_size)
        target_dict.update({w_qkv_key: w_qkv_value_mg})


def _concat_ffn_weight(w1_keys, w3_keys, model_config, qkv_dict, condition, target_dict):
    r"""
    concat ffn weight from dicts.

    Args:
        w1_keys: ffn w1 weight name.
        w3_keys: ffn w3 weight name.
        model_config: model config.
        qkv_dict: query, key, value weight dict.
        condition: condition to manager context.
        target_dict: converted weight dict.

    Returns:

    """
    from mindformers.utils.convert_utils import ffn_concat_hf2mg

    intermediate_size = model_config.intermediate_size
    ffn_dim_multiplier = model_config.ffn_dim_multiplier
    multiple_of = model_config.multiple_of or 256
    ffn_hidden_size = model_config.hidden_size * 4
    if intermediate_size is not None:
        ffn_hidden_size = intermediate_size
    else:
        if ffn_dim_multiplier is not None:
            ffn_hidden_size = int((ffn_dim_multiplier + 0.01) * ffn_hidden_size)
        ffn_hidden_size = int(2 * ffn_hidden_size / 3)
        ffn_hidden_size = multiple_of * \
            ((ffn_hidden_size + multiple_of - 1) // multiple_of)

    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for w3_key in w3_keys:
        w1_key = w3_key.replace('linear_fc1', 'gating')
        if w1_key not in w1_keys:
            with condition:
                qkv_dict[w3_key] = target_dict.pop(w3_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat ffn
    for w1_key in w1_keys:
        w3_key = w1_key.replace('gating', 'linear_fc1')
        w1_value = target_dict.pop(w1_key)
        w3_value = target_dict.pop(w3_key, None)

        # get missing weight from shared dict
        if w3_value is None:
            with condition:
                condition.wait_for(lambda: w3_key in qkv_dict.keys())
                w3_value = qkv_dict.pop(w3_key)

        w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
        w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)
        # ffn weight format: hf -> mg
        w_gate_hidden_value_mg = ffn_concat_hf2mg(w_gate_hidden_value, ffn_hidden_size)
        target_dict.update({w_gate_hidden_key: w_gate_hidden_value_mg})
