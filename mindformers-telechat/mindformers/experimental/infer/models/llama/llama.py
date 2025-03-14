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
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Condition

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, ops, mint, mutable
from mindspore.communication import get_group_size
from mindspore.communication._comm_helper import _is_initialized

from mindformers.experimental.infer.core.layers import ColumnParallelLinear
from mindformers.experimental.infer.core.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.parallel_state import get_group_info, initialize_model_parallel
from mindformers.models.llama.llama import LlamaPreTrainedModel
from mindformers.modules import Linear
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode, is_pynative
from mindformers.tools.logger import logger

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
        if get_group_info('tp').group is None and _is_initialized():
            initialize_model_parallel(get_group_size(), order='tp')
        self.config = convert_model_config(config)
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.is_pynative = is_pynative()

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.slice = ops.StridedSlice()
        self.not_equal = ops.NotEqual()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.ones = ops.Ones()
        self.gather = ops.Gather(1) if self.is_pynative else ops.Gather()
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
        self.npu_mem_size = config.npu_mem_size if hasattr(config, "npu_mem_size") else 2

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
        """Prepare inputs for dynamic shape."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)

        def get_input():
            if self.npu_mem_size > 0:
                return None
            cache_list = []
            for _ in self.model.layers:
                cache_list.append(Tensor(shape=[None, None, None, None], dtype=self.config.compute_dtype))
            return mutable(cache_list)
        key_cache = get_input()
        value_cache = get_input()
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None, key_cache, value_cache)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None, None, key_cache, value_cache)
        logger.info("Set dynamic input for llama.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                  key_cache=None, value_cache=None):
        """
        Forward of llama model.
        """
        bsz, _ = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
            else:
                batch_valid_length = self.reshape(batch_valid_length, (-1,))
        output = self.model(input_ids, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping, prefix_keys_values, key_cache=key_cache, value_cache=value_cache)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            if not self.is_pynative:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
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

    @classmethod
    def convert_name(cls, weight_name):
        """convert HuggingFace weight name to MindFormers weight name"""
        origin_name = weight_name
        weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('.norm.', '.norm_out.')
        weight_name = weight_name.replace('output.', 'lm_head.')
        weight_name = weight_name.replace('.tok_embeddings.weight', '.tok_embeddings.embedding_weight')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        """convert HuggingFace weight dict to MindFormers weight dict"""
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
                if part[-2] == 'wq':
                    wq_keys.append(k)
                if part[-2] == 'wk':
                    wk_keys.append(k)
                if part[-2] == 'wv':
                    wv_keys.append(k)
                if part[-2] == 'w1':
                    w1_keys.append(k)
                if part[-2] == 'w3':
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
        """convert HuggingFace map dict to MindFormers map dict"""
        qkv_concat = kwargs.pop("qkv_concat", False)
        target_dict = {}
        wq_keys = []
        w1_keys = []

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'wq':
                    wq_keys.append(k)
                if part[-2] == 'w1':
                    w1_keys.append(k)

        if qkv_concat:
            for wq_key in wq_keys:
                wk_key = wq_key.replace('wq', 'wk')
                wv_key = wq_key.replace('wq', 'wv')
                wq_value = target_dict.pop(wq_key)
                target_dict.pop(wk_key)
                target_dict.pop(wv_key)

                w_qkv_key = wq_key.replace('wq', 'w_qkv')
                w_qkv_value = wq_value
                target_dict.update({w_qkv_key: w_qkv_value})
            for w1_key in w1_keys:
                w3_key = w1_key.replace('w1', 'w3')
                w1_value = target_dict.pop(w1_key)
                target_dict.pop(w3_key)

                w_gate_hidden_key = w1_key.replace('w1', 'w_gate_hidden')
                w_gate_hidden_value = w1_value
                target_dict.update({w_gate_hidden_key: w_gate_hidden_value})

        return target_dict

    @classmethod
    def obtain_qkv_ffn_concat_keys(cls):
        qkv_key = "w_qkv"
        ffn_key = "w_gate_hidden"
        concat_keys = [qkv_key, ffn_key]
        logger.info(f"{cls.__name__} qkv/ffn concat keys are {concat_keys}")
        return concat_keys

    def clear_kv_cache(self):
        return self.model.clear_kv_cache()


def _concat_qkv_weight(wq_keys, wk_keys, wv_keys, model_config, qkv_dict, condition, target_dict):
    """concat qkv weight from dicts"""
    from mindformers.utils.convert_utils import qkv_concat_hf2mg

    num_heads = model_config.num_heads
    n_kv_heads = model_config.n_kv_heads or num_heads
    hidden_size = model_config.hidden_size

    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for wk_key in wk_keys:
        wq_key = wk_key.replace('wk', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wk_key] = target_dict.pop(wk_key)  # add extra weight to shared dict
                condition.notify_all()
    for wv_key in wv_keys:
        wq_key = wv_key.replace('wv', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wv_key] = target_dict.pop(wv_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat qkv
    for wq_key in wq_keys:
        wk_key = wq_key.replace('wq', 'wk')
        wv_key = wq_key.replace('wq', 'wv')
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

        w_qkv_key = wq_key.replace('wq', 'w_qkv')
        w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
        # qkv weight format: hf -> mg
        w_qkv_value_mg = qkv_concat_hf2mg(w_qkv_value, num_heads, n_kv_heads, hidden_size)
        target_dict.update({w_qkv_key: w_qkv_value_mg})


def _concat_ffn_weight(w1_keys, w3_keys, model_config, qkv_dict, condition, target_dict):
    """concat ffn weight from dicts"""
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
        w1_key = w3_key.replace('w3', 'w1')
        if w1_key not in w1_keys:
            with condition:
                qkv_dict[w3_key] = target_dict.pop(w3_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat ffn
    for w1_key in w1_keys:
        w3_key = w1_key.replace('w1', 'w3')
        w1_value = target_dict.pop(w1_key)
        w3_value = target_dict.pop(w3_key, None)

        # get missing weight from shared dict
        if w3_value is None:
            with condition:
                condition.wait_for(lambda: w3_key in qkv_dict.keys())
                w3_value = qkv_dict.pop(w3_key)

        w_gate_hidden_key = w1_key.replace('w1', 'w_gate_hidden')
        w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)
        # ffn weight format: hf -> mg
        w_gate_hidden_value_mg = ffn_concat_hf2mg(w_gate_hidden_value, ffn_hidden_size)
        target_dict.update({w_gate_hidden_key: w_gate_hidden_value_mg})
