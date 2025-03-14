# Copyright 2023 Huawei Technologies Co., Ltd
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
"""yizhao model."""
import copy

import numpy as np
from mindpet.delta.ptuning2 import PrefixEncoder
from mindspore import Tensor, mint, ops
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from yizhao_config import YiZhaoConfig
from yizhao_loss import DPOLoss, DPOCrossEntropy
from yizhao_modules import YiZhaoFreqsMgr
from yizhao_transformer import YiZhaoTransformer

from mindformers.core.loss import CrossEntropyLoss
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode
from mindformers.version_control import get_dropout
from mindformers.version_control import get_tril


class YiZhaoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = YiZhaoConfig
    base_model_prefix = "YiZhao"


class YiZhaoModel(YiZhaoPreTrainedModel):
    r"""
    The backbone of YiZhao network

    Args:
        config (YiZhaoConfig): The config of network.
    """

    def __init__(self, config: YiZhaoConfig, **kwargs):
        super(YiZhaoModel, self).__init__(config, **kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        self.is_first_iteration = True
        self.use_llama_rope = config.use_llama_rope

        # mask
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_past=config.use_past)

        # vocab embedding
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel
        self.embedding = LlamaEmbedding(vocab_table_size=config.vocab_size, embedding_size=config.hidden_size,
                                        param_init_type=config.param_init_type, parallel_optimizer=True)
        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.freqs_mgr = YiZhaoFreqsMgr(
            dim=rotary_dim // 2,
            seq_length=config.seq_length,
            rotary_dtype=config.rotary_dtype,
            base=10000,
            rope_ratio=config.rope_ratio
        )

        self.encoder = YiZhaoTransformer(config)

        self.output_layer = Linear(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=False,
                                   param_init_type=config.param_init_type,
                                   compute_dtype=config.compute_dtype)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.embedding.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.embedding.set_comm_fusion(2)
                self.output_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1
            else:
                self.embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            self.embedding.shard(config.parallel_config)
            if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
                self.output_layer.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.output_layer.shard(strategy_matmul=((1, 1), (dp * cp * mp, 1)))
                if dp * cp >= 32 and dp * cp % 8 == 0:
                    self.output_layer.shard(strategy_matmul=((dp * cp * mp // 8, 1), (8, 1)))

        self.tril = get_tril()
        self.ones = P.Ones()
        self.less = P.Less()
        self.gather = P.Gather()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.tile = ops.Tile()
        low_triangle = np.tril(np.ones((1, self.seq_length, self.seq_length)))
        self.low_triangle = Tensor(low_triangle, mstype.int32)
        self.cast = P.Cast()
        self.dropout = get_dropout(config.hidden_dropout)
        self.dropout.dropout.recompute(False)

    def get_masks(self, batch_size, padding_mask=None, input_position=None):
        """Get attention mask."""
        # [1, seq_length, seq_length] -> [batch_size, seq_length, seq_length]
        low_triangle = self.tile(self.low_triangle, (batch_size, 1, 1))
        if padding_mask is not None:
            low_triangle = self.mul(low_triangle, self.expand_dims(padding_mask, 1))
        if self.use_past and padding_mask is not None:
            low_triangle -= self.expand_dims(padding_mask, -1) - 1
        attention_mask = self.less(low_triangle, 0.5)
        if self.use_past and not self.is_first_iteration:
            # [bs, 1, seq_len] for incremental infer
            attention_mask = self.gather(attention_mask.view(-1, self.seq_length), input_position, 0)
        # [bs, 1, seq_len, seq_len] for normal, [bs, 1, 1, seq_len] for incremental infer
        attention_mask = self.reshape(attention_mask, (batch_size, 1, -1, self.seq_length))
        return attention_mask

    # pylint: disable=W0613
    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, batch_valid_length=None, full_attention_mask=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):
        """YiZhao model."""
        _ = position_ids
        _, seq_len = input_ids.shape
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill()
                mask = self.casual_mask.prefill()
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            freqs_cis = self.freqs_mgr(seq_len)
            if full_attention_mask is None:
                full_attention_mask = attention_mask

            mask = full_attention_mask
        if input_embeds is None:
            input_embeds = self.embedding(input_ids)  # (bs, seq_len, hs)
        input_embeds = self.dropout(input_embeds)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, mask, freqs_cis,
            batch_valid_length=batch_valid_length, prefix_key_values=prefix_key_values, block_tables=block_tables,
            slot_mapping=slot_mapping)
        return hidden_states


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class YiZhaoForCausalLM(YiZhaoPreTrainedModel):
    r"""
    Provide gpt training loss or logits through network.

    Args:
        config (YiZhaoConfig): The config of YiZhaoModel.

    Returns:
        Tensor, the loss or logits of the network.
    """

    @lazy_inline
    def __init__(self, config: YiZhaoConfig, **kwargs):
        super(YiZhaoForCausalLM, self).__init__(config, **kwargs)
        self.transformer = YiZhaoModel(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel

        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        else:
            loss_parallel_config = copy.deepcopy(config.parallel_config)
            loss_parallel_config.model_parallel = dp * mp * cp
            loss_parallel_config.data_parallel = 1
            loss_parallel_config.context_parallel = 1
            if dp * cp >= 32 and dp * cp % 8 == 0:  # For large scale training
                loss_parallel_config.model_parallel = 8
                loss_parallel_config.data_parallel = dp * mp * cp // 8
            self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)

        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.rl_config = config.rl_config
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.load_checkpoint(config)
        self.vocab_size = config.padded_vocab_size
        self.predict_run_mode = get_predict_run_mode()
        self.sub_batch_valid_len = P.Sub()

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get YiZhao model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs = input_ids.shape[0]

        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        batch_valid_length = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return (input_ids, labels, None, None, None, None, None, None, batch_valid_length,
                None, None, slot_mapping, None, None)

    # pylint: disable=W0613
    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, None,
                        dynamic_batch_valid_length, None, dynamic_block_tables, dynamic_slot_mapping, None, None)
        logger.info("Set dynamic input for YiZhao.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.encoder.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.rotary_embedding.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, attention_mask=None, input_mask=None, input_position=None,
                  position_ids=None, input_embeds=None, init_reset=None, batch_valid_length=None,
                  prefix_key_values=None,
                  block_tables=None, slot_mapping=None, batch_index=None, zactivate_len=None):
        """YiZhao for conditional generation model."""
        bs, seq_len = input_ids.shape

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        hidden_states = self.transformer(
            input_ids=input_ids,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            hidden_states = self.gather(hidden_states, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        lm_logits = self.transformer.output_layer(hidden_states)
        outputs = (lm_logits,)
        if self.rl_config is not None:
            return lm_logits

        # train
        if labels is not None:
            logits = lm_logits.to(mstype.float32)
            labels = labels.reshape((-1,))
            logits = logits.reshape((-1, logits.shape[-1]))
            if input_mask is None:
                input_mask = self.not_equal(labels, -100).to(mstype.float32)
            input_mask = input_mask.reshape((-1,))

            if self.training:
                # if training, return loss directly
                outputs = self.loss(logits, labels, input_mask)
            else:
                # eval in train ppl
                # pre-shift to fit mindformers/core/metric/utils.py:PerplexityCell
                zeros = ops.zeros((bs, 1, self.vocab_size), dtype=logits.dtype)
                logits = logits.reshape((bs, seq_len, self.vocab_size))
                logits = ops.cat((logits, zeros), axis=1)

                zeros = ops.zeros((bs, 1), dtype=labels.dtype)
                labels = labels.reshape((bs, seq_len))
                labels = ops.cat((zeros, labels), axis=1)

                zeros = zeros.to(input_mask.dtype)
                input_mask = input_mask.reshape((bs, seq_len))
                input_mask = ops.cat((zeros, input_mask), axis=1)

                outputs = logits, labels, input_mask

        # generation process
        if not self.training:
            lm_logits = self.cast(lm_logits, mstype.float32)
            if self.predict_run_mode:
                lm_logits = self.reshape(lm_logits, (-1, lm_logits.shape[-1]))
                outputs = (lm_logits,)

        return outputs


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class YiZhaoDPO(YiZhaoForCausalLM):
    """
        YiZhao Model for pretraining with DPO

        Args:
            config (YiZhaoConfig): The config of network.
    """
    @lazy_inline
    def __init__(self, config: YiZhaoConfig, **kwargs):
        super().__init__(config, **kwargs)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.dpo_loss = DPOLoss(config)
        else:
            loss_parallel_config = copy.deepcopy(config)
            loss_parallel_config.parallel_config.model_parallel = dp * mp
            loss_parallel_config.parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0:  # For large scale training
                loss_parallel_config.parallel_config.model_parallel = 8
                loss_parallel_config.parallel_config.data_parallel = dp * mp // 8
            self.dpo_loss = DPOLoss(loss_parallel_config)
        self.alpha = config.alpha
        self.beta = config.beta
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.sft_loss = DPOCrossEntropy(parallel_config=config.parallel_config)
        else:
            loss_parallel_config = copy.deepcopy(config.parallel_config)
            loss_parallel_config.model_parallel = dp * mp
            loss_parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0:  # For large scale training
                loss_parallel_config.model_parallel = 8
                loss_parallel_config.data_parallel = dp * mp // 8
            self.sft_loss = DPOCrossEntropy(parallel_config=loss_parallel_config)

    # pylint: disable=W0221
    def construct(self, input_ids, labels=None,
                  attention_mask=None, loss_mask=None,
                  ref_chosen_logps=None, ref_rejected_logps=None,
                  input_position=None, position_ids=None, input_embeds=None,
                  batch_valid_length=None, block_tables=None, slot_mapping=None, prefix_key_values=None):
        """YiZhao for conditional generation model."""
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        # [bs, seq_len, hidden_size]
        hidden_states = self.transformer(
            input_ids=input_ids,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
        # [bs, seq_len, vocab_size]
        policy_logits = self.transformer.output_layer(hidden_states)
        policy_logits = self.cast(policy_logits, mstype.float32)

        dpo_loss, sft_loss = self.dpo_loss(policy_logits, labels, loss_mask, ref_chosen_logps.reshape((-1,)),
                                           ref_rejected_logps.reshape((-1,)))
        return self.alpha * dpo_loss + self.beta * sft_loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class YiZhaoWithPtuning2(YiZhaoForCausalLM):
    """
    YiZhao Model for pretraining with p-tuning-v2

    Args:
        config (YiZhaoConfig): The config of network.
    """

    def __init__(self, config: YiZhaoConfig = None, **kwargs):
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        config.pre_seq_len = config.pet_config.pre_seq_len

        super().__init__(config, **kwargs)

        # get Pet tuning model.
        self.use_past = config.use_past
        config.pet_config.num_layers = config.num_layers
        config.pet_config.kv_channels = config.kv_channels
        if config.multi_query_attention:
            config.pet_config.num_heads = config.multi_query_group_num
        else:
            config.pet_config.num_heads = config.num_attention_heads
        self.prefix_encoder = PrefixEncoder(
            config.pet_config.pre_seq_len,
            config.pet_config.num_layers,
            config.pet_config.num_heads,
            config.pet_config.kv_channels,
            config.pet_config.prefix_projection,
            config.pet_config.projection_dim,
            config.pet_config.dropout_prob
        )

        if ckpt_cfg:
            # load ckpt
            config.checkpoint_name_or_path = ckpt_cfg
            self.load_checkpoint(config)

        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, config.pet_config.pet_type)

    # pylint: disable=W0613
    # pylint: disable=W0221
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None, batch_index=None, zactivate_len=None):
        if not self.use_past or self.is_first_iteration:
            batch_size = input_ids.shape[0]
            prefix_key_values = self.prefix_encoder(batch_size)

        return super().construct(
            input_ids=input_ids,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

    def kvcache(self, layer_idx):
        key_cache = \
            self.transformer.encoder.layers[layer_idx].self_attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = \
            self.transformer.encoder.layers[layer_idx].self_attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
