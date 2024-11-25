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
"""ChatGLM2 model."""
import copy
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindpet.delta.ptuning2 import PrefixEncoder

import numpy as np
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.layers import Linear
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_disable_custom_fa
from mindformers.core.loss import CrossEntropyLoss
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.version_control import get_tril
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.tools.utils import get_predict_run_mode

from ..utils import lazy_inline
from .glm2_config import ChatGLM2Config
from .glm2_modules import FreqsMgr
from .glm2_transformer import ChatGLM2Transformer
from ...tools.logger import logger

__all__ = ['ChatGLM2ForConditionalGeneration', 'ChatGLM2Model', 'ChatGLM2WithPtuning2']


class GLM2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChatGLM2Config
    base_model_prefix = "glm2"


class ChatGLM2Model(GLM2PreTrainedModel):
    r"""
    The backbone of ChatGLM2 network

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2Model, self).__init__(config, **kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        self.is_first_iteration = True
        # default open internal kernel boost
        self.disable_custom_fa = get_disable_custom_fa()
        logger.info("Open prefill flatten and disable custom flash attention op:{}".format(self.disable_custom_fa))

        # mask
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention)

        # vocab embedding
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.embedding = LlamaEmbedding(vocab_table_size=config.vocab_size, embedding_size=config.hidden_size,
                                        param_init_type=config.param_init_type, parallel_optimizer=True)
        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.freqs_mgr = FreqsMgr(
            dim=rotary_dim // 2,
            seq_length=config.seq_length,
            rotary_dtype=config.rotary_dtype,
            base=10000,
            rope_ratio=config.rope_ratio)

        self.encoder = ChatGLM2Transformer(config)

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
                self.output_layer.shard(strategy_matmul=((dp, 1), (1, 1)))
            else:
                self.output_layer.shard(strategy_matmul=((1, 1), (dp * mp, 1)))

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

    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, batch_valid_length=None, full_attention_mask=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):
        """ChatGLM2 model."""
        _ = position_ids
        batch_size, seq_len = input_ids.shape

        mask = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(batch_size, seq_len)
                if self.use_flash_attention:
                    if self.disable_custom_fa:
                        mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
                        mask = self.cast(mask, mstype.float16)
                else:
                    mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            freqs_cis = self.freqs_mgr(seq_len)
            if full_attention_mask is None:
                # (bs, 1, seq_len, seq_len)
                full_attention_mask = self.get_masks(batch_size, attention_mask, input_position)
                full_attention_mask = full_attention_mask.type(mstype.uint8)
            mask = full_attention_mask
        if input_embeds is None:
            input_embeds = self.embedding(input_ids)  # (bs, seq_len, hs)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, mask, freqs_cis,
            batch_valid_length=batch_valid_length, prefix_key_values=prefix_key_values, block_tables=block_tables,
            slot_mapping=slot_mapping)
        return hidden_states


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2ForConditionalGeneration(GLM2PreTrainedModel):
    """
    Provide ChatGLM2 training loss or logits through network.

    Args:
        config (ChatGLM2Config): The config of ChatGLM2Model.
        kwargs (dict, optional): A variable number of keyword parameters reserved for the keyword parameters to be
            expanded.

    Inputs:
        - **input_ids** (Tensor, optional) - A tokenized input tensor, which is of int32 integer type and has a shape of
          (batch, seq_length). Default: ``None`` .
        - **labels** (Tensor, optional) - A tokenized label tensor, which is of int32 integer type and has a shape of
          (batch, seq_length). Default: ``None`` .
        - **input_position** (Tensor, optional) - The current position, used in predict. Default: ``None`` .
        - **position_ids** (Tensor, optional) - Keep the parameter unused. Default: ``None`` .
        - **attention_mask** (Tensor, optional) - Keep the parameter unused. Default: ``None`` .
        - **input_embeds** (Tensor, optional) - Keep the parameter unused. Default: ``None`` .
        - **init_reset** (Tensor, optional) - A bool tensor with shape [1], used to clear previous key-value pairs in
          incremental inference. Default: ``None`` .
        - **batch_valid_length** (Tensor, optional) - In incremental inference, a tensor used for calculating the index
          of the previous step. It is of int32 type and has a shape of [batch_size]. Default: ``None`` .
        - **prefix_key_values** (Tensor, optional) - A set of additional key-value pairs added before the regular
          key-value pairs. These prefix key-value pairs can be used to capture long-term dependencies or provide prior
          knowledge, thereby helping the model better understand and generate sequences. Default: ``None`` .
        - **block_tables** (Tensor, optional) - Store the mapping table for each sequence. Default: ``None`` .
        - **slot_mapping** (Tensor, optional) - Store the physical slot index of the sequence cache. Default: ``None`` .
        - **batch_index** (Tensor, optional) - Keep the parameter unused. Default: ``None`` .
        - **zactivate_len** (Tensor, optional) - Keep the parameter unused. Default: ``None`` .

    Outputs:
        output(Tensor), including an on-line loss value or a logical value, a sequence of predictive text, an input
        mask.

    Examples:
        >>> from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration
        >>> config = ChatGLM2Config(batch_size=2)
        >>> network = ChatGLM2ForConditionalGeneration(config=config)
        >>> type(network)
        <class 'mindformers.models.glm2.glm2.ChatGLM2ForConditionalGeneration'>
        >>> from mindformers import ChatGLM2ForConditionalGeneration
        >>> network = ChatGLM2ForConditionalGeneration.from_pretrained('glm3_6b')
        >>> type(network)
        <class 'mindformers.models.glm2.glm2.ChatGLM2ForConditionalGeneration'>
    """
    _support_list = MindFormerBook.get_model_support_list()['glm2']
    _support_list.extend(MindFormerBook.get_model_support_list()['glm3'])
    _support_list.extend(MindFormerBook.get_model_support_list()['codegeex2'])

    @lazy_inline
    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2ForConditionalGeneration, self).__init__(config, **kwargs)
        self.transformer = ChatGLM2Model(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.loss = CrossEntropyLoss(parallel_config=config.parallel_config,
                                         check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                         calculate_per_token_loss=calculate_per_token_loss)
        else:
            loss_parallel_config = copy.deepcopy(config.parallel_config)
            loss_parallel_config.model_parallel = dp * mp
            loss_parallel_config.data_parallel = 1
            self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                         check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                         calculate_per_token_loss=calculate_per_token_loss)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.load_checkpoint(config)
        self.vocab_size = config.padded_vocab_size
        self.predict_run_mode = get_predict_run_mode()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get ChatGLM2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        batch_valid_length = Tensor(np.array([seq] * bs), mstype.int32)
        return input_ids, labels, None, None, None, None, None, batch_valid_length, None, None, slot_mapping, None, None

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                        dynamic_batch_valid_length, None, dynamic_block_tables, dynamic_slot_mapping, None, None)
        logger.info("Set dynamic input for glm.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.encoder.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.rotary_embedding.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None, batch_index=None, zactivate_len=None):
        """ChatGLM2 for conditional generation model."""
        # input_ids: (bs, seq_len)
        # position_ids: (bs, seq_len)
        # attention_mask: (bs, seq_len)
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
        lm_logits = self.transformer.output_layer(hidden_states)
        if lm_logits.dtype == mstype.bfloat16:
            lm_logits = self.cast(lm_logits, mstype.float32)
        outputs = (lm_logits,)

        # train
        if labels is not None:
            logits = lm_logits.to(mstype.float32)
            labels = labels.reshape((-1,))
            logits = logits.reshape((-1, logits.shape[-1]))
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

        if not self.training:
            lm_logits = self.cast(lm_logits, mstype.float32)
            if self.predict_run_mode:
                lm_logits = self.reshape(lm_logits, (-1, lm_logits.shape[-1]))
                outputs = (lm_logits,)

        return outputs


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2WithPtuning2(ChatGLM2ForConditionalGeneration):
    """
    ChatGLM2 Model for pretraining with p-tuning-v2

    Args:
        config (ChatGLM2Config): The config of network.
    """

    def __init__(self, config: ChatGLM2Config = None, **kwargs):
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
