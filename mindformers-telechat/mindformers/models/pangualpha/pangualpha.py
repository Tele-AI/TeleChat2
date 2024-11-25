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
"""PanGuAlpha model"""
import copy
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.layers import LayerNorm
from mindformers.modules.transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, \
    AttentionMask, MoEConfig
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook

from .pangualpha_config import PanguAlphaConfig


__all__ = ['PanguAlphaHeadModel', 'PanguAlphaModel', 'PanguAlphaPromptTextClassificationModel']


class PanguAlphaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PanguAlphaConfig
    base_model_prefix = "pangualpha"


class EmbeddingLayer(nn.Cell):
    r"""Embedding layer of the PanGuAlpha Model"""
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        # Only for the pipeline mode, the embedding needs to be row sliced.
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("normal", [config.vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.hidden_size,
                                                 param_init=initializer("normal",
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=mstype.float32),
                                                 parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.dropout = nn.Dropout(keep_prob=1 - config.embedding_dropout_prob)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size

    def construct(self, input_ids, input_position, batch_valid_length):
        r"""forward pass of the layer"""
        word_embedding, word_table = self.word_embedding(input_ids)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = F.shape(input_ids)
            input_position = batch_valid_length.view(self.batch_size, seq_length)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table

    def get_word_embedding_weight(self):
        r"""get word embedding weight"""
        return self.word_embedding.embedding_table


class QueryLayer(TransformerEncoderLayer):
    r"""Query Layer at the final layer."""
    def __init__(self, batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 param_init_type=mstype.float32,
                 hidden_act='fast_gelu',
                 use_past=False,
                 parallel_config=None,
                 softmax_compute_type=mstype.float32):
        super(QueryLayer, self).__init__(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         ffn_hidden_size=ffn_hidden_size,
                                         num_heads=num_heads,
                                         seq_length=seq_length,
                                         attention_dropout_rate=attention_dropout_rate,
                                         hidden_dropout_rate=hidden_dropout_rate,
                                         post_layernorm_residual=post_layernorm_residual,
                                         param_init_type=param_init_type,
                                         hidden_act=hidden_act,
                                         use_past=use_past,
                                         parallel_config=parallel_config.dp_mp_config,
                                         softmax_compute_type=softmax_compute_type)

    # pylint: disable=W0221
    def construct(self, x, query_vector, input_mask, init_reset=True, batch_valid_length=None):
        r"""
        The forward process of the block.
        """
        # [bs * seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(query_vector, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class PanguAlphaHead(nn.Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super(PanguAlphaHead, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embed):
        r"""forward pass of the head"""
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id
    logger.info("pipeline stage id is %s", pp_id)

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class PanguAlphaModel(PanguAlphaPreTrainedModel):
    r"""The base backbone of the PanGuAlpha model"""
    def __init__(self, config):
        super(PanguAlphaModel, self).__init__(config)
        self.is_pipeline = config.parallel_config.pipeline_stage > 1
        self.embedding = EmbeddingLayer(config)
        self.config = config
        self.layernorm = LayerNorm((config.hidden_size,)).to_float(mstype.float32)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1
        # Configure the shard configure of the Embedding layer
        self.embedding.pipeline_stage = 0
        self.num_layers = config.num_layers
        if config.use_moe:
            moe_config = MoEConfig(expert_num=config.expert_num,
                                   num_experts_chosen=config.per_token_num_experts_chosen)
        else:
            moe_config = MoEConfig(expert_num=1)
        # The shard setting of Transformer is set within the class StackedTransformer
        self.blocks = TransformerEncoder(num_layers=config.num_layers - 1,
                                         batch_size=config.batch_size,
                                         hidden_size=config.hidden_size,
                                         ffn_hidden_size=config.ffn_hidden_size,
                                         num_heads=config.num_heads,
                                         seq_length=config.seq_length,
                                         attention_dropout_rate=config.attention_dropout_rate,
                                         hidden_dropout_rate=config.hidden_dropout_rate,
                                         lambda_func=set_parallel_configure_for_layer,
                                         hidden_act=config.hidden_act,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config,
                                         softmax_compute_type=config.softmax_compute_type).blocks
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.top_query_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                  embedding_size=config.hidden_size,
                                                  param_init=initializer("normal",
                                                                         [config.seq_length, config.hidden_size],
                                                                         dtype=mstype.float32),
                                                  parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.top_query_embedding.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if config.parallel_config.pipeline_stage > 1:
            self.top_query_embedding.set_comm_fusion(2)
        else:
            self.top_query_embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.top_query_layer = QueryLayer(batch_size=config.batch_size,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.ffn_hidden_size,
                                          num_heads=config.num_heads,
                                          seq_length=config.seq_length,
                                          attention_dropout_rate=config.attention_dropout_rate,
                                          hidden_dropout_rate=config.hidden_dropout_rate,
                                          hidden_act=config.hidden_act,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          softmax_compute_type=config.softmax_compute_type,
                                          parallel_config=config.parallel_config)
        if isinstance(config.parallel_config.recompute, bool):
            if config.parallel_config.recompute:
                self.top_query_layer.recompute()
        else:
            if config.parallel_config.recompute.recompute:
                self.top_query_layer.recompute(
                    recompute_slice_activation=config.parallel_config.recompute.recompute_slice_activation)
        self.top_query_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.top_query_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.dtype = mstype.float16

        self.use_past = config.use_past
        self.add = P.TensorAdd()

    def construct(self, input_ids,
                  input_position,
                  encoder_masks,
                  init_reset=True,
                  batch_valid_length=None):
        r"""forward pass of the model"""
        embed, word_table = self.embedding(input_ids, input_position, batch_valid_length)
        hidden_state = P.Cast()(embed, self.dtype)
        hidden_shape = F.shape(hidden_state)
        hidden_state = F.reshape(hidden_state, (-1, hidden_shape[-1]))
        # the input of the incremental prediction is 3d
        if self.blocks is not None:
            for i in range(self.num_layers - 1):
                hidden_state, _ = self.blocks[i](hidden_state, encoder_masks, init_reset, batch_valid_length)
        if self.use_past and not self.is_first_iteration:
            input_position = self.add(input_position, F.expand_dims(batch_valid_length, -1))
        if self.is_pipeline:
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(hidden_state, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)
            encoder_output = self.layernorm(encoder_output)
        else:
            hidden_state = self.reshape_to_2d(hidden_state)
            encoder_output = self.layernorm(hidden_state)
            encoder_output = P.Cast()(encoder_output, self.dtype)
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(encoder_output, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)

        return encoder_output, word_table

    def reshape_to_2d(self, x):
        r"""reshape nd tensor to 2d, if n <= 2, keep original shape."""
        shape = F.shape(x)
        if len(shape) <= 2:
            return x
        x = F.reshape(x, (-1, shape[-1]))
        return x


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class PanguAlphaHeadModel(PanguAlphaPreTrainedModel):
    """
    The PanguAlpha network consisting of two parts the backbone and the head
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_position(Tensor): current position, used by model.predict.
        attention_mask(Tensor): input sentences padding mask, where 0 indicates padding position.
        position_ids(Tensor): used to identify each token's position in the list of tokens.
        input_embeds(Tensor): reserved param, not used.
        labels(Tensor): the labels of corresponding input sequences.
        init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
            past value parameter used in the incremental prediction. Default True.
        batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
            prediction. Tensor of shape :math:`(batch_size,)`. Default None.

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)

    Examples:
        >>> # input model name, load model and weights
        >>> model_a = PanguAlphaHeadModel.from_pretrained('pangualpha_2_6b')
        >>> # input config, load model without weights
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('pangualpha_2_6b')
        >>> model_b = PanguAlphaHeadModel(config)
    """
    _support_list = MindFormerBook.get_model_support_list()['pangualpha']

    def __init__(self, config: PanguAlphaConfig = None):
        config = config if config is not None else PanguAlphaConfig()
        super(PanguAlphaHeadModel, self).__init__(config)
        logger.info(config.parallel_config)
        # Network head to get logits over vocabulary
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        logger.info(copied_parallel_config)

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False

        dp = config.parallel_config.data_parallel

        self.pad_token_id = config.pad_token_id

        self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)

        self.head = PanguAlphaHead(hidden_size=config.hidden_size,
                                   parallel_config=copied_parallel_config)
        self.head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.backbone = PanguAlphaModel(config)
        self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        loss_parallel_config = copy.deepcopy(config.parallel_config)
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile().shard(((dp,),))
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)
        self.expand = P.ExpandDims()
        self.add = P.Add().shard(((dp, 1), ()))
        self.gather = P.Gather()

        self.use_past = config.use_past
        self.is_first_iteration = True

        self.position_ids = Tensor(np.arange(config.seq_length), mstype.int32)
        self.all_ones_attention_mask = P.Ones()((1, 1, 1), mstype.float32)

        self.load_checkpoint(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_position": input_position
        }

    # pylint: disable=W0613
    def construct(self, input_ids, input_position=None, attention_mask=None, position_ids=None,
                  input_embeds=None, labels=None, init_reset=True, batch_valid_length=None):
        r"""forward pass of the model"""
        batch_size, seq_length = input_ids.shape

        if self.training:
            seq_length = seq_length - 1
            tokens = self.slice(input_ids, (0, 0), (batch_size, seq_length), (1, 1))
            input_mask = F.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
            input_position = self.slice(input_position, (0, 0), (batch_size, seq_length), (1, 1))
            attention_mask = self.cast(attention_mask, mstype.float32)
            attention_mask = self.slice2(attention_mask, (0, 0, 0),
                                         (batch_size, seq_length, seq_length),
                                         (1, 1, 1))
        else:
            tokens = input_ids
            input_mask = F.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

            if attention_mask is None:
                attention_mask = self.get_attention_mask(input_mask)
                if not self.is_first_iteration:
                    attention_mask = self.tile(self.all_ones_attention_mask, (batch_size, 1, 1))

            if input_position is None or self.is_first_iteration:
                if batch_size == 1:
                    input_position = F.reshape(self.position_ids, (1, seq_length))
                else:
                    input_position = self.tile(self.position_ids, (batch_size, 1))
            # when incremental reasoning is not the first iteration, it goes into the following logic.
            else:
                bias = Tensor(np.arange(batch_size) * self.config.seq_length, mstype.int32)
                input_position = F.sub(input_position, bias)
                input_position = F.reshape(input_position, (batch_size, 1))

        # [batch_size, seq_length, vocab_size]
        output_states, word_table = self.backbone(tokens, input_position, attention_mask,
                                                  init_reset, batch_valid_length)
        logits = self.head(output_states, word_table)

        if not self.training:
            logits = logits.reshape((batch_size, seq_length, -1))
            logits = logits.reshape((-1, logits.shape[-1]))
            if (not self.use_past or self.is_first_iteration) and input_position is not None:
                logits = self.gather(logits, input_position, 0)
            return logits, tokens, input_mask

        labels = self.slice(input_ids, (0, 1), (batch_size, seq_length + 1), (1, 1))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)

        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class PanguAlphaPromptTextClassificationModel(PanguAlphaHeadModel):
    """The PanguAlpha network for prompt text classification consisting of two parts the backbone and the head
        Args:
            config(PanguAlphaConfig): the config of network
        Inputs:
            input_ids: the tokenized inputs, shape is [num_labels, seq_length]
            labels: the index of the true label, shape is [1,]
            attention_mask: the mask indicating whether each position is a valid input and is not the added prompt,
                            shape is [num_labels, seq_length]
        Returns:
            logits: Tensor: corresponding outputs for calculating metrics

        Examples:
            >>> # input model name, load model and weights
            >>> model_a = PanguAlphaHeadModel.from_pretrained('pangualpha_2_6b')
            >>> # input config, load model without weights
            >>> from mindformers import AutoConfig
            >>> config = AutoConfig.from_pretrained('pangualpha_2_6b')
            >>> model_b = PanguAlphaPromptTextClassificationModel(config)
    """
    def __init__(self, config: PanguAlphaConfig = None):
        config = config if config is not None else PanguAlphaConfig()
        super().__init__(config)
        self.num_labels = config.num_labels

    # pylint: disable=arguments-differ
    def construct(self, input_ids=None, labels=None, attention_mask=None, position_ids=None,
                  input_embeds=None, input_position=None, init_reset=True, batch_valid_length=None):
        r"""forward pass of the model"""
        if self.training:
            raise ValueError("PanguAlphaPromptTextClassificationModel just supports evaluate mode, "
                             "please set run_mode to eval")
        if input_ids is None and input_embeds is None:
            raise ValueError("input_ids and input_embeds can not be all None")
        batch_size, num_labels_mul_seq_length = input_ids.shape
        seq_length = num_labels_mul_seq_length // self.num_labels

        input_ids = self.reshape(input_ids, (-1, seq_length))
        input_mask = self.cast(self.reshape(attention_mask, (-1, seq_length)), mstype.float32)

        attention_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
        attention_mask = self.get_attention_mask(attention_mask)

        if position_ids is None:
            position_ids = F.tuple_to_array(F.make_range(seq_length))
            position_ids = self.expand(position_ids, 0)
            position_ids = self.tile(position_ids, (self.num_labels * batch_size, 1))

        logits, vocab_table = self.backbone(input_ids, position_ids, attention_mask, init_reset, batch_valid_length)
        logits = self.head(logits, vocab_table)
        logits = self.reshape(logits, (batch_size, self.num_labels, seq_length, -1))

        return logits, input_ids, input_mask, labels
