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
"""Bloom model"""
import copy
import os
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules.transformer import VocabEmbedding
from mindformers.modules.layers import LayerNorm, AlibiTensor
from mindformers.core.loss import CrossEntropyLoss
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.mindformer_book import MindFormerBook

from .layers import BloomBlocks, CausalMask
from .bloom_config import BloomConfig
from ..utils import convert_mstype, lazy_inline


class BloomPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "bloom"


def jit_inference_with_condition():
    """allow jit inference"""

    def decorator(func):
        if os.getenv("JIT_INFERENCE", "NOT_FOUND") == "NOT_FOUND":
            return func
        from mindspore import jit, JitConfig
        dec = jit(jit_config=JitConfig(jit_level="O2"))
        return dec(func)

    return decorator


class BloomEmbeddingLayer(nn.Cell):
    """The Embedding Layer of Bloom network."""

    def __init__(self, config=None):
        super(BloomEmbeddingLayer, self).__init__(auto_prefix=False)
        parallel_config = copy.deepcopy(config.parallel_config)
        embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % embedding_mp != 0:
            logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
                           vocab_size, embedding_mp)
            logger.warning("Now, model_parallel will be changed: mp = 1")
            parallel_config.embedding_dp_mp_config.model_parallel = 1

        self.word_embedding = VocabEmbedding(vocab_size=vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("normal",
                                                                    [vocab_size, config.hidden_size],
                                                                    dtype=config.embedding_init_type),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.norm = LayerNorm((config.hidden_size,)).shard(((1, 1, 1), (1,), (1,)))

    def construct(self, input_ids):
        """The forward compute of Embedding Layer."""
        word_embedding, word_table = self.word_embedding(input_ids)
        embedding = self.norm(word_embedding)
        embedding = embedding.astype(mstype.float16)
        return embedding, word_table


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers, use_select_recompute=False):
    """
    Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

    Args:
        network(Cell) - Represents the transformer block
        parallel_config(dict) - Parallel Config
        layer_id(int) - Means the layer index for the current module, counts from zero.
        offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
        layers(int) - The total layers used for the model.
        use_select_recompute(bool) - Indicates whether to use the select recompute mode.
    """

    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)

    network.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute and not use_select_recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute and not use_select_recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class BloomModel(BloomPreTrainedModel):
    """
    The backbone of Bloom network

    Args:
        config(BloomConfig): The config of network

    Inputs:
        input_ids(Tensor): The tokenized inputs with datatype int32
        input_mask(Tensor): The mask indicating whether each position is a valid input

    Returns:
        output_state(Tensor): The output logit of backbone
        embedding_table(Tensor): The embedding table for the vocabulary
    """

    def __init__(self, config):
        super(BloomModel, self).__init__(config)

        self.embedding = BloomEmbeddingLayer(config)
        self.embedding.pipeline_stage = 0

        self.make_causal_attention = CausalMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)

        self.build_alibi_tensor = AlibiTensor(seq_length=config.seq_length,
                                              num_heads=config.num_heads,
                                              parallel_config=config.parallel_config)

        self.blocks = BloomBlocks(hidden_size=config.hidden_size,
                                  batch_size=config.batch_size,
                                  ffn_hidden_size=config.hidden_size * config.expand_ratio,
                                  seq_length=config.seq_length,
                                  num_layers=config.num_layers,
                                  num_heads=config.num_heads,
                                  attention_dropout_rate=config.attention_dropout_rate,
                                  hidden_dropout_rate=config.hidden_dropout_rate,
                                  hidden_act=config.hidden_act,
                                  lambda_func=set_parallel_configure_for_layer,
                                  param_init_type=config.param_init_type,
                                  layernorm_compute_type=config.layernorm_compute_type,
                                  softmax_compute_type=config.softmax_compute_type,
                                  use_past=config.use_past,
                                  use_seq_parallel=config.use_seq_parallel,
                                  use_select_recompute=config.use_select_recompute,
                                  use_flash_attention=config.use_flash_attention,
                                  parallel_config=config.parallel_config).blocks
        self.num_layers = config.num_layers
        self.ln_f = LayerNorm((config.hidden_size,)).to_float(config.layernorm_compute_type)
        if config.parallel_config.pipeline_stage > 1:
            self.ln_f.set_comm_fusion(2)
        else:
            self.ln_f.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.ln_f.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.ln_f.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.use_past = config.use_past
        self.dtype = convert_mstype(config.param_init_type)
        self.mul_init_reset = P.Mul().shard(
            ((config.parallel_config.data_parallel, config.parallel_config.model_parallel, 1, 1), (1,)))

        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, input_ids, input_mask, init_reset=True, batch_valid_length=None):
        """Bloom model"""
        input_embedding, embedding_table = self.embedding(input_ids)
        hidden_states = input_embedding
        hidden_states_shape = hidden_states.shape
        hidden_states = self.reshape(hidden_states, (-1, hidden_states_shape[-1]))

        causal_mask = self.make_causal_attention(input_mask)
        alibi_tensor = self.build_alibi_tensor(input_mask, hidden_states.dtype)

        if self.use_past:
            init_reset = self.cast(init_reset, self.dtype)
            init_reset = self.mul_init_reset(self.blocks[0].key_past, init_reset)
        for i in range(self.num_layers):
            hidden_states, _ = self.blocks[i](hidden_states, alibi_tensor, causal_mask, init_reset, batch_valid_length)
        hidden_states = self.reshape(hidden_states, hidden_states_shape)
        output_state = self.ln_f(hidden_states)

        return output_state, embedding_table


class BloomHead(nn.Cell):
    """Head for Bloom to get the logits of each token in the vocab."""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_type="float16",
                 parallel_config=None):
        super(BloomHead, self).__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        mp = copied_parallel_config.model_parallel
        if vocab_size % mp != 0:
            logger.warning("The vocab size of BloomHead MatMul is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of BloomHead MatMul will be changed: mp = 1")
            copied_parallel_config.model_parallel = 1

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        if copied_parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (
                copied_parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = convert_mstype(compute_type)

    def construct(self, state, embedding_table):
        ori_dtype = state.dtype
        state = state.reshape((-1, self.hidden_size))
        logits = self.matmul(state.astype(self.dtype), embedding_table.astype(self.dtype))
        return logits.astype(ori_dtype)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BloomLMHeadModel(BloomPreTrainedModel):
    """
    Provide bloom training loss or logits through network.

    Args:
        config (BloomConfig): The config of BloomModel.

    Returns:
        Tensor, the loss or logits of the network.
    """

    _support_list = MindFormerBook.get_model_support_list()['bloom']

    @lazy_inline
    def __init__(self, config=None):
        config = config if config is not None else BloomConfig()
        super(BloomLMHeadModel, self).__init__(config, auto_prefix=False)
        self.use_past = self.config.use_past
        self.is_sample_acceleration = self.config.is_sample_acceleration

        if self.use_past:
            self.input_mask_all_ones = Tensor(
                np.ones((self.config.batch_size, self.config.seq_length), np.float32), mstype.float32)

        if self.is_sample_acceleration:
            self.p_all_ones = Tensor(np.ones((self.config.batch_size, 1), np.float32), mstype.float32)

        self.eos_token_id = self.config.eos_token_id
        parallel_config = self.config.parallel_config
        self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.gt = P.Greater().shard(((parallel_config.data_parallel, 1), ()))
        self.mul = P.Mul().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.abs = P.Abs().shard(((parallel_config.data_parallel, 1),))
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), ()))
        self.gather = P.Gather().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel,)))

        self.transformer = BloomModel(self.config)
        self.head = BloomHead(hidden_size=config.hidden_size,
                              vocab_size=config.vocab_size,
                              parallel_config=self.config.parallel_config)
        if parallel_config.pipeline_stage > 1:
            self.head.pipeline_stage = parallel_config.pipeline_stage - 1
            self.transformer.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Bloom Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Bloom Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(self.config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)
        self.seq_length = config.seq_length
        self.load_checkpoint(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.blocks:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    @jit_inference_with_condition()
    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, labels=None, init_reset=True, batch_valid_length=None):
        """
        construct function for Language Modeling

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_position(Tensor): current position, used by model.predict. Default None.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            labels(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                      otherwise, return the computed loss.
        """

        batch_size, seq_length = input_ids.shape

        if self.training:
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
        else:
            tokens = input_ids

        input_mask = self.not_equal(tokens, self.eos_token_id).astype(mstype.float32) \
            if not self.use_past else self.input_mask_all_ones

        loss_mask = self.mul(input_mask, self.gt(tokens, 0).astype(mstype.float32))
        tokens = self.abs(tokens)

        # [batch_size, seq_length, vocab_size]
        output_states, embedding_table = self.transformer(tokens, input_mask, init_reset, batch_valid_length)
        logits = self.head(output_states, embedding_table)

        if not self.training:
            if self.is_sample_acceleration:
                return self.get_top_token_id(logits, current_index=input_position)
            input_mask = self.add(input_mask, 1)
            if (not self.use_past or self.is_first_iteration) and input_position is not None:
                logits = logits.reshape(-1, logits.shape[-1])
                index = input_position.view(-1,)
                logits = self.gather(logits, index, 0)
                logits = logits.reshape(batch_size, -1, logits.shape[-1])
            return logits, tokens, input_mask

        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        labels = labels.reshape((-1,))
        loss_mask = loss_mask.reshape((-1,))
        loss = self.loss(logits, labels, loss_mask)
        return loss

    def get_top_token_id(self, logits, current_index=None):
        """get_top_token_id"""
        logits = logits.reshape(-1, logits.shape[-1])
        if self.use_past and not self.is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = P.Gather()(logits, index, 0)
        probabilities = P.Softmax(-1)(logits)
        top_token_id = P.Argmax(-1)(probabilities)
        top_token_id = top_token_id.view(-1, 1)
        return self.p_all_ones, top_token_id
