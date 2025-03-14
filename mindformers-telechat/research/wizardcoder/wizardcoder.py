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

"""WizardCoder model"""
import copy

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

from mindformers.models.utils import lazy_inline
from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.layers import LayerNorm
from mindformers.version_control import get_dropout
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.transformer import AttentionMask
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.logger import logger
from wizardcoder_config import WizardCoderConfig
from wizardcoder_modules import WizardCoderTransformerDecoderLayer, WizardCoderVocabEmbedding


__all__ = ['WizardCoderLMHeadModel']


class WizardCoderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = WizardCoderConfig
    base_model_prefix = "wizardcoder"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class WizardCoderLMHeadModel(WizardCoderPreTrainedModel):
    """Provide wizardcoder training loss or logits through network.
        Args:
            config (WizardCoderConfig): The config of WizardCoderModel.

        Returns:
            Tensor, the loss or logits of the network.
    """
    @lazy_inline
    def __init__(self, config: WizardCoderConfig = None):
        config = config if config is not None else WizardCoderConfig()
        super(WizardCoderLMHeadModel, self).__init__(config, auto_prefix=True)
        self.use_past = config.use_past
        self.eos_token = self.config.eos_token
        self.pad_token = self.config.pad_token
        self.eos_token_tensor = Tensor((np.ones((1, 1)) * self.eos_token).astype(np.int32))
        self.seq_length = config.seq_length

        parallel_config = self.config.parallel_config
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        self.stridedslice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))

        # AttentionMask default compute_dtype is fp16
        # if assign compute_dtype to fp32, loss will error
        # if assign compute_dtype to bf16, lower_triangle_mask is fp32, will error
        self.get_attention_mask = AttentionMask(
            seq_length=config.seq_length, parallel_config=parallel_config.dp_mp_config).to_float(config.compute_dtype)

        self.backbone = WizardCoderModel(config)
        self.dtype = config.compute_dtype
        self.head = WizardCoderHead(vocab_size=config.vocab_size,
                                    compute_dtype=self.dtype,
                                    parallel_config=self.config.parallel_config)

        if parallel_config.pipeline_stage > 1:
            self.head.pipeline_stage = parallel_config.pipeline_stage - 1
            self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)

        if vocab_size % mp != 0:
            logger.warning("The vocab size of WizardCoder Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of WizardCoder Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1

        check_for_nan_in_loss_and_grad = getattr(self.config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     eps_const=1e-24)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.load_checkpoint(config)
        self.add = P.Add().shard(((dp, 1), ()))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.tile = P.Tile()
        self.gather = P.Gather()
        self.concat = P.Concat(axis=-1)
        self.ones = P.Ones()
        self.compute_dtype = config.compute_dtype

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_position": input_position
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Wizardcoder model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        return input_ids, labels, None, None, None, None

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.backbone.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.backbone.blocks:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    def construct(self, input_ids, labels=None, input_mask=None, input_position=None,
                  init_reset=True, batch_valid_length=None):
        r"""
            construct function for Language Modeling

            Args:
                input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
                labels (Tensor): the indices of labels in the vocabulary.

            Returns:
                logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        batch_size, seq_length = self.shape(input_ids)
        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((batch_size, 1), mstype.int32)

        if self.training:
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
        else:
            tokens = input_ids

        input_mask = self.cast(self.not_equal(tokens, self.pad_token), self.dtype)
        attention_mask = self.get_attention_mask(input_mask)
        # if do not cast to bf16, loss will error
        attention_mask = self.cast(attention_mask, self.dtype)

        # [batch_size, seq_length, vocab_size]
        output_states, table = self.backbone(tokens, attention_mask, input_position, init_reset=init_reset,
                                             batch_valid_length=batch_valid_length)
        logits = self.head(output_states, table)

        if not self.training:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
            if (not self.use_past or self.is_first_iteration) and input_position is not None:
                logits = self.gather(logits, input_position, 0)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            # cast logits from bf16 to fp32 is caused by bf16 cannot asnumpy in text_generator.py
            logits = self.cast(logits, mstype.float32)
            return logits, tokens, input_mask

        if labels is None:
            labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        else:
            if self.training:
                labels = self.stridedslice(labels, (0, 1), (batch_size, seq_length), (1, 1))
            label_mask = self.cast(self.not_equal(labels, -100), self.dtype)
            input_mask = self.mul(input_mask, label_mask)

        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        # cast input_mask from bf16 to fp32 is caused by loss_reduce is fp32 in loss.py,
        # if you do not change it, it will error in pynative mode, but it will run success in graph mode.
        loss = self.loss(logits, labels, self.cast(input_mask, mstype.float32))

        return loss


class WizardCoderEmbeddingLayer(nn.Cell):
    r"""The Embedding Layer of WizardCoder network."""

    def __init__(self, config: WizardCoderConfig = None):
        super(WizardCoderEmbeddingLayer, self).__init__()
        parallel_config = copy.deepcopy(config.parallel_config)
        embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % embedding_mp != 0:
            logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
                           vocab_size, embedding_mp)
            logger.warning("Now, model_parallel will be changed: mp = 1")
            parallel_config.embedding_dp_mp_config.model_parallel = 1

        self.word_embedding = WizardCoderVocabEmbedding(vocab_size=vocab_size,
                                                        embedding_size=config.embedding_size,
                                                        param_init=initializer('normal',
                                                                               [vocab_size, config.embedding_size],
                                                                               dtype=config.param_init_type),
                                                        parallel_config=parallel_config.embedding_dp_mp_config)
        self.word_embedding.embedding_table.parallel_optimizer = True
        new_parallel_config = copy.deepcopy(parallel_config)
        new_parallel_config.vocab_emb_dp = True

        self.position_embedding = WizardCoderVocabEmbedding(vocab_size=config.n_position,
                                                            embedding_size=config.embedding_size,
                                                            param_init=initializer('normal',
                                                                                   [config.n_position,
                                                                                    config.embedding_size],
                                                                                   dtype=config.param_init_type),
                                                            parallel_config=new_parallel_config.embedding_dp_mp_config)
        dp = parallel_config.data_parallel
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.dropout = get_dropout(config.dropout_prob)
        self.dropout.dropout.shard(((dp, 1, 1),))

    def construct(self, input_ids, input_position):
        """The forward compute of Embedding Layer."""
        word_embedding, word_table = self.word_embedding(input_ids)
        position_embedding, _ = self.position_embedding(input_position)
        embedding = self.add(word_embedding, position_embedding)
        embedding = self.dropout(embedding)
        return embedding, word_table


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers, use_select_recompute):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    pp = parallel_config.pipeline_stage
    pp_dis = max(int(np.ceil((layers - 1) / pp)), 1)
    pp_remainder = layers % pp
    if pp_remainder > 0 and pp_dis != 1:
        if layer_id < (pp - pp_remainder) * (pp_dis - 1):
            pp_dis = pp_dis - 1
        else:
            layer_id = layer_id + pp - pp_remainder

    pp_id = min((layer_id + offset) // pp_dis, pp - 1)
    network.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if pp > 1:
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if not use_select_recompute:
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute:
                network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    else:
        network.attention.set_select_recompute()


class WizardCoderModel(WizardCoderPreTrainedModel):
    """
    The backbone of WizardCoder network

    Args:
        config(WizardCoderConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(WizardCoderModel, self).__init__(config)

        self.embedding = WizardCoderEmbeddingLayer(config)
        self.embedding.pipeline_stage = 0
        self.cast_rec = P.Cast()
        self.reshape_rec = P.Reshape()
        self.config = config
        self.is_first_iteration = True
        self.use_past = config.use_past

        self.layernorm = LayerNorm((config.embedding_size,), param_init_type=config.layernorm_dtype)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

        if config.use_select_recompute:
            self.layernorm.layer_norm.add_prim_attr("recompute_comm_op", True)

        if not hasattr(config.parallel_config, "moe_config"):
            config.parallel_config.moe_config = default_moe_config
        moe_config = config.parallel_config.moe_config

        self.blocks = nn.CellList()
        for i in range(config.num_layers):
            block = WizardCoderTransformerDecoderLayer(
                hidden_size=config.embedding_size,
                batch_size=config.batch_size,
                ffn_hidden_size=config.embedding_size * config.expand_ratio,
                seq_length=config.seq_length,
                num_heads=config.num_heads,
                attention_dropout_rate=config.attention_probs_dropout_prob,
                hidden_dropout_rate=config.hidden_dropout_prob,
                hidden_act=config.hidden_act,
                use_past=config.use_past,
                compute_dtype=config.compute_dtype,
                param_init_type=config.param_init_type,
                layernorm_compute_type=config.layernorm_dtype,
                softmax_compute_type=config.softmax_dtype,
                parallel_config=config.parallel_config.dp_mp_config,
                use_seq_parallel=config.use_seq_parallel,
                use_flash_attention=config.use_flash_attention,
                moe_config=moe_config)
            set_parallel_configure_for_layer(
                block, layer_id=i, layers=config.num_layers,
                offset=0, parallel_config=config.parallel_config,
                use_select_recompute=config.use_select_recompute)
            self.blocks.append(block)

        self.tile = P.Tile().shard(((config.parallel_config.data_parallel,),))
        self.dtype = config.compute_dtype
        self.num_layers = config.num_layers
        self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)
        self.bias = Tensor(np.arange(config.batch_size) * self.config.seq_length, mstype.int32)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.sub = P.Sub()

    def construct(self, input_ids, attention_mask, input_position=None, init_reset=False, batch_valid_length=None):
        """wizardcoder model"""
        batch_size, seq_length = self.shape(input_ids)
        if input_position is None or self.is_first_iteration:
            if batch_size == 1:
                input_position = self.reshape_rec(self.input_position, (1, seq_length))
            else:
                input_position = self.tile(self.input_position, (batch_size, 1))
        else:
            bias = Tensor(np.arange(batch_size) * self.config.seq_length, mstype.int32)
            input_position = self.sub(input_position, bias)
            input_position = self.reshape(input_position, (batch_size, 1))
        input_embedding, embedding_table = self.embedding(input_ids, input_position)

        hidden_states = self.cast_rec(input_embedding, self.dtype)
        hidden_shape = self.shape(hidden_states)
        hidden_states = self.reshape_rec(hidden_states, (-1, hidden_shape[-1]))

        for i in range(self.num_layers):
            hidden_states = self.blocks[i](hidden_states, attention_mask, init_reset=init_reset,
                                           batch_valid_length=batch_valid_length)
        output_state = self.layernorm(hidden_states)
        return output_state, embedding_table


class WizardCoderHead(nn.Cell):
    r"""Head for wizardcoder to get the logits of each token in the vocab."""

    def __init__(self,
                 vocab_size,
                 compute_dtype,
                 parallel_config=None):
        super().__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        mp = copied_parallel_config.model_parallel
        if vocab_size % mp != 0:
            logger.warning("The vocab size of WizardCoderHead MatMul is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of WizardCoderHead MatMul will be changed: mp = 1")
            copied_parallel_config.model_parallel = 1

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        dp, mp = copied_parallel_config.data_parallel, copied_parallel_config.model_parallel
        if copied_parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (mp, 1)))
        self.dtype = compute_dtype
        self.cast = P.Cast()

    def construct(self, state, table):
        logits = self.matmul(self.cast(state, self.dtype), self.cast(table, self.dtype))
        return logits
