# Copyright 2022 Huawei Technologies Co., Ltd
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
# This file was refer to project:
# https://gitee.com/mindspore/models/blob/master/official/nlp/Bert/src/bert_model.py
# ============================================================================
"""BERT Model."""
import numpy as np
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore import Tensor, Parameter
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindformers.modules.layers import LayerNorm
from mindformers.modules.transformer import Transformer, VocabEmbedding
from mindformers.modules.transformer.moe import default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.bert.bert_config import BertConfig
from .bert_config import BertConfig

__all__ = ['BertConfig', 'BertModel', 'BertForPreTraining', 'BertForTokenClassification', 'BertForMultipleChoice',
           'BertForQuestionAnswering']


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertForTokenClassification(BertPreTrainedModel):
    """
    Bert with dense layer for name entity recoginition task.

    Args:
        config (BertConfig): The config of BertForTokenClassification.

    Returns:
        Tensor, loss, logits.

    Examples:
        >>> from mindformers import BertForTokenClassification
        >>> model = BertForTokenClassification.from_pretrained('tokcls_bert_base_chinese')
        >>> type(model)
        <class 'mindformers.models.bert.bert.BertForTokenClassification'>
    """

    _support_list = MindFormerBook.get_model_support_list()['tokcls']['bert']

    def __init__(self, config=BertConfig()):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, config.use_one_hot_embeddings)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels).to_float(config.compute_dtype)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.reshape = P.Reshape()
        self.load_checkpoint(config)

    def construct(self, input_ids, input_mask, token_type_ids, label_ids=None):
        """Get Training Loss or Logits"""
        bert_outputs = self.bert(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.training:
            logits = self.reshape(logits, (-1, self.num_labels))
            label_ids = self.reshape(label_ids, (-1,))
            output = self.cross_entropy_loss(logits, label_ids)
        else:
            if label_ids is None:
                output = logits
            else:
                output = (logits, label_ids)

        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertForPreTraining(BertPreTrainedModel):
    """
    Provide bert pre-training loss through network.

    Args:
        config (BertConfig): The config of BertForPreTraining.
    Returns:
        Tensor, the loss of the network.

    Examples:
        >>> from mindformers import BertForPreTraining
        >>> model = BertForPreTraining.from_pretrained('bert_base_uncased')
        >>> type(model)
        <class 'mindformers.models.bert.bert.BertForPreTraining'>
    """

    _support_list = MindFormerBook.get_model_support_list()['bert']

    def __init__(self, config=BertConfig()):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config, config.use_one_hot_embeddings)
        self.mlmloss = GetMaskedLMOutput(config)
        self.nsploss = GetNextSentenceOutput(config)
        self.loss = BertLoss(config)
        self.cast = P.Cast()
        self.use_moe = (config.parallel_config.moe_config.expert_num > 1)
        self.add = P.Add().shard(((1,), ()))
        self.load_checkpoint(config)

    def bert_forward(self, input_ids, input_mask, token_type_id,
                     masked_lm_positions=None):
        """connect backbone and heads."""
        moe_loss = 0
        if self.use_moe:
            sequence_output, pooled_output, embedding_table, moe_loss = \
                self.bert(input_ids, token_type_id, input_mask)
        else:
            sequence_output, pooled_output, embedding_table = \
                self.bert(input_ids, token_type_id, input_mask)

        prediction_scores = self.mlmloss(sequence_output,
                                         embedding_table,
                                         masked_lm_positions)
        seq_relationship_score = self.nsploss(pooled_output)
        if not self.training:
            return sequence_output, pooled_output, prediction_scores, seq_relationship_score
        return prediction_scores, seq_relationship_score, moe_loss

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels=None,
                  masked_lm_positions=None,
                  masked_lm_ids=None,
                  masked_lm_weights=None):
        """Get pre-training loss"""
        if not self.training:
            return self.bert_forward(input_ids, input_mask, token_type_id, masked_lm_positions)

        prediction_scores, seq_relationship_score, moe_loss = \
            self.bert_forward(input_ids, input_mask, token_type_id, masked_lm_positions)
        total_loss = self.loss(prediction_scores, seq_relationship_score,
                               masked_lm_ids, masked_lm_weights, next_sentence_labels)
        if self.use_moe:
            total_loss = self.add(total_loss, moe_loss)
        return self.cast(total_loss, mstype.float32)


class BertModel(BertPreTrainedModel):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 config=None,
                 use_one_hot_embeddings=False):
        super(BertModel, self).__init__(config)
        if not self.training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.get_attention_mask = CreateAttentionMaskFromInputMask(config)
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None
        if not hasattr(config.parallel_config, "moe_config"):
            config.parallel_config.moe_config = default_moe_config
        moe_config = config.parallel_config.moe_config
        self.use_moe = (config.parallel_config.moe_config.expert_num > 1)

        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("truncatedNormal",
                                                                    [config.vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)

        output_embedding_shape = [-1, config.seq_length, self.embedding_size]

        self.embedding_postprocessor = EmbeddingPostprocessor(config,
                                                              embedding_size=self.embedding_size,
                                                              embedding_shape=output_embedding_shape,
                                                              use_relative_positions=config.use_relative_positions,
                                                              use_token_type=True,
                                                              token_type_vocab_size=config.type_vocab_size,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              max_position_embeddings=config.max_position_embeddings,
                                                              dropout_prob=config.hidden_dropout_prob)


        self.bert_encoder = Transformer(
            hidden_size=config.hidden_size,
            batch_size=config.batch_size,
            ffn_hidden_size=config.intermediate_size,
            src_seq_length=config.seq_length,
            tgt_seq_length=config.seq_length,
            num_heads=config.num_attention_heads,
            encoder_layers=config.num_hidden_layers,
            parallel_config=config.parallel_config,
            decoder_layers=0,
            moe_config=moe_config,
            param_init_type=config.compute_dtype,
            layernorm_compute_type=config.layernorm_dtype,
            softmax_compute_type=config.softmax_dtype,
            post_layernorm_residual=config.post_layernorm_residual,
            attention_dropout_rate=config.attention_probs_dropout_prob,
            hidden_dropout_rate=config.hidden_dropout_prob,)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_dtype)
        self.slice = P.StridedSlice().shard(((1, 1, 1),))

        self.squeeze_1 = P.Squeeze(axis=1)
        self.dense = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_dtype)
        self.print = P.Print()

    def construct(self, input_ids, token_type_ids, input_mask):
        """Bidirectional Encoder Representations from Transformers."""
        # embedding
        word_embeddings, embedding_tables = self.word_embedding(input_ids)
        embedding_output = self.embedding_postprocessor(token_type_ids, word_embeddings)

        # attention mask [batch_size, seq_length, seq_length](4, 1, 128) -> (4, 128, 128)
        input_mask = P.Cast()(input_mask, self.dtype)
        attention_mask = self.get_attention_mask(input_mask)
        # bert encoder
        encoder_output = self.bert_encoder(self.cast_compute_type(embedding_output), attention_mask)
        sequence_output = encoder_output[0]
        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)
        moe_loss = 0
        if self.use_moe:
            moe_loss = encoder_output[-1]
            return sequence_output, pooled_output, embedding_tables, moe_loss
        return sequence_output, pooled_output, embedding_tables


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertForMultipleChoice(BertPreTrainedModel):
    """
    Bert with dense layer for txt classification task.

    Args:
        config (BertConfig): The config of BertForMultipleChoice.

    Returns:
        Tensor, loss, logits.

    Examples:
        >>> from mindformers import BertForMultipleChoice
        >>> model = BertForMultipleChoice.from_pretrained('txtcls_bert_base_uncased')
        >>> type(model)
        <class 'mindformers.models.bert.bert.BertForMultipleChoice'>
    """
    _support_list = MindFormerBook.get_model_support_list()['txtcls']['bert']

    def __init__(self, config=BertConfig()):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, config.use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.classifier = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                   has_bias=True).to_float(config.compute_dtype)
        self.dropout = nn.Dropout(1 - config.dropout_prob)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.dtype = config.dtype
        self.compute_dtype = config.compute_dtype
        self.load_checkpoint(config)

    def construct(self, input_ids, input_mask, token_type_id, label_ids=None):
        """Get Training Loss or Logits"""
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.training:
            logits = self.reshape(logits, (-1, self.num_labels))
            label_ids = self.reshape(label_ids, (-1,))
            output = self.cross_entropy_loss(logits, label_ids)
        else:
            if label_ids is None:
                output = logits
            else:
                label_ids = self.reshape(label_ids, (-1,))
                output = (logits, label_ids)

        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertForQuestionAnswering(BertPreTrainedModel):
    """Bert with dense layer for question answering task.

        Args:
            config (BertConfig): The config of BertModel.

        Returns:
            Tensor, loss, logits.

        Examples:
            >>> from mindformers import BertForQuestionAnswering
            >>> model = BertForQuestionAnswering.from_pretrained('qa_bert_base_uncased')
            >>> type(model)
            <class 'mindformers.models.bert.bert.BertForQuestionAnswering'>
    """
    _support_list = MindFormerBook.get_model_support_list()['qa']['bert']

    def __init__(self, config=BertConfig()):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config, config.use_one_hot_embeddings)
        self.qa_outputs = nn.Dense(config.hidden_size, 2).to_float(config.compute_dtype)
        self.load_checkpoint(config)
        self.split = P.Split(output_num=2, axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, start_position=None, end_position=None, unique_id=None):
        """Get Training Loss or Logits"""
        bert_outputs = self.bert(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_id)
        sequence_output = bert_outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = self.split(logits)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.training:
            if len(start_position.shape) > 1:
                start_position = start_position.squeeze(-1)
            if len(end_position.shape) > 1:
                end_position = end_position.squeeze(-1)
            ignored_index = start_logits.shape[1]
            start_position = ops.clip_by_value(start_position, 0, ignored_index)
            end_position = ops.clip_by_value(end_position, 0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_position)
            end_loss = loss_fct(end_logits, end_position)
            total_loss = (start_loss + end_loss) / 2
            output = total_loss
        else:
            output = (unique_id, start_logits, end_logits)

        return output


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: False.
        token_type_vocab_size (int): Size of token type vocab. Default: 16.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self, config,
                 embedding_size,
                 embedding_shape,
                 use_relative_positions=False,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_one_hot_embeddings=False,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.token_type_embedding = VocabEmbedding(vocab_size=token_type_vocab_size,
                                                   embedding_size=embedding_size,
                                                   param_init=initializer("truncatedNormal",
                                                                          [token_type_vocab_size, embedding_size],
                                                                          dtype=mstype.float32),
                                                   parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.shape_flat = (-1,)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.gather = P.Gather()
        self.use_relative_positions = use_relative_positions
        self.slice = P.StridedSlice().shard(((1, 1),))
        _, seq, _ = self.shape
        self.full_position_embedding = VocabEmbedding(vocab_size=config.max_position_embeddings,
                                                      embedding_size=embedding_size,
                                                      param_init=initializer("truncatedNormal",
                                                                             [config.max_position_embeddings,
                                                                              embedding_size],
                                                                             dtype=mstype.float32),
                                                      parallel_config=config.parallel_config.embedding_dp_mp_config)

        self.layernorm = LayerNorm((embedding_size,)).to_float(mstype.float32)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.position_ids = Tensor(np.arange(seq).reshape(-1, seq).astype(np.int32))
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.slice = P.StridedSlice().shard(((1, 1),))

    def construct(self, token_type_ids, word_embeddings):
        """Postprocessors apply positional and token type embeddings to word embeddings."""
        output = word_embeddings
        if self.use_token_type:
            token_type_embeddings, _ = self.token_type_embedding(token_type_ids)
            output = self.add(output, token_type_embeddings)
        if not self.use_relative_positions:
            shape = F.shape(output)
            shape_position = F.shape(self.position_ids)
            position_ids = self.slice(self.position_ids, (0, 0), (shape_position[0], shape[1]), (1, 1))
            position_embeddings, _ = self.full_position_embedding(position_ids)
            output = self.add(output, position_embeddings)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """
    def __init__(self, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertForPreTraining.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask = None
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.tile = mindspore.ops.Tile().shard(((config.parallel_config.data_parallel, 1, 1),))

    def construct(self, input_mask):
        seq_length = F.shape(input_mask)[1]
        attention_mask = self.cast(self.reshape(input_mask, (-1, 1, seq_length)), mstype.float16)
        attention_mask = self.tile(attention_mask, (1, seq_length, 1))
        return attention_mask


class BertLoss(nn.Cell):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertForPreTraining.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, config):
        super(BertLoss, self).__init__()
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot().shard(((config.parallel_config.data_parallel, 1), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum().shard(((config.parallel_config.data_parallel,),))
        self.reduce_sum1 = P.ReduceSum().shard(((config.parallel_config.data_parallel, 1),))
        self.reduce_mean = P.ReduceMean().shard(((config.parallel_config.data_parallel,),))
        self.reduce_mean2 = P.ReduceMean().shard(((1,),))
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg().shard(((config.parallel_config.data_parallel,),))
        self.cast = P.Cast()
        self.div = P.RealDiv().shard(((), (1,)))
        self.add = P.Add().shard(((1,), ()))
        self.mul = P.Mul().shard(((config.parallel_config.data_parallel, 1), (config.parallel_config.data_parallel, 1)))
        self.mul2 = P.Mul().shard(((config.parallel_config.data_parallel,), (config.parallel_config.data_parallel,)))

    def construct(self, prediction_scores, seq_relationship_score, masked_lm_ids,
                  masked_lm_weights, next_sentence_labels):
        """Defines the computation performed."""
        label_ids = self.reshape(masked_lm_ids, self.last_idx)
        label_weights = self.cast(self.reshape(masked_lm_weights, self.last_idx), mstype.float32)
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum1(self.mul(prediction_scores, one_hot_labels), self.last_idx))
        numerator = self.reduce_sum(self.mul2(label_weights, per_example_loss), ())
        denominator = self.reduce_sum(label_weights, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        masked_lm_loss = self.div(numerator, denominator)

        # next_sentence_loss
        labels = self.reshape(next_sentence_labels, self.last_idx)
        one_hot_labels = self.onehot(labels, 2, self.on_value, self.off_value)
        per_example_loss = self.neg(self.reduce_sum1(
            self.mul(one_hot_labels, seq_relationship_score), self.last_idx))
        next_sentence_loss = self.reduce_mean2(per_example_loss, self.last_idx)
        # total_loss
        total_loss = self.add(masked_lm_loss, next_sentence_loss)

        return total_loss


class GetMaskedLMOutput(nn.Cell):
    """
    Get masked lm output.

    Args:
        config (BertConfig): The config of BertForPreTraining.

    Returns:
        Tensor, masked lm output.
    """

    def __init__(self, config):
        super(GetMaskedLMOutput, self).__init__()
        self.width = config.hidden_size
        self.reshape = P.Reshape()
        self.gather = P.Gather()
        self.gather.shard(((1, 1), (1,)))
        weight_init = TruncatedNormal(config.initializer_range)


        self.dense = nn.Dense(self.width,
                              config.hidden_size,
                              weight_init=weight_init,
                              activation=config.hidden_act).to_float(config.compute_dtype)

        if config.parallel_config.vocab_emb_dp:
            self.dense.matmul.shard(((config.parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.dense.matmul.shard(((config.parallel_config.data_parallel, 1), (
                1, config.parallel_config.model_parallel)))

        self.layernorm = LayerNorm((config.hidden_size,)).to_float(config.compute_dtype)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.output_bias = Parameter(
            initializer(
                'zero',
                config.vocab_size))
        self.matmul = P.MatMul(transpose_b=True)
        self.matmul.shard(((config.parallel_config.data_parallel, 1), (1, 1)))
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.log_softmax.log_softmax.shard(((config.parallel_config.data_parallel, 1),))
        self.shape_flat_offsets = (-1, 1)
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (-1, self.width)
        self.cast = P.Cast()
        self.compute_dtype = config.compute_dtype
        self.dtype = config.dtype

    def construct(self,
                  input_tensor,
                  output_weights,
                  positions):
        """Get output log_probs"""
        input_shape = P.Shape()(input_tensor)
        rng = F.tuple_to_array(F.make_range(input_shape[0]))
        flat_offsets = self.reshape(rng * input_shape[1], self.shape_flat_offsets)
        flat_position = self.reshape(positions + flat_offsets, self.last_idx)
        flat_sequence_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.gather(flat_sequence_tensor, flat_position, 0)
        input_tensor = self.cast(input_tensor, self.compute_dtype)
        output_weights = self.cast(output_weights, self.compute_dtype)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.layernorm(input_tensor)
        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)
        logits = logits + self.output_bias
        log_probs = self.log_softmax(logits)
        return log_probs


class GetNextSentenceOutput(nn.Cell):
    """
    Get next sentence output.

    Args:
        config (BertConfig): The config of Bert.

    Returns:
        Tensor, next sentence output.
    """

    def __init__(self, config):
        super(GetNextSentenceOutput, self).__init__()
        self.log_softmax = P.LogSoftmax()
        self.log_softmax.shard(((config.parallel_config.data_parallel, 1),))
        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = nn.Dense(config.hidden_size, 2,
                              weight_init=weight_init, has_bias=True).to_float(config.compute_dtype)

        if config.parallel_config.vocab_emb_dp:
            self.dense.matmul.shard(((config.parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.dense.matmul.shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                     (config.parallel_config.model_parallel, 1)))

        self.dtype = config.dtype
        self.cast = P.Cast()

    def construct(self, input_tensor):
        logits = self.dense(input_tensor)
        logits = self.cast(logits, self.dtype)
        log_prob = self.log_softmax(logits)
        return log_prob
