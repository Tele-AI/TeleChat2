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
"""qformer implementation."""

import math
import os
from collections import OrderedDict
from typing import Optional

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, Parameter, Tensor, load_param_into_net, load_checkpoint

from mindformers import MindFormerBook
from mindformers.models.blip2.qformer import BertPreTrainedModel, BertOnlyMLMHead, CrossEntropyLoss, BertModel
from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.modules.layers import Dropout, LayerNorm, Linear
from mindformers.tools.download_tools import download_with_progress_bar
from mindformers.tools.logger import logger
from mindformers.tools.utils import try_sync_file

ACT2CLS = {
    "gelu": nn.GELU,
    "gelu_fast": nn.FastGelu,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class ClassInstanter(OrderedDict):
    """ClassInstanter for OrderedDict func-mapping input.

    Args:
        OrderedDict : function mapping.
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstanter(ACT2CLS)


def recursive_apply(module: nn.Cell, function_call):
    """apply cetain function to a nn.Cell
    module, recursively.

    Args:
        module (nn.Cell): model input.
        fn (function): function call
    """
    for submodule in module.cells():
        recursive_apply(submodule, function_call)
    function_call(module)


class BertEmbeddings(nn.Cell):
    """forward the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # delete for visualglm
        # self.layernorm = LayerNorm(
        #     (config.hidden_size,), eps=config.layer_norm_eps)
        # self.layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.dropout = Dropout(1. - config.hidden_dropout_prob)
        self.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.concat = P.Concat(axis=1)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        position_embeds = Tensor(
            [[i for i in range(config.max_position_embeddings)]], dtype=mstype.int32)
        self.position_ids = Parameter(
            position_embeds,
            requires_grad=False
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def construct(self, input_ids=None, position_ids=None, query_embeds=None, past_key_values_length=0):
        """forward the embeddings from word and position embeddings."""
        if input_ids is not None:
            seq_length = input_ids.shape[1]
        else:
            seq_length = 0

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length:
                                                 seq_length + past_key_values_length].copy()

            if self.position_embedding_type == "absolute" and self.position_embeddings:
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = self.concat((query_embeds, embeddings))
        else:
            embeddings = query_embeds

        # [bz, query_size, qformer_hidden_size]
        # delete for visualglm
        # embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Cell):
    """ BertSelfAttention """

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        self.softmax_dtype = config.softmax_dtype
        self.compute_dtype = config.compute_dtype
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if config.parallel_config:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
        else:
            dp = mp = 1

        self.query = Linear(in_channels=config.hidden_size,
                            out_channels=self.all_head_size,
                            compute_dtype=config.compute_dtype,
                            param_init_type=config.dtype
                            )
        self.query.shard(strategy_matmul=((dp, 1), (mp, 1)),
                         strategy_bias=((dp, mp), (mp,)))
        if is_cross_attention:
            self.key = Linear(
                in_channels=config.encoder_width,
                out_channels=self.all_head_size,
                compute_dtype=config.compute_dtype,
                param_init_type=config.dtype)

            self.value = Linear(
                in_channels=config.encoder_width,
                out_channels=self.all_head_size,
                compute_dtype=config.compute_dtype,
                param_init_type=config.dtype)
        else:
            self.key = Linear(
                in_channels=config.hidden_size,
                out_channels=self.all_head_size,
                compute_dtype=config.compute_dtype,
                param_init_type=config.dtype)
            self.value = Linear(
                in_channels=config.hidden_size,
                out_channels=self.all_head_size,
                compute_dtype=config.compute_dtype,
                param_init_type=config.dtype)
        self.key.shard(strategy_matmul=((dp, 1), (mp, 1)),
                       strategy_bias=((dp, mp), (mp,)))
        self.value.shard(strategy_matmul=((dp, 1), (mp, 1)),
                         strategy_bias=((dp, mp), (mp,)))

        self.dropout = Dropout(1. - config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" \
                or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1,
                                                   self.attention_head_size)
        self.save_attention = False

        self.einsum = P.Einsum("bhld,lrd->bhlr")
        self.einsum2 = P.Einsum("bhrd,lrd->bhlr")

        self.divider = math.sqrt(self.attention_head_size)
        self.cast = P.Cast()

        self.concat = P.Concat(axis=2)
        self.batch_matmul = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

        self.softmax = nn.Softmax(axis=-1)
        self.softmax.softmax.shard(((dp, mp, 1, 1),))
        self.transpose = P.Transpose().shard(((1, 1, 1, 1),))

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """ transpose input for scores output.

        Args:
            x (Tensor): input

        Returns:
            Tensor: output
        """
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return self.transpose(x, (0, 2, 1, 3))

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """ BertSelfAttention forwarding """

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        hidden_states = self.cast(hidden_states, self.compute_dtype)
        if is_cross_attention:
            # [batch_size, vit_seq_length, encoder_hidden_width]
            encoder_hidden_states = self.cast(
                encoder_hidden_states, self.compute_dtype)
            # [batch_size, num_attention_heads, vit_seq_length, attention_head_size]
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            # [batch_size, num_attention_heads, vit_seq_length, attention_head_size]
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = self.concat([past_key_value[0], key_layer])
            value_layer = self.concat([past_key_value[1], value_layer])
        else:
            # [batch_size, num_attention_heads, query_size, attention_head_size]
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # [batch_size, num_attention_heads, query_size, attention_head_size]
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # [batch_size, query_size, qformer_hidden_size]
        mixed_query_layer = self.query(hidden_states)
        #  [batch_size, num_attention_heads, query_size, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(tmp_shape) [batch_size, num_attention_heads, attention_head_size, query_size]
        # query_layer: [batch_size, num_attention_heads, query_size, attention_head_size]
        trans_key_layer = self.transpose(key_layer, (0, 1, 3, 2))
        attention_scores = self.batch_matmul(query_layer, trans_key_layer)

        if (
                self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.shape[1]
            position_ids_l = Tensor(
                [i for i in range(seq_length)], dtype=mstype.int32).view(-1, 1)
            position_ids_r = Tensor(
                [i for i in range(seq_length)], dtype=mstype.int32).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )

            if self.position_embedding_type == "relative_key":
                relative_position_scores = self.einsum(
                    query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = self.einsum(
                    query_layer, positional_embedding)
                relative_position_scores_key = self.einsum2(
                    key_layer, positional_embedding)
                attention_scores = (attention_scores + relative_position_scores_query + relative_position_scores_key)

        attention_scores /= self.divider
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.  [batch_size, num_heads, query_size, query_size]
        attention_scores = self.cast(attention_scores, self.softmax_dtype)
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        attention_probs_dropped = self.cast(
            attention_probs_dropped, self.compute_dtype)
        context_layer = self.batch_matmul(attention_probs_dropped, value_layer)

        # [batch_size, num_heads, query_size, attention_head_size]
        context_layer = self.transpose(context_layer, (0, 2, 1, 3)).copy()
        # [batch_size, query_size, all_head_size]
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Cell):
    """ BertSelfOutput """

    def __init__(self, config):
        super().__init__()
        if config.parallel_config:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
        else:
            dp = mp = 1

        self.dtype = config.dtype
        self.dense = Linear(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.dtype
        )
        self.dense.shard(strategy_matmul=((dp, mp), (1, mp)))
        self.layernorm = LayerNorm(
            (config.hidden_size,), eps=config.layer_norm_eps).shard(((dp, mp, 1),))
        self.dropout = Dropout(1. - config.hidden_dropout_prob)
        self.cast = P.Cast()

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.cast(hidden_states, self.dtype)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Cell):
    """ BertAttention """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self_att = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """
        hidden_states: [batch_size, query_size, qformer_hidden_size]
        attention_mask: [batch_size, 1, 1, query_size]
        encoder_hidden_states: [batch_size, vit_seq_length, vit_hidden_size]
        encoder_attention_mask: [batch_size, 1, 1, vit_seq_length]
        """

        # self_outputs.shape ([batch_size, query_size, qformer_hidden_size],
        # ([batch_size, num_head, query_size, head_size], [batch_size, num_head, query_size, head_size]))
        self_outputs = self.self_att(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)

        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Cell):
    """ BertIntermediate """

    def __init__(self, config):
        super().__init__()
        if config.parallel_config:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
        else:
            dp = mp = 1

        self.dense = Linear(
            in_channels=config.hidden_size,
            out_channels=config.intermediate_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.dtype
        )
        self.dense.shard(strategy_matmul=((dp, mp), (1, mp)))

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
            if isinstance(self.intermediate_act_fn, nn.GELU):
                self.intermediate_act_fn = nn.GELU(approximate=False)
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):
    """ BertOutput """

    def __init__(self, config):
        super().__init__()
        if config.parallel_config:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
        else:
            dp = mp = 1

        self.dense = Linear(
            in_channels=config.intermediate_size,
            out_channels=config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.dtype)
        self.dense.shard(strategy_matmul=((dp, mp), (1, mp)))

        self.layernorm = LayerNorm(
            (config.hidden_size,), eps=config.layer_norm_eps).shard(((dp, 1, 1),))
        self.dropout = Dropout(1. - config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BertLayer(nn.Cell):
    """ BertLayer """

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.input_layernorm = LayerNorm(
            (config.hidden_size,), eps=config.layer_norm_eps).shard(((config.parallel_config.data_parallel, 1, 1),))
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0:
            self.crossattention = BertAttention(config,
                                                is_cross_attention=self.config.add_cross_attention)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

        self.concat = P.Concat(axis=1)
        self.concat_seq = P.Concat(axis=self.seq_len_dim)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            query_length=0,
    ):
        """
        hidden_states: [batch_size, query_size, qformer_hidden_size]
        attention_mask: [batch_size, 1, 1, query_size]
        encoder_hidden_states: [batch_size, vit_seq_length, vit_hidden_size ]
        encoder_attention_mask: [batch_size, 1, 1, vit_seq_length]
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        # add by visualglm
        hidden_states = self.input_layernorm(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            None,
            None,
            self_attn_past_key_value,
            output_attentions,
        )
        # [batch_size, query_size, qformer_hidden_size]
        attention_output = self_attention_outputs[0]
        # ([batch_size, num_head, query_size, head_size], [batch_size, num_head, query_size, head_size]))
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            # [batch_size, query_size, qformer_hidden_size]
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers.")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    output_attentions,
                )
                # [batch_size, query_size, qformer_hidden_size]
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            # [batch_size, query_size, qformer_hidden_size]
            layer_output = self.apply_chunking_to_forward(self.feed_forward_chunk_query,
                                                          query_attention_output)
            if attention_output.shape[1] > query_length:
                layer_output_text = self.apply_chunking_to_forward(self.feed_forward_chunk,
                                                                   attention_output[:, query_length:, :])
                layer_output = self.concat([layer_output, layer_output_text])
        else:
            layer_output = self.apply_chunking_to_forward(
                self.feed_forward_chunk, attention_output)
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """ apply feed_forward with chunks """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        """ apply feed_forward with chunks (query) """
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

    def apply_chunking_to_forward(self, forward_fn, *input_tensors):
        """ apply chunking to forward computation """
        if input_tensors is None:
            raise ValueError(f"input_tensors has to be a tuple/list of tensors.")

        if self.chunk_size_feed_forward > 0:
            tensor_shape = input_tensors[0].shape[self.seq_len_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[self.seq_len_dim] != tensor_shape:
                    raise ValueError(
                        f"All input tenors have to be of the same shape: {tensor_shape}, "
                        f"found shape {input_tensor.shape[self.seq_len_dim]}"
                    )

            if input_tensors[0].shape[self.seq_len_dim] % self.chunk_size_feed_forward != 0:
                raise ValueError(
                    f"The dimension to be chunked {input_tensors[0].shape[self.seq_len_dim]} "
                    f"has to be a multiple of the chunk size {self.chunk_size_feed_forward}"
                )

            num_chunks = input_tensors[0].shape[self.seq_len_dim] // self.chunk_size_feed_forward

            # chunk input tensor into tuples
            input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=self.seq_len_dim)
                                         for input_tensor in input_tensors)
            # apply forward fn to every tuple
            output_chunks = tuple(forward_fn(*input_tensors_chunk)
                                  for input_tensors_chunk in zip(*input_tensors_chunks))
            # concatenate output at same dimension
            return self.concat_seq(output_chunks)

        return forward_fn(*input_tensors)


class BertEncoder(nn.Cell):
    """ BertEncoder """

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.CellList(
            [BertLayer(config, i)
             for i in range(self.config.num_hidden_layers)]
        )
        self.num_hidden_layers = [
            i for i in range(self.config.num_hidden_layers)]

        # add for layernorm
        self.final_layernorm = LayerNorm(
            (config.hidden_size,), eps=config.layer_norm_eps).shard(((config.parallel_config.data_parallel, 1, 1),))

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            query_length=0,
    ):
        """
        attention_mask: [batch_size, 1, 1, query_size]
        encoder_hidden_states: [batch_size, vit_seq_length, encoder_hidden_width]
        encoder_attention_mask: [batch_size, 1, 1, vit_seq_length]
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        for i in self.num_hidden_layers:
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # layer_outputs shape  ([batch_size, query_size, qformer_hidden_size],
            # ([batch_size, num_head, query_size, head_size], [batch_size, num_head, query_size, head_size]))
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                query_length,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + \
                                       (layer_outputs[2],)

        # add for visualglm
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        value_list = [hidden_states, next_decoder_cache,
                      all_hidden_states, all_self_attentions, all_cross_attentions]
        return tuple(value_list)


class BertModelVisualGLM(BertModel):
    """
    BertModel adaptor for visualglm
    """

    def __init__(self, config):
        super(BertModelVisualGLM, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)


class BertLMHeadModel(BertPreTrainedModel):
    """ BertLMHeadModel, the main model for Qformer

    Args:
        config (QFormerConfig): config for qformer, see qformer_config.py.

    Raises:
        ValueError: config type Error.

    Returns:
        a BertLMHeadModel instance.
    """
    _support_list = ["bert_base_uncased", "bert_base_uncased_resized"]

    def __init__(self, config: QFormerConfig):
        super(BertLMHeadModel, self).__init__(config)
        if not isinstance(config, QFormerConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` "
                "should be an instance of class `QFormerConfig`. "
                "To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        self.config = config
        self.bert = BertModelVisualGLM(config)
        self.cls = BertOnlyMLMHead(config)

        if self.config.checkpoint_name_or_path:
            self.load_checkpoint(config)

        # for lm_loss reduction - GRAPH_MODE
        self.reduction = config.loss_reduction
        self.loss = CrossEntropyLoss(
            reduction=self.reduction, label_smoothing=0.1)
        self.vocab_size = self.config.vocab_size

    def convert_bert_model_params(self, bert_model_params: OrderedDict):
        """
        convert params from BertModel in MindFormers, some param names are altered.
        """
        dict_mapping = {'layer.': 'blocks.',
                        'encoder.': 'bert_encoder.encoder.',
                        'self_att.query.': 'dense1.',
                        'self_att.key.': 'dense2.',
                        'self_att.value.': 'dense3.',
                        'attention.output.dense.': 'attention.projection.',
                        'attention.output.layernorm.gamma': 'layernorm2.gamma',
                        'attention.output.layernorm.beta': 'layernorm2.beta',
                        'intermediate.dense.weight': 'output.mapping.weight',
                        'intermediate.dense.bias': 'output.mapping.bias',
                        'output.dense.weight': 'output.projection.weight',
                        'output.dense.bias': 'output.projection.bias',
                        'output.layernorm.gamma': 'layernorm1.gamma',
                        'output.layernorm.beta': 'layernorm1.beta',
                        'embeddings.position_embeddings.embedding_table': \
                            'embedding_postprocessor.full_position_embedding.embedding_table',
                        'embeddings.layernorm.gamma': 'embedding_postprocessor.layernorm.gamma',
                        'embeddings.layernorm.beta': 'embedding_postprocessor.layernorm.beta',
                        'embeddings.word_embeddings.embedding_table': 'word_embedding.embedding_table',
                        'cls.predictions.transform.dense': 'bert.mlmloss.dense',
                        'cls.predictions.transform.layernorm': 'bert.mlmloss.layernorm',
                        'cls.predictions.decoder': 'bert.mlmloss.vocab_dense'}

        param_dict = self.parameters_dict()
        for name, data in param_dict.items():
            new_name = name
            for replace_from, replace_to in dict_mapping.items():
                new_name = new_name.replace(replace_from, replace_to)
            if new_name not in bert_model_params.keys():
                logger.warning("%s not loaded.", name)
                continue
            new_data = bert_model_params[new_name]
            if name.endswith("intermediate.dense.weight") or name.endswith("output.dense.weight"):
                new_data = new_data.T
            data.assign_value(new_data)

    def load_bert_model_params(self, config: QFormerConfig, param):
        """
        load parameters for BertLMHeadModel, if the weights come from
        mindformers.models.bert.BertModel, param conversion is needed.

        Args:
            config (QFormerConfig): config for the Q-Former model.
            param (OrderedDict): the params to be loaded.
        """
        if config.resize_token_embeddings and config.convert_param_from_bert:
            self.convert_bert_model_params(param)
        else:
            load_param_into_net(self, param)

    def load_checkpoint(self, config: QFormerConfig):
        """
        load checkpoint for BertLMHeadModel. (we can use the param for BertModel on obs,
        but we need to alter the names of some param)

        Args:
            config (ModelConfig): QFormerConfig instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        # the relevant file will be downloaded from the Obs platform.
        if not os.path.exists(checkpoint_name_or_path):
            if checkpoint_name_or_path not in self._support_list:
                raise ValueError(f"{checkpoint_name_or_path} is not a supported default model"
                                 f" or a valid path to checkpoint,"
                                 f" please select from {self._support_list}.")
            # on Atlas 800T A2, load the 'resized' checkpoint.
            if not config.resize_token_embeddings and not checkpoint_name_or_path.endswith("_resized"):
                checkpoint_name_or_path = checkpoint_name_or_path + "_resized"
            checkpoint_name = checkpoint_name_or_path
            default_checkpoint_download_folder = os.path.join(
                MindFormerBook.get_default_checkpoint_download_folder(),
                checkpoint_name_or_path.split("_")[0])
            if not os.path.exists(default_checkpoint_download_folder):
                os.makedirs(default_checkpoint_download_folder, exist_ok=True)

            ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name + ".ckpt")
            if not os.path.exists(ckpt_file):
                url = MindFormerBook.get_model_ckpt_url_list()[checkpoint_name_or_path][0]
                succeed = download_with_progress_bar(url, ckpt_file)
                if not succeed:
                    logger.info("checkpoint download failed, and pretrained weights are unloaded.")
                    return
            try_sync_file(ckpt_file)
            self.default_checkpoint_download_path = ckpt_file
            logger.info("start to read the ckpt file: %s", os.path.getsize(ckpt_file))
        else:
            ckpt_file = checkpoint_name_or_path
        param = load_checkpoint(ckpt_file)
        try:
            self.load_bert_model_params(config, param)
            logger.info("weights in %s are loaded", ckpt_file)
        except RuntimeError:
            logger.error("the given config and weights in %s are"
                         " mismatched, and weights load failed", ckpt_file)

    def get_input_embeddings(self) -> nn.Cell:
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.bert.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Cell:
        return self.cls.predictions.decoder

    def set_output_embeddings(self, value):
        self.cls.predictions.decoder = value

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        resize token embeddings, inherit from super class.
        """
        old_num_tokens = self.get_input_embeddings().embedding_table.shape[0]
        logger.info("resize_token_embeddings from %d to %d.", old_num_tokens, new_num_tokens)
        super(BertLMHeadModel, self).resize_token_embeddings(new_num_tokens)

    def tie_weights(self):
        """
        tie encoder and decoder weights, inherit from super class.
        """
        logger.info("weights tied.")
        super(BertLMHeadModel, self).tie_weights()

    # pylint: disable=W0613
    def construct(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                  query_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None,
                  past_key_values=None, use_cache=True, output_attentions=None, output_hidden_states=None,
                  return_dict=None, return_logits=False, is_decoder=True):
        """
        construct function for QFormer.

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            position_ids (Tensor): used to identify each token's position in the list of tokens.
            attention_mask (Tensor): used when batching sequences together.
            query_embeds (Tensor): to be supplemented.
            return_dict(bool): Reserved param, not used.
            head_mask (Tensor): to be supplemented.
            encoder_hidden_states (`Tensor` of shape : (batch_size, sequence_length, hidden_size)`)
                Sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention if the model is configured as a decoder.
            encoder_attention_mask (`Tensor` of shape : (batch_size, sequence_length)`, `optional`))
                Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in ``[0, 1]``:
                1 for tokens that are **not masked**,
                0 for tokens that are **masked**.
            past_key_values: Reserved param, not used.
            labels (`Tensor(mstype.int32)` of shape : (batch_size, sequence_length)`, `optional`))
                Labels for computing the left-to-right language modeling loss (next word prediction).
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is
                only computed for the tokens with labels n ``[0, ..., config.vocab_size]``,
                past_key_values (:obj:`tuple(tuple(Tensor(mstype.float)))` of length:
                `config.n_layers` with each tuple having 4 tensors of shape
                (batch_size, num_heads, sequence_length - 1, embed_size_per_head)),
                Contains precomputed key and value hidden states of the attention blocks.
                Can be used to speed up decoding. If :obj:`past_key_values` are used, the user
                can optionally input only the last :obj:`decoder_input_ids`
                (those that don't have their past key value states given to this model) of
                shape (batch_size, 1)` instead of all :obj:`decoder_input_ids` of shape
                (batch_size, sequence_length)`.
            use_cache (bool, `optional`, default is True):
                If set to :obj:`True`, :obj:`past_key_values` key value states are returned
                and can be used to speed up decoding (see :obj:`past_key_values`).
            output_attentions (bool, `optional`, default is None):
                whether to append self-attentions as a part of outputs in the BertSelfAttention layer.
            output_hidden_states (bool, `optional`, default is None):
                whether to return all hidden states in the output of the BertEncoder layer.
            return_logits (bool, `optional`, default is False):
                whether to only return prediction_scores other than lm_loss as output.
            is_decoder (bool, `optional`, default is True):
                specify whether the BertModel is encoder or decoder.

        Returns:
            output (tuple of Tensors):
                if return_logits is True, directly return prediction_scores as output.
                if label input is not None, return lm_loss, prediction_scores and BertModel outputs
                (except sequence_output), otherwise return prediction_scores and BertModel outputs
                (except sequence_output) as output.
        """

        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].copy()

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].copy()
            labels = labels[:, 1:].copy()
            lm_loss = self.loss(
                shifted_prediction_scores.view(-1, self.vocab_size),
                labels.view(-1),
            )
            if self.reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        output = (prediction_scores,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output
