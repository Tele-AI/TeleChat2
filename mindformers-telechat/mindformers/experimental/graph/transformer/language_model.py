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
""" For language model """
from typing import Union

from mindspore import nn
from mindspore.ops import operations as P

from mindformers.experimental.graph.transformer.enums import AttnMaskType
from mindformers.experimental.utils import init_method_normal, scaled_init_method_normal
from mindformers.experimental.graph.tensor_parallel.layers import ColumnParallelLinear
from mindformers.experimental.graph.transformer.dropout import Dropout
from mindformers.experimental.graph.tensor_parallel.layers import VocabParallelEmbedding
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.graph.transformer.transformer import ParallelTransformer, CausalMaskGenerate

__all__ = [
    "Pooler",
    "Embedding",
    "TransformerLanguageModel",
    "get_language_model"
]


class Pooler(nn.Cell):
    r"""
    Pooler layer.
    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Args:
        hidden_size: hidden states size for embedding layer
        init_method: weight initialization method for the linear layer. bias is set to zero.
        config: configuration

    Inputs:
        hidden_states: hidden_size
        sequence_index: sequence index to pools
    """

    def __init__(self, hidden_size, init_method, config: TransformerConfig):
        super(Pooler, self).__init__()
        self.compute_dtype = config.compute_dtype
        self.dense = ColumnParallelLinear(hidden_size, hidden_size, config, bias=True,
                                          compute_dtype=self.compute_dtype,
                                          is_expert=False,
                                          skip_bias_add=True,
                                          init_method=init_method
                                          )

    def construct(self, hidden_states, sequence_index=0):
        """Pooler construct"""
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
        pooled, _ = self.dense(pooled)
        return pooled


class Embedding(nn.Cell):
    r"""
    A embedding layer contain word embediing, position embedding and tokentypes embedding.

    Args:
        hidden_size: hidden states size for embedding layer
        vocab_size: vocabulary size
        max_sequence_length: if using position embedding, it is necessary to set the maximum sequence length
        embedding_dropout_prob: dropout rate for embedding layer
        config: configuration
        num_tokentypes: if > 0, using tokentypes embedding

    Inputs:
        input_ids: input ids
        position_ids: position ids
        tokentype_ids: tokentype ids

    Outputs:
        embeddings: the embedding output
        word_embedding_table: the word embedding table

    Supported Platforms:
        Ascend
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config: TransformerConfig,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.embedding_init_type = config.embedding_init_type
        self.compute_dtype = config.compute_dtype
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method

        # Word embedding
        self.word_embeddings = VocabParallelEmbedding(vocab_size,
                                                      hidden_size,
                                                      config,
                                                      init_method=self.init_method,
                                                      init_type=self.embedding_init_type)
        # Position embedding
        self.add_position_embedding = config.position_embedding_type == 'learned_absolute'
        if self.add_position_embedding:
            self.position_embeddings = VocabParallelEmbedding(max_sequence_length,
                                                              hidden_size,
                                                              config,
                                                              init_method=self.init_method,
                                                              init_type=self.embedding_init_type)
        # tokentypes embedding
        if num_tokentypes > 0:
            self.tokentype_embedding = VocabParallelEmbedding(num_tokentypes,
                                                              hidden_size,
                                                              config,
                                                              init_method=self.init_method,
                                                              init_type=self.embedding_init_type)
        else:
            self.tokentype_embedding = None
        # Embedding dropout
        self.embedding_dropout_prob = embedding_dropout_prob
        self.embedding_dropout = Dropout(self.embedding_dropout_prob)
        # operations
        self.add_pe = P.Add()
        self.add_te = P.Add()
        self.cast = P.Cast()

        self.shard(config)

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """embedding construct"""
        words_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = self.add_pe(words_embeddings, position_embeddings)
        else:
            embeddings = words_embeddings
        if tokentype_ids is not None:
            if self.tokentype_embedding is None:
                raise RuntimeError("Embedding layer got 'tokentype_ids' input, "
                                   "but 'tokentype_embeddings' layer is not initialized")
            embeddings = self.add_te(embeddings, self.tokentype_embedding(tokentype_ids))
        else:
            if self.tokentype_embedding is not None:
                raise RuntimeError("The 'tokentype_ids' input for Embedding layer is None, "
                                   "but 'tokentype_embeddings' layer is initialized")

        # Dropout
        if self.embedding_dropout_prob > 0:
            embeddings = self.embedding_dropout(embeddings)

        embeddings = self.cast(embeddings, self.compute_dtype)
        return embeddings

    def shard(self, config: TransformerConfig):
        """sharding parameters"""
        dp = 1 if config is None else config.data_parallel
        cp = 1 if config is None else config.context_parallel

        if config.vocab_emb_dp:
            self.add_pe.shard(((dp, cp, 1), (dp, cp, 1)))
            self.add_te.shard(((dp, cp, 1), (dp, cp, 1)))
            strategy_dropout = (dp, cp, 1)
            self.embedding_dropout.shard(strategy=strategy_dropout)
        else:
            self.add_pe.shard(((1, 1, 1), (1, 1, 1)))
            self.add_te.shard(((1, 1, 1), (1, 1, 1)))
            strategy_dropout = (1, 1, 1)
            self.embedding_dropout.shard(strategy=strategy_dropout)


class TransformerLanguageModel(nn.Cell):
    r"""
    Transformer language model.

    Args:
        config: model config
        encoder_attn_mask_type: encoder attention mask type
        num_tokentypes: if > 0, using tokentypes embedding
        add_encoder: if True, use encoder
        add_decoder: if True, use decoder
        decoder_attn_mask_type: decoder attention mask type
        add_pooler: if True, use pooler
        pre_process: when using pipeline parallel, indicate whether it's the first stage
        post_process: when using pipeline parallel, indicate whether it's the last stage

    Inputs:
        enc_input_ids: encoder input ids.
        enc_position_ids: encoder position ids.
        enc_attn_mask: encoder attention mask.

    Outputs:
        encoder_output: the hidden states
        embedding_table: the word embedding table
        pooled_output: the pooler layer output

    Supported Platforms:
        Ascend
    """

    def __init__(self,
                 config: TransformerConfig,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=None,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        super(TransformerLanguageModel, self).__init__()
        if add_decoder:
            raise NotImplementedError('add_decoder is not supported for now.')
        if encoder_attn_mask_type is not None:
            raise NotImplementedError("encoder_attn_mask_type is not supported for now.")
        if decoder_attn_mask_type is not None:
            raise NotImplementedError("decoder_attn_mask_type is not supported for now.")

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_tokentypes = num_tokentypes
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.add_pooler = add_pooler
        self.encoder_attn_mask_type = encoder_attn_mask_type

        # get value from config
        self.init_method = config.init_method
        self.compute_dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.vocab_size = config.padded_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_dropout = config.hidden_dropout

        # Embeddings
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       self.vocab_size,
                                       self.max_position_embeddings,
                                       self.hidden_dropout,
                                       config,
                                       self.num_tokentypes)

        # rope
        self.use_rotary_position_embeddings = config.position_embedding_type == 'rope'
        if self.use_rotary_position_embeddings:
            self.seq_length = config.seq_length
            rotary_dim = self.hidden_size // config.num_attention_heads
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=rotary_dim, rotary_percent=config.rotary_percent,
                seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor)

        # Encoder
        if self.add_encoder:
            self.encoder = ParallelTransformer(config, model_type=None, self_attn_mask_type=self.encoder_attn_mask_type,
                                               pre_process=False, post_process=False)
        else:
            self.encoder = None

        # pooler
        if self.post_process:
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, config)

        # causel mask
        self.causal_mask = CausalMaskGenerate(seq_length=config.seq_length,
                                              compute_type=config.compute_dtype,
                                              is_dynamic=config.is_dynamic,
                                              pad_token_id=config.pad_token_id,
                                              use_flash_attention=config.use_flash_attn,
                                              use_attn_mask_compression=config.use_attn_mask_compression,
                                              config=config
                                              )

        # operations
        self.cast = P.Cast()
        self.concat_prefix = P.Concat(-1)
        self.zeros = P.Zeros()
        self.shape = P.Shape()

        self.shard(config)

    def construct(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                  dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  enc_dec_attn_mask=None, tokentype_ids=None,
                  inference_params=None,
                  pooling_sequence_index=0,
                  enc_hidden_states=None, output_enc_hidden=False,
                  prefix_keys_values=None):
        """TransformerLanguageModel construct"""
        self._check_inputs(dec_input_ids, dec_position_ids, dec_attn_mask, retriever_input_ids, retriever_position_ids,
                           retriever_attn_mask, enc_dec_attn_mask, inference_params, output_enc_hidden)

        # Encoder embedding
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

        # rope
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        if enc_attn_mask is None:
            enc_attn_mask = self.causal_mask(enc_input_ids)

        if prefix_keys_values is not None:
            if enc_attn_mask is None:
                raise ValueError("enc_attn_mask should not be None when prefix_keys_values is not None!")
            if self.config.use_attn_mask_compression or enc_attn_mask.ndim != 4:
                raise ValueError("use_attn_mask_compression should be False when prefix_keys_values is not None! "
                                 "And enc_attn_mask.ndim should be 4, but got {}".format(enc_attn_mask.ndim))

            # prefix_key_values shape num_layers*(2, B, prefix_len, kv_num*kv_channel)
            bs, seq_len = self.shape(enc_input_ids)
            prefix_length = self.shape(prefix_keys_values[0])[2]
            prefix_mask = self.zeros((bs, 1, seq_len, prefix_length), enc_attn_mask.dtype)
            # (B, 1, S, S) -> (B, 1, S, S+prefix_len)
            enc_attn_mask = self.concat_prefix((prefix_mask, enc_attn_mask))

        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input, enc_attn_mask, rotary_pos_emb, prefix_keys_values)
            else:
                encoder_output = None
        else:
            encoder_output = self.cast(enc_hidden_states, encoder_input.dtype)

        if self.post_process and self.add_pooler:
            pooled_output = self.pooler(encoder_output, pooling_sequence_index)
            return encoder_output, pooled_output

        return encoder_output

    def _check_inputs(self, dec_input_ids, dec_position_ids, dec_attn_mask, retriever_input_ids, retriever_position_ids,
                      retriever_attn_mask, enc_dec_attn_mask, inference_params, output_enc_hidden):
        """check inputs function"""
        if dec_input_ids is not None:
            raise NotImplementedError("dec_input_ids is not supported for now.")
        if dec_position_ids is not None:
            raise NotImplementedError("dec_position_ids is not supported for now.")
        if dec_attn_mask is not None:
            raise NotImplementedError("dec_attn_mask is not supported for now.")
        if retriever_input_ids is not None:
            raise NotImplementedError("dec_input_ids is not supported for now.")
        if retriever_position_ids is not None:
            raise NotImplementedError("dec_position_ids is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("dec_attn_mask is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")
        if output_enc_hidden:
            raise NotImplementedError("output_enc_hidden is not supported for now.")

    def shard(self, config: TransformerConfig):
        """sharding parameters"""
        dp = 1 if config is None else config.data_parallel
        cp = 1 if config is None else config.context_parallel

        self.concat_prefix.shard(((dp, 1, cp, 1), (dp, 1, cp, 1)))


def get_language_model(config: TransformerConfig,
                       num_tokentypes: int,
                       add_pooler: bool,
                       encoder_attn_mask_type: Union[AttnMaskType, None],
                       add_encoder: bool = True,
                       add_decoder: bool = False,
                       decoder_attn_mask_type: Union[AttnMaskType, None] = AttnMaskType.causal,
                       pre_process: bool = True,
                       post_process: bool = True
                       ) -> (TransformerLanguageModel, str):
    """Build language model and its key for checkpoints.

    Args:
        config (TransformerConfig): Configuration for the model.
        num_tokentypes (int): Number of token types.
        add_pooler (bool): Whether to add a pooler.
        encoder_attn_mask_type (AttnMaskType): Attention mask type for encoder.
        add_encoder (bool): Whether to add an encoder.
        add_decoder (bool): Whether to add a decoder.
        decoder_attn_mask_type (AttnMaskType): Attention mask type for decoder.
        pre_process (bool): Whether to add pre-process.
        post_process (bool): Whether to add post-process.

    Returns:
        language_model (TransformerLanguageModel): Language model.
        language_model_key (str): Key used for checkpoints.
    """
    if config.init_method is None:
        config.init_method = init_method_normal(config.init_method_std, config.params_dtype)

    if config.output_layer_init_method is None:
        config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                    config.num_layers,
                                                                    config.params_dtype)

    # Language model.
    language_model = TransformerLanguageModel(
        config,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key
