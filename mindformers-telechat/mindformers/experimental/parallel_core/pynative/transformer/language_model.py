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
""" Language Model """
import mindspore as ms
from mindspore import mint
from mindspore.nn import Cell
import mindspore.ops as P
import mindspore.nn as nn

from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromSequenceParallelRegion, \
    VocabParallelEmbedding, ScatterToSequenceParallelRegion, ColumnParallelLinear
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import get_rng_tracer
from mindformers.experimental.parallel_core.pynative.parallel_state import get_pipeline_model_parallel_world_size
from mindformers.experimental.parallel_core.pynative.transformer.enums import ModelType, AttnMaskType

from .module import Module
from .transformer import ParallelTransformer
from .mlp import ParallelMLP
from .rotary_pos_embedding import RotaryEmbedding


class Pooler(Cell):
    """
    Add a linear transformation to the hidden state corresponding to a specific token.

    Args:
        hidden_size: hidden states size for dense layer
        init_method: dense layer weight init method
        config: model config
    """

    def __init__(self, hidden_size, init_method, config, **kwargs):
        super().__init__(**kwargs)
        param_init_dtype = config.params_dtype
        self.dense = nn.Dense(hidden_size,
                              hidden_size,
                              weight_init=init_method,
                              bias_init='zeros',
                              dtype=param_init_dtype,
                              activation='tanh')
        self.sequence_parallel = config.parallel_config.sequence_parallel
        self.gather_from_sequence_parallel_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=False,
            tensor_parallel_output_grad=False
        )

    def construct(self, hidden_states, sequence_index=0):
        """ pooler forward """
        if self.sequence_parallel:
            hidden_states = self.gather_from_sequence_parallel_region(hidden_states)

        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        return pooled


class Embedding(Module):
    """
    An embedding layer contain word embedding, position embedding and tokentypes embedding.

    Args:
        hidden_size (int): hidden states size for embedding layer.
        vocab_size (int): vocabulary size.
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding. if using position
            embedding, it is necessary to set the maximum sequence length.
        embedding_dropout_prob (float): dropout rate for embedding layer.
        config (TransformerConfig): The transformer configuration include init_method, parallel_config, etc.
        num_tokentypes (int, optional): size of the token-type embeddings. If > 0, using tokentypes embedding.

    Inputs:
        - **input_ids** (Tensor) - The tokenized inputs with datatype int32, shape :math:`(B, S)`.
        - **position_ids** (Tensor) - Position ids for position embedding, shape :math:`(B, S)`.
        - **tokentype_ids** (Tensor) - Token type IDs used to distinguish different types of tokens (e.g., sentence A
          and sentence B in BERT), with datatype int32, shape :math:`(B, S)`.

    Outputs:
        - **embeddings** (Tensor)- The embedding output, shape :math:`(B, S, H)`.

    Raises:
        NotImplementedError: If `config.fp32_residual_connection` or `config.clone_scatter_output_in_embedding` is True.
        RuntimeError: If `tokentype_ids` is not None and `tokentype_embeddings` is None.
            If `tokentype_ids` is None and `tokentype_embeddings` is not None.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindformers.experimental.parallel_core.pynative.config import TransformerConfig, ModelParallelConfig
        >>> from mindformers.experimental.parallel_core.pynative.transformer.language_model import (
        ...     Embedding as Pynative_Embedding
        ... )
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core import Embedding
        >>> from mindformers.core.context import build_context
        >>> from mindspore.common.initializer import initializer
        >>> from mindspore.numpy import array_equal
        >>> from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import (
        ...     apply_rotary_pos_emb as pynative_apply_rotary_pos_emb
        ... )
        >>> from mindformers.experimental.parallel_core import apply_rotary_pos_emb
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> init()
        >>> initialize_model_parallel()
        >>> def get_config():
        ...     parallel_config = ModelParallelConfig(tensor_model_parallel_size=1)
        ...     config = TransformerConfig(vocab_size=1,
        ...                                num_layers=1,
        ...                                num_attention_heads=1,
        ...                                hidden_size=1,
        ...                                ffn_hidden_size=1,
        ...                                parallel_config=parallel_config)
        ...     return config
        >>> build_context(config={'context': {'mode': 'PYNATIVE_MODE'}, 'parallel': {}})
        >>> x = initializer('normal', (1, 8, 4096, 64), ms.dtype.bfloat16)
        >>> freqs = initializer('normal', (1, 1, 4096, 64), ms.dtype.bfloat16)
        >>> config = get_config()
        >>> assert array_equal(apply_rotary_pos_emb(x, freqs, None), pynative_apply_rotary_pos_emb(x, freqs, None))
        >>> assert isinstance(Embedding(hidden_size=16, vocab_size=1600, config=config,
        ...                             max_sequence_length=64, embedding_dropout_prob=0.),
        ...                   Pynative_Embedding)
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_init_dtype = config.params_dtype
        self.compute_dtype = config.compute_dtype
        self.init_method = config.init_method
        self.sequence_parallel = config.parallel_config.sequence_parallel
        self.num_tokentypes = num_tokentypes
        self.data_layout = config.dataset_config.data_layout
        self.fp32_residual_connection = config.fp32_residual_connection
        self.clone_scatter_output_in_embedding = config.clone_scatter_output_in_embedding

        # init word embedding
        self.word_embeddings = VocabParallelEmbedding(vocab_size,
                                                      hidden_size,
                                                      config=config,
                                                      init_method=self.init_method,
                                                      param_init_dtype=self.param_init_dtype)

        # init position embedding
        self.use_position_embedding = config.position_embedding_type == 'learned_absolute'
        self.parallel_position_embedding = config.parallel_position_embedding
        if self.use_position_embedding:
            if not self.parallel_position_embedding:
                self.position_embeddings = nn.Embedding(max_sequence_length,
                                                        hidden_size,
                                                        embedding_table=self.init_method,
                                                        dtype=ms.int32)
            else:
                self.position_embeddings = VocabParallelEmbedding(max_sequence_length,
                                                                  hidden_size,
                                                                  config=config,
                                                                  init_method=self.init_method,
                                                                  param_init_dtype=self.param_init_dtype)
        # init tokentypes embedding
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = nn.Embedding(num_tokentypes,
                                                     hidden_size,
                                                     embedding_table=self.init_method,
                                                     dtype=ms.int32)
        else:
            self.tokentype_embeddings = None

        # init dropout
        self.embedding_dropout_prob = embedding_dropout_prob
        self.dropout = mint.nn.Dropout(self.embedding_dropout_prob)

        # init comm op
        self.scatter_to_sequence_parallel_region = ScatterToSequenceParallelRegion(
            need_to_swapaxes=self.data_layout == "BSH"
        )

        self.fp32_residual_connection = config.fp32_residual_connection
        if self.fp32_residual_connection:
            raise NotImplementedError("fp32_residual_connection is not supported for now.")
        self.clone_scatter_output_in_embedding = config.clone_scatter_output_in_embedding
        if self.clone_scatter_output_in_embedding:
            raise NotImplementedError("clone_scatter_output_in_embedding is not supported for now.")

    def zero_parameters(self):
        """ set zero value for all embedding parameters """
        P.assign(self.word_embeddings, P.zeros_like(self.word_embeddings))
        self.word_embeddings.weight.shared = True
        if self.use_position_embedding:
            P.assign(self.position_embeddings, P.zeros_like(self.position_embeddings))
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            P.assign(self.tokentype_embeddings, P.zeros_like(self.tokentype_embeddings))
            self.tokentype_embeddings.weight.shared = True

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """ embedding layer forward """
        # word embedding
        words_embeddings = self.word_embeddings(input_ids)

        # position embedding
        if self.use_position_embedding:
            position_embedding = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embedding
        else:
            embeddings = words_embeddings

        # tokentype embedding
        if tokentype_ids is not None:
            if self.tokentype_embeddings is None:
                raise RuntimeError("Embedding layer got 'tokentype_ids' input, "
                                   "but 'tokentype_embeddings' layer is not initialized")
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids)
            embeddings = embeddings + tokentype_embedding
        else:
            if self.tokentype_embeddings is not None:
                raise RuntimeError("The 'tokentype_ids' input for Embedding layer is None, "
                                   "but 'tokentype_embeddings' layer is initialized")

        if self.data_layout == "SBH":
            embeddings = embeddings.swapaxes(0, 1)

        if self.fp32_residual_connection:
            raise NotImplementedError("`fp32_residual_connection` is not supported for now.")

        # dropout
        if self.sequence_parallel:
            embeddings = self.scatter_to_sequence_parallel_region(embeddings)
            if self.clone_scatter_output_in_embedding:
                raise NotImplementedError("`clone_scatter_output_in_embedding` is not supported for now.")
            with get_rng_tracer().rng_fork():
                embeddings = self.dropout(embeddings)
        else:
            embeddings = self.dropout(embeddings)

        # convert dtype to compute dtype
        embeddings = embeddings.astype(self.compute_dtype)
        return embeddings


class TransformerLanguageModel(Module):
    """
    Transformer language model.

    Args:
        config (TransformerConfig): The transformer configuration includes init_method, parallel_config, etc.
        encoder_attn_mask_type (int): Encoder attention mask type.
        num_tokentypes (int): If > 0, using tokentypes embedding.
        add_encoder (bool): If True, use encoder.
        use_decoder (bool): If True, use decoder.
        decoder_attn_mask_type (int): Decoder attention mask type.
        add_pooler (bool): If True, use pooler.
        pre_process (bool): When using pipeline parallel, indicate whether it's the first stage.
        post_process (bool): When using pipeline parallel, indicate whether it's the last stage.
        visual_encoder (nn.Cell): Visual encoder.

    Inputs:
        - **enc_input_ids** (Tensor) - Encoder input indexes. Shape :math:`(B, S)`.
        - **enc_position_ids** (Tensor) - Encoder position offset. Shape :math:`(B, S)`.
        - **enc_attn_mask** (Tensor) - Encoder attention mask. Shape :math:`(B, S)`.
        - **dec_input_ids** (Tensor) - Decoder input indexes. Shape :math:`(B, S)`.
        - **dec_position_ids** (Tensor, optional) - Decoder input position indices. Shape :math:`(B, S)`.
        - **dec_attn_mask** (Tensor, optional) - Decoder attention mask. Shape :math:`(B, S)`.
        - **retriever_input_ids** (Tensor, optional) - Retriever input token indices. Shape: Depends on the input shape
          of the retrieval task.
        - **retriever_position_ids** (Tensor, optional) - Retriever input position indices. Shape: Depends on the input
          shape of the retrieval task.
        - **retriever_attn_mask** (Tensor, optional) - Retriever attention mask. Used to control the attention range in
          the retriever when calculating attention. Shape: Depends on the attention calculation shape of the retriever.
        - **enc_dec_attn_mask** (Tensor, optional) - Encoder-decoder attention mask. Shape: Depends on the attention
          calculation between the encoder and decoder.
        - **tokentype_ids** (Tensor, optional) - List of token type ids to be fed to a model. Shape :math:`(B, S)`.
        - **inference_params** (InferenceParams) - Inference parameters. Used to specify specific settings during
          inference, such as maximum generation length, max batch size, etc.
        - **pooling_sequence_index** (int) - Pooling sequence index.
        - **enc_hidden_states** (Tensor, optional) - Encoder hidden states. Shape: Depends on the output shape of the
          encoder.
        - **output_enc_hidden** (bool, optional) - Whether to output encoder hidden states.
        - **input_image** (Tensor, optional) - Tensor of the input image. Shape :math:`(N, C_{in}, H_{in}, W_{in})` or
          :math:`(N, H_{in}, W_{in}, C_{in}, )` depending on `data_format`.
        - **delimiter_position** (Tensor, optional) - Delimiter position tensor. Shape :math:`(B, N)`, where :math:`N`
          represents the number of delimiters.
        - **image_embedding** (Tensor, optional) - Image embedding tensor. The shape depends on the dimension of the
          image embedding, for example (batch_size, embedding_dim).

    Outputs:
        - **encoder_output** - Output Tensor of shape :math:`(B, S, H)` or :math:`(S, B, H)`.

    Raises:
        ValueError: If config.untie_embeddings_and_output_weights and add_decoder are both True.
        RuntimeError: If the length of the input is not 1.
        NotImplementedError: If `config.retro_add_retriever` is True.
        NotImplementedError: if `visual_encoder` or `add_decoder` is True.
        NotImplementedError: If `dec_input_ids`, `dec_position_ids`, `dec_attn_mask`, `retriever_input_ids`,
            `retriever_position_ids`, `retriever_attn_mask`, `enc_dec_attn_mask`, `input_image`, `delimiter_position`
            or `image_embedding` is not None.
        NotImplementedError: if `output_enc_hidden` is True.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.transformer import TransformerLanguageModel
        >>> from mindformers.experimental.parallel_core.pynative.config import (
        ...     init_configs_from_yaml,
        ...     TrainingConfig,
        ...     ModelParallelConfig,
        ...     TransformerConfig,
        ...     DatasetConfig,
        ... )
        >>> os.environ['HCCL_BUFFSIZE'] = "1"
        >>> CONFIG_PATH = "test_language_model.yaml"
        >>> training_config, parallel_config, dataset_config, model_config = init_configs_from_yaml(
        ...     CONFIG_PATH, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig]
        ... )
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")
        >>> init()
        >>> initialize_model_parallel()
        >>> language_model = TransformerLanguageModel(model_config, encoder_attn_mask_type=None)
        >>> input_data = Tensor(np.random.random((model_config.seq_length, model_config.seq_length)).astype(np.float32))
        >>> label_data = Tensor(np.zeros((model_config.seq_length, model_config.seq_length)).astype(np.int32))
        >>> hidden_states = language_model(input_data, None, label_data)
        >>> print(hidden_states.shape)
        (32, 32, 64)
    """
    # pylint: disable=W0613, C0111
    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 visual_encoder=None,
                 **kwargs):
        super().__init__(config, share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
                         **kwargs)
        if config.untie_embeddings_and_output_weights and add_decoder:
            raise ValueError("When 'untie_embeddings_and_output_weights' is True, 'add_decoder' can't be True")
        if config.retro_add_retriever:
            raise NotImplementedError("retriever is not supported for now.")
        if visual_encoder is not None:
            raise NotImplementedError("visual_encoder is not supported for now.")

        self.pre_process = pre_process
        self.post_process = post_process
        self.pipeline_parallel = get_pipeline_model_parallel_world_size() > 1
        self.add_encoder = add_encoder
        self.num_tokentypes = num_tokentypes
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.init_method = config.init_method
        self.add_decoder = add_decoder

        # get value from config
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        post_norm = config.use_final_norm
        param_init_dtype = config.params_dtype
        self.hidden_size = config.hidden_size
        padded_vocab_size = config.vocab_size
        hidden_dropout_rate = config.hidden_dropout

        if self.pre_process:
            # init embedding layer
            self.embedding = Embedding(self.hidden_size,
                                       padded_vocab_size,
                                       self.seq_length,
                                       hidden_dropout_rate,
                                       config,
                                       self.num_tokentypes)

            # init visual encoder
            if visual_encoder is not None:
                self.visual_encoder = visual_encoder
                self.visual_mlp = ParallelMLP(config.visual_config)
            else:
                self.visual_encoder = None

        # init rotary embeddings
        self.use_rotary_embeddings = config.position_embedding_type == 'rope'
        if self.use_rotary_embeddings:
            rotary_dim = config.hidden_size // config.num_attention_heads \
                if config.kv_channels is None else config.kv_channels
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=config.seq_len_interpolation_factor,
                rotary_base=config.rotary_base)

        # init encoder

        if self.add_encoder:
            self.encoder = ParallelTransformer(config,
                                               model_type=ModelType.encoder_or_decoder,
                                               self_attn_mask_type=self.encoder_attn_mask_type,
                                               pre_process=self.pre_process,
                                               post_process=self.post_process,
                                               post_norm=post_norm
                                               )
        else:
            self.encoder = None

        if self.add_decoder:
            raise NotImplementedError("add_decoder is not supported for now.")

        # init pooler
        if self.post_process:
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, config)

            if self.untie_embeddings_and_output_weights or self.pipeline_parallel:
                init_method = self.init_method if self.untie_embeddings_and_output_weights else 'zeros'
                self.output_layer = ColumnParallelLinear(
                    self.hidden_size,
                    padded_vocab_size,
                    config=config,
                    init_method=init_method,
                    bias=False,
                    param_init_dtype=param_init_dtype
                )

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

        if len(input_tensor) != 1:
            raise RuntimeError("When using `set_input_tensor` function, "
                               "length of `input_tensor` must be equal to 1")
        self.encoder.set_input_tensor(input_tensor[0])

    def visual_forward(self, input_image):
        """ visual encoder forward """
        n_image = 1
        if input_image.ndim == 5:
            bs, n_image, channel, height, width = input_image.shape
            input_image = input_image.reshape(-1, channel, height, width)
        image_embedding = self.visual_encoder(input_image)
        image_embedding, _ = self.visual_mlp(image_embedding)
        image_embedding = image_embedding.reshape(bs, n_image, image_embedding.shape[1], image_embedding.shape[2])
        image_embedding = image_embedding.astype(self.compute_dtype)
        return image_embedding

    def mixed_embedding(self, text_embedding, image_embedding, delimiter_position):
        """ mixing text embedding and image embedding """
        mix_embeddings = []
        for cur_batch in range(text_embedding.shape[0]):
            mix_embedding = []
            image_num = int(len(delimiter_position[cur_batch]) / 2)
            image_delimiter_position = [i + 1 for i in range(image_num)]
            split_text_embedding = P.tensor_split(text_embedding[cur_batch], delimiter_position[cur_batch], axis=0)
            split_image_embedding = P.tensor_split(image_embedding[cur_batch], image_delimiter_position, axis=0)
            split_image_embedding = [split_image_embedding[i][0] for i in range(image_num)]
            for i, embedding in enumerate(split_text_embedding):
                mix_embedding.append(embedding if i % 2 == 0 else split_image_embedding[int((i - 1) / 2)])
            mix_embedding = mint.cat(mix_embedding, dim=0)
            mix_embeddings.append(mix_embedding)
        mix_embeddings = mint.cat(mix_embeddings, dim=0)
        return mix_embeddings

    def construct(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                  dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                  retriever_input_ids=None, retriever_position_ids=None, retriever_attn_mask=None,
                  enc_dec_attn_mask=None, tokentype_ids=None, inference_params=None,
                  pooling_sequence_index=0, enc_hidden_states=None, output_enc_hidden=False,
                  input_image=None, delimiter_position=None, image_embedding=None):
        """ language model forward """
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
        if output_enc_hidden:
            raise NotImplementedError("output_enc_hidden is not supported for now.")
        if input_image is not None:
            raise NotImplementedError("input_image is not supported for now.")
        if delimiter_position is not None:
            raise NotImplementedError("delimiter_position is not supported for now.")
        if image_embedding is not None:
            raise NotImplementedError("image_embedding is not supported for now.")

        # visual encoder
        image_embedding_out = None
        if self.pre_process:
            if image_embedding is not None:
                image_embedding_out = image_embedding
            else:
                if self.visual_encoder is not None:
                    if input_image is None:
                        raise TypeError("When 'visual_encoder' is not None, 'input_image' can't be None")
                    image_embedding_out = self.visual_forward(input_image)

        # encoder
        text_embedding_out = None
        if self.pre_process:
            text_embedding_out = self.embedding(enc_input_ids, enc_position_ids,
                                                tokentype_ids=tokentype_ids)

            # mix embedding out if image_embedding_out is not None
            # Now, only support below mix order:
            # one_text_embedding -> one_image_embedding (loop)
            if image_embedding_out is None:
                encoder_input = text_embedding_out
            else:
                if delimiter_position is None:
                    raise TypeError("When 'visual_encoder' is not None, 'delimiter_position' can't be None")
                encoder_input = self.mixed_embedding(text_embedding_out, image_embedding_out, delimiter_position)
        else:
            encoder_input = None

        # rotary embedding
        rotary_pos_emb = None
        if self.use_rotary_embeddings:
            if inference_params is not None:
                raise NotImplementedError("inference_params is not supported for now.")
            rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        # encoder
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input,
                                              enc_attn_mask,
                                              retriever_input=None,
                                              retriever_attn_mask=retriever_attn_mask,
                                              inference_params=inference_params,
                                              rotary_pos_emb=rotary_pos_emb)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.astype(encoder_input.dtype)

        # pooler
        if self.post_process and self.add_pooler:
            pooled_output = self.pooler(encoder_output,
                                        pooling_sequence_index)

        if self.add_pooler and self.post_process:
            return encoder_output, pooled_output
        return encoder_output


def get_language_model(config, num_tokentypes, add_pooler,
                       encoder_attn_mask_type,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=None,
                       pre_process=True, post_process=True):
    """
    get language model

    Args:
        - **config** : model config
        - **num_tokentypes** : if > 0, using tokentypes embedding
        - **add_pooler** : if True, use pooler
        - **encoder_attn_mask_type** : encoder attention mask type
        - **add_encoder** : if True, use encoder
        - **add_decoder** : if True, use decoder
        - **decoder_attn_mask_type** : decoder attention mask type
        - **pre_process** : when using pipeline parallel, indicate whether it's the first stage
        - **post_process** : when using pipeline parallel, indicate whether it's the last stage
    """
    language_model = TransformerLanguageModel(
        config=config,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    language_model_key = 'language_model'
    return language_model, language_model_key
