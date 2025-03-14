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
"""visualglm language model."""
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.ops import operations as P

from mindformers import CrossEntropyLoss
from mindformers.models.glm.attention import default_dpmp_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.glm.glm import GLMModel, GLMForPreTraining
from mindformers.models.glm.glm_config import GLMConfig

from layers import ImageTextEmbeddingPreparationMixIn
from attention import SelfAttentionAdapter


class GLMModelForBlip2(GLMModel):
    """
    The backbone of GLM network

    Args:
        config (GLMConfig): The config of network.
        op_parallel_config (optional): Operator parallel strategy. Default: `OpParallelConfig()`.
        embed_parallel_config (optional): Operator parallel strategy. Default: `EmbeddingOpParallelConfig()`.
    """

    def __init__(self, config):
        super().__init__(config)

        op_parallel_config = default_dpmp_config
        if config.parallel_config:
            op_parallel_config = config.parallel_config

        # adapter
        self.modify_attention_fn(config, op_parallel_config)

    def modify_attention_fn(self, config, op_parallel_config):
        """replace default attention func"""
        for i in range(config.num_layers):
            layer = self.layers[i]
            layer_id = i + 1
            layer.attention = SelfAttentionAdapter(
                config.hidden_size,
                config.batch_size,
                config.num_heads,
                op_parallel_config,
                config.attention_dropout_rate,
                config.hidden_dropout_rate,
                layer_id,
                max_seq_len=config.seq_length,
                hidden_size_per_attention_head=config.hidden_size_per_attention_head,
                position_encoding_2d=config.position_encoding_2d,
                bias=True,
                params_dtype=config.param_init_type,
                softmax_dtype=config.softmax_compute_type,
                compute_dtype=config.compute_dtype,
                use_past=config.use_past
            )

    def construct(self, input_embeddings, position_ids, attention_mask, init_reset=True, batch_valid_length=None):
        """
        Get output logits

        Inputs:
            input_ids (Tensor): The tokenized inputs with dtype int32.
            input_mask (Tensor): The mask indicating whether each position is a valid input.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            attention_mask (Tensor): Used when batching sequences together.
            init_reset (bool, optional): Default: True.
            batch_valid_length (Tensor, optional): Default: None.

        Returns:
            logits (Tensor): The output logit of backbone.
            table (Tensor): The embedding table for the vocabulary.
        """
        if attention_mask is None:
            attention_mask = ops.ones((1, 1), mstype.int32)

        hidden_states = input_embeddings
        for i in range(self.num_layers):
            layer_ret = self.layers[i](hidden_states, attention_mask, position_ids, init_reset, batch_valid_length)

            if isinstance(layer_ret, tuple):
                layer_ret = layer_ret[0]
            hidden_states = layer_ret

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        return logits


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTrainingForBlip2(GLMForPreTraining, ImageTextEmbeddingPreparationMixIn):
    r"""
    Provide glm training loss or logits through network.

    Args:
        config (GLMConfig): The config of GLMModel.

    Examples:
        >>> from mindformers import GLMForPreTraining
        >>> model = GLMForPreTraining.from_pretrained("glm_6b")
        >>> type(model)
        <class 'mindformers.models.glm.glm.GLMForPreTraining'>
    """

    def __init__(self, config: GLMConfig):
        checkpoint_name_or_path = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = ""

        GLMForPreTraining.__init__(self, config=config)
        ImageTextEmbeddingPreparationMixIn.__init__(self, config=config)

        self.transformer = GLMModelForBlip2(config)

        self.config.checkpoint_name_or_path = checkpoint_name_or_path

        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)
        self.cast_1d = P.Cast()
        self.mul_1d = P.Mul().shard(((1,), (1,)))
        self.reshape = P.Reshape()
        self.not_equal_1d = P.NotEqual().shard(((1,), ()))
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.load_checkpoint(config)  # lite推理注释，ms放开

    def to_text_embeddings(self, text_input_ids):
        """
        create text embeddings from input ids
        :param text_input_ids: text input id
        :return: text embedding
        """
        input_embeds_raw = self.transformer.word_embeddings(text_input_ids)
        input_embeds = input_embeds_raw[0]
        input_embeds = self.transformer.embedding_dropout(input_embeds)
        return input_embeds

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        return self.prepare_image_text_embedding(input_ids, **kwargs)

    # pylint: disable=W0613
    def construct(self, input_embeddings=None, input_ids=None, labels=None, position_ids=None, attention_mask=None,
                  input_position=None, input_embeds=None, init_reset=True, batch_valid_length=None):
        """
        Extract logits and calculate loss

        Inputs:
            input_ids (Tensor): the tokenized inputs with dtype int32.
            labels (Tensor): the indices of input sequence tokens in the vocabulary.
            position_ids (Tensor): used to identify each token's position in the list of tokens.
            attention_mask (Tensor): used when batching sequences together.
            input_position(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset (bool, optional): Default: True.
            batch_valid_length(Tensor, optional): Default: None.

        Returns:
            Training phase:
                loss: Training loss.
            Other phase:
                logits (Tensor): The output logit of backbone.
        """

        if input_embeddings is None and input_ids is not None:  # for incremental infer
            input_embeddings = self.to_text_embeddings(input_ids)

        output_states = self.transformer(input_embeddings, position_ids, attention_mask, init_reset, batch_valid_length)
        logits = self.lm_head(output_states)

        seq_length = output_states.shape[1]
        logits_shape = logits.shape
        if not self.training:
            logits = logits.reshape((-1, logits_shape[-1]))
            # only gather in auto-aggressive generate or first iteration
            if (not self.use_past or self.is_first_iteration) and input_position is not None:
                logits = self.gather(logits, input_position, 0)
            return (logits,)

        logits_reshape = logits.reshape((self.batch_size, seq_length, self.vocab_size))

        shift_logits = logits_reshape[..., :-1, :]
        shift_labels = labels[..., 1:]

        logits_view = shift_logits.view((-1, shift_logits.shape[-1]))
        labels_view = shift_labels.view(-1)

        input_mask = self.cast_1d(self.not_equal_1d(shift_labels, -100), mstype.float32)
        input_mask = self.reshape(input_mask, (-1,))

        loss = self.loss(logits_view, labels_view, input_mask)
        # loss = self.loss(logits_view, labels_view)
        return loss
