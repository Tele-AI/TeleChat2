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
"""GLM model."""
import os
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer import VocabEmbedding, EmbeddingOpParallelConfig, OpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.layers import LayerNorm
from mindformers.version_control import get_dropout
from mindformers.models.modeling_utils import PreTrainedModel

from .glm_config import GLMConfig
from .layers import DeepNormWithGLULayer

#  Get MS backend: 0 vm 1 GE
is_ge = os.getenv('MS_ENABLE_GE')
if is_ge == '1':
    jit_level = "O3"
else:
    jit_level = "O1"

default_dpmp_config = OpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()

__all__ = ['GLMForPreTraining', 'GLMChatModel']


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """

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


class ProcessLogits(nn.Cell):
    r"""Process logits into probability distribution."""

    def __init__(self, use_past=False):
        super(ProcessLogits, self).__init__()
        self.e = ms.Tensor(np.e)
        self.gather = P.Gather()
        self.logsoftmax = P.LogSoftmax()
        self.reshape = P.Reshape()
        self.use_past = use_past

    def construct(self, logits, current_index=None, is_first_iteration=False):
        logits = logits.reshape(-1, logits.shape[-1])
        if self.use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax(logits)
        outputs = F.tensor_pow(self.e, outputs)
        return outputs


class GLMModel(GLMPreTrainedModel):
    """
    The backbone of GLM network

    Args:
        config (GLMConfig): The config of network.
        op_parallel_config (optional): Operator parallel strategy. Default: `OpParallelConfig()`.
        embed_parallel_config (optional): Operator parallel strategy. Default: `EmbeddingOpParallelConfig()`.
    """

    def __init__(self,
                 config,
                 op_parallel_config=default_dpmp_config,
                 embed_parallel_config=default_embedding_parallel_config):
        super(GLMModel, self).__init__(config)
        # recording parameters
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.seq_length = config.seq_length
        self.use_past = config.use_past
        layernorm = LayerNorm
        if config.parallel_config:
            op_parallel_config = config.parallel_config

        # create embedding parameters
        self.embedding_dropout = get_dropout(config.embedding_dropout_prob)

        embed_parallel_config.data_parallel = op_parallel_config.data_parallel
        embed_parallel_config.model_parallel = op_parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = False
        self.word_embeddings = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                              parallel_config=embed_parallel_config)
        self.word_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.matmul = ops.MatMul().shard(((1, 1), (1, embed_parallel_config.model_parallel)))
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))

        def get_layer(layer_id):
            return DeepNormWithGLULayer(
                self.num_layers,
                self.hidden_size,
                self.num_heads,
                config.batch_size,
                config.attention_dropout_rate,
                config.hidden_dropout_rate,
                config.layernorm_epsilon,
                layer_id,
                max_seq_len=self.seq_length,
                inner_hidden_size=config.inner_hidden_size,
                hidden_size_per_attention_head=config.hidden_size_per_attention_head,
                layernorm_order=config.layernorm_order,
                layernorm=layernorm,
                use_bias=True,
                activation_func=config.activation_func,
                position_encoding_2d=config.position_encoding_2d,
                params_dtype=config.param_init_type,
                layernorm_dtype=config.layernorm_compute_type,
                softmax_dtype=config.softmax_compute_type,
                compute_dtype=config.compute_dtype,
                use_past=self.use_past,
                parallel_config=op_parallel_config,
            )

        self.layers = nn.CellList()
        for i in range(config.num_layers):
            layer = get_layer(i + 1)
            set_parallel_configure_for_layer(layer, layer_id=i, layers=config.num_layers,
                                             offset=0, parallel_config=op_parallel_config)
            self.layers.append(layer)

        # Final layer norm before output.
        self.use_final_layernorm = config.use_final_layernorm
        if config.use_final_layernorm:
            self.final_layernorm = layernorm(config.hidden_size, eps=config.layernorm_epsilon)
            self.final_layernorm.shard(((op_parallel_config.data_parallel, 1, 1),))

    # pylint: disable=W0613
    def construct(self, input_ids, position_ids, attention_mask, init_reset=True, batch_valid_length=None,
                  labels=None, input_position=None, input_embeds=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
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

        hidden_states, table = self.word_embeddings(input_ids)

        hidden_states = self.embedding_dropout(hidden_states)

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

        return logits, table


class GLMHead(nn.Cell):
    r"""Head for GLM to get the logits of each token in the vocab."""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 embed_parallel_config=None):
        super(GLMHead, self).__init__()
        self.param_init_type = param_init_type
        self.compute_dtype = compute_dtype
        self.weight = Parameter(initializer("normal", [vocab_size, hidden_size], compute_dtype), name="weight")
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))
        self.matmul = ops.MatMul(transpose_b=True).shard(
            ((embed_parallel_config.data_parallel, 1), (embed_parallel_config.model_parallel, 1)))

    def construct(self, state, embedding_table=None):
        """Get vocab probs"""
        state = F.reshape(state, (-1, F.shape(state)[-1]))
        state = ops.cast(state, self.compute_dtype)
        if embedding_table is None:
            embedding_table = self.weight
        embedding_table = self.cast(embedding_table, self.compute_dtype)
        logits_parallel = self.matmul(state, embedding_table)
        return logits_parallel


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTraining(GLMPreTrainedModel):
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
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMForPreTraining, self).__init__(config)
        self.config = config
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = GLMModel(config)
        self.lm_head = GLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            embed_parallel_config=config.parallel_config)
        self.stridedslice = ops.StridedSlice().shard(((1, 1),))
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     eps_const=3.4e-38,
                                     calculate_per_token_loss=calculate_per_token_loss)
        self.seq_length = config.seq_length
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.ones = P.Ones()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.ignore_index = config.ignore_index
        self.load_checkpoint(config)

    def get_masks_np(self, input_ids):
        """get attention mask using numpy."""
        batch_size, seq_length = input_ids.shape
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = np.tril(np.ones((batch_size, seq_length, seq_length)))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = np.expand_dims(attention_mask, axis=1)
        attention_mask = np.array(attention_mask < 0.5, np.bool_)
        return attention_mask

    def get_position_ids_np(self, input_ids, mask_positions, use_gmasks=None):
        """Get position ids from input_ids and mask_positions with numpy"""
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        if self.config.position_encoding_2d:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, np.int32),
                np.arange(seq_length - context_length, dtype=np.int32) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0)
            position_ids = np.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    def create_position_ids_np(self, input_ids):
        """Get position ids from input_ids with numpy"""
        mask, gmask = self.config.mask_token_id, self.config.gmask_token_id
        seqs = list(input_ids)

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gmask if gmask in seq else mask
            use_gmask = mask_token == gmask
            mask_positions.append(list(seq).index(mask_token))
            use_gmasks.append(use_gmask)
        position_ids = self.get_position_ids_np(input_ids, mask_positions, use_gmasks=None)
        return position_ids

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        """update glm kwargs before generate."""
        # for GLM `attention_mask` and `position_ids` generation
        attention_mask = self.get_masks_np(input_ids)
        position_ids = self.create_position_ids_np(input_ids)
        # update in model kwargs
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["position_ids"] = position_ids

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "attention_mask": Tensor(attention_mask, mstype.int32),
            "position_ids": Tensor(position_ids, mstype.int32),
            "input_position": input_position
        }

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        """used for non-first iterations, slice the inputs to length 1."""
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        position_ids = model_inputs.pop("position_ids")
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.asnumpy()
        if isinstance(attention_mask, Tensor):
            attention_mask = attention_mask.asnumpy()
        if isinstance(position_ids, Tensor):
            position_ids = position_ids.asnumpy()
        inputs_tmp = []
        position_ids_tmp = []
        attention_mask_tmp = []
        for i, index_value in enumerate(current_index):
            current_index_tmp = (
                int(index_value) - i * input_ids.shape[1]
            )  # multibatch
            # use numpy to slice array to avoid complie ascend slice op
            inputs_tmp.append(input_ids[i][current_index_tmp: current_index_tmp + 1])
            position_ids_tmp.append(position_ids[i][..., current_index_tmp:current_index_tmp + 1])
            attention_mask_tmp.append(attention_mask[i][:, current_index_tmp:current_index_tmp + 1, :])
        inputs_tmp = np.array(inputs_tmp, dtype=np.int32)
        position_ids_tmp = np.array(position_ids_tmp, dtype=np.int32)
        attention_mask_tmp = np.array(attention_mask_tmp, dtype=np.int32)
        model_inputs["input_ids"] = Tensor(inputs_tmp, mstype.int32)
        model_inputs["position_ids"] = Tensor(position_ids_tmp, mstype.int32)
        model_inputs["attention_mask"] = Tensor(attention_mask_tmp, mstype.int32)

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, position_ids=None, attention_mask=None,
                  input_position=None, input_embeds=None, init_reset=True, batch_valid_length=None,
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None):
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
        batch_size, seq_length = input_ids.shape

        if self.training:
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length), (1, 1))
        else:
            tokens = input_ids

        output_states, _ = self.transformer(tokens, position_ids,
                                            attention_mask, init_reset, batch_valid_length)
        logits = self.lm_head(output_states)

        logits_shape = logits.shape
        if not self.training:
            logits = logits.reshape((-1, logits_shape[-1]))
            # only gather in auto-aggressive generate or first iteration
            if (not self.use_past or self.is_first_iteration) and input_position is not None:
                logits = self.gather(logits, input_position, 0)
            return (logits,)

        labels = labels.reshape((-1,))
        logits = logits.reshape((-1, logits_shape[-1]))
        input_mask = self.not_equal(labels, self.ignore_index).astype(logits.dtype)
        input_mask = input_mask.reshape((-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMChatModel(GLMForPreTraining):
    r"""
    Provide glm chat capability through network.
    Args:
        config (GLMConfig): The config of GLMModel.

    Returns:
        Tensor, the probability distribution of network loss.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMChatModel, self).__init__(config)
        self.e = ms.Tensor(np.e, dtype=mstype.float32)
        self.pow = P.Pow()
        self.topk = P.TopK(sorted=True)
        self.cumsum = P.CumSum()
        self.sum = P.ReduceSum(keep_dims=False)
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.frequency_list = ms.Tensor([[0 for _ in range(self.vocab_size)]])
        self.post_logits = ProcessLogits(use_past=config.use_past)
        # seems not supported yet.
        self.top_p = 1
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty
        self.is_first_iteration = False
        self.is_sample_acceleration = config.is_sample_acceleration

    def sample(self, log_probs):
        """Convert the log_probs to probability"""

        if self.repetition_penalty != 1:
            log_probs = log_probs - self.frequency_list * self.repetition_penalty - \
                        (self.frequency_list > 0) * self.repetition_penalty

        # Process sample in graph to accelerate generate
        logits = self.pow(self.e, log_probs)

        # If top_p is less than 1.0, use top_p sampling
        # seems not supported yet.
        if self.top_p < 1.0:
            sorted_logits, index = self.topk(logits, 5000)
            cumsum_logits = self.cumsum(sorted_logits, 1)
            top_p_num = self.sum((cumsum_logits < self.top_p).astype(mstype.int32), -1) + 1
            top_p_num = int(top_p_num)
            # Get the corresponding probs and indices
            probs = sorted_logits[:, :top_p_num]
            p_args = index[:, :top_p_num]
            p = probs / self.sum(probs, -1, keepdim=True)

        # if top_p is set to 1.0, use top_k sampling
        else:
            probs, p_args = self.topk(logits, self.top_k)
            p = probs

        return p, p_args

    # pylint:disable=arguments-differ,W0613
    def construct(self, input_ids, position_ids=None, attention_mask=None, init_reset=True, batch_valid_length=None,
                  labels=None, input_position=None, input_embeds=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """Get probs and p_args"""
        # model forward
        output_states, _ = self.transformer(input_ids, position_ids, attention_mask, init_reset, batch_valid_length)
        logits = self.lm_head(output_states)

        if not self.is_sample_acceleration:
            return (logits,)

        # logit post process
        log_probs = self.post_logits(logits, input_position, self.is_first_iteration)

        # logit sort and sample
        probs, p_args = self.sample(log_probs)

        return probs, p_args
