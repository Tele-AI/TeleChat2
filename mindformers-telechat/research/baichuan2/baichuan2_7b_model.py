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
"""Baichuan2_7b models' APIs."""
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn, ops, mint
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.common.initializer import initializer, HeUniform

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline, LayerSetting
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.layers import FreqsMgr

from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode

__all__ = ['Baichuan7BV2ForCausalLM', 'Baichuan7BV2Model']


class Baichuan2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "baichuan2"


class Baichuan7BV2Model(Baichuan2PreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of llama decoderlayer
    """

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_flash_attention = config.use_flash_attention
        if self.use_flash_attention:
            logger.info("Enable flash attention.")

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)
        self.freqs_mgr.shard(config.parallel_config)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_past=config.use_past)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type,
                                             parallel_optimizer=True)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = LLamaDecodeLayer(config.seq_length,
                                     layer_id,
                                     dim=config.hidden_size,
                                     n_heads=config.num_heads,
                                     n_kv_heads=config.n_kv_heads,
                                     intermediate_size=config.intermediate_size,
                                     multiple_of=config.multiple_of,
                                     ffn_dim_multiplier=config.ffn_dim_multiplier,
                                     norm_eps=config.rms_norm_eps,
                                     qkv_has_bias=config.qkv_has_bias,
                                     qkv_concat=config.qkv_concat,
                                     compute_dtype=config.compute_dtype,
                                     layernorm_compute_dtype=config.layernorm_compute_type,
                                     softmax_compute_dtype=config.softmax_compute_type,
                                     rotary_dtype=config.rotary_dtype,
                                     param_init_type=config.param_init_type,
                                     use_past=config.use_past,
                                     use_flash_attention=self.use_flash_attention,
                                     is_dynamic=config.is_dynamic,
                                     block_size=config.block_size,
                                     num_blocks=config.num_blocks,
                                     use_rope_slice=config.use_rope_slice,
                                     parallel_config=config.parallel_config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.norm_out.shard((dp, 1, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """
        Forward of llama model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.

        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask.prefill()
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            freqs_cis = self.freqs_mgr(seq_len)
            mask = self.casual_mask(tokens)  # mask: [bs, seq, seq]

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                               slot_mapping=slot_mapping)
        output = self.norm_out(h)
        return output


class NormHead(nn.Cell):
    """
    NormHead Layer.

        Args:
            hidden_size (int): The hidden size of the input.
            vocab_size (int): Size of the dictionary of embeddings.
            compute_type (dtype.Number): The compute type.
            eps (number): A small positive value prevents division by zero.

        Inputs:
            - hidden_states (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, vocab_size)`.
    """
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 use_past,
                 compute_dtype=mstype.float16,
                 eps=1e-5):
        super().__init__()
        self.weight = Parameter(
            initializer(HeUniform(negative_slope=math.sqrt(5)),
                        [vocab_size, hidden_size],
                        mstype.float16),
            name='weight',
            parallel_optimizer=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.real_div = P.RealDiv()
        self.reshape = P.Reshape()
        self.sum = P.ReduceSum()
        self.eps = Tensor([eps], mstype.float16)
        self.is_first_iteration = True
        self.use_past = use_past

        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.assign = P.Assign()

    def construct(self, hidden_states):
        """Forward process of the NormHead"""
        out_shape = P.Shape()(hidden_states)[:-1] + (self.vocab_size,)
        hidden_states = self.reshape(hidden_states, (-1, self.hidden_size))

        if self.is_first_iteration:
            variance = self.square(self.weight)
            variance = self.sum(variance, 1)
            variance = self.reshape(variance, (-1, 1))
            variance_eps = self.sqrt(self.add(variance, self.eps))
            norm_weight = self.real_div(self.weight, variance_eps)
            if self.use_past:
                norm_weight = ops.depend(norm_weight, norm_weight)
                self.assign(self.weight, norm_weight)
        else:
            norm_weight = self.weight
            self.assign(self.weight, norm_weight)
            norm_weight = ops.depend(norm_weight, norm_weight)

        ori_type = hidden_states.dtype
        out = self.matmul(hidden_states.astype(self.compute_dtype),
                          norm_weight.astype(self.compute_dtype))
        out = self.reshape(out, out_shape)
        return self.cast(out, ori_type)

    def shard(self, parallel_config):
        """sharding for norm head"""
        self.square.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.sqrt.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.add.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1), (1,)))
        self.real_div.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),
                             (parallel_config.model_parallel * parallel_config.data_parallel, 1)))
        self.sum.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.matmul.shard(((1, 1),
                           (parallel_config.model_parallel * parallel_config.data_parallel, 1)))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan7BV2ForCausalLM(Baichuan2PreTrainedModel):
    r"""Provide baichuan2_7b training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan2_7b model.

        Inputs:
            input_ids(Tensor): The tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): The tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): Current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): The past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.

        Returns:
            Tensor, the loss or logits of the network.
    """
    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan7BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = Baichuan7BV2Model(config=config)
        self.lm_head = NormHead(hidden_size=config.hidden_size,
                                vocab_size=config.vocab_size,
                                use_past=config.use_past,
                                compute_dtype=config.compute_dtype)

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        loss_parallel_config.model_parallel = loss_parallel_config.model_parallel * loss_parallel_config.data_parallel
        loss_parallel_config.data_parallel = 1
        if vocab_size % (loss_parallel_config.model_parallel) != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, loss_parallel_config.model_parallel)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            self.lm_head.shard(config.parallel_config)

            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.lm_head.set_comm_fusion(2)
            else:
                self.lm_head.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                        dynamic_batch_valid_length, None, None, dynamic_block_tables, dynamic_slot_mapping)
        logger.info("Set dynamic input for baichuan2.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """Baichuan7BV2 ForCausalLM forward."""
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables, slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                return logits
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
