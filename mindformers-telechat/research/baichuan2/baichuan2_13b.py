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
"""Baichuan2_13b models' APIs."""
from typing import Optional
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import Parameter
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, HeUniform
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.flash_attention import FlashAttention
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.layers import Linear, _check_input_dtype, build_alibi_tensor_v2
from mindformers.modules.transformer import TransformerOpParallelConfig, LowerTriangularMaskWithDynamic
from mindformers.modules.infer_attention import InferAttention
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.models.utils import LayerSetting
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaFeedForward, LlamaRMSNorm
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode

__all__ = ['Baichuan13BV2ForCausalLM', 'Baichuan13BV2Model']


class Baichuan2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "baichuan2"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan13BV2ForCausalLM(Baichuan2PreTrainedModel):
    r"""Provide baichuan2_13B training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan2_13B model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from research.baichuan2.baichuan2_13b import Baichuan13BV2ForCausalLM
            >>> config = LlamaConfig(batch_size=2)
            >>> network = Baichuan13BV2ForCausalLM(config=config)
    """
    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan13BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.dtype = config.compute_dtype

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = Baichuan13BV2Model(config=config)
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
            self.lm_head.shard(config.parallel_config)
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))

            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.lm_head.set_comm_fusion(2)
            else:
                self.lm_head.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                        dynamic_batch_valid_length, None, None, dynamic_block_tables, dynamic_slot_mapping)
        logger.info("Set dynamic input for baichuan2.")

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Baichuan13BV2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, None, None, None, None, None, None, None, None, None, None, slot_mapping

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
        """Baichuan13BV2ForCausalLM forward."""
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
        output = self.model(tokens, batch_valid_length, block_tables, slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
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
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache


class Baichuan13BV2Model(Baichuan2PreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Baichuan13BV2DecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of baichuan2_13b decoderlayer
    """

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_flash_attention = config.use_flash_attention
        # only support flash attention in train and prefill predict process.
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        # only support paged attention in predict process.
        self.block_size = config.block_size
        self.num_blocks = config.num_blocks

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.mul_mask = P.Mul()
        self.mul_alibi = P.Mul()
        self.sub = P.Sub()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()
        self.transpose = P.Transpose()
        self.slice = P.StridedSlice()
        self.ones = P.Ones()

        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention and not
                                                          config.use_past)
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
            layer = Baichuan13BDecodeLayer(config.batch_size,
                                           config.seq_length,
                                           layer_id,
                                           dim=config.hidden_size,
                                           n_heads=config.num_heads,
                                           n_kv_heads=config.n_kv_heads,
                                           intermediate_size=config.intermediate_size,
                                           multiple_of=config.multiple_of,
                                           ffn_dim_multiplier=config.ffn_dim_multiplier,
                                           norm_eps=config.rms_norm_eps,
                                           compute_dtype=config.compute_dtype,
                                           layernorm_compute_dtype=config.layernorm_compute_type,
                                           softmax_compute_dtype=config.softmax_compute_type,
                                           param_init_type=config.param_init_type,
                                           use_past=config.use_past,
                                           is_dynamic=config.is_dynamic,
                                           use_flash_attention=self.use_flash_attention,
                                           block_size=self.block_size,
                                           num_blocks=self.num_blocks,
                                           parallel_config=config.parallel_config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)
        self.alibi_tensor = build_alibi_tensor_v2(seq_len=config.seq_length,
                                                  num_heads=config.num_heads,
                                                  return_tensors='ms',
                                                  dtype=self.dtype)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
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
            self.sub.shard(((1,), (dp, 1, 1)))
            self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
            self.mul_alibi.shard(((1, mp, 1, 1), (dp, 1, 1, 1)))

            self.expand_dims.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.gather.shard(((1, mp, 1, 1), (1,)))
            self.norm_out.shard((dp, 1, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, block_tables=None, slot_mapping=None):
        """Forward of baichuan2_13b model."""
        # preprocess
        bs, seq_len = self.shape(tokens)

        if not self.use_past:
            mask = self.casual_mask(tokens)  # mask: mask: [bs , 1, seq, seq]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float16)
            alibi_tensor = self.mul_alibi(self.alibi_tensor, self.reshape(input_mask, (bs, 1, -1, 1)))
        else:
            mask = None
            if self.is_first_iteration:
                if not self.use_flash_attention:
                    mask = self.casual_mask(tokens)  # mask: [bs , 1, seq, seq]
                alibi_tensor = self.slice(self.alibi_tensor, (0, 0, 0, 0),
                                          (1, self.alibi_tensor.shape[1], seq_len, seq_len), (1, 1, 1, 1))
            else:
                alibi_tensor = self.gather(self.alibi_tensor, batch_valid_length, 2)
                alibi_tensor = self.transpose(alibi_tensor, (2, 1, 0, 3))
                alibi_tensor = self.slice(alibi_tensor,
                                          (0, 0, 0, 0),
                                          (alibi_tensor.shape[0], alibi_tensor.shape[1], alibi_tensor.shape[2],
                                           block_tables.shape[1] * self.block_size),
                                          (1, 1, 1, 1),
                                          )
        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, alibi_tensor, mask, batch_valid_length, block_tables, slot_mapping)
        output = self.norm_out(h)
        return output


class Baichuan13BAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in Baichuan.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **alibi_tensor** (Tensor) - Alibi Tensor for position embedding used in attention.
            - **mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 is_dynamic=False,
                 use_flash_attention=False,
                 block_size: int = 128,
                 num_blocks: int = 224,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.block_size = block_size
        self.num_blocks = num_blocks

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.add_alibi = P.Add()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()

        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)
        self.wq = Linear(self.hidden_size,
                         self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)
        self.wk = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)
        self.wv = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.add_alibi.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))

            self.wq.shard(((dp, 1), (mp, 1)))
            self.wk.shard(((dp, 1), (mp, 1)))
            self.wv.shard(((dp, 1), (mp, 1)))
            self.wo.shard(((dp, mp), (1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
        if parallel_config.recompute.select_recompute:
            self.batch_matmul_q_k.recompute()
            self.mul.recompute()
            self.add_alibi.recompute()
            self.softmax.recompute()
            self.batch_matmul.recompute()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=n_heads,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  input_layout='BNSD',
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  use_alibi_mask=True)
            self.flash_attention.shard(parallel_config)
        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.head_dim,
                                                  self.n_kv_head,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=65536,
                                                  next_tokens=65536,
                                                  block_size=self.block_size,
                                                  num_blocks=self.num_blocks,
                                                  use_alibi_mask=True,
                                                  use_rope_rotary_emb=False,
                                                  use_flash_attention=use_flash_attention,
                                                  compute_dtype=compute_dtype).shard(parallel_config)
            self.infer_attention.shard(parallel_config)

    def construct(self, x: Tensor, alibi_tensor: Tensor, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)

        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
        value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp

        # key and value for current token(s)
        if self.use_past:
            attention = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                             None, mask, alibi_tensor)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            if self.use_flash_attention:
                attention = self.flash_attention(query, key, value, mask, alibi_tensor)
                attention = self._merge_heads(attention)
            else:
                key = self._repeat_kv(key, self.n_rep)
                value = self._repeat_kv(value, self.n_rep)
                attention = self._attn(query, key, value, mask, alibi_tensor)

        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        # [bs, seq/1, hidden_dim]
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask, alibi_tensor):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add_alibi(score, alibi_tensor)

        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class Baichuan13BDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            multiple_of(int): The SwiGLU hidden layer size multiple of large power of 2.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **alibi_tensor** (Tensor) - Alibi Tensor for position embedding used in attention.
            - **mask** (Tensor) - Float Tensor, If the use_past is
            False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, head_dim, seq_length),
              (batch_size, num_heads, seq_length, head_dim)).

    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 is_dynamic=False,
                 use_flash_attention=False,
                 block_size: int = 128,
                 num_blocks: int = 224,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads

        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.is_dynamic = is_dynamic
        self.key_past = None
        self.value_past = None
        self.use_seq_parallel = parallel_config.use_seq_parallel

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.add = P.Add()
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = Baichuan13BAttention(batch_size=batch_size,
                                              seq_length=seq_length,
                                              dim=dim,
                                              n_heads=n_heads,
                                              n_kv_heads=n_kv_heads,
                                              compute_dtype=compute_dtype,
                                              softmax_compute_dtype=softmax_compute_dtype,
                                              param_init_type=param_init_type,
                                              use_past=use_past,
                                              is_dynamic=is_dynamic,
                                              use_flash_attention=use_flash_attention,
                                              block_size=block_size,
                                              num_blocks=num_blocks,
                                              parallel_config=parallel_config)
        self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                             intermediate_size=intermediate_size,
                                             hidden_dim=4 * self.hidden_size,
                                             multiple_of=multiple_of,
                                             ffn_dim_multiplier=ffn_dim_multiplier,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type,
                                             is_dynamic=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.add.shard(((dp, mp, 1), (dp, mp, 1)))
                self.attention_norm.shard((dp, mp, 1))
                self.ffn_norm.shard((dp, mp, 1))
                self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

    def construct(self, x, alibi_tensor, mask=None, batch_valid_length=None, block_tables=None, slot_mapping=None):
        """ Forward of transformer block. """
        self._check_input(x, alibi_tensor, mask)
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, alibi_tensor, mask, batch_valid_length, block_tables, slot_mapping)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out

    def _check_input(self, x, alibi_tensor, mask):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(alibi_tensor.dtype, "alibi_tensor",
                           [mstype.float32, mstype.float16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask", [mstype.float32, mstype.float16, mstype.uint8], self.cls_name)
        return True


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
                 compute_dtype=mstype.float32,
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
