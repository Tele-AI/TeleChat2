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
"""Qwen models' APIs."""

import copy
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline
from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_use_rope_self_define
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks,\
    FreqsMgr
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaSiLU, LlamaRMSNorm
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.models.utils import LayerSetting
from mindformers.version_control import check_valid_flash_attention

from .qwen_config import QwenConfig


class QwenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = QwenConfig
    base_model_prefix = "qwen"


class MatMulPad(nn.Cell):
    """
    Run MatMul with padding the x and weight to satisfy the value is divisible by 512 when enable_emb_opt is True.
    """
    def __init__(self, matmul, vocab_size, align_size, enable_emb_opt=False):
        super().__init__()
        self.matmul = matmul
        self.enable_emb_opt = enable_emb_opt
        if self.enable_emb_opt:
            matmul_in_strategy = self.matmul.attrs.get('in_strategy', None)
            self.zeros = P.Zeros()
            self.concat_weight = P.Concat(axis=0)
            self.strided_slice = P.StridedSlice()
            if matmul_in_strategy is not None and _get_parallel_mode() in ParallelMode.SEMI_AUTO_PARALLEL:
                self.concat_weight.shard((matmul_in_strategy[1],
                                          matmul_in_strategy[1])).add_prim_attr("skip_redistribution", True)
                self.strided_slice.shard(((matmul_in_strategy[0][0],
                                           matmul_in_strategy[1][0]),)).add_prim_attr("skip_redistribution", True)
                align_size = align_size * matmul_in_strategy[1][0]
            _, remainder = divmod(vocab_size, align_size)
            if remainder > 0:
                self.pad_length = align_size - remainder
            else:
                self.enable_emb_opt = False
                logger.warning("The vocab_size is already aligned, no need to pad.")

    def construct(self, x, weight):
        vocab_size, hidden_size = weight.shape
        if self.enable_emb_opt:
            pad_weight = self.zeros((self.pad_length, hidden_size), P.DType()(weight))
            weight = self.concat_weight([weight, pad_weight])
        output = self.matmul(x, weight)
        if self.enable_emb_opt:
            output = self.strided_slice(output, (0, 0), (x.shape[0], vocab_size), (1, 1))
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenForCausalLM(QwenPreTrainedModel):
    """Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.
    """
    @lazy_inline
    def __init__(self, config=None):
        super().__init__(config)

        self.transformer = QwenModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        loss_parallel_config.model_parallel = loss_parallel_config.model_parallel * loss_parallel_config.data_parallel
        loss_parallel_config.data_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)

        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.ignore_token_id = config.ignore_token_id
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.ones = P.Ones()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub_batch_valid_len = P.Sub()
        self.gather = P.Gather(1)
        self.enable_slice_dp = config.enable_slice_dp

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if config.enable_emb_opt:
            lm_head_matmul = self.lm_head.matmul
            self.lm_head.matmul = MatMulPad(lm_head_matmul, config.vocab_size, 512, config.enable_emb_opt)
        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Qwen model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        input_embeds = Tensor(kwargs["input_embeds"]) if "input_embeds" in kwargs else None
        bs = input_ids.shape[0]
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return input_ids, labels, None, None, None, input_embeds, None, None, None, None, None, slot_mapping

    # pylint: disable=W0613
    def set_dynamic_inputs(self, **kwargs):
        """Set inputs when is_dynamic=True."""
        # dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_input_embeds = Tensor(shape=[None, None, None], dtype=mstype.float32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(None, None, None, None, None,
                        dynamic_input_embeds, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)
        logger.info("Set dynamic input for Qwen.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """construct"""

        if input_ids is None and input_embeds is None:
            raise ValueError()

        if input_ids is not None:
            bsz, seqlen = input_ids.shape
            if self.training:
                tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
            else:
                tokens = input_ids

            input_embeds = self.to_embeddings(tokens)

            if attention_mask is None:
                input_attention_masks = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
            else:
                input_attention_masks = attention_mask
        else:
            # pass embeds, and attn_mask, label
            bsz, seqlen, _ = input_embeds.shape
            input_attention_masks = attention_mask

        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        output = self.transformer(input_embeds=input_embeds, input_attention_masks=input_attention_masks,
                                  init_reset=init_reset, batch_valid_length=batch_valid_length,
                                  batch_index=batch_index, zactivate_len=zactivate_len,
                                  block_tables=block_tables, slot_mapping=slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_attention_masks, 1)
            return logits, input_mask

        input_mask = input_attention_masks
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    _, label_seqlen = labels.shape
                    labels = self.slice(labels, (0, 1), (bsz, label_seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_attention_masks, label_mask)

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def to_embeddings(self, input_ids):
        input_embeds = self.transformer.wte(input_ids)
        input_embeds = self.transformer.drop(input_embeds)
        return input_embeds

    def shard(self, parallel_config):
        """sharding for feedforward"""

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.enable_slice_dp:
            self.slice.shard(((dp, 1),))
        else:
            self.slice.shard(((1, 1),))
        self.not_equal.shard(((dp, 1), ()))
        self.mul.shard(((dp, 1), (dp, 1)))
        self.add.shard(((dp, 1), ()))
        self.sub_batch_valid_len.shard(((1,), ()))
        self.gather.shard(((dp, 1, 1), (dp,)))

        if parallel_config.vocab_emb_dp:
            self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
        else:
            self.lm_head.shard(strategy_matmul=((1, 1), (dp * mp, 1)))

    def kvcache(self, layer_idx):
        key_cache = self.transformer.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.transformer.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache


class QwenModel(QwenPreTrainedModel):
    """transformer"""

    def __init__(self, config):
        super().__init__(config)
        self.dtype = config.compute_dtype
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_layers
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.seq_length = config.seq_length
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_heads
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        embedding_parallel_optimizer = config.embedding_parallel_optimizer

        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention and check_valid_flash_attention(
                        import_fa_valid=True, fa_type='FlashAttention')

        # 1. wte
        self.wte = LlamaEmbedding(self.vocab_size, self.embed_dim, param_init_type=config.param_init_type,
                                  parallel_optimizer=embedding_parallel_optimizer)

        # 2. drop
        self.drop = nn.Dropout(p=config.emb_dropout_prob)

        # 4. h hidden layers for transformer
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = QwenDecodeLayer(layer_id,
                                    dim=config.hidden_size,
                                    n_heads=config.num_heads,
                                    intermediate_size=config.intermediate_size,
                                    norm_eps=config.rms_norm_eps,
                                    compute_dtype=config.compute_dtype,
                                    layernorm_compute_dtype=config.layernorm_compute_type,
                                    softmax_compute_dtype=config.softmax_compute_type,
                                    rotary_dtype=config.rotary_dtype,
                                    param_init_type=config.param_init_type,
                                    qkv_has_bias=True,
                                    use_past=config.use_past,
                                    use_flash_attention=self.use_flash_attention,
                                    block_size=config.block_size,
                                    num_blocks=config.num_blocks,
                                    parallel_config=config.parallel_config,
                                    qkv_concat=config.qkv_concat)

            self.layer_setting(layer, layer_id)

            self.layers.append(layer)

        self.use_rope_self_define = get_use_rope_self_define()
        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=self.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method)
        self.casual_mask = CausalMaskForQwen(seq_length=config.seq_length,
                                             compute_type=config.compute_dtype,
                                             is_dynamic=config.is_dynamic,
                                             pad_token_id=config.pad_token_id,
                                             use_flash_attention=self.use_flash_attention)

        # 5. ln_f
        self.ln_f = LlamaRMSNorm(
            self.embed_dim,
            eps=config.rms_norm_eps,
            compute_type=config.layernorm_compute_type
        )

        self.shape = P.Shape()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

            self.wte.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.ln_f.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.wte.set_comm_fusion(2)
                self.ln_f.set_comm_fusion(2)
            else:
                self.wte.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.ln_f.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

    # pylint: disable=W0613
    def construct(self, input_embeds: Tensor, input_attention_masks: Tensor,
                  init_reset=True, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        """construct"""
        # 2. rotary_emb
        hidden_states = input_embeds
        bs, seq_len, _ = self.shape(hidden_states)
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                if not self.use_flash_attention:
                    mask = self.casual_mask(masks=input_attention_masks)
                    mask = self.casual_mask.post_process(mask)
                if self.use_rope_self_define:
                    freqs_cis = self.freqs_mgr(seq_len)
                else:
                    freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            freqs_cis = self.freqs_mgr(seq_len)
            mask = self.casual_mask(masks=input_attention_masks)
            mask = self.casual_mask.post_process(mask)  # mask: [bs, 1, seq, seq]

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                           block_tables=block_tables, slot_mapping=slot_mapping)

        # 5. ln_f
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def shard(self, parallel_config):
        """sharding for feedforward"""
        self.wte.shard(parallel_config)
        self.casual_mask.shard(parallel_config)
        self.ln_f.shard((parallel_config.data_parallel, 1, 1))


class QwenDecodeLayer(LLamaDecodeLayer):
    """Qwen decode layer"""

    def __init__(self,
                 layer_id,
                 intermediate_size,
                 parallel_config,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 **kwargs):
        super().__init__(layer_id,
                         intermediate_size=intermediate_size,
                         parallel_config=parallel_config,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         **kwargs)

        self.feed_forward = QwenFeedForward(dim=self.hidden_size,
                                            intermediate_size=intermediate_size,
                                            compute_dtype=compute_dtype,
                                            param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            if kwargs.get('qkv_concat'):
                self.attention.w.bias_add.shard(((dp, mp), (mp,)))
            else:
                self.attention.wq.bias_add.shard(((dp, mp), (mp,)))
                self.attention.wk.bias_add.shard(((dp, mp), (mp,)))
                self.attention.wv.bias_add.shard(((dp, mp), (mp,)))


class QwenFeedForward(nn.Cell):
    r"""
    Qwen FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                intermediate_size=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 intermediate_size=0,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 is_dynamic=False):
        super().__init__()

        hidden_dim = intermediate_size
        self.dtype = compute_dtype
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.silu = LlamaSiLU()

        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x)  # dp, 1 -> dp, mp
        hidden = self.w3(x)  # dp, 1 -> dp, mp
        hidden = self.mul(gate, self.silu(hidden).astype(self.dtype))  # dp, mp -> dp, mp
        output = self.w2(hidden)  # dp, mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))
        self.silu.shard(((dp, 1, mp),))


class CausalMaskForQwen(nn.Cell):
    r""" Get the Lower triangular matrix from the input_ids.
            [[[1. 0. 0. 0. 0]
              [1. 1. 0. 0. 0]
              [1. 1. 1. 0. 0]
              [1. 1. 1. 1. 0]
              [1. 1. 1. 1. 0]]]"""

    def __init__(self, seq_length, compute_type=mstype.float16,
                 is_dynamic=False, pad_token_id=0, use_flash_attention=False):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length))), mstype.float32)

        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.mul_post = P.Mul()
        self.expand_dim_post = P.ExpandDims()

    def construct(self, tokens=None, masks=None):
        """Forward process of the CausalMask"""
        if tokens is not None:
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)

        shape_right = (bs, 1, seq_len)
        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.mul(mask_right, lower_triangle)
        return attention_mask

    def increment(self, seq_range, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def increment_slice(self, seq_range, seq_length, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        else:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, seq_length), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range_mask, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def post_process(self, mask):
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        mask = self.expand_dim_post(mask, 1)
        if not self.use_flash_attention:
            mask = self.mul_post(mask, self.multiply_data)
        else:
            mask = self.cast(mask, mstype.uint8)
        return mask

    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.not_equal.shard(((dp, 1), ()))
        self.expand_dim.shard(((1, 1),))
        self.mul.shard(((dp, 1, 1), (1, 1, 1)))
        self.less_equal.shard(((1, 1, 1), (1, 1, 1)))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.mul_post.shard(((dp, 1, 1, 1), (1,)))
        self.expand_dim_post.shard(((dp, 1, 1),))
