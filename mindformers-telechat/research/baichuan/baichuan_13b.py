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
"""Baichuan_13b models' APIs."""
from typing import Optional
import math
import numpy as np

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn, ops
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.flash_attention import FlashAttention
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer import AttentionMask, TransformerOpParallelConfig
from mindformers.modules.layers import Linear, _check_input_dtype, AlibiTensor
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from mindformers.models.utils import LayerSetting
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaFeedForward, LlamaRMSNorm
from mindformers.tools.logger import logger


__all__ = ['Baichuan13BForCausalLM', 'Baichuan13BModel']


class BaichuanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "baichuan"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan13BForCausalLM(BaichuanPreTrainedModel):
    r"""Provide baichuan_13B training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan_13B model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from research.baichuan.baichuan_13b import Baichuan13BForCausalLM
            >>> config = LlamaConfig(batch_size=2)
            >>> network = Baichuan13BForCausalLM(config=config)
    """
    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan13BForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id

        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.model = Baichuan13BModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")  # meta default: xavier_normal
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel

        self.slice.shard(((dp, 1),))
        self.not_equal.shard(((dp, 1), ()))
        self.mul.shard(((dp, 1), (dp, 1)))
        self.add.shard(((dp, 1), ()))
        if config.parallel_config.vocab_emb_dp:
            self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
        else:
            self.lm_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
        if config.parallel_config.pipeline_stage > 1:
            self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """Baichuan13BForCausalLM forward."""
        bsz, seqlen = input_ids.shape
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position,
                            init_reset, batch_valid_length)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(
            tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(
                    labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if not self.training:
            logits = self.reshape(logits, (bsz, seqlen, -1))
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


class Baichuan13BModel(BaichuanPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Baichuan13BDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of baichuan_13b decoderlayer
    """

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.num_layers = config.num_layers
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        self.get_attention_mask = AttentionMask(
            config.seq_length, parallel_config=config.parallel_config.dp_mp_config).to_float(config.compute_dtype)
        self.multiply_data = Tensor([-10000.0], dtype=config.compute_dtype)
        self.one = Tensor([1.0], dtype=config.compute_dtype)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.mul_mask = P.Mul()
        self.sub = P.Sub()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()

        self.tok_embeddings = LlamaEmbedding(
            config.vocab_size, config.hidden_size, param_init_type=config.param_init_type)
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
                                           multiple_of=config.multiple_of,
                                           n_kv_heads=config.n_kv_heads,
                                           ffn_dim_multiplier=config.ffn_dim_multiplier,
                                           norm_eps=config.rms_norm_eps,
                                           compute_dtype=config.compute_dtype,
                                           layernorm_compute_dtype=config.layernorm_compute_type,
                                           softmax_compute_dtype=config.softmax_compute_type,
                                           param_init_type=config.param_init_type,
                                           use_past=config.use_past,
                                           use_flash_attention=config.use_flash_attention,
                                           compute_in_2d=config.compute_in_2d,
                                           use_past_shard=config.use_past_shard,
                                           parallel_config=config.parallel_config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)

        self.build_alibi_tensor = AlibiTensor(
            seq_length=config.seq_length, num_heads=config.num_heads, parallel_config=config.parallel_config)

        dp = config.parallel_config.data_parallel
        self.tok_embeddings.pipeline_stage = 0
        if config.parallel_config.pipeline_stage > 1:
            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            self.tok_embeddings.set_comm_fusion(2)
            self.norm_out.set_comm_fusion(2)
        else:
            self.tok_embeddings.set_comm_fusion(
                config.parallel_config.gradient_aggregation_group)
            self.norm_out.set_comm_fusion(
                config.parallel_config.gradient_aggregation_group)

        self.tok_embeddings.shard(config.parallel_config)

        self.tile.shard(((1, 1, 1, 1), ()))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
        self.expand_dims.shard(((dp, 1, 1),))
        self.not_equal.shard(((dp, 1), ()))
        if config.compute_in_2d:
            self.norm_out.shard((dp, 1))
        else:
            self.norm_out.shard((dp, 1, 1))

        if self.use_past:
            seq_range = np.arange(config.seq_length).reshape(1, 1, -1)
            self.ones = P.Ones()
            self.range = Tensor(
                np.tile(seq_range, (config.batch_size, 1, 1)), mstype.int32)
            self.le_past = P.LessEqual()
            self.input_mask_all_ones = Tensor(
                np.ones((self.config.batch_size, self.config.seq_length), np.float32), mstype.float32)

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_position=None, init_reset=True, batch_valid_length=None):
        """Forward of baichuan_13b model."""
        # preprocess
        input_mask = self.cast(self.not_equal(
            tokens, self.pad_token_id), self.dtype)

        if self.is_first_iteration:
            mask = self.get_attention_mask(input_mask)
            alibi_tensor = self.build_alibi_tensor(input_mask, self.dtype)
            # mask: [bs, seq, seq]
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            mask = self.cast(self.le_past(
                self.range, valid_length), self.dtype)
            alibi_tensor = self.build_alibi_tensor(self.input_mask_all_ones, self.dtype)
            # mask: [bs, 1, 1]
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)

        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, alibi_tensor, mask,
                                  init_reset=init_reset, batch_valid_length=batch_valid_length)
        output = self.norm_out(h)
        return output


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
            - **mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

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
                 multiple_of: int = 256,
                 n_kv_heads: Optional[int] = None,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 compute_in_2d=False,
                 use_past_shard=False,
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
        self.compute_in_2d = compute_in_2d
        self.key_past = None
        self.value_past = None

        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.add = P.Add()
        self.attention_norm = LlamaRMSNorm(
            self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(
            self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = Baichuan13BAttention(batch_size=batch_size,
                                              seq_length=seq_length,
                                              dim=dim,
                                              n_heads=n_heads,
                                              n_kv_heads=n_kv_heads,
                                              compute_dtype=compute_dtype,
                                              softmax_compute_dtype=softmax_compute_dtype,
                                              param_init_type=param_init_type,
                                              use_past=use_past,
                                              use_flash_attention=use_flash_attention,
                                              compute_in_2d=compute_in_2d,
                                              use_past_shard=use_past_shard,
                                              parallel_config=parallel_config)
        self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                             hidden_dim=4 * self.hidden_size,
                                             multiple_of=multiple_of,
                                             ffn_dim_multiplier=ffn_dim_multiplier,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.feed_forward.shard(parallel_config)
        if self.compute_in_2d:
            self.add.shard(((dp, 1), (dp, 1)))
            self.attention_norm.shard((dp, 1))
            self.ffn_norm.shard((dp, 1))
        else:
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            if self.compute_in_2d:
                self.add.shard(((dp * mp, 1), (dp * mp, 1)))
                self.attention_norm.shard((dp * mp, 1))
                self.ffn_norm.shard((dp * mp, 1))
            else:
                self.add.shard(((dp, mp, 1), (dp, mp, 1)))
                self.attention_norm.shard((dp, mp, 1))
                self.ffn_norm.shard((dp, mp, 1))
            self.feed_forward.w2.shard(
                ((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if self.use_past:
            kv_shape = (batch_size, self.n_kv_head, seq_length, self.head_dim)
            self.key_past = Parameter(
                Tensor(np.zeros(kv_shape), self.dtype), name="key_past")
            self.value_past = Parameter(
                Tensor(np.zeros(kv_shape), self.dtype), name="value_past")
            self.ones = P.Ones()
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (1,)))
            self.assign_past = P.Assign().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.mul_past.shard(((dp, mp, 1, 1), (1,)))
                self.assign_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

    def construct(self, x, alibi_tensor, mask=None, init_reset=True, batch_valid_length=None):
        """ Forward of transformer block. """
        self._check_input(x, alibi_tensor, mask,
                          init_reset, batch_valid_length)
        # [bs, seq/1, hidden_dim] (first) [bs * seq/1, hidden_dim] (others)
        if self.compute_in_2d and x.ndim != 2:
            x = self.reshape(x, (-1, x.shape[-1]))
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        input_x = self.attention_norm(x)

        key_reset = None
        value_reset = None
        if self.use_past and self.is_first_iteration:
            # reset states, init_reset True for reuse and False for reset
            self.assign_past(self.key_past, self.mul_past(
                self.key_past, self.cast(init_reset, self.dtype)))
            self.assign_past(self.value_past, self.mul_past(
                self.value_past, self.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = ops.depend(input_x, key_reset)
            input_x = ops.depend(input_x, value_reset)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        h, layer_present = self.attention(input_x, alibi_tensor, mask,
                                          self.key_past, self.value_past, batch_valid_length)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign_past(self.key_past, key_present)
            self.assign_past(self.value_past, value_present)
            key_update = self.key_past
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = ops.depend(key_update, key_reset)
            value_update = ops.depend(value_update, value_reset)

        # add dependency for desired execution order
        ffn_out = ops.depend(ffn_out, value_update)
        ffn_out = ops.depend(ffn_out, key_update)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out, layer_present

    def _check_input(self, x, alibi_tensor, mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(alibi_tensor.dtype, "alibi_tensor",
                           [mstype.float32, mstype.float16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                bs = x.shape[0]
                batch_valid_length = self.ones((bs, 1), mstype.int32)
            _check_input_dtype(init_reset.dtype, "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True


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
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

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
                 use_flash_attention=False,
                 compute_in_2d=False,
                 use_past_shard=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
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
        self.compute_in_2d = compute_in_2d
        self.use_flash_attention = use_flash_attention

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(
            1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.add_alibi = P.Add()
        self.softmax = nn.Softmax().to_float(softmax_compute_dtype)
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()

        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wq = Linear(self.hidden_size,
                         self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wk = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wv = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.transpose.shard(((dp, 1, mp, 1),))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        self.add_alibi.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.softmax.softmax.shard(((dp, mp, 1, 1),))
        self.tile_kv.shard(((dp * mp, 1, 1, 1),))

        self.wq.shard(((dp, 1), (mp, 1)))
        self.wk.shard(((dp, 1), (mp, 1)))
        self.wv.shard(((dp, 1), (mp, 1)))
        self.wo.shard(((dp, mp), (1, mp)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.wo.shard(((dp, mp), (1, mp)),
                          out_strategy_matmul=((dp * mp, 1),))
        if parallel_config.recompute.select_recompute:
            self.tile_kv.recompute()
            self.batch_matmul_q_k.recompute()
            self.mul.recompute()
            self.add.recompute()
            self.cast_attn.recompute()
            self.softmax.softmax.recompute()
            self.batch_matmul.recompute()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=n_heads,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  input_layout='BNSD',
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  use_alibi_mask=True)
            self.flash_attention.shard(parallel_config)
            if parallel_config.recompute.select_recompute:
                self.flash_attention.recompute()

        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(seq_length).reshape(1, 1, -1)
            self.range = Tensor(
                np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
            self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
            self.add_past = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.equal = P.Equal().shard(((dp, 1, 1), (dp, 1, 1)))
            self.less = P.Less().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.add_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul_past.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    def construct(self, x: Tensor, alibi_tensor: Tensor, mask=None,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        x = self.reshape(x, (-1, x.shape[-1]))
        # [bs * seq/1, hidden_dim]
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key = self.cast(self.wk(x), self.dtype)    # dp, 1 -> dp, mp
        value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        query = self.reshape(query, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_head, self.head_dim))
        key = self.reshape(key, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                 self.n_kv_head, self.head_dim))
        value = self.reshape(value, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_kv_head, self.head_dim))
        # [bs, seq/1, n_head/n_kv_head, head_dim]
        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))

        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (
                    self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul_past(
                    key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul_past(
                    value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = (self.equal(
                    self.range, valid_length)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul_past(
                    key, self.expand_dims(valid_length_vector, 3))
                current_value = self.mul_past(
                    value, self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add_past(key_past, current_key)
                value = self.add_past(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            mask = self.expand_dim_post(mask, 1)
            mask = self.cast(mask, mstype.uint8)
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, alibi_tensor, mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output, layer_present

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
        x = self.reshape(x, (bs * n_kv_head, 1, seqlen, head_dim))
        x = self.tile_kv(x, (1, rep, 1, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        x_shape = x.shape
        if self.compute_in_2d:
            # [bs * seq/1, hidden_dim]
            new_shape = (-1, x_shape[-2] * x_shape[-1])
        else:
            # [bs, seq/1, hidden_dim]
            new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, alibi_tensor, mask):
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

        attention_probs = self.softmax(
            self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(
            self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge
