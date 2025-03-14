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
"""LLaMA models' APIs."""
import copy
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Condition

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, mint, Parameter
from mindspore.common.initializer import initializer
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid, check_use_3d_tensor_parallel_valid
from mindformers.modules.layers import Linear, FreqsMgr
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode

from .llama_config import LlamaConfig
from .llama_layer import LlamaEmbedding, LlamaRMSNorm
from .llama_transformer import LLamaDecodeLayer
from .llama_interleave import LLamaDecodeLayerInterleave
from ..utils import lazy_inline
from ...tools.logger import logger

__all__ = ['LlamaModel', 'LlamaForCausalLM']


class LlamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "llama"


class LlamaModel(LlamaPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Returns:
            output: Tensor, the output of llama decoderlayer

    Examples:
        >>> from mindformers import LlamaModel
        >>> network = LlamaModel.from_pretrained('llama_7b')
        >>> type(network)
        <class 'mindformers.models.llama.llama.LlamaModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.chunk_prefill = config.chunk_prefill
        self.use_past = config.use_past
        self.use_eod_attn_mask_compression = config.use_eod_attn_mask_compression
        self.use_flash_attention = config.use_flash_attention
        self.use_ring_attention = config.use_ring_attention
        self.parallel_decoding = config.parallel_decoding_params is not None
        self.concat = P.Concat(-1)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.rmsnorm_compute_2d = config.rmsnorm_compute_2d
        self.rl_config = config.rl_config

        if config.moe_config.expert_num > 1:
            logger.info("MoE config is provided, use MoE FFN")
        else:
            logger.info("MoE config is None, use normal FFN")
        if not self.use_flash_attention and self.use_ring_attention:
            raise ValueError(f"When the ring_attention = True, the flash_attention must be True.")
        if not self.use_flash_attention and self.use_eod_attn_mask_compression:
            raise ValueError(f"When the use_eod_attn_mask_compression = True, the flash_attention must be True.")
        self.seq_split_num = config.parallel_config.seq_split_num
        self.seq_pipe = self.seq_split_num > 1
        if self.seq_pipe:
            dp = config.parallel_config.data_parallel
            if self.use_ring_attention:
                raise ValueError(f"When the seq_pipe = True, the use_ring_attention cannot be True.")
            if config.use_attn_mask_compression or config.use_eod_attn_mask_compression:
                raise ValueError(f"Currently, when the seq_pipe = True, "
                                 f"both use_attn_mask_compression and use_eod_attn_mask_compression cannot be True.")
            self.n_kv_head = self.n_head if config.n_kv_heads is None else config.n_kv_heads
            kv_shape = (config.batch_size * dp, self.n_kv_head, config.seq_length, self.head_dim)
            self.zeros = initializer('zeros', kv_shape, dtype=self.dtype)
            self.seq_update = Tensor(1, dtype=mstype.int32)
            self.seq_zero = Tensor(0, dtype=mstype.int32)
            self.seq_seg_len = config.seq_length // self.seq_split_num
            kv_mask = np.zeros((1, self.n_kv_head, config.seq_length, self.head_dim), np.int32)
            for s in range(self.seq_split_num):
                kv_mask[:, :, s * self.seq_seg_len: (s + 1) * self.seq_seg_len, :] = s
            self.kv_mask = Tensor(kv_mask)
            self.seq_chunk = Parameter(Tensor(0, dtype=mstype.int32), name="seq_chunk",
                                       requires_grad=False, parallel_optimizer=False)
            cp = config.parallel_config.context_parallel
            mp = config.parallel_config.model_parallel
            self.equal_kv = P.Equal().shard(((dp, mp, cp, 1), ()))
            self.kv_mask_add = P.Add().shard(((dp, mp, cp, 1), (1, mp, cp, 1)))
            self.assign_add_count = P.AssignAdd()
            self.assign_count = P.Assign()
            self.assign_mask = P.Assign().shard(((dp, 1), (dp, 1)))
            self.mask_zeros = Tensor(np.zeros((config.batch_size * dp, config.seq_length)), mstype.float32)

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  parallel_config=config.parallel_config,
                                  is_dynamic=config.is_dynamic)
        self.residual_cast_flag = config.residual_dtype != self.dtype
        if self.residual_cast_flag:
            logger.info(f"residual in llama model cast flag: {self.residual_cast_flag}, "
                        f"residual dtype: {config.residual_dtype}")
        self.freqs_mgr.shard(config.parallel_config)
        total_batch_size_in_dp = config.batch_size * config.parallel_config.data_parallel
        use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          batch_size=total_batch_size_in_dp,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_attn_mask_compression=use_attn_mask_compression,
                                                          use_past=config.use_past,
                                                          seq_split_num=self.seq_split_num,
                                                          chunk_prefill=config.chunk_prefill)

        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             init_method_std=config.init_method_std,
                                             param_init_type=config.embedding_init_type,
                                             parallel_optimizer=config.parallel_optimizer,
                                             rmsnorm_compute_2d=config.rmsnorm_compute_2d)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.use_3d_tensor_parallel = check_use_3d_tensor_parallel_valid(config)
        self.tp_x = getattr(config, "tp_x", 1)
        self.tp_y = getattr(config, "tp_y", 1)
        self.tp_z = getattr(config, "tp_z", 1)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num,
                                          config.start_stage,
                                          config.stage_num)
        for layer_id in range(config.num_layers):
            if self.fine_grain_interleave:
                layer = LLamaDecodeLayerInterleave(config.batch_size,
                                                   config.seq_length,
                                                   layer_id,
                                                   dim=config.hidden_size,
                                                   n_heads=config.num_heads,
                                                   num_layers=config.num_layers,
                                                   multiple_of=config.multiple_of,
                                                   n_kv_heads=config.n_kv_heads,
                                                   intermediate_size=config.intermediate_size,
                                                   ffn_dim_multiplier=config.ffn_dim_multiplier,
                                                   norm_eps=config.rms_norm_eps,
                                                   qkv_has_bias=config.qkv_has_bias,
                                                   attn_proj_has_bias=config.attn_proj_has_bias,
                                                   qkv_concat=config.qkv_concat,
                                                   compute_dtype=config.compute_dtype,
                                                   layernorm_compute_dtype=config.layernorm_compute_type,
                                                   softmax_compute_dtype=config.softmax_compute_type,
                                                   rotary_dtype=config.rotary_dtype,
                                                   param_init_type=config.param_init_type,
                                                   residual_dtype=config.residual_dtype,
                                                   use_flash_attention=config.use_flash_attention,
                                                   use_ring_attention=config.use_ring_attention,
                                                   use_attn_mask_compression=config.use_attn_mask_compression,
                                                   use_eod_attn_mask_compression=config.use_eod_attn_mask_compression,
                                                   fine_grain_interleave=config.fine_grain_interleave,
                                                   init_method_std=config.init_method_std,
                                                   parallel_config=config.parallel_config)
            else:
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
                                         attn_proj_has_bias=config.attn_proj_has_bias,
                                         qkv_concat=config.qkv_concat,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         rotary_dtype=config.rotary_dtype,
                                         param_init_type=config.param_init_type,
                                         residual_dtype=config.residual_dtype,
                                         use_past=config.use_past,
                                         is_dynamic=config.is_dynamic,
                                         use_flash_attention=config.use_flash_attention,
                                         use_ring_attention=config.use_ring_attention,
                                         use_attn_mask_compression=config.use_attn_mask_compression,
                                         use_eod_attn_mask_compression=config.use_eod_attn_mask_compression,
                                         block_size=config.block_size,
                                         num_blocks=config.num_blocks,
                                         use_rope_slice=config.use_rope_slice,
                                         rmsnorm_compute_2d=config.rmsnorm_compute_2d,
                                         batch_size=config.batch_size,
                                         moe_config=config.moe_config,
                                         parallel_config=config.parallel_config,
                                         parallel_decoding=self.parallel_decoding,
                                         rl_config=self.rl_config,
                                         fused_kernel=config.fused_rms_norm,
                                         init_method_std=config.init_method_std,
                                         chunk_prefill=config.chunk_prefill,
                                         use_3d_tensor_parallel=self.use_3d_tensor_parallel,
                                         tp_x=self.tp_x,
                                         tp_y=self.tp_y,
                                         tp_z=self.tp_z
                                         )
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type,
                                     fused_kernel=config.fused_rms_norm)
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel

        self.tok_embeddings.pipeline_stage = config.start_stage
        if config.parallel_config.pipeline_stage > 1:
            if config.stage_num == 0:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            else:
                self.norm_out.pipeline_stage = config.start_stage + config.stage_num - 1
            self.tok_embeddings.set_comm_fusion(2)
            self.norm_out.set_comm_fusion(2)
        else:
            self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            if self.fine_grain_interleave or config.rmsnorm_compute_2d:
                self.norm_out.shard((dp * cp, 1))
            else:
                self.norm_out.shard((dp, cp, 1))
        else:
            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.concat.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if self.fine_grain_interleave or config.rmsnorm_compute_2d:
                if self.rl_config is not None:
                    self.norm_out.shard((dp * cp * mp, 1))
                else:
                    self.norm_out.shard((dp * cp, 1))
            else:
                self.norm_out.shard((dp, cp, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_embeds=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  attention_mask=None, position_ids=None, q_seq_lens=None, seq_range=None, actual_seq_len=None):
        """
        Forward of llama model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_embeds: the embedding Tensor of tokens, Tensor of shape:math:`(batch_size, seq/_length, hidden_size)`.
                Default None.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        if actual_seq_len is not None:
            actual_seq_len = self.reshape(actual_seq_len, (-1,))
        kv_mask = None
        seq_chunk = None
        rmsnorm_compute_2d = self.training and self.rmsnorm_compute_2d
        if self.chunk_prefill and self.is_first_iteration:
            # get chunk + decode masks
            if attention_mask is not None:
                mask = attention_mask
            else:
                mask = self.casual_mask.chunk_masks(seq_range)
            # get chunk + decode pos
            freqs_cis = self.freqs_mgr.chunk_with_decode(seq_range)
        elif self.parallel_decoding:
            # FA with TH layout, mask is 2D, FA with BSH layout, mask is 4D
            if self.is_first_iteration:
                mask = self.casual_mask.prefill()
            else:
                mask = attention_mask
            freqs_cis = self.freqs_mgr.increment_multi_ids(position_ids)
        elif self.use_eod_attn_mask_compression and not self.use_ring_attention:
            mask = self.casual_mask()
            freqs_cis = self.freqs_mgr(seq_len, position_ids)
        elif attention_mask is not None:
            mask = attention_mask
            mask = self.cast(mask, mstype.uint8)
            freqs_cis = self.freqs_mgr(seq_len, position_ids)
        else:
            mask = None
            if self.use_past:
                if self.is_first_iteration:
                    freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                    if self.use_flash_attention:
                        mask = self.casual_mask.prefill()
                    else:
                        mask = self.casual_mask(tokens)
                    if prefix_keys_values is not None:
                        if mask is None:
                            mask = self.casual_mask(tokens)
                        prefix_length = prefix_keys_values[0].shape[2]
                        prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                        mask = self.concat((prefix_mask, mask))
                else:
                    freqs_cis = self.freqs_mgr.increment(batch_valid_length)
            else:
                if self.seq_pipe:
                    mask = self.casual_mask(tokens, seq_chunk=self.seq_chunk)
                    seq_chunk = P.ReLU()(self.seq_chunk)
                    kv_mask = self.cast(self.equal_kv(self.kv_mask_add(self.zeros, self.kv_mask), seq_chunk),
                                        self.dtype)
                    seq_update = F.depend(self.seq_update, mask)
                    seq_update = F.depend(seq_update, kv_mask)
                    mask = F.depend(mask, self.assign_add_count(self.seq_chunk, seq_update))
                elif not self.use_ring_attention:
                    mask = self.casual_mask(tokens)
                freqs_cis = self.freqs_mgr(seq_len, seq_chunk=seq_chunk)
                if prefix_keys_values is not None:
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        if input_embeds is not None:
            h = self.cast(input_embeds, self.dtype)
        else:
            h = self.cast(self.tok_embeddings(tokens), self.dtype)
        if not rmsnorm_compute_2d:
            h = self.reshape(h, (bs, seq_len, self.hidden_size))    # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                               slot_mapping=slot_mapping, prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                               kv_mask=kv_mask, seq_chunk=seq_chunk, actual_seq_len=actual_seq_len)
        if rmsnorm_compute_2d:
            h = self.reshape(h, (bs * seq_len, -1))
        output = self.norm_out(h)
        return output

    def clear_kv_cache(self):
        zeros = 0.0
        return_tuple = ()
        return_tuple += (self.assign_count(self.seq_chunk, self.seq_zero),)
        return_tuple += (self.assign_mask(self.casual_mask.mask_cache, self.mask_zeros),)
        return F.depend(zeros, return_tuple)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForCausalLM(LlamaPreTrainedModel):
    r"""
    Provide llama training loss or logits through network.

    Args:
        config (LlamaConfig, optional): The config of llama model. Default: `None` .

    Inputs:
        - **input_ids** (Tensor) - the indices of input sequence tokens in the vocabulary with data type Int64/Int32,
          Tensor of shape :math:`(batch, seq\_length)`.
        - **labels** (Tensor, optional) - the labels of inputs with data type Int64/Int32, Tensor of
          shape :math:`(batch, seq\_length)` . Default: ``None``.
        - **input_position** (Tensor, optional) - the position ids of inputs (at incremental reasoning mode) which is
          an increasing sequence with data type Int64/Int32, Tensor :math:`(batch, seq\_length)`.
          Default: ``None``.
        - **position_ids** (Tensor, optional) - the position ids of inputs which is
          an increasing sequence with data type
          Int64/Int32, Tensor :math:`(batch, seq\_length)`. Default: ``None``.
        - **attention_mask** (Tensor, optional) - input sentences padding mask, where 0 indicates padding position with
          data type Int64/Int32, Tensor of shape :math:`(batch, seq\_length)`. Default: ``None``.
        - **input_embeds** (Tensor, optional) - the embedding of inputs with data type Float32/Float16, Tensor of
          shape :math:`(batch, seq\_length, hidden\_size)`. Default: ``None``.
        - **init_reset** (Tensor, optional) - A Bool tensor with shape [1], used to clear the past key parameter and
          past value parameter used in the incremental prediction. Only valid when use_past is True.
          Tensor of shape :math:`(1)`. Default: ``Tensor([True])``.
        - **batch_valid_length** (Tensor, optional) - Int32 tensor with shape [batch_size]
          the past calculated the index.
          Used for incremental prediction when the use_past is True. Default: ``None``.
        - **batch_index** (Tensor, optional) - Discard argument. Will be deleted in the future. Default: ``None``.
        - **zactivate_len** (Tensor, optional) - Discard argument. Will be deleted in the future. Default: ``None``.
        - **block_tables** (Tensor, optional) - Int64 type Tensor, store mapping tables for each sequence.
          Default: ``None``.
        - **slot_mapping** (Tensor, optional) - Int32 type Tensor, token cache physical slot index. Default: ``None``.
        - **prefix_keys_values** (Tensor, optional) - Discard argument. Will be deleted in the future.
          Default: ``None``.
        - **llm_boost_inputs** (Tensor, optional) - Discard argument. Will be deleted in the future. Default: ``None``.
        - **q_seq_lens** (Tensor, optional) - In parallel decoding, the query may be flattened.
          The Paged Attention operator need `q_seq_lens` to obtain the length information. Default: ``None`` .
        - **loss_mask** (Tensor, optional) - Float32/Int32 type tensor, which is used to determine
          whether the corresponding token position participates in the loss calculation. If the value is :math:`(1)`,
          the loss of the position is calculated, and :math:`(0)` is not calculated. Default: ``None``.
        - **gather_index** (Tensor, optional) - Int32 type Tensor, used to obtain the last latent vector of
          each sequence. Default: ``None``.
        - **seq_range** (Tensor, optional) - Int32 type Tensor, used to obtain Mask and positional encoding of
          valid tokens for each sequence. Default: ``None``.
        - **actual_seq_len** (Tensor, optional) - Int32 type Tensor, used to automatically generate attention mask
          within FlashAttention for eod text. Default: ``None``.

    Outputs:
        Tensor. If it is in training mode, the output Tensor contains loss;
        If it is in prediction mode, the output Tensor contains logits;
        If it is in evaluation mode, the output Tensor contains logits, tokens, and input masks.

    Examples:
        >>> from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
        >>> import mindspore as ms
        >>> ms.set_context(mode=0)
        >>> config = LlamaConfig(batch_size=2)
        >>> network = LlamaForCausalLM(config=config)
        >>> type(network)
        <class 'mindformers.models.llama.llama.LlamaForCausalLM'>
        >>> from mindformers import LlamaForCausalLM
        >>> network = LlamaForCausalLM.from_pretrained('llama2_7b')
        >>> type(network)
        <class 'mindformers.models.llama.llama.LlamaForCausalLM'>
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(LlamaForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.chunk_prefill = config.chunk_prefill
        self.rl_config = config.rl_config

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.prefill_gather_flatten = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.predict_run_mode = get_predict_run_mode()
        logger.info("Predict run mode: {}".format(self.predict_run_mode))
        if self.predict_run_mode and self.config.is_dynamic:
            logger.info("use_flash_attention is set to True when run_mode is predict and is_dynamic is True.")
            self.config.use_flash_attention = True
        self.model = LlamaModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")  # meta default: xavier_normal
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.embedding_weight

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        loss_parallel_config.data_parallel *= loss_parallel_config.context_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss,
                                     seq_split_num=config.parallel_config.seq_split_num)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))
        else:
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.prefill_gather_flatten.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                if self.rl_config is not None:
                    self.lm_head.shard(strategy_matmul=((dp * cp * mp, 1), (1, 1)))
                else:
                    self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))

        if config.parallel_config.pipeline_stage > 1:
            self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)
        self.parallel_decoding = config.parallel_decoding_params is not None
        self.input_sliced_sig = config.input_sliced_sig

    def to_embeddings(self, tokens):
        """return embedding tokens"""
        return self.model.tok_embeddings(tokens)

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        position_ids = Tensor(np.zeros(shape=tuple([bs, seq])), mstype.int32) if self.parallel_decoding else None
        mask = Tensor(np.zeros(shape=tuple([seq, seq])), mstype.float16) if self.parallel_decoding else None
        q_seq_lens = Tensor(np.zeros(shape=tuple([bs])), mstype.int32) if self.parallel_decoding else None
        outputs = (input_ids, labels, None, position_ids, mask, None, None, None, None, None, None, slot_mapping,
                   prefix_keys_values, None, q_seq_lens)
        return outputs

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        dynamic_position_ids = Tensor(shape=[None, None], dtype=mstype.int32) if self.parallel_decoding else None
        dynamic_mask = Tensor(shape=[None, None], dtype=mstype.float16) if self.parallel_decoding else None
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32) if self.parallel_decoding else None
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None, dynamic_q_seq_lens, None,
                            None, None, None)
        elif self.use_past:
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None, None, dynamic_q_seq_lens, None, None, None, None)
        elif kwargs.get("pre_gather", False):
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, None, None, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            None, None, None, None, None, None)
        logger.info("Set dynamic input for llama.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    def pre_gather_func(self, pre_gather, output, batch_valid_length, gather_index=None):
        """Pre gather operation in infer mode."""
        if not pre_gather:
            return output
        if pre_gather:
            if self.chunk_prefill and self.is_first_iteration:
                output = output.reshape(-1, output.shape[-1])
                output = output[self.sub_batch_valid_len(gather_index, 1)]
            elif self.config.is_dynamic:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = self.prefill_gather_flatten(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
            else:
                output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        return output

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None, q_seq_lens=None,
                  loss_mask=None, gather_index=None, seq_range=None, actual_seq_len=None):
        r"""LlamaForCausalLM forward."""
        has_loss_mask = loss_mask is not None
        input_sliced_sig = self.input_sliced_sig
        if self.training and input_sliced_sig and labels is None:
            input_sliced_sig = False

        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if not input_sliced_sig and self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
            if has_loss_mask:
                loss_mask = self.slice(loss_mask, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))

        output = self.model(tokens, input_embeds, batch_valid_length, batch_index, zactivate_len, block_tables, \
                            slot_mapping, prefix_keys_values, attention_mask, position_ids, q_seq_lens, \
                            seq_range, actual_seq_len)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        output = self.pre_gather_func(pre_gather, output, batch_valid_length, gather_index)
        logits = self.lm_head(output)
        input_mask = loss_mask if has_loss_mask \
            else self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if self.rl_config is not None:
            return logits
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if not input_sliced_sig and self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                if not has_loss_mask:
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

    @classmethod
    def convert_name(cls, weight_name):
        """convert HuggingFace weight name to MindFormers weight name"""
        origin_name = weight_name
        weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
        weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
        weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
        weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('.norm.', '.norm_out.')
        weight_name = weight_name.replace('output.', 'lm_head.')
        weight_name = weight_name.replace('.tok_embeddings.weight', '.tok_embeddings.embedding_weight')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        """convert HuggingFace weight dict to MindFormers weight dict"""
        model_config = kwargs.get("model_config")
        qkv_concat = model_config.qkv_concat
        target_dict = {}
        wq_keys = []
        wk_keys = []
        wv_keys = []
        w1_keys = []
        w3_keys = []

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'wq':
                    wq_keys.append(k)
                if part[-2] == 'wk':
                    wk_keys.append(k)
                if part[-2] == 'wv':
                    wv_keys.append(k)
                if part[-2] == 'w1':
                    w1_keys.append(k)
                if part[-2] == 'w3':
                    w3_keys.append(k)

        if qkv_concat:
            qkv_dict = kwargs.get('qkv_dict', None)
            if not isinstance(qkv_dict, DictProxy):
                raise ValueError(f'qkv_queue must be a queue, when qkv_concat is True, but got {qkv_dict}.')
            condition = kwargs.get('condition', None)
            if not isinstance(condition, Condition):
                raise ValueError(f'condition must be a Condition, when qkv_concat is True, but got {condition}.')
            _concat_qkv_weight(wq_keys, wk_keys, wv_keys, model_config, qkv_dict, condition, target_dict)
            _concat_ffn_weight(w1_keys, w3_keys, model_config, qkv_dict, condition, target_dict)

        return target_dict

    @classmethod
    def convert_map_dict(cls, source_dict, **kwargs):
        """convert HuggingFace map dict to MindFormers map dict"""
        qkv_concat = kwargs.pop("qkv_concat", False)
        target_dict = {}
        wq_keys = []
        w1_keys = []

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'wq':
                    wq_keys.append(k)
                if part[-2] == 'w1':
                    w1_keys.append(k)

        if qkv_concat:
            for wq_key in wq_keys:
                wk_key = wq_key.replace('wq', 'wk')
                wv_key = wq_key.replace('wq', 'wv')
                wq_value = target_dict.pop(wq_key)
                target_dict.pop(wk_key)
                target_dict.pop(wv_key)

                w_qkv_key = wq_key.replace('wq', 'w_qkv')
                w_qkv_value = wq_value
                target_dict.update({w_qkv_key: w_qkv_value})
            for w1_key in w1_keys:
                w3_key = w1_key.replace('w1', 'w3')
                w1_value = target_dict.pop(w1_key)
                target_dict.pop(w3_key)

                w_gate_hidden_key = w1_key.replace('w1', 'w_gate_hidden')
                w_gate_hidden_value = w1_value
                target_dict.update({w_gate_hidden_key: w_gate_hidden_value})

        return target_dict

    @classmethod
    def obtain_qkv_ffn_concat_keys(cls):
        qkv_key = "w_qkv"
        ffn_key = "w_gate_hidden"
        concat_keys = [qkv_key, ffn_key]
        logger.info(f"{cls.__name__} qkv/ffn concat keys are {concat_keys}")
        return concat_keys

    def clear_kv_cache(self):
        return self.model.clear_kv_cache()


def _concat_qkv_weight(wq_keys, wk_keys, wv_keys, model_config, qkv_dict, condition, target_dict):
    """concat qkv weight from dicts"""
    from mindformers.utils.convert_utils import qkv_concat_hf2mg

    num_heads = model_config.num_heads
    n_kv_heads = model_config.n_kv_heads or num_heads
    hidden_size = model_config.hidden_size

    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for wk_key in wk_keys:
        wq_key = wk_key.replace('wk', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wk_key] = target_dict.pop(wk_key)  # add extra weight to shared dict
                condition.notify_all()
    for wv_key in wv_keys:
        wq_key = wv_key.replace('wv', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wv_key] = target_dict.pop(wv_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat qkv
    for wq_key in wq_keys:
        wk_key = wq_key.replace('wq', 'wk')
        wv_key = wq_key.replace('wq', 'wv')
        wq_value = target_dict.pop(wq_key)
        wk_value = target_dict.pop(wk_key, None)
        wv_value = target_dict.pop(wv_key, None)

        # get missing weight from shared dict
        if wk_value is None:
            with condition:
                condition.wait_for(lambda: wk_key in qkv_dict.keys())
                wk_value = qkv_dict.pop(wk_key)
        if wv_value is None:
            with condition:
                condition.wait_for(lambda: wv_key in qkv_dict.keys())
                wv_value = qkv_dict.pop(wv_key)

        w_qkv_key = wq_key.replace('wq', 'w_qkv')
        w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
        # qkv weight format: hf -> mg
        w_qkv_value_mg = qkv_concat_hf2mg(w_qkv_value, num_heads, n_kv_heads, hidden_size)
        target_dict.update({w_qkv_key: w_qkv_value_mg})


def _concat_ffn_weight(w1_keys, w3_keys, model_config, qkv_dict, condition, target_dict):
    """concat ffn weight from dicts"""
    from mindformers.utils.convert_utils import ffn_concat_hf2mg

    intermediate_size = model_config.intermediate_size
    ffn_dim_multiplier = model_config.ffn_dim_multiplier
    multiple_of = model_config.multiple_of or 256
    ffn_hidden_size = model_config.hidden_size * 4
    if intermediate_size is not None:
        ffn_hidden_size = intermediate_size
    else:
        if ffn_dim_multiplier is not None:
            ffn_hidden_size = int((ffn_dim_multiplier + 0.01) * ffn_hidden_size)
        ffn_hidden_size = int(2 * ffn_hidden_size / 3)
        ffn_hidden_size = multiple_of * \
                          ((ffn_hidden_size + multiple_of - 1) // multiple_of)

    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for w3_key in w3_keys:
        w1_key = w3_key.replace('w3', 'w1')
        if w1_key not in w1_keys:
            with condition:
                qkv_dict[w3_key] = target_dict.pop(w3_key)  # add extra weight to shared dict
                condition.notify_all()

    # concat ffn
    for w1_key in w1_keys:
        w3_key = w1_key.replace('w1', 'w3')
        w1_value = target_dict.pop(w1_key)
        w3_value = target_dict.pop(w3_key, None)

        # get missing weight from shared dict
        if w3_value is None:
            with condition:
                condition.wait_for(lambda: w3_key in qkv_dict.keys())
                w3_value = qkv_dict.pop(w3_key)

        w_gate_hidden_key = w1_key.replace('w1', 'w_gate_hidden')
        w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)
        # ffn weight format: hf -> mg
        w_gate_hidden_value_mg = ffn_concat_hf2mg(w_gate_hidden_value, ffn_hidden_size)
        target_dict.update({w_gate_hidden_key: w_gate_hidden_value_mg})
