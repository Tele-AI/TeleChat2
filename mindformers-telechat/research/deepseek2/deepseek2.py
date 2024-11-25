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
"""DeepseekV2 models' APIs."""
import copy
import numpy as np
import mindspore.common.dtype as mstype
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear, FreqsMgr
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import lazy_inline, LayerSetting, check_fine_grain_interleave_valid
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode, get_use_rope_self_define
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm

from research.deepseek2.deepseek2_config import DeepseekV2Config
from research.deepseek2.deepseek2_transformer import DeepSeekV2DecodeLayer


__all__ = ['DeepseekV2ForCausalLM', 'DeepseekV2Model']


class DeepseekV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV2Config
    base_model_prefix = "deepseekv2"


class DeepseekV2Model(DeepseekV2PreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepSeekV2DecoderLayer`]
    Args:
        config(DeepseekV2Config): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of deepseek decoderlayer
    """

    def __init__(self,
                 config: DeepseekV2Config = None):
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
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.max_position_embeddings = config.max_position_embeddings  # used for yarn rotary embedding.

        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_flash_attention = config.use_flash_attention
        # only support flash attention in train and prefill predict process.
        if self.use_past:
            self.use_flash_attention = False
        if self.use_flash_attention:
            logger.info("Enable flash attention.")

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()

        self.use_rope_self_define = get_use_rope_self_define()
        self.freqs_mgr = FreqsMgr(head_dim=self.qk_rope_head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embeddings,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method)

        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention)

        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type,
                                             parallel_optimizer=True)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = DeepSeekV2DecodeLayer(layer_id,
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
                                          use_flash_attention=config.use_flash_attention,
                                          block_size=config.block_size,
                                          num_blocks=config.num_blocks,
                                          is_dynamic=config.is_dynamic,
                                          use_rope_slice=config.use_rope_slice,
                                          parallel_config=config.parallel_config,
                                          moe_config=config.moe_config,
                                          kv_lora_rank=config.kv_lora_rank,
                                          q_lora_rank=config.q_lora_rank,
                                          qk_rope_head_dim=config.qk_rope_head_dim,
                                          v_head_dim=config.v_head_dim,
                                          qk_nope_head_dim=config.qk_nope_head_dim,
                                          max_position_embeddings=config.max_position_embeddings,
                                          scaling_factor=config.scaling_factor)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type,
                                     fused_kernel=not get_predict_run_mode())

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
                  block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  extra_loss=None):
        """
        Forward of deepseekv2 model.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.

        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = None
        if self.use_past:
            if self.is_first_iteration:
                if self.use_rope_self_define:
                    freqs_cis = self.freqs_mgr(seq_len)
                else:
                    freqs_cis = self.freqs_mgr.prefill(bs, seq_len)

                if self.use_flash_attention:
                    if self.disable_custom_fa:  # only support fp16
                        mask = self.casual_mask(tokens)  # mask: [bs, seq, seq]
                        mask = self.cast(mask, mstype.float16)
                else:
                    mask = self.casual_mask(tokens)  # mask: [bs, seq, seq]

                if prefix_keys_values is not None:
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        h = self.cast(self.tok_embeddings(tokens), self.dtype)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))

        #h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h, extra_loss = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                           block_tables=block_tables, slot_mapping=slot_mapping,
                                           prefix_keys_values=prefix_kv, extra_loss=extra_loss)
        output = self.norm_out(h)
        return output, extra_loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    r"""
    Provide DeepseekV2 training loss or logits through network.
    Args:
        config (DeepseekV2Config): The config of DeepseekV2 model.

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
    def __init__(self, config: DeepseekV2Config = None):
        super(DeepseekV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.init_extra_loss = Tensor([0], mstype.float32)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = DeepseekV2Model(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              skip_redistribution=config.is_dynamic,
                              weight_init="normal")

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()
        logger.info("Predict run mode:{}".format(self.predict_run_mode))

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor.from_numpy(input_ids.astype(np.int32))
        }

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get deepseekv2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        """Mindspore's feature, Set dynamic input for DeepSeekV2."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_input_position = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_init_reset = Tensor([False], mstype.bool_)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)
        logger.info("Set dynamic input for DeepSeekV2.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model when the use_past is enabled."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """DeepseekV2ForCausalLM forward.
        """
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
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output, extra_loss = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables,
                                        slot_mapping, prefix_keys_values, self.init_extra_loss)
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
        loss = self.loss(logits, labels, input_mask) + extra_loss
        return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
