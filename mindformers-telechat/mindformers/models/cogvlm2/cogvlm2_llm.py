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
# This file was refer to project:
# https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B
# ============================================================================
"""CogVLM2 LLM APIs."""
import copy
import numpy as np

from mindspore import Tensor, nn, Parameter
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear, SeqExtendMethod
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_disable_custom_fa, get_predict_run_mode, get_use_rope_self_define
from mindformers.tools.logger import logger

from ..utils import lazy_inline
from ..llama.llama import LlamaPreTrainedModel
from ..llama.llama_config import LlamaConfig
from ..llama.llama_layer import LlamaEmbedding, LlamaRMSNorm
from ..llama.llama_transformer import LLamaDecodeLayer
from ..llama.llama_interleave import LLamaDecodeLayerInterleave

__all__ = ['CogVLM2VideoLM', 'CogVLM2VideoLMModel']


class FreqsMgr(nn.Cell):
    r"""cogvlm2 freqs manager."""

    def __init__(self,
                 head_dim,
                 seq_length=None,
                 max_position_embedding=4096,
                 rotary_dtype=mstype.float16,
                 theta=10000,
                 scaling_factor=1.0,
                 extend_method=SeqExtendMethod.NONE.value):
        super().__init__()
        if seq_length is not None and seq_length > max_position_embedding:
            max_position_embedding = seq_length
        if extend_method == SeqExtendMethod.NTK.value:
            theta *= scaling_factor
        self.rotary_dtype = rotary_dtype
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (theta ** (freqs_base / head_dim))  # (head_dim // 2, )
        self.freqs = Parameter(Tensor(freqs, self.rotary_dtype), requires_grad=False, name='freqs')
        logger.info("init parameter freqs in FreqsMgr.")
        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, max_position_embedding / scaling_factor, 1 / scaling_factor).astype(np.float32)
        else:
            t = np.arange(0, max_position_embedding, 1).astype(np.float32)
        self.t = Tensor(t, dtype=self.rotary_dtype)
        swap_mask = FreqsMgr.get_swap_mask(head_dim)

        self.head_dim = head_dim
        self.swap_mask = Tensor(swap_mask, dtype=self.rotary_dtype)

        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.tile = P.Tile()
        self.add = P.Add()
        self.concat = P.Concat(axis=-1)
        self.cos = P.Cos()
        self.sin = P.Sin()

    def construct(self, position_ids):
        """Gather freqs_cos and freqs_sin from input position_ids."""
        # prepare freqs
        freqs = ops.outer(self.t, self.freqs)
        emb = self.concat((freqs, freqs))
        freqs_cos = self.cast(self.cos(emb), self.rotary_dtype)
        freqs_sin = self.cast(self.sin(emb), self.rotary_dtype)

        # 1. not use_past, position_ids -> (bs, seq_length)
        # 2. use_past,     position_ids -> (bs, seq_length/1)
        # freqs_cos, freqs_sin          -> (bs, seq_length, head_dim)
        freqs_cos = self.gather(freqs_cos, position_ids, 0)
        freqs_sin = self.gather(freqs_sin, position_ids, 0)
        # freqs_cos, freqs_sin          -> (bs, 1, seq_length, head_dim)
        freqs_cos = self.expand_dims(freqs_cos, 1)
        freqs_sin = self.expand_dims(freqs_sin, 1)
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask(head_dim):
        """Swap matrix."""
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])

    def shard(self):
        self.gather.shard(((1, 1), (1, 1)))
        self.tile.shard(((1, 1),))


class CogVLM2VideoLMModel(LlamaPreTrainedModel):
    """
    Provide CogVLM2VideoLMModel Layers.

    Args:
        config (LlamaConfig): The config of CogVLM2VideoLMModel model.

    Returns:
        output: Tensor, the output of CogVLM2VideoLMModel.
    """

    def __init__(self, config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        self.concat = P.Concat(-1)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        # default open internal kernel boost
        self.disable_custom_fa = get_disable_custom_fa()
        logger.info("disable custom flash attention score op:{}".format(self.disable_custom_fa))
        if config.moe_config.expert_num > 1:
            logger.info("MoE config is provided, use MoE FFN")
        else:
            logger.info("MoE config is None, use normal FFN")
        self.use_rope_self_define = get_use_rope_self_define()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_attn_mask_compression=config.use_attn_mask_compression)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.embedding_init_type,
                                             parallel_optimizer=config.parallel_optimizer)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            layer = self.build_decoderlayer(layer_id, config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
        if cp > 1:
            raise ValueError("CogVLM2 does not support cp > 1.")
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
            self.concat.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if self.fine_grain_interleave:
                self.norm_out.shard((dp * cp, 1))
            else:
                self.norm_out.shard((dp, cp, 1))
            self.freqs_mgr.shard()

            for layer in self.layers:
                if self.use_past:
                    layer.attention.infer_attention.rotary_embedding.mul.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))
                    layer.attention.infer_attention.rotary_embedding.mul_inc.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))
                else:
                    layer.attention.apply_rotary_emb.mul.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))
                    layer.attention.apply_rotary_emb.mul_inc.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    def build_decoderlayer(self, layer_id, config):
        """Build llama decoderlayer."""
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
                                               qkv_concat=config.qkv_concat,
                                               compute_dtype=config.compute_dtype,
                                               layernorm_compute_dtype=config.layernorm_compute_type,
                                               softmax_compute_dtype=config.softmax_compute_type,
                                               rotary_dtype=config.rotary_dtype,
                                               param_init_type=config.param_init_type,
                                               use_flash_attention=config.use_flash_attention,
                                               use_attn_mask_compression=config.use_attn_mask_compression,
                                               fine_grain_interleave=config.fine_grain_interleave,
                                               parallel_config=config.parallel_config)
        else:
            layer = LLamaDecodeLayer(layer_id,
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
                                     use_attn_mask_compression=config.use_attn_mask_compression,
                                     block_size=config.block_size,
                                     num_blocks=config.num_blocks,
                                     is_dynamic=config.is_dynamic,
                                     use_rope_slice=config.use_rope_slice,
                                     moe_config=config.moe_config,
                                     parallel_config=config.parallel_config)
        return layer

    # pylint: disable=W0613
    def construct(self, tokens: Tensor = None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  input_embeds=None, input_attention_masks=None, position_ids=None):
        """CogVLM2VideoLMModel Forward."""
        # preprocess
        if tokens is None and input_embeds is None:
            raise ValueError("tokens and input_embeds should not be None at the same time.")

        if tokens is not None:
            h = self.cast(self.tok_embeddings(tokens), self.dtype)
            input_attention_masks = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        else:
            if input_embeds is None or input_attention_masks is None:
                raise ValueError("input_embeds and input_attention_masks should not be None when tokens is None.")
            h = self.cast(input_embeds, self.dtype)

        bs, seq_len, _ = self.shape(h)
        freqs_cis = self.freqs_mgr(position_ids)
        mask = None
        if self.use_past and self.is_first_iteration:
            if self.use_flash_attention:
                if self.disable_custom_fa:  # only support fp16
                    mask = self.casual_mask(masks=input_attention_masks)  # mask: [bs, seq, seq]
                    mask = self.cast(mask, mstype.float16)
            else:
                mask = self.casual_mask(masks=input_attention_masks)  # mask: [bs, seq, seq]

            if prefix_keys_values is not None:
                if mask is None:
                    mask = self.casual_mask(masks=input_attention_masks)
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))
        elif not self.use_past:
            mask = self.casual_mask(masks=input_attention_masks)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                               slot_mapping=slot_mapping, prefix_keys_values=prefix_kv)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class CogVLM2VideoLM(LlamaPreTrainedModel):
    """
    Provide CogVLM2VideoLM Model.

    Args:
        config (LlamaConfig): The config of CogVLM2VideoLM model.

    Returns:
        output: Tensor, the output of CogVLM2VideoLM.
    """

    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(CogVLM2VideoLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
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
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = CogVLM2VideoLMModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")  # meta default: xavier_normal

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
                                     calculate_per_token_loss=calculate_per_token_loss)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
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

        logger.info(f"Predict kbk mode: {self.predict_run_mode}")

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs = input_ids.shape[0]
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        """Set dynamic inputs for model."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)
        logger.info("Set dynamic input for llama.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    def to_embeddings(self, input_ids):
        """Get token embedding from sub-model."""
        return self.model.tok_embeddings(input_ids)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """Forward of CogVLM2VideoLM."""
        if input_ids is None and input_embeds is None:
            raise ValueError("input_ids and input_embeds should not be None at the same time.")

        if input_ids is not None:
            bsz, seqlen = self.shape(input_ids)
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
        output = self.model(None, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping, prefix_keys_values, input_embeds=input_embeds,
                            input_attention_masks=input_attention_masks, position_ids=position_ids)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = input_attention_masks
        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                return logits
            return logits, input_mask

        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            labels_bsz, labels_seqlen = self.shape(labels)
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (labels_bsz, labels_seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def kvcache(self, layer_idx):
        """Get kvcache with input layer index."""
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
