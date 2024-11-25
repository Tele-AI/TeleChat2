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

import mindspore as ms
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear, RotaryEmbedding
from mindformers.modules.transformer import (
    LowerTriangularMaskWithDynamic,
    TransformerOpParallelConfig,
)
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    get_disable_custom_fa,
    get_predict_run_mode,
    get_use_rope_self_define,
)

from .cogvlm2_llm import FreqsMgr
from ..llama.llama import LlamaPreTrainedModel
from ..llama.llama_config import LlamaConfig
from ..llama.llama_layer import LlamaEmbedding, LlamaRMSNorm, LlamaFeedForward


class VisionExpertAttention(nn.Cell):
    """ Vision Expert Attention """
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            num_multi_query_heads,
            compute_dtype,
            rotary_dtype,
            softmax_compute_type,
            param_init_type,
            use_flash_attention=False,
            block_size=None,
            num_blocks=None,
            use_past=False,
            use_attn_mask_compression=False,
            parallel_config=TransformerOpParallelConfig(),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_dim = self.head_dim * self.num_multi_query_heads
        self.stride = [
            self.num_attention_heads,
            self.num_multi_query_heads,
            self.num_multi_query_heads,
        ]
        self.qkv_size = (
            self.hidden_size + self.kv_dim * 2
        )
        self.scaling = self.head_dim**-0.5
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_past = use_past
        self.is_first_iteration = True

        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.compute_dtype = compute_dtype
        self.rotary_dtype = rotary_dtype
        self.softmax_dtype = softmax_compute_type

        self.apply_rotary_emb = RotaryEmbedding(self.head_dim, self.compute_dtype)
        self.vision_expert_query_key_value = Linear(
            self.hidden_size,
            self.qkv_size,
            has_bias=True,
            compute_dtype=self.compute_dtype,
            param_init_type=param_init_type,
        )
        self.vision_expert_dense = Linear(
            self.hidden_size,
            self.hidden_size,
            has_bias=False,
            compute_dtype=self.compute_dtype,
            param_init_type=param_init_type,
        )
        self.language_expert_query_key_value = Linear(
            self.hidden_size,
            self.qkv_size,
            has_bias=False,
            compute_dtype=self.compute_dtype,
            param_init_type=param_init_type,
        )
        self.language_expert_dense = Linear(
            self.hidden_size,
            self.hidden_size,
            has_bias=False,
            compute_dtype=self.compute_dtype,
            param_init_type=param_init_type,
        )

        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if self.use_past:
            self.infer_attention = InferAttention(
                self.num_attention_heads,
                self.head_dim,
                self.num_multi_query_heads,
                pa_n_head_split=self.num_attention_heads // mp,
                pa_n_kv_head_split=self.num_multi_query_heads // mp,
                scale_value=self.scaling,
                pre_tokens=2147483647,
                next_tokens=0,
                block_size=self.block_size,
                num_blocks=self.num_blocks,
                use_flash_attention=self.use_flash_attention,
                rotary_cos_format=2,
                rotary_dtype=rotary_dtype,
                compute_dtype=compute_dtype,
            )
            self.infer_attention.shard(parallel_config)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.stride_slice = P.StridedSlice()
        self.get_type = P.DType()

        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul()
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.cast_attn = P.Cast()
        self.merger_head_transpose = P.Transpose()

        self.update = P.TensorScatterUpdate()
        self.expand_dims = P.ExpandDims()
        self.tile_kv = P.Tile()
        self.split_qkv = ms.ops.auto_generate.SplitWithSize()

        if self.use_flash_attention:
            self.input_layout = "BSH" if cp > 1 else "BNSD"
            self.sparse_mode = 2 if self.use_attn_mask_compression else 0
            self.flash_attention = FlashAttention(head_num=self.num_attention_heads,
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  input_layout=self.input_layout,
                                                  keep_prob=1.,
                                                  scale_value=self.scaling,
                                                  sparse_mode=self.sparse_mode,
                                                  use_attention_mask=True)
            self.flash_attention.shard(parallel_config)
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(parallel_config)

    def shard(self, parallel_config):
        """set parallel config to ops"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.vision_expert_query_key_value.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
        self.language_expert_query_key_value.shard(((dp, 1), (mp, 1)))
        self.stride_slice.shard(((dp, 1, mp),))
        self.apply_rotary_emb.shard(parallel_config)
        self.tile_kv.shard(((dp, mp, 1, 1),))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        self.softmax.shard(((dp, mp, 1, 1),))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.vision_expert_dense.shard(((dp, mp), (1, mp)))
        self.language_expert_dense.shard(((dp, mp), (1, mp)))
        self.split_qkv.shard(((1, 1, 1),))
        if self.use_past:
            self.infer_attention.rotary_embedding.mul.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))
        else:
            self.apply_rotary_emb.mul.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    def construct(
            self,
            hidden_states,
            freqs_cis,
            vision_token_mask,
            language_token_mask,
            vision_indices,
            language_indices,
            attention_mask=None,
            batch_valid_length=None,
            block_tables=None,
            slot_mapping=None,
    ):
        """forward"""
        bsz, q_len, _ = self.shape(hidden_states)

        shape = (bsz, q_len, self.qkv_size)

        if (self.use_past and not self.is_first_iteration) or vision_token_mask is None:
            mixed_raw_layer = self.language_expert_query_key_value(hidden_states)
        else:
            mixed_raw_layer = self.zeros(shape, self.get_type(hidden_states))
            vision = hidden_states[vision_token_mask]
            vision = self.reshape(self.vision_expert_query_key_value(vision), (bsz, -1, self.qkv_size))
            mixed_raw_layer = self.update(mixed_raw_layer, vision_indices, vision)
            language = hidden_states[language_token_mask]
            language = self.reshape(self.language_expert_query_key_value(language), (bsz, -1, self.qkv_size))
            mixed_raw_layer = self.update(mixed_raw_layer, language_indices, language)

        query_states, key_states, value_states = self.split_qkv(
            mixed_raw_layer, (self.hidden_size, self.kv_dim, self.kv_dim), 2)

        if self.use_past:
            context_layer = self.infer_attention(
                query_states,
                key_states,
                value_states,
                batch_valid_length,
                block_tables,
                slot_mapping,
                freqs_cis,
                attention_mask,
            )
        else:
            query_states = self._transpose_for_scores(query_states, self.num_attention_heads)  # (B, heads, S, Hhead)
            key_states = self._transpose_for_scores(key_states, self.num_multi_query_heads)  # (B, mqh, S, Hhead)
            value_states = self._transpose_for_scores(value_states, self.num_multi_query_heads)  # (B, mqh, S, Hhead)

            query_states, key_states = self.apply_rotary_emb(query_states, key_states, freqs_cis)
            if self.use_flash_attention:
                context_layer = self.flash_attention(query_states, key_states, value_states, attention_mask)
                context_layer = self._merge_heads(context_layer)
            else:
                key_states = self.broadcast_to_query_shape(key_states)
                value_states = self.broadcast_to_query_shape(value_states)
                context_layer = self._attn(query_states, key_states, value_states, attention_mask)

        shape = self.shape(context_layer)
        if (self.use_past and not self.is_first_iteration) or vision_token_mask is None:
            attn_output = self.language_expert_dense(context_layer)
        else:
            attn_output = self.zeros(shape, self.get_type(context_layer))
            vision = context_layer[vision_token_mask]
            vision = self.reshape(self.vision_expert_dense(vision), (bsz, -1, self.hidden_size))
            attn_output = self.update(attn_output, vision_indices, vision)
            language = context_layer[language_token_mask]
            language = self.reshape(self.language_expert_dense(language), (bsz, -1, self.hidden_size))
            attn_output = self.update(attn_output, language_indices, language)

        return attn_output

    def _transpose_for_scores(self, tensor, num_heads):
        bsz, q_len, _ = self.shape(tensor)
        shape = (bsz, q_len, num_heads, self.head_dim)
        return self.transpose(self.reshape(tensor, shape), (0, 2, 1, 3))

    def broadcast_to_query_shape(self, tensor):
        """broadcast_to_query_shape"""
        rep = self.num_attention_heads // self.num_multi_query_heads
        bs, n_kv_head, seqlen, head_dim = self.shape(tensor)
        x = self.reshape(tensor, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 3d output

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
        new_shape = (bs, seq_len, n_head * head_dim)  # dp,1,mp,1 -> dp,1,mp ?
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
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
        query = self.mul(query, self.scaling)
        score = self.batch_matmul_q_k(query, key)

        # score: [bs, n_head, seq/1, seq]
        if mask is not None:
            score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.compute_dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class VisionExpertMLP(nn.Cell):
    """Vision Expert MLP"""
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            compute_dtype,
            param_init_type,
            use_past=False,
            parallel_config=TransformerOpParallelConfig()
    ):
        super().__init__()
        self.use_past = use_past
        self.hidden_size = hidden_size
        self.is_first_iteration = True
        self.language_mlp = LlamaFeedForward(
            hidden_size,
            intermediate_size=intermediate_size,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            parallel_config=parallel_config
        )
        self.vision_mlp = LlamaFeedForward(
            hidden_size,
            intermediate_size=intermediate_size,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type,
            parallel_config=parallel_config
        )

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.zeros_like = P.ZerosLike()

        self.update = P.TensorScatterUpdate()
        self.expand_dims = P.ExpandDims()

    def construct(
            self,
            hidden_states,
            vision_token_mask,
            language_token_mask,
            vision_indices,
            language_indices,
    ):
        """forward"""
        bsz, _, _ = self.shape(hidden_states)
        if (self.use_past and not self.is_first_iteration) or vision_token_mask is None:
            output = self.language_mlp(hidden_states)
        else:
            output = self.zeros_like(hidden_states)
            vision = hidden_states[vision_token_mask]
            vision = self.reshape(self.vision_mlp(vision), (bsz, -1, self.hidden_size))
            output = self.update(output, vision_indices, vision)
            language = hidden_states[language_token_mask]
            language = self.reshape(self.language_mlp(language), (bsz, -1, self.hidden_size))
            output = self.update(output, language_indices, language)

        return output


class CogVLMDecoderLayer(nn.Cell):
    """CogVLMDecoderLayer"""
    def __init__(
            self,
            layer_id,
            hidden_size,
            num_attention_heads,
            num_multi_query_heads,
            intermediate_size,
            rms_norm_eps,
            compute_dtype,
            rotary_dtype,
            softmax_compute_type,
            param_init_type,
            layernorm_compute_dtype,
            use_flash_attention=False,
            block_size=None,
            num_blocks=None,
            use_past=False,
            parallel_config=TransformerOpParallelConfig(),
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.self_attn = VisionExpertAttention(
            hidden_size,
            num_attention_heads,
            num_multi_query_heads,
            compute_dtype,
            rotary_dtype,
            softmax_compute_type,
            param_init_type,
            use_flash_attention=use_flash_attention,
            block_size=block_size,
            num_blocks=num_blocks,
            use_past=use_past,
            parallel_config=parallel_config,
        )
        self.mlp = VisionExpertMLP(
            hidden_size,
            intermediate_size,
            compute_dtype,
            param_init_type,
            use_past=use_past,
        )
        self.input_layernorm = LlamaRMSNorm(
            hidden_size, eps=rms_norm_eps, compute_type=layernorm_compute_dtype
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, eps=rms_norm_eps, compute_type=layernorm_compute_dtype
        )

        self.add = P.Add()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            dp = parallel_config.data_parallel
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.input_layernorm.shard((dp, 1, 1))
            self.post_attention_layernorm.shard((dp, 1, 1))

    def construct(
            self,
            hidden_states,
            freqs_cis,
            vision_token_mask,
            language_token_mask,
            vision_indices,
            language_indices,
            attention_mask=None,
            batch_valid_length=None,
            block_tables=None,
            slot_mapping=None,
    ):
        """forward"""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            freqs_cis,
            vision_token_mask,
            language_token_mask,
            vision_indices,
            language_indices,
            attention_mask=attention_mask,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )
        hidden_states = self.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            vision_token_mask,
            language_token_mask,
            vision_indices,
            language_indices,
        )
        hidden_states = self.add(residual, hidden_states)

        return hidden_states


class LlamaModelForCogVLM2Image(LlamaPreTrainedModel):
    """
    Provide LlamaModelForCogVLM2Image Layers.

    Args:
        config (LlamaConfig): The config of LlamaModelForCogVLM2Image model.

    Returns:
        output: Tensor, the output of llama decoderlayer.
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
        logger.info("disable custom flash attention core op:{}".format(self.disable_custom_fa))
        if config.moe_config.expert_num > 1:
            logger.info("MoE config is provided, use MoE FFN")
        else:
            logger.info("MoE config is None, use normal FFN")
        self.use_rope_self_define = get_use_rope_self_define()

        self.freqs_mgr = FreqsMgr(
            head_dim=self.head_dim,
            seq_length=config.seq_length,
            max_position_embedding=config.max_position_embedding,
            rotary_dtype=config.rotary_dtype,
            theta=config.theta,
            scaling_factor=config.scaling_factor,
            extend_method=config.extend_method,
        )
        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
            use_attn_mask_compression=config.use_attn_mask_compression,
        )
        self.tok_embeddings = LlamaEmbedding(
            vocab_table_size=config.vocab_size,
            embedding_size=config.hidden_size,
            param_init_type=config.embedding_init_type,
            parallel_optimizer=config.parallel_optimizer,
        )
        self.fine_grain_interleave = check_fine_grain_interleave_valid(
            config.fine_grain_interleave, config.parallel_config
        )
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(
            config.num_layers,
            config.offset,
            config.parallel_config,
            config.pp_interleave_num,
        )
        for layer_id in range(config.num_layers):
            layer = self.build_decoderlayer(layer_id, config)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            compute_type=config.layernorm_compute_type,
        )
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(
                    config.parallel_config.gradient_aggregation_group
                )
                self.norm_out.set_comm_fusion(
                    config.parallel_config.gradient_aggregation_group
                )

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.concat.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if self.fine_grain_interleave:
                self.norm_out.shard((dp * cp, 1))
            else:
                self.norm_out.shard((dp, cp, 1))
            self.freqs_mgr.shard()

    def build_decoderlayer(self, layer_id, config):
        """Build llama decoderlayer."""
        if self.fine_grain_interleave:
            raise NotImplementedError("fine_grain_interleave is not supported.")
        layer = CogVLMDecoderLayer(
            layer_id,
            config.hidden_size,
            config.num_heads,
            config.n_kv_heads,
            config.intermediate_size,
            config.rms_norm_eps,
            config.compute_dtype,
            config.rotary_dtype,
            config.softmax_compute_type,
            config.param_init_type,
            config.layernorm_compute_type,
            use_flash_attention=config.use_flash_attention,
            block_size=config.block_size,
            num_blocks=config.num_blocks,
            use_past=config.use_past,
            parallel_config=config.parallel_config,
        )
        return layer

    # pylint: disable=W0613
    def construct(
            self,
            tokens: Tensor = None,
            batch_valid_length=None,
            batch_index=None,
            zactivate_len=None,
            block_tables=None,
            slot_mapping=None,
            prefix_keys_values=None,
            input_embeds=None,
            input_attention_masks=None,
            position_ids=None,
            vision_token_mask=None,
            language_token_mask=None,
            vision_indices=None,
            language_indices=None,
    ):
        """Forward of LlamaModelForCogVLM2Video model."""
        # preprocess
        if tokens is None and input_embeds is None:
            raise ValueError("tokens and input_embeds should not be None at the same time.")

        if tokens is not None:
            h = self.cast(self.tok_embeddings(tokens), self.dtype)
            input_attention_masks = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        else:
            if input_embeds is None or input_attention_masks is None:
                raise ValueError(
                    "input_embeds and input_attention_masks should not be None when tokens is None."
                )
            h = self.cast(input_embeds, self.dtype)
        freqs_cis = self.freqs_mgr(position_ids)

        bs, seq_len, _ = self.shape(h)

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
            h = self.layers[i](
                h,
                freqs_cis,
                vision_token_mask,
                language_token_mask,
                vision_indices,
                language_indices,
                attention_mask=mask,
                batch_valid_length=batch_valid_length,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
            )
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForCausalLMForCogVLM2Image(LlamaPreTrainedModel):
    """
    Provide LlamaForCogVLM2Video Model.

    Args:
        config (LlamaConfig): The config of LlamaForCogVLM2Video model.

    Returns:
        output: Tensor, the output of LlamaModelForCogVLM2Video.
    """

    def __init__(self, config: LlamaConfig = None):
        super(LlamaForCausalLMForCogVLM2Image, self).__init__(config, auto_prefix=True)
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
        self.model = LlamaModelForCogVLM2Image(config=config)
        self.lm_head = Linear(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            has_bias=False,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type,
            weight_init="normal",
        )  # meta default: xavier_normal

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning(
                "The vocab size of Loss is: %s, it is not divide by model_parallel: %s", vocab_size, mp)
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

        logger.info("Predict run mode:{}".format(self.predict_run_mode))

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.add_flags(is_first_iteration=is_first_iteration)
            layer.mlp.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.add_flags(
                is_first_iteration=is_first_iteration
            )

    def to_embeddings(self, input_ids):
        """Get token embedding from sub-model."""
        return self.model.tok_embeddings(input_ids)

    # pylint: disable=W0613
    def construct(
            self,
            input_ids=None,
            labels=None,
            input_position=None,
            position_ids=None,
            attention_mask=None,
            input_embeds=None,
            init_reset=True,
            batch_valid_length=None,
            batch_index=None,
            zactivate_len=None,
            block_tables=None,
            slot_mapping=None,
            prefix_keys_values=None,
            vision_token_mask=None,
            language_token_mask=None,
            vision_indices=None,
            language_indices=None,
    ):
        """LlamaForCogVLM2Video forward."""
        if input_ids is None and input_embeds is None:
            raise ValueError(
                "input_ids and input_embeds should not be None at the same time."
            )

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

        if self.use_past and not isinstance(batch_valid_length, Tensor):
            batch_valid_length = self.ones((bsz,), mstype.int32)

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        output = self.model(
            None,
            batch_valid_length,
            batch_index,
            zactivate_len,
            block_tables,
            slot_mapping,
            prefix_keys_values,
            input_embeds=input_embeds,
            input_attention_masks=input_attention_masks,
            position_ids=position_ids,
            vision_token_mask=vision_token_mask,
            language_token_mask=language_token_mask,
            vision_indices=vision_indices,
            language_indices=language_indices,
        )
        pre_gather = (
            not self.use_past or self.is_first_iteration
        ) and batch_valid_length is not None
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
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
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
        key_cache = self.model.layers[
            layer_idx
        ].self_attn.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[
            layer_idx
        ].self_attn.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
