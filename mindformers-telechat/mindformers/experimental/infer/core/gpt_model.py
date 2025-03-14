# Copyright 2025 Huawei Technologies Co., Ltd
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
"""GPT Transformer language model"""
from typing import Literal, Optional
import numpy as np

from mindspore import nn, Tensor, ops, mint
import mindspore.common.dtype as mstype

from mindformers.modules import Linear
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.core.transformer import VocabEmbedding
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, VocabParallelEmbedding
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec
from mindformers.experimental.infer.core.rotary_embedding import RotaryEmbedding, Llama3RotaryEmbedding
from mindformers.experimental.infer.core.transformer_block import TransformerBlock


class GPTModel(nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            pre_process: bool = True,
            post_process: bool = True,
            position_embedding_type: Literal['rope', 'llama3', 'none'] = 'rope',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            seq_len_interpolation_factor: Optional[float] = None,
    ):
        super(GPTModel, self).__init__()

        self.config = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.position_embedding_type = position_embedding_type
        self.compute_dtype = self.config.compute_dtype

        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_dim = config.hidden_size // config.num_attention_heads
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.tp_group_size = get_tp_world_size()

        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.sub = ops.Sub()
        self.reshape = ops.Reshape()

        self.is_prefill = True

        if self.pre_process:
            if self.config.vocab_emb_dp or self.tp_group_size == 1:
                self.embedding = VocabEmbedding(
                    num_embeddings=self.vocab_size,
                    embedding_dim=self.config.hidden_size,
                    param_init_type=self.config.embedding_init_type,
                    param_init="normal",
                )
            else:
                self.embedding = VocabParallelEmbedding(
                    num_embeddings=self.vocab_size,
                    embedding_dim=self.config.hidden_size,
                    parallel_config=config,
                    init_method="normal",
                    init_type=self.config.embedding_init_type,
                )

        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=self.config.seq_length,
            compute_type=self.config.compute_dtype,
            pad_token_id=self.config.pad_token_id,
        )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.hidden_dim,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rotary_cos_format=2,
                rotary_dtype=self.config.rotary_dtype,
            )
        elif self.position_embedding_type == 'llama3':
            self.rotary_pos_emb = Llama3RotaryEmbedding(
                kv_channels=self.hidden_dim,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=self.config.factor,
                low_freq_factor=self.config.low_freq_factor,
                high_freq_factor=self.config.high_freq_factor,
                orig_max_position=self.config.orig_max_position,
                rotary_cos_format=2,
                rotary_dtype=self.config.rotary_dtype,
            )

        # Transformer
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec
        )

        # Output
        if post_process:
            if self.config.vocab_emb_dp:
                self.output_layer = Linear(
                    in_channels=self.config.hidden_size,
                    out_channels=self.vocab_size,
                    weight_init="normal",
                    has_bias=False,
                    param_init_type=self.config.params_dtype,
                    compute_dtype=self.config.compute_dtype
                )
            else:
                self.output_layer = ColumnParallelLinear(
                    input_size=self.config.hidden_size,
                    output_size=self.vocab_size,
                    config=self.config,
                    bias=False,
                    gather_output=True,
                    param_init_type=self.config.params_dtype,
                    compute_dtype=self.config.compute_dtype,
                )
            if self.config.tie_word_embeddings:
                self.output_layer.weight = self.embedding.embedding_weight

    def pre_gather_func(self, output, context_lens_tensor, seq_lens_tensor):
        """Pre gather operation in infer mode."""
        if self.is_prefill:
            q_seq_lens_tensor = self.sub(seq_lens_tensor, context_lens_tensor)
            gather_index = self.sub(mint.cumsum(q_seq_lens_tensor, 0), 1)
            output = self.gather(output, gather_index, 1)
        return output

    # pylint: disable=W0613
    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None,
                  block_tables=None, slot_mapping=None, kv_cache=None, attention_mask=None, attn_metadata=None):
        """ Construct function of GPTModel. """
        input_ids = self.reshape(input_ids, (1, -1))
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_prefill(self.max_position_embeddings)
        else:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_decode(positions, self.max_position_embeddings)
        if attention_mask is None:
            attention_mask = self.casual_mask(positions)

        hidden_states = self.cast(self.embedding(input_ids), self.compute_dtype)

        hidden_states = self.decoder(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            prefix_keys_values=None,
            kv_cache=kv_cache
        )

        # return hidden_states
        output = self.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)

        logits = self.output_layer(output)
        logits = self.cast(logits.squeeze(0), mstype.float32)
        return logits


class LowerTriangularMaskWithDynamic(nn.Cell):
    """
        Get the Strictly Lower triangular matrix from the input_ids.
    """

    def __init__(self, seq_length, compute_type=mstype.float16, pad_token_id=0):
        super().__init__()
        self.compute_dtype = compute_type
        self.pad_token_id = pad_token_id
        self.seq_length = seq_length
        self.is_prefill = True
        mask_coeff = 1.0 if self.compute_dtype is mstype.bfloat16 else -10000.0
        self.pa_lower_triangle_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length), dtype=np.int8)),
                                             dtype=self.compute_dtype) * -10000
        self.fa_lower_triangle_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff,
                                             dtype=self.compute_dtype)
        self.gather = ops.Gather()

    def construct(self, positions):
        """Forward process of the CausalMask"""
        if self.is_prefill:
            return self.prefill()

        return self.decode(positions)

    def prefill(self):
        return self.fa_lower_triangle_mask

    def decode(self, positions):
        return self.gather(self.pa_lower_triangle_mask, positions, 0)
