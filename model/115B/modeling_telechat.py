# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch TELECHAT model implementation similar to that in transformers."""

from typing import Optional, Tuple, Union, List, Dict
from threading import Thread
import math
import copy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import GenerationConfig

from .configuration_telechat import TelechatConfig
from .generation_utils import History, TelechatIterTextStreamer

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "telechat"
_CONFIG_FOR_DOC = "TelechatConfig"

TELECHAT_PRETRAINED_MODEL_ARCHIVE_LIST = []

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None


class RotaryEmbedding(nn.Module):
    r"""
    Rotary positional embeddings used in the TELECHAT model.

    This implements the rotary position embeddings as described in:
    "Rotary Positional Embeddings" (Su et al. 2021).

    Args:
        dim (`int`): The dimension of the rotary embeddings.
        config (`TelechatConfig`): The model config.
        base (`int`, *optional*, defaults to 10000): Base for computing rotary frequencies.
        precision (`torch.dtype`, *optional*, defaults to `torch.half`): The precision used for computation.

    """
    def __init__(self, dim, config, base=10000, precision=torch.half):
        super().__init__()
        self.config = config
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().half() / dim)).cuda()
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def get_mscale(self, scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_ntk_alpha(self, true_seq_len):
        # Compute ntk_alpha to handle sequence context growth
        context_value = math.log(true_seq_len / self.config.base_seqlen, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        seq_len = max(seq_len, self.config.training_seqlen)
        ntk_alpha = self.get_ntk_alpha(seq_len)
        self.mscale = float(self.get_mscale(seq_len / self.config.training_seqlen))
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        if self.precision == torch.bfloat16:
            emb = emb.float()
        self.cos_cached = self.mscale * emb.cos()[:, None, :].half()
        self.sin_cached = self.mscale * emb.sin()[:, None, :].half()
        if self.precision == torch.bfloat16:
            self.cos_cached = self.cos_cached.bfloat16()
            self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    # Helper function for rotary embeddings
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):
    """
    Apply rotary position embeddings to query and key tensors.
    """
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MixedFusedRMSNorm(nn.Module):
    r"""
    Mixed precision fused RMSNorm layer used by TELECHAT.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FlashSelfAttention(nn.Module):
    r"""
    A wrapper for FlashAttention implementation of scaled dot product attention.
    Requires `flash_attn` and `einops` installed.
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            'Please install FlashAttention first, e.g., with `pip install flash-attn`.'
        )
        assert rearrange is not None, 'Please install einops first, e.g., `pip install einops`.'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """
        Forward pass of flash self-attention.
        """
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # In training, q/k/v have the same length.
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # In inference, we may have causal= True only for the first token step.
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p=dropout_p, softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


def _make_causal_mask(input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int) -> torch.BoolTensor:
    """
    Create a causal mask for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length
    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Apply dropout to tensor x and then add to residual.
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def telechat_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Forward part of the custom GELU used in TELECHAT.
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def telechat_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Backward part of custom GELU approximation.
    """
    x = x[0]
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return telechat_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = telechat_gelu_back(grad_output, input)
        return tmp


class TelechatGelu(nn.Module):
    r"""
    TELECHAT-specific GELU activation function to handle training vs inference modes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return telechat_gelu_forward(x)


class TelechatAttention(nn.Module):
    r"""
    Self-attention module used in TELECHAT model blocks.
    """
    def __init__(self, config: TelechatConfig, layer_idx):
        super().__init__()
        self.kv_cache = None
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.kv_projection_size = self.head_dim * self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_value = nn.Linear(self.hidden_size, self.kv_projection_size * 2, bias=False)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config=config)

        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=config.attention_dropout
        )

    def repeat_kv(self, hidden_states, n_rep):
        # Repeat key/value tensors for multi-group heads
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            slen, batch, num_key_value_heads_per_partition, n_rep, head_dim
        )
        return hidden_states.reshape(slen, batch, num_key_value_heads_per_partition * n_rep, head_dim)

    def split_tensor_along_last_dim(self,
                                    tensor: torch.Tensor,
                                    num_partitions: int,
                                    contiguous_split_chunks: bool = False):
        # Helper function to split tensor
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
        return tensor_list

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Merge attention heads back to [batch, seq_len, hidden_size]
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        # Transpose to shape [seq_len, batch, hidden_size]
        hidden_states = hidden_states.transpose(1, 0)
        query_layer = self.query(hidden_states)
        new_tensor_shape = query_layer.size()[:-1] + (self.num_heads, self.head_dim)
        query_layer = query_layer.view(*new_tensor_shape)

        mixed_kv_layer = self.key_value(hidden_states)
        new_tensor_shape = mixed_kv_layer.size()[:-1] + (self.num_key_value_heads, 2 * self.head_dim)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
        (key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_kv_layer, 2)

        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
            key_layer.size(2),
        )

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[4], -1)

        apply_rotary_fn = apply_rotary_pos_emb_torch

        seq_len = key_layer.shape[0]
        offset = 0

        # Handle past key values if using cache
        if use_cache and layer_past is not None:
            past_key, past_value = layer_past
            offset = past_key.shape[0]
            seq_len += offset

        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        if use_cache:
            if layer_past is not None:
                past_key, past_value = layer_past
                key_layer = torch.cat((past_key, key_layer[-1, ...].unsqueeze(0)), dim=0)
                value_layer = torch.cat((past_value, value_layer[-1, ...].unsqueeze(0)), dim=0)
            layer_past = (key_layer, value_layer)

        s_value, bz, kv_head, dim = value_layer.shape
        s_key = key_layer.shape[0]
        s_query = query_layer.shape[0]
        q_head = output_size[1]

        query_layer = query_layer.reshape((s_query, bz, q_head, dim))
        key_layer = key_layer.reshape((s_key, bz, kv_head, dim))

        key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
        value_layer = self.repeat_kv(value_layer, self.num_key_value_groups)

        if self.config.flash_attn:
            # Use flash attention if configured
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in
                       (query_layer, key_layer, value_layer)]
            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            # Fallback to normal attention if flash_attn is not used
            query_layer = query_layer.reshape(s_query, bz * self.num_heads, dim)
            key_layer = key_layer.reshape(s_key, bz * self.num_heads, dim)
            matmul_result = (1.0 / math.sqrt(self.head_dim)) * torch.einsum(
                'bik,bkj->bij',
                query_layer.transpose(0, 1),
                key_layer.transpose(0, 1).transpose(1, 2)
            )

            attention_scores = matmul_result.view(bz, self.num_heads, s_query, s_key)
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16:
                attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(
                attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min
            )
            attention_probs = F.softmax(attn_weights, dim=-1).to(input_dtype)
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs_reshaped = attention_probs.view(bz * self.num_heads, s_query, s_key)

            value_layer = value_layer.reshape(s_key, bz * self.num_heads, dim)
            context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))
            context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        present = None
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output_tensor, layer_past


class TelechatMLP(nn.Module):
    r"""
    Telechat MLP module applied after attention.
    """
    def __init__(self, config: TelechatConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, hidden_size, bias=True)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Gated feed-forward network with SiLU
        intermediate_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output


class TelechatBlock(nn.Module):
    r"""
    Single Transformer block of TELECHAT model, consisting of:
    - RMSNorm
    - Self-Attention
    - Post-Attention RMSNorm
    - MLP
    """
    def __init__(self, config: TelechatConfig, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = MixedFusedRMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.layer_idx = layer_idx
        self.self_attention = TelechatAttention(config, layer_idx)
        self.post_attention_layernorm = MixedFusedRMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TelechatMLP(config)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        # Layer norm before attention
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self-attention
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # Layer norm after attention
        layernorm_output = self.post_attention_layernorm(attention_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs


class TelechatPreTrainedModel(PreTrainedModel):
    r"""
    Base class for all Telechat models.
    """
    config_class = TelechatConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TelechatBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for linear layers and embeddings."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, TelechatModel):
            module.gradient_checkpointing = value


class TelechatModel(TelechatPreTrainedModel):
    r"""
    The bare Telechat Model outputting raw hidden-states without any specific head.
    """
    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        if self.config.embed_layernorm:
            self.word_embeddings_layernorm = MixedFusedRMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.h = nn.ModuleList([TelechatBlock(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = MixedFusedRMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # Prepare combined attention mask for causal decoding
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        # Main forward pass for TelechatModel
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        if self.config.embed_layernorm:
            hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    causal_mask,
                    layer_past,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TelechatForCausalLM(TelechatPreTrainedModel):
    r"""
    Telechat Model with a language modeling head on top.
    """
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: TelechatConfig):
        super().__init__(config)
        self.transformer = TelechatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # Prepare inputs during generation (like GPT-2/LLM style)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        # Forward pass for causal language modeling
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def chat(self, tokenizer, question: str = '', history: Union[List[Dict], History] = None, stream: bool = False,
             generation_config: Optional[GenerationConfig] = None, **kwargs):
        r"""
        Generate a reply for the given question with a chat-like interface.

        Args:
            tokenizer: A tokenizer compatible with this model.
            question (`str`): The user query.
            history (`List[Dict]` or `History`, *optional*): The conversation history.
            stream (`bool`, *optional*, default=False): Whether to return a streamer for tokens or final output.
            generation_config (`GenerationConfig`, *optional*): Generation parameters for the model.
            **kwargs: Additional generation parameters.

        Returns:
            `str` or `TelechatIterTextStreamer`: The generated answer (or a streamer if `stream=True`).
        """
        generation_config = generation_config or self.generation_config
        if not generation_config:
            logger.error("generation_config is None")
            raise ValueError("generation_config must not be None")
        if not question:
            logger.error("question is empty")
            raise ValueError("question must not be empty")
        if history is None:
            history = []

        generation_config = copy.deepcopy(generation_config)
        user_id = generation_config.user_token_id
        bot_id = generation_config.bot_token_id
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()

        if not isinstance(history, History):
            history = History(tokenizer, history)

        inputs = self.build_inputs_for_chat(tokenizer, question, history, generation_config, user_id, bot_id)
        history.append({"role": "user", "content": question})
        if stream:
            # Return a streamer for iterative token output
            streamer = TelechatIterTextStreamer(tokenizer, history, skip_prompt=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=inputs.to(self.device), streamer=streamer,
                generation_config=generation_config, **model_kwargs
            )).start()
            return streamer
        else:
            # Direct generation and return the final response
            outputs = self.generate(inputs.to(self.device), generation_config=generation_config, **model_kwargs)
            response = tokenizer.decode(outputs[0][len(inputs[0]):-1])
            history.append({"role": "bot", "content": response})
            return response, history

    def build_inputs_for_chat(self, tokenizer, question, history, generation_config, usr_id, bot_id):
        r"""
        Build input tokens for chat-based inference.

        This method reconstructs the user-bot conversation turns from the given `history` and the `question`,
        then truncates according to model maximum context length.

        Args:
            tokenizer: The tokenizer instance.
            question (`str`): The question from the user.
            history: Current conversation history (as a `History` instance or list of dicts).
            generation_config (`GenerationConfig`): Generation config to respect max length constraints.
            usr_id (`int`): The user id token.
            bot_id (`int`): The bot id token.

        Returns:
            `torch.LongTensor`: The input token ids for the model.
        """
        q_token = tokenizer(question)
        qa_history = copy.deepcopy(history)

        model_max_length = self.config.seq_length
        build_max_length = max(0, model_max_length - generation_config.max_new_tokens - 1) \
            if generation_config.max_new_tokens else max(0, generation_config.max_length)
        if build_max_length < 3:
            logger.warning("the model can not meet the requirements of input length, Please check config")
            raise ValueError("")

        # Start building the input tokens from the last user query
        input_tokens = [usr_id] + q_token["input_ids"][-build_max_length + 1:] + [bot_id]
        length = len(input_tokens)

        # Prepend historical turns until max length reached
        while len(qa_history) != 0:
            message = qa_history.pop()
            if message["role"] == "user":
                tokens = [usr_id] + message["input_ids"]
            elif message["role"] == "bot":
                tokens = [bot_id] + message["input_ids"] + [generation_config.eos_token_id]
            else:
                tokens = []
            if len(tokens) + length >= build_max_length:
                break
            else:
                input_tokens = tokens + input_tokens

        # Add BOS token at the start
        input_tokens = [generation_config.bos_token_id] + input_tokens

        return torch.tensor([input_tokens], dtype=torch.int64)
