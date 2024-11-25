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
"""For transformer"""

__all__ = [
    "get_attention_mask",
    "ParallelMLP",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformer",
    "ParallelLMLogits",
]

import math
import copy
from collections import OrderedDict
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn, ops, mint, Parameter
import mindspore.ops.functional as F

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_data_parallel_world_size,
    get_context_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import get_rng_tracer
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    CopyToModelParallelRegion,
    GatherFromModelParallelRegion,
    LinearWithGradAccumulationAndAsyncCommunication
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.lora_layers import (
    ColumnParallelLoRA,
    RowParallelLoRA
)
from mindformers.experimental.parallel_core.pynative.transformer.utils import get_attn_mask_func
from mindformers.experimental.parallel_core.pynative.transformer.norm import get_norm
from mindformers.experimental.parallel_core.pynative.transformer.moe.moe_layer import MoELayer

from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from mindformers.experimental.parallel_core.pynative.transformer.scale_mask_softmax import (
    ScaleMaskSoftmax,
)
from mindformers.experimental.parallel_core.pynative.transformer.enums import (
    AttnType, AttnMaskType, LayerType, ModelType
)
from mindformers.experimental.parallel_core.pynative.context_parallel.ring_attention import (
    RingAttention,
)
from mindformers.experimental.parallel_core.pynative.context_parallel.flash_sp import FlashSP

from mindformers.experimental.parallel_core.pynative.recompute import (
    CheckpointedRecomputeOrientedCell,
)

from mindformers.experimental.parallel_core.pynative.utils import divide
from mindformers.tools import logger

from .module import Module
from .mlp import ParallelMLP


def get_attention_mask(input_mask, data_layout='BSH'):
    """get attention mask base on input_mask"""
    input_shape = input_mask.shape
    if data_layout == 'BSH':
        batch_size = input_shape[0]
        seq_length = input_shape[1]
    elif data_layout == 'SBH':
        batch_size = input_shape[1]
        seq_length = input_shape[0]
    else:
        raise TypeError(f"Do not support data layout: {data_layout}")
    ones = mint.ones((seq_length, seq_length), dtype=input_mask.dtype)
    attention_mask_left = input_mask.reshape((batch_size, seq_length, 1))
    attention_mask_right = input_mask.reshape((batch_size, 1, seq_length))
    attention_mask = mint.matmul(attention_mask_left, attention_mask_right)
    lower_triangle_mask = ops.tril(ones).unsqueeze(0)
    attention_mask = mint.mul(attention_mask, lower_triangle_mask)
    return attention_mask


def _merge_heads(x):
    """Merge attention heads."""
    # [B, N, S, D] -> [B, S, N, D]
    x = x.transpose(0, 2, 1, 3)
    bs, seq_len, num_heads, head_dim = x.shape
    # [B, S, N, D] -> [B, S ,H]
    merged_shape = (bs, seq_len, num_heads * head_dim)
    x_merged = x.reshape(merged_shape)
    return x_merged


class CoreAttention(nn.Cell):
    r"""
    This class can get the weighted score along the seq_length. It is used within a larger framework to compute the
    context layer based on query, key, and value layers and an attention mask.

    Args:
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        config (dict): A configuration dictionary that provides various settings for the attention mechanism.
        attn_mask_type (int): Attention mask type. Support [AttnMaskType::padding = 1, AttnMaskType::causal = 2].
            Default: ``1``.

    Inputs:
        - **query_layer** (Tensor) - The shape of query Tensor is :math:`(B, N, S, D)`.
        - **key_layer** (Tensor) - The shape of key Tensor is :math:`(B, N, S, D)`.
        - **value_layer** (Tensor) - The shape of value Tensor is :math:`(B, N, S, D)`.
        - **attention_mask** (Tensor) - The shape of attention mask Tensor is :math:`(B, N, S_q, S_k)'.

    Outputs:
        - **context_layer** (Tensor) - The shape of context_layer Tensor is :math:`(B, S, H)`.

    Raises:
        NotImplementedError: If 'masked_softmax_fusion' in config is true.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.transformer import CoreAttention
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> class MyNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(MyNet, self).__init__()
        ...         self.core_attn = CoreAttention(layer_number=1, config=config)
        ...     def construct(self, x, mask):
        ...         out = self.core_attn(x, x, x, mask)
        ...         return out
        ...
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=2)
        >>> config = TransformerConfig(vocab_size=1,
        >>>                            num_layers=1,
        >>>                            num_attention_heads=8,
        >>>                            num_query_groups=4,
        >>>                            hidden_size=256,
        >>>                            ffn_hidden_size=256,
        >>>                            parallel_config=parallel_config)
        >>> input_shape = (1024, 32, 4, 256)
        >>> input = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> mask = np.ones((32, 1024, 1024), dtype=np.uint8)
        >>> mask = Tensor(np.expand_dims(mask, axis=1))
        >>> out = MyNet(config=config)(input, mask)
    """

    def __init__(self, layer_number, config, attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        self.config = config
        self.layer_index = max(1, layer_number)
        self.param_init_dtype = self.config.params_dtype
        self.attention_softmax_in_fp32 = self.config.attention_softmax_in_fp32
        self.softmax_compute_dtype = self.config.softmax_compute_dtype
        self.sequence_parallel = self.config.parallel_config.sequence_parallel
        self.attn_mask_type = attn_mask_type
        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling
        self.head_dim = self.config.kv_channels
        projection_size = self.config.kv_channels * self.config.num_attention_heads
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size,
                                                world_size)

        if self.config.masked_softmax_fusion:
            raise NotImplementedError(
                "`masked_softmax_fusion` is not supported for now."
            )

        coeff = None
        norm_factor = math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
            coeff = self.layer_index
            norm_factor *= coeff
        self.norm_factor = norm_factor

        self.mask_func = get_attn_mask_func(self.config.mask_func_type)
        self.scale_mask_softmax = ScaleMaskSoftmax(
            self.mask_func, softmax_compute_type=mstype.float32 \
                if self.attention_softmax_in_fp32 else self.softmax_compute_dtype
        )

        self.attention_dropout = mint.nn.Dropout(p=self.config.attention_dropout)

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """ CoreAttention forward."""
        # [b, np, sq, sk]
        output_size = (query_layer.shape[1],
                       query_layer.shape[2],
                       query_layer.shape[0],
                       key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2],
                                          output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        matmul_input_buffer = mint.zeros((output_size[0] * output_size[1], output_size[2], output_size[3]),
                                         dtype=query_layer.dtype)

        matmul_result = mint.baddbmm(matmul_input_buffer,
                                     query_layer.swapaxes(0, 1),
                                     key_layer.swapaxes(0, 1).swapaxes(1, 2),
                                     beta=0.0, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        if not self.sequence_parallel:
            with get_rng_tracer().rng_fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1],
                       value_layer.shape[2],
                       query_layer.shape[0],
                       value_layer.shape[3])

        # change view [b * np, sq, sk]
        value_layer = value_layer.view(value_layer.shape[0],
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = ops.bmm(attention_probs.astype(value_layer.dtype),
                                value_layer.swapaxes(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        # contiguous()
        context_layer = context_layer.permute(2, 0, 1, 3)

        new_context_layer_shape = context_layer.shape[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class ParallelAttention(Module):
    r"""
    This class represents a parallel attention mechanism. It can handle different attention types and is configurable
    with various parameters.

    Args:
        config (dict): Configuration dictionary for the parallel attention.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        attention_type (int): Attention type. Support [AttnType::self_attn = 1, AttnType::cross_attn = 2].
            Default: ``1``.
        attn_mask_type (int): Attention mask type. Support [AttnMaskType::padding = 1, AttnMaskType::causal = 2].
            Default: ``1``.

    Inputs:
        - **hidden_states** (Tensor) - The shape of hidden_states Tensor is :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - The shape of attention_mask Tensor is :math:`(B, N, S_q, S_k)'.
        - **encoder_output** (Tensor) - Tensor of encoder output used for cross attention. Default: None.
        - **inference_params** (Tensor) - Tensor of inference params. Currently not supported. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - The shape of output Tensor is :math:`(B, S, H)`.
        - **bias** (Tensor) - The trainable bias parameter.

    Raises:
        ValueError: If 'attention_type' is neither 'AttnType::self_attn' nor 'AttnType::cross_attn'.
        NotImplementedError: If 'attention_type' is 2 and 'group_query_attention' in config is true.
        ValueError: If 'hidden_size' is not equal to 'kv_hidden_size' and 'attention_type' is 2.
        NotImplementedError: If 'inference_params' is not none.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.transformer import ParallelAttention
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> class MyNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(MyNet, self).__init__()
        ...         self.attention = ParallelAttention(layer_number=1, config=config)
        ...     def construct(self, x, attention_mask):
        ...         output, _ = self.attention(x, attention_mask)
        ...         return output
        ...
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=2)
        >>> config = TransformerConfig(vocab_size=1,
        >>>                            num_layers=1,
        >>>                            num_attention_heads=8,
        >>>                            num_query_groups=4,
        >>>                            hidden_size=256,
        >>>                            ffn_hidden_size=256,
        >>>                            parallel_config=parallel_config)
        >>> input_shape = (32, 1024, 256)
        >>> input = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> mask = np.ones((32, 1024, 1024), dtype=np.uint8)
        >>> mask = Tensor(np.expand_dims(mask, axis=1))
        >>> out = MyNet(config=config)(input, mask)
        >>> print(out.shape)
        (32, 1024, 256)
    """

    def __init__(self, config, layer_number, attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        self.config = config
        self.layer_index = max(1, layer_number)
        self.param_init_dtype = self.config.params_dtype
        self.compute_dtype = self.config.compute_dtype

        self.attn_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.group_query_attention = self.config.group_query_attention
        self.num_heads = self.config.num_attention_heads
        self.num_query_groups = self.config.num_query_groups if self.group_query_attention else self.num_heads
        self.hidden_size = self.config.hidden_size
        self.kv_channels = self.config.kv_channels
        self.kv_hidden_size = self.kv_channels * self.num_query_groups

        self.sequence_parallel = self.config.parallel_config.sequence_parallel
        self.use_flash_attention = self.config.use_flash_attention and attention_type == AttnType.self_attn \
            and self.attn_mask_type == AttnMaskType.causal
        if self.use_flash_attention:
            if self.attn_type != AttnType.self_attn:
                raise NotImplementedError(
                    'FlashAttention code path only supports self-attention for now.'
                )
            self.fa_config = self.config.fa_config
        self.enable_flash_sp = self.config.enable_flash_sp
        self.norm_factor = math.sqrt(self.kv_channels)

        tp_group_size = get_tensor_model_parallel_world_size()
        self.tp_group_size = tp_group_size
        self.num_heads_per_partition = divide(self.num_heads, tp_group_size)
        self.use_lora = self.config.lora_config.use_lora
        self.data_layout = self.config.dataset_config.data_layout

        if self.group_query_attention:
            if self.num_query_groups % tp_group_size != 0:
                raise ValueError(
                    "The num_query_groups should be "
                    "a multiple of the tensor parallel size"
                )
            self.kv_num_heads_per_partition = divide(self.num_query_groups, tp_group_size)
        else:
            self.kv_num_heads_per_partition = self.num_heads_per_partition

        if self.attn_type == AttnType.self_attn:
            self.qkv_proj = self._init_qkv_proj(
                self.hidden_size,
                self.hidden_size + 2 * self.kv_hidden_size,
                cell_name="qkv_proj",
            )

        else:
            if self.attn_type != AttnType.cross_attn:
                raise ValueError(
                    "attention_type should be self_attn or cross_attn"
                )
            if self.group_query_attention:
                raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
            if self.hidden_size != self.kv_hidden_size:
                raise ValueError("hidden_size should equal to kv_hidden_size when using cross_attn.")
            self.q_proj = self._init_qkv_proj(
                self.hidden_size,
                self.hidden_size,
                cell_name="q_proj",
            )
            self.kv_proj = self._init_qkv_proj(
                self.hidden_size,
                2 * self.kv_hidden_size,
                cell_name="kv_proj",
            )

        if get_context_parallel_world_size() > 1:
            if not self.enable_flash_sp:
                self.ring_attention = RingAttention(
                    self.num_heads,
                    input_layout="BNSD",
                    scale_value=1 / self.norm_factor,
                    sparse_mode=0
                )
            else:
                self.flash_sp = FlashSP(
                    self.num_heads,
                    input_layout="BSH",
                    scale_value=1 / self.norm_factor,
                    dp=get_data_parallel_world_size(),
                    mp=get_tensor_model_parallel_world_size(),
                    sp=get_context_parallel_world_size(),
                )

        self.core_attention = CoreAttention(self.layer_index, self.config, self.attn_mask_type)

        self.out_proj = self._init_out_proj(
            self.hidden_size,
            self.hidden_size,
            cell_name="out_proj",
            input_is_parallel=True,
        )

    def _init_qkv_proj(self, input_size, output_size, cell_name):
        """Construct qkv projection cell."""
        proj = ColumnParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.qkv_has_bias,
            gather_output=False,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            proj_lora = self._get_cell_lora_config(self.config, cell_name)
            if proj_lora is not None:
                proj = ColumnParallelLoRA(
                    input_size,
                    output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.config.qkv_has_bias,
                    gather_output=False,
                    skip_bias_add=False,
                    bias_init=self.config.bias_init,
                    lora_rank=proj_lora["rank"],
                    lora_alpha=proj_lora["alpha"],
                    lora_dropout=proj_lora['dropout']
                )
        return proj

    def _init_out_proj(self, input_size, output_size, cell_name, input_is_parallel=True
                       ):
        """Construct out projection cell."""
        out_proj = RowParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.out_proj_has_bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=True,
            bias_init=self.config.bias_init,
        )
        if self.use_lora:
            out_proj_lora = self._get_cell_lora_config(self.config, cell_name)
            if out_proj_lora is not None:
                out_proj = RowParallelLoRA(
                    input_size,
                    output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    bias=self.config.out_proj_has_bias,
                    input_is_parallel=input_is_parallel,
                    skip_bias_add=False,
                    bias_init=self.config.bias_init,
                    lora_rank=out_proj_lora["rank"],
                    lora_alpha=out_proj_lora["alpha"],
                    lora_dropout=out_proj_lora['dropout']
                )
        return out_proj

    def construct(self,
                  hidden_states,
                  attention_mask,
                  encoder_output=None,
                  inference_params=None,
                  rotary_pos_emb=None,
                  ):
        """Construct function of attention block."""
        if inference_params:
            raise NotImplementedError("inference_params is not supported for now.")
        # hidden_states: [B, S, H]
        ori_dtype = hidden_states.dtype
        if self.data_layout == "SBH":
            seq_len, bs, _ = hidden_states.shape
        else:
            bs, seq_len, _ = hidden_states.shape
        # apply query, key, value projection
        if self.attn_type == AttnType.self_attn:
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size
            # [B, S, H] --> [B, S, H + 2 * kv_H]
            qkv, _ = self.qkv_proj(hidden_states)
            new_tensor_shape = qkv.shape[:-1] + (
                self.kv_num_heads_per_partition,
                (
                    (self.num_heads_per_partition // self.kv_num_heads_per_partition + 2)
                    * self.kv_channels
                ),
            )
            mixed_x_layer = qkv.view(*new_tensor_shape)

            (query,
             key,
             value) = mint.split(
                 mixed_x_layer,
                 [
                     (
                         self.num_heads_per_partition // self.kv_num_heads_per_partition
                         * self.kv_channels
                     ),
                     self.kv_channels,
                     self.kv_channels
                 ],
                 dim=3
             )
            query = query.reshape(query.shape[0], query.shape[1], -1,
                                  self.kv_channels)
        else:
            kv, _ = self.kv_proj(encoder_output)

            new_tensor_shape = kv.shape[:-1] + \
                (self.num_heads_per_partition,
                 2 * self.kv_channels)
            kv = kv.view(*new_tensor_shape)

            last_dim = kv.ndim - 1
            (key, value) = mint.split(kv, split_size_or_sections=kv.shape[last_dim] // 2, dim=last_dim)

            query, _ = self.q_proj(hidden_states)
            new_tensor_shape = query.shape[:-1] + \
                (self.num_heads_per_partition,
                 self.kv_channels)
            query = query.view(*new_tensor_shape)

        if self.data_layout == "BSH":
            query = query.swapaxes(0, 1)
            key = key.swapaxes(0, 1)
            value = value.swapaxes(0, 1)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        # expand the key_layer and value_layer [B, S, kv_N_per_tp, D]
        # to [B, S, N_per_tp, D]
        if self.num_heads_per_partition // self.kv_num_heads_per_partition > 1:
            repeat_num = divide(
                self.num_heads_per_partition, self.kv_num_heads_per_partition
            )
            key = mint.repeat_interleave(key, repeat_num, dim=2)
            value = mint.repeat_interleave(value, repeat_num, dim=2)

        # apply rotary position embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb, self.config)
            key = apply_rotary_pos_emb(key, k_pos_emb, self.config)

        if not self.use_flash_attention:
            context_layer = self.core_attention(query, key, value, attention_mask)
        elif get_context_parallel_world_size() <= 1:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.expand_dims(axis=1)
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)
            attention_mask = attention_mask.astype(mstype.uint8)

            # SBND -> BNSD
            query = query.transpose(1, 2, 0, 3)
            key = key.transpose(1, 2, 0, 3)
            value = value.transpose(1, 2, 0, 3)

            if self.fa_config:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                    **self.fa_config,
                )
            else:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                )
            context_layer = _merge_heads(output)

            # BSH -> SBH
            context_layer = context_layer.swapaxes(0, 1)
        else:
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)

            # SBND -> BNSD
            query = query.transpose(1, 2, 0, 3)
            key = key.transpose(1, 2, 0, 3)
            value = value.transpose(1, 2, 0, 3)

            if not self.enable_flash_sp:
                output = self.ring_attention(query, key, value)
            else:
                # BNSD to BSH
                query = query.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)
                key = key.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)
                value = value.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)

                output = self.flash_sp(query, key, value)
                # BSH to BNSD
                output = output.reshape(bs, seq_len, -1, self.kv_channels).transpose(
                    (0, 2, 1, 3)
                )

            context_layer = _merge_heads(output)
            # BSH -> SBH
            context_layer = context_layer.swapaxes(0, 1)

        if self.data_layout == "BSH":
            context_layer = context_layer.swapaxes(0, 1)

        # apply output projection
        output, bias = self.out_proj(context_layer)
        output = ops.cast(output, ori_dtype)

        return output, bias


class ParallelTransformerLayer(Module):
    r"""
    This class represents a parallel transformer layer. It combines normalization, attention,
    cross attention (if applicable), and an MLP to process input hidden states.

    Args:
        config (dict): Configuration dictionary for the transformer layer.
        layer_number (int): Number which indicates the index of this transformer layer in the
            whole transformer block.
        layer_type (int): Type of the layer. Support [LayerType::encoder = 1, LayerType::decoder = 2,
            LayerType::retro_encoder = 3, LayerType::retro_decoder = 4, LayerType::retro_decoder_with_retriever = 5].
            Default: ``1``.
        self_attn_mask_type (int): Attention mask type. Support [AttnMaskType::padding = 1, AttnMaskType::causal = 2].
            Default: ``1``.
        drop_path_rate（float）：Drop path rate. Currently not supported if greater than 0. Default: ``0.0``

    Inputs:
        - **hidden_states** (Tensor) - The shape of hidden_states tensor is :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Encoder output tensor. Currently not supported. Default: None.
        - **enc_dec_attn_mask** (Tensor) - Encoder-decoder attention mask tensor. Currently not supported.
          Default: None.
        - **retriever_input** (Tensor) - Retriever input tensor. Currently not supported. Default: None.
        - **retriever_output** (Tensor) - Retriever output tensor. Currently not supported. Default: None.
        - **retriever_attn_mask** (Tensor) - Retriever attention mask tensor. Currently not supported. Default: None.
        - **inference_params** (Tensor) - Tensor of inference params. Currently not supported. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **output** (Tensor) - The shape of output tensor is :math:`(B, S, H)`.

    Raises:
        NotImplementedError: If 'bias_dropout_fusion' in config is true.
        NotImplementedError: If `drop_path_rate` greater than 0.
        NotImplementedError: If 'retro_add_retriever' in config is true.
        NotImplementedError: If `encoder_output`, `enc_dec_attn_mask`, `retriever_input`, `retriever_output`,
            `retriever_attn_mask` or `inference_params` is not none.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformerLayer
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> class MyNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(MyNet, self).__init__()
        ...         self.layer = ParallelTransformerLayer(layer_number=1, config=config)
        ...     def construct(self, x, attention_mask):
        ...         output = self.layer(x, attention_mask)
        ...         return output
        ...
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=2)
        >>> config = TransformerConfig(vocab_size=1,
        >>>                            num_layers=1,
        >>>                            num_attention_heads=8,
        >>>                            num_query_groups=4,
        >>>                            hidden_size=256,
        >>>                            ffn_hidden_size=256,
        >>>                            parallel_config=parallel_config)
        >>> input_shape = (32, 1024, 256)
        >>> input = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> mask = Tensor(np.triu(np.ones((1024, 1024)), 1), mstype.uint8)
        >>> out = MyNet(config=config)(input, mask).shape
        >>> print(out)
        (32, 1024, 256)
    """
    # pylint: disable=W0613
    def __init__(self,
                 config,
                 layer_number,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.0,
                 ):
        super(ParallelTransformerLayer, self).__init__(config)
        self.config = config

        self.layer_index = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_norm = (
            self.config.apply_residual_connection_post_norm
        )
        self.fp32_residual_connection = self.config.fp32_residual_connection

        self.residual_connection_dtype = self.config.residual_connection_dtype
        # Normalize the input data.
        self.input_norm = get_norm(config)

        # Attention.
        attention_config = copy.deepcopy(config)
        use_lora = config.lora_config.use_lora
        if use_lora:
            attention_config.update_lora_config("attention")

        self.attention = ParallelAttention(attention_config, layer_number, attention_type=AttnType.self_attn,
                                           attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        if self.bias_dropout_fusion:
            raise NotImplementedError(
                "bias_dropout_fusion is not supported for now."
            )
        if drop_path_rate > 0.0:
            raise NotImplementedError(
                "`drop_path_rate > 0` is not supported for now, "
                "but got `drop_path_rate={}`".format(drop_path_rate)
            )

        # Normalize the attention output
        self.post_attention_norm = get_norm(config)

        # Cross attention.
        if self.layer_type in (LayerType.decoder,
                               LayerType.retro_decoder,
                               LayerType.retro_decoder_with_retriever,
                               LayerType.retro_encoder):
            self.inter_attention = ParallelAttention(
                config,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Normalize the attention output.
            self.post_inter_attention_norm = get_norm(config)

        # MLP
        if self.config.moe_config is not None and self.config.moe_config.num_experts > 1:
            moe_config = copy.deepcopy(config)
            self.mlp = MoELayer(moe_config)
        else:
            mlp_config = copy.deepcopy(config)
            if use_lora:
                mlp_config.update_lora_config('mlp')
            self.mlp = ParallelMLP(mlp_config)

        # Retriever (bidirectional transformer with cross attention)
        if layer_type == LayerType.retro_decoder_with_retriever:
            self.retriever = ParallelTransformer(
                config=config,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever'
        else:
            self.retriever = None

        if self.config.retro_add_retriever:
            raise NotImplementedError(
                "retro_add_retriever is not supported for now."
            )

        self.hidden_states_dropout = mint.nn.Dropout(p=self.hidden_dropout)

        # selective recompute
        if self.config.recompute_granularity == "selective" or self.config.select_recompute:
            self._set_selective_recompute()

    def _set_selective_recompute(self):
        """Set selective recompute for transformer layer."""
        if not self.config.use_flash_attention:
            self.attention.core_attention.recompute()
        else:
            if self.mlp.act_func and isinstance(self.mlp.act_func, nn.Cell):
                self.mlp.act_func.recompute()

    def construct(self,
                  hidden_states,
                  attention_mask,
                  encoder_output=None,
                  enc_dec_attn_mask=None,
                  retriever_input=None,
                  retriever_output=None,
                  retriever_attn_mask=None,
                  inference_params=None,
                  rotary_pos_emb=None,
                  ):
        """Construct function of transformer layer."""
        if encoder_output is not None:
            raise NotImplementedError("encoder_output is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if retriever_input is not None:
            raise NotImplementedError("retriever_input is not supported for now.")
        if retriever_output is not None:
            raise NotImplementedError("retriever_output is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("retriever_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")
        # hidden_states: [B, S, H]
        # layernorm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)

        # attention.
        attention_output, attention_bias = self.attention(
            hidden_states=norm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        # attention bias add
        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
            attention_output = attention_output + attention_bias

        with get_rng_tracer().rng_fork():
            out = self.hidden_states_dropout(attention_output)
        norm_input = residual + out

        # layernorm post attention.
        norm_output = self.post_attention_norm(norm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(norm_output)

        # mlp bias add
        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
            mlp_output = mlp_output + mlp_bias

        # residual-connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        with get_rng_tracer().rng_fork():
            out = self.hidden_states_dropout(mlp_output)
        output = residual + out

        return output


def _get_custom_num_layers(num_layer_list, pp_stage, pp_rank, vpp_stage=None, vpp_rank=0):
    """get transformer layers nums for current rank according to custom num layer list"""
    pp_layout = (1,)
    if vpp_stage is not None:
        pp_layout = (vpp_stage, pp_stage)
    elif pp_stage is not None:
        pp_layout = (pp_stage,)
    num_layer_array = np.array(num_layer_list)
    if num_layer_array.shape != pp_layout:
        raise ValueError("The shape of num_layer_list {} must equal to"
                         "pp_layout {}".format(num_layer_array.shape, pp_layout))
    if vpp_stage is None:
        num_layers = num_layer_array[pp_rank]
        offset = num_layer_array[:pp_rank].sum()
        return num_layers, offset

    num_layers = num_layer_array[vpp_rank][pp_rank]
    offset = num_layer_array[:vpp_rank].sum() + num_layer_array[vpp_rank][:pp_rank].sum()
    return num_layers, offset


# pylint: disable=W0613
def _get_num_layers(config, model_type, is_decoder=False):
    """get transformer layers nums for current rank"""
    vpp = get_virtual_pipeline_model_parallel_world_size() \
        if get_virtual_pipeline_model_parallel_world_size() is not None else 1
    pp_split_num = vpp * get_pipeline_model_parallel_world_size()
    if config.num_layers < pp_split_num:
        raise RuntimeError(f"The number of model layers is {config.num_layers}, "
                           f"but using pipeline parallel requires at least "
                           f"'pp({get_pipeline_model_parallel_world_size()}) "
                           f"* vpp({vpp}) = {pp_split_num}' layers for splitting")
    standalone_embedding_stage = config.parallel_config.standalone_embedding_stage
    if config.encoder_num_layers or config.decoder_num_layers:
        raise NotImplementedError(
            "`encoder_num_layers` and `decoder_num_layers` are not supported for now."
        )
    if get_pipeline_model_parallel_world_size() > 1:
        if standalone_embedding_stage and get_pipeline_model_parallel_rank() == 0:
            num_layers = 0
            offset = 0
        else:
            if config.parallel_config.num_layer_list:
                pp_stage = get_pipeline_model_parallel_world_size()
                pp_rank = get_pipeline_model_parallel_rank()
                if standalone_embedding_stage:
                    pp_stage = get_pipeline_model_parallel_world_size() - 1
                    pp_rank = get_pipeline_model_parallel_rank() - 1
                vpp_stage = get_virtual_pipeline_model_parallel_world_size()
                vpp_rank = get_virtual_pipeline_model_parallel_rank() if vpp_stage is not None else 0
                num_layer_array = np.array(config.parallel_config.num_layer_list)
                if num_layer_array.sum() != config.num_layers:
                    raise ValueError(f"The number of model layers is {config.num_layers}, "
                                     f"but the sum of num_layer_list  "
                                     f"{config.parallel_config.num_layer_list} is {num_layer_array.sum()}.")
                if not np.all(num_layer_array > 0):
                    raise ValueError("num_layer_array has element <= 0")
                num_layers, offset = _get_custom_num_layers(config.parallel_config.num_layer_list,
                                                            pp_stage, pp_rank, vpp_stage, vpp_rank)
                if vpp_stage is not None:
                    logger.info("Custom num layer list is {}. "
                                "Num_layers in vpp_rank:{}"
                                ", pp_rank:{} is {}.\n".format(num_layer_array, vpp_rank, pp_rank, num_layers))
                else:
                    logger.info("Custom num layer list is {}. "
                                "Num_layers in pp_rank:{} is {}.\n".format(num_layer_array, pp_rank, num_layers))
                return num_layers, offset

            def divide_layers(num_layers, stage, rank):
                num_layer_list = [num_layers // stage] * stage
                remain_layer_nums = num_layers - sum(num_layer_list)
                for i in range(remain_layer_nums):
                    num_layer_list[-i - 2] += 1
                num_layers = num_layer_list[rank]
                offset = sum(num_layer_list[:rank])
                return num_layers, offset

            num_layers = config.num_layers
            offset = 0

            vpp_stage = get_virtual_pipeline_model_parallel_world_size()
            if vpp_stage is not None:
                vpp_rank = get_virtual_pipeline_model_parallel_rank()
                num_layers, offset_vpp = divide_layers(num_layers, vpp_stage, vpp_rank)
                offset = offset + offset_vpp

            pp_stage = get_pipeline_model_parallel_world_size() - 1 \
                if standalone_embedding_stage else get_pipeline_model_parallel_world_size()
            pp_rank = get_pipeline_model_parallel_rank() - 1 \
                if standalone_embedding_stage else get_pipeline_model_parallel_rank()
            num_layers, offset_pp = divide_layers(num_layers, pp_stage, pp_rank)
            offset = offset + offset_pp
    else:
        num_layers = config.num_layers
        offset = get_pipeline_model_parallel_rank() * num_layers
    return num_layers, offset


class ParallelTransformer(Module):
    r"""
    This class represents a parallel transformer. It consists of multiple transformer layers and can handle various
    configurations and processing steps.

    Args:
        config (dict): Configuration dictionary for the parallel transformer.
        model_type (int): Type of the model. Support [ModelType::encoder_or_decoder = 1,
            ModelType::encoder_and_decoder = 2, ModelType::retro_encoder = 3, ModelType::retro_decoder = 4].
        layer_type (int): Type of the layer. Support [LayerType::encoder = 1, LayerType::decoder = 2,
            LayerType::retro_encoder = 3, LayerType::retro_decoder = 4, LayerType::retro_decoder_with_retriever = 5].
            Default: ``1``.
        self_attn_mask_type (int): Attention mask type. Support [AttnMaskType::padding = 1, AttnMaskType::causal = 2].
            Default: ``1``.
        post_norm (bool): Insert normalization layer at the end of transformer block. Default: ``True``.
        pre_process (bool): When using pipeline parallel, indicate whether it's the first stage. Default: ``False``.
        post_process (bool): When using pipeline parallel, indicate whether it's the last stage. Default: ``False``.
        drop_path_rate (float): Drop path rate. Currently not supported if greater than 0. Default: ``0.0``

    Inputs:
        - **hidden_states** (Tensor) - The shape of hidden_states tensor is :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **encoder_output** (Tensor) - Encoder output tensor. Currently not supported. Default: None.
        - **enc_dec_attn_mask** (Tensor) - Encoder-decoder attention mask tensor. Currently not supported.
          Default: None.
        - **retriever_input** (Tensor) - Retriever input tensor. Currently not supported. Default: None.
        - **retriever_output** (Tensor) - Retriever output tensor. Currently not supported. Default: None.
        - **retriever_attn_mask** (Tensor) - Retriever attention mask tensor. Currently not supported. Default: None.
        - **inference_params** (Tensor) - Tensor of inference params. Currently not supported. Default: None.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - The shape of hidden_states tensor is :math:`(B, S, H)`.

    Raises:
        NotImplementedError: If `drop_path_rate` greater than 0.
        NotImplementedError: If 'distribute_saved_activations' in config is true and 'sequence_parallel' in config is
            false.
        NotImplementedError: If `transformer_impl` in config is 'transformer_engine'.
        NotImplementedError: If `fp8` in config is not none.
        NotImplementedError: If `retro_add_retriever` in config is true.
        NotImplementedError: If `model_type` equal 3 or 4.
        NotImplementedError: If `encoder_output`, `enc_dec_attn_mask`, `retriever_input`, `retriever_output`,
            `retriever_attn_mask` or `inference_params` is not none.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> class MyNet(nn.Cell):
        ...     def __init__(self, config):
        ...         super(MyNet, self).__init__()
        ...         self.transformer = ParallelTransformer(config=config, model_type=None)
        ...     def construct(self, x, attention_mask):
        ...         output = self.transformer(x, attention_mask)
        ...         return output
        ...
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=2)
        >>> config = TransformerConfig(seq_length=16,
        >>>                            vocab_size=1,
        >>>                            num_layers=1,
        >>>                            num_attention_heads=8,
        >>>                            num_query_groups=4,
        >>>                            hidden_size=256,
        >>>                            ffn_hidden_size=256,
        >>>                            parallel_config=parallel_config)
        >>> input_shape = (32, 1024, 256)
        >>> input = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> mask = Tensor(np.triu(np.ones((1024, 1024)), 1), mstype.uint8)
        >>> out = MyNet(config=config)(input, mask).shape
        >>> print(out)
        (32, 1024, 256)
    """

    def __init__(self,
                 config,
                 model_type,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_norm=True,
                 pre_process=False,
                 post_process=False,
                 drop_path_rate=0.0,
                 ):
        super(ParallelTransformer, self).__init__(config)
        if drop_path_rate > 0.0:
            raise NotImplementedError(
                "`drop_path_rate > 0` is not supported for now, "
                "but got `drop_path_rate={}`".format(drop_path_rate)
            )
        self.layer_type = layer_type
        self.attn_mask_type = self_attn_mask_type
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_norm = post_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.config = config
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = config.transformer_impl
        self.retro_add_retriever = config.retro_add_retriever

        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel
        if self.distribute_saved_activations:
            raise NotImplementedError(
                "`distribute_saved_activations` is not supported for now."
            )

        if self.transformer_impl == 'transformer_engine':
            raise NotImplementedError(
                "`transformer_impl=transformer_engine` is not supported for now."
            )

        self.use_fp8 = config.fp8 is not None
        if self.use_fp8:
            raise NotImplementedError(
                "fp8 format is not supported for now."
            )

        # number of layers.
        self.num_layers, offset = _get_num_layers(
            config, model_type=None, is_decoder=False
        )

        seq_length = config.seq_length
        if config.parallel_config.sequence_parallel:
            seq_length = seq_length // get_tensor_model_parallel_world_size()

        if config.retro_add_retriever:
            raise NotImplementedError(
                "retro_add_retriever is not supported for now."
            )

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            raise NotImplementedError("ModelType `retro_decoder` is not supported for now.")
        if model_type == ModelType.retro_encoder:
            raise NotImplementedError("ModelType `retro_encoder` is not supported for now.")

        layers_config = copy.deepcopy(config)
        use_lora = config.lora_config.use_lora

        # user defined recompute
        recompute_config = self.config.parallel_config.recompute_config
        if recompute_config is not None:
            (full_recompute_layers, select_recompute_layers,
             select_comm_recompute_layers) = self._get_recompute_nums(recompute_config)
            # enable full layer recompute
            layers_index_config = []
            if full_recompute_layers != 0:
                self.config.recompute_method = "block"
                self.config.recompute_num_layers = full_recompute_layers
                self.config.recompute_granularity = "full"

            for i in range(full_recompute_layers):
                layer_config_new = copy.deepcopy(self.config)
                layers_index_config.append(layer_config_new)

            # enable select、select_comm according to self.config.select_recompute/select_comm_recompute
            if select_comm_recompute_layers < select_recompute_layers:
                common_select_recompute_layer_nums = select_comm_recompute_layers
            else:
                common_select_recompute_layer_nums = select_recompute_layers
            for i in range(common_select_recompute_layer_nums):
                layer_config_new = copy.deepcopy(layers_config)
                layer_config_new.select_recompute = True
                layer_config_new.select_comm_recompute = True
                layers_index_config.append(layer_config_new)

            diff_select_recompute_layer_nums = abs(select_comm_recompute_layers - select_recompute_layers)
            for i in range(diff_select_recompute_layer_nums):
                layer_config_new = copy.deepcopy(layers_config)
                if select_comm_recompute_layers > select_recompute_layers:
                    layer_config_new.select_comm_recompute = True
                else:
                    layer_config_new.select_recompute = True
                layers_index_config.append(layer_config_new)
            remain_layer_nums = (self.num_layers - common_select_recompute_layer_nums -
                                 diff_select_recompute_layer_nums - full_recompute_layers)

            # remain layers are not recompute.
            for i in range(remain_layer_nums):
                layer_config_new = copy.deepcopy(layers_config)
                layers_index_config.append(layer_config_new)


        if use_lora:
            layers_config.update_lora_config("layers")
            if recompute_config is None:
                layers_index_config = []
            for i in range(self.num_layers):
                if recompute_config is None:
                    layer_config_new = copy.deepcopy(layers_config)
                    layer_config_new.update_lora_config(f"{i}")
                    layers_index_config.append(layer_config_new)
                else:
                    layer_config_new = copy.deepcopy(layers_index_config[i])
                    layer_config_new.update_lora_config(f"{i}")
                    layers_index_config[i] = layer_config_new

        # ensure the Parameter of each rank init as correct name
        layers_dict = OrderedDict()
        for i in range(self.num_layers):
            layers_dict[str(i + offset)] = ParallelTransformerLayer(
                config=layers_index_config[i] if use_lora or recompute_config else layers_config,
                layer_number=i + 1 + offset,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=drop_path_rate
            )
        self.layers = nn.SequentialCell(layers_dict)

        # gradient checkpointing for recompute.
        self.checkpointed_recompute = (
            self.config.recompute_method is not None
            and self.config.recompute_granularity is not None
            and self.config.recompute_num_layers is not None
            and self.config.recompute_granularity == "full"
        )
        if self.checkpointed_recompute:
            self._set_checkpointed_recompute(
                self.config.recompute_method, self.config.recompute_num_layers
            )
        if self.post_process and self.post_norm:
            # final layernorm before output.
            self.final_norm = get_norm(config)

        self.pipeline_parallel = get_pipeline_model_parallel_world_size() > 1
        if self.pipeline_parallel:
            batch_size = config.dataset_config.batch_size
            if config.dataset_config.data_layout == "BSH":
                hidden_states_shape = (batch_size, seq_length, config.hidden_size)
            else:
                hidden_states_shape = (seq_length, batch_size, config.hidden_size)
            self.set_hidden_states = Parameter(
                mint.zeros(
                    hidden_states_shape, dtype=config.compute_dtype
                ),
                requires_grad=False,
                name="set_hidden_states",
            )


    def _get_recompute_nums(self, recompute_config):
        """Get recompute layers nums for current rank"""
        # Get vpp_rank and pp_rank
        pp_rank = 0
        vpp_stage = None
        pp_layout = (1,)
        if get_pipeline_model_parallel_world_size() > 1:
            standalone_embedding_stage = self.config.parallel_config.standalone_embedding_stage
            pp_stage = get_pipeline_model_parallel_world_size()
            pp_rank = get_pipeline_model_parallel_rank()
            if standalone_embedding_stage:
                pp_stage = get_pipeline_model_parallel_world_size() - 1
                pp_rank = get_pipeline_model_parallel_rank() - 1
            vpp_stage = get_virtual_pipeline_model_parallel_world_size()
            vpp_rank = get_virtual_pipeline_model_parallel_rank() if vpp_stage is not None else 0
            if vpp_stage is not None:
                pp_layout = (vpp_stage, pp_stage)
            elif pp_stage is not None:
                pp_layout = (pp_stage,)

        full_recompute_layers = 0
        select_recompute_layers = 0
        select_comm_recompute_layers = 0
        self.config.recompute_method = None
        self.config.recompute_granularity = None
        self.config.recompute_num_layers = None
        def _get_recompute_layer_nums(recompute_num_list):
            recompute_num_array = np.array(recompute_num_list)
            assert np.all(recompute_num_array >= 0)
            if recompute_num_array.shape != pp_layout:
                raise ValueError("The shape of recompute_num_list {} must equal to "
                                 "pp_layout {}".format(recompute_num_array.shape, pp_layout))
            if vpp_stage is not None:
                recompute_layers = recompute_num_array[vpp_rank][pp_rank]
                return recompute_layers
            return recompute_num_array[pp_rank]
        if recompute_config.recompute:
            full_recompute_layers = _get_recompute_layer_nums(recompute_config.recompute)

        if recompute_config.select_recompute:
            select_recompute_layers = _get_recompute_layer_nums(recompute_config.select_recompute)

        if recompute_config.select_comm_recompute:
            select_comm_recompute_layers = _get_recompute_layer_nums(recompute_config.select_comm_recompute)

        if select_comm_recompute_layers + full_recompute_layers > self.num_layers:
            raise ValueError("Recompute layers must less or equal num layers, but got "
                             "select_comm_recompute_layers {} + full_recompute_layers {} > "
                             "num_layers {}.".format(select_comm_recompute_layers,
                                                     full_recompute_layers, self.num_layers))
        if select_recompute_layers + full_recompute_layers > self.num_layers:
            raise ValueError("Recompute layers must less or equal num layers, but got "
                             "select_recompute_layers {} + full_recompute_layers {} > "
                             "num_layers {}.".format(select_recompute_layers, full_recompute_layers, self.num_layers))
        if vpp_stage is not None:
            logger.info("in vpp_rank:{}, pp_rank:{}, full_recompute_layers is {}, "
                        "select_recompute_layers is {}, select_comm_recompute_layers is {}"
                        .format(vpp_rank, pp_rank, full_recompute_layers, select_recompute_layers,
                                select_comm_recompute_layers))
        else:
            logger.info("in pp_rank:{}, full_recompute_layers is {}, "
                        "select_recompute_layers is {}, select_comm_recompute_layers is {}"
                        .format(pp_rank, full_recompute_layers, select_recompute_layers, select_comm_recompute_layers))
        return full_recompute_layers, select_recompute_layers, select_comm_recompute_layers


    def _set_checkpointed_recompute(self, recompute_method, recompute_num_layers):
        """Set checkpointed recompute for transformer."""
        self.checkpointed_recompute = True
        self.checkpointed_layer_groups = nn.CellList()
        if recompute_method == "uniform":
            for idx in range(0, self.num_layers, recompute_num_layers):
                checkpointed_layer_group = CheckpointedRecomputeOrientedCell(
                    self.layers[idx: idx + recompute_num_layers]
                )
                checkpointed_layer_group.recompute()
                self.checkpointed_layer_groups.append(checkpointed_layer_group)
        elif recompute_method == "block":
            for idx in range(0, min(self.num_layers, recompute_num_layers)):
                self.layers[idx].recompute()
            self.checkpointed_layer_groups = self.layers
        else:
            raise NotImplementedError(
                f"recompute_method should be uniform or blocks, but got {recompute_method}"
            )

    def set_input_tensor(self, input_tensor):
        """
        In pipeline parallel, the receiving data from previous stage will be set into class.
        Construct function's input will be replace by self.set_hidden_states.
        """
        self.set_hidden_states.set_data(input_tensor, slice_shape=True)

    def construct(self,
                  hidden_states,
                  attention_mask,
                  encoder_output=None,
                  enc_dec_attn_mask=None,
                  retriever_input=None,
                  retriever_output=None,
                  retriever_attn_mask=None,
                  inference_params=None,
                  rotary_pos_emb=None,
                  ):
        """Construct function of transformer."""
        if encoder_output is not None:
            raise NotImplementedError("encoder_output is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if retriever_input is not None:
            raise NotImplementedError("retriever_input is not supported for now.")
        if retriever_output is not None:
            raise NotImplementedError("retriever_output is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("retriever_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")

        if self.checkpointed_recompute:
            layers = self.checkpointed_layer_groups
        else:
            layers = self.layers

        # self.hidden_states instead of input
        if not self.pre_process and self.pipeline_parallel:
            hidden_states = self.set_hidden_states.value()

        for layer in layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
            )

        # final layernorm.
        if self.post_process and self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


class ParallelLMLogits(nn.Cell):
    r"""
    Head to get the logits of each token in the vocab.

    Args:
        config (dict): Parallel configuration.
        bias (bool): Specifies whether the layer uses a bias vector. Default: ``False``.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: ``None``.

    Inputs:
        - **input_** (Tensor) - Tensor of hidden states.
        - **word_embedding_table** (Parameter) - Weight matrix passed from embedding layer.
        - **parallel_output** (bool) - Specifies whether return paralleled output on each tensor parallel rank.
          Default: True.
        - **bias** (Tensor) - The trainable bias parameter.

    Outputs:
        Tensor of logits.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindformers.experimental.parallel_core.pynative.config import ModelParallelConfig, TransformerConfig
        >>> from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
        >>> from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits
        >>> init()
        >>> initialize_model_parallel()
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
        >>> config = TransformerConfig      #The config of Transformer model. For details, please refer to TransformerConfig
        >>> model = ParallelLMLogits(config=config, bias=False, compute_dtype=ms.float32)
        >>> input = Tensor(np.random.random((2, 3, 3)).astype(np.float32))
        >>> weight = Tensor(np.random.random((3, 3)).astype(np.float32))
        >>> logits = model(input, weight, parallel_output=True)
        >>> print(logits)
    """

    def __init__(self, config, bias=False, compute_dtype=None):
        super(ParallelLMLogits, self).__init__()
        self.compute_dtype = (
            compute_dtype if compute_dtype else config.compute_dtype
        )
        self.config = config
        self.is_tensor_parallel = get_tensor_model_parallel_world_size() > 1
        if self.is_tensor_parallel or self.config.parallel_config.sequence_parallel:
            self.allreduce_dgrad = self.is_tensor_parallel and not config.parallel_config.sequence_parallel
        else:
            self.allreduce_dgrad = False

        self.gradient_accumulation_fusion = config.parallel_config.gradient_accumulation_fusion
        self.forward_impl_ = LinearWithGradAccumulationAndAsyncCommunication(
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel=self.config.parallel_config.sequence_parallel,
            allreduce_dgrad=self.allreduce_dgrad,
            data_layout=config.dataset_config.data_layout
        )
        self.copy_to_mp_region = CopyToModelParallelRegion()
        self.gather_from_mp_region = GatherFromModelParallelRegion()

    def construct(self, input_, word_embeddings_weight, parallel_output=True, bias=None
                  ):
        """LM logits using word embedding table"""
        if self.is_tensor_parallel or self.config.parallel_config.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = self.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = ops.cast(word_embeddings_weight, self.compute_dtype)
        weight_param = None
        if self.gradient_accumulation_fusion and isinstance(word_embeddings_weight, Parameter
                                                            ):
            weight_param = word_embeddings_weight
        input_parallel = ops.cast(input_parallel, self.compute_dtype)

        bias = ops.cast(bias, self.compute_dtype) if bias else None

        # Matrix multiply.
        logits_parallel = self.forward_impl_(
            input_parallel, weight, bias, weight_param=weight_param
        )
        logits_parallel = ops.cast(logits_parallel, origin_dtype)

        # Gather if needed.
        if parallel_output:
            return logits_parallel

        return self.gather_from_mp_region(logits_parallel)
