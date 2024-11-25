# Copyright 2020-2024 Huawei Technologies Co., Ltd
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

"""
Fused softmax for transformer.
"""
from typing import Callable
from mindspore import nn, dtype, Tensor
from mindspore.ops import operations as P
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.transformer.enums import AttnMaskType

__all__ = [
    'FusedScaleMaskSoftmax'
]


class FusedScaleMaskSoftmax(nn.Cell):
    """Fused operation: scaling + mask + softmax

    Args:
        config (TransformerConfig): The transformer configuration.
        mask_func (Callable): The mask function.
        scale (float): The scale factor.
        softmax_in_fp32 (bool): Whether to use fp32 for softmax.
        input_in_fp16 (bool): Whether the input is in fp16.
        input_in_bf16 (bool): Whether the input is in bf16.
        attn_mask_type (AttnMaskType): The attention mask type.
        scaled_masked_softmax_fusion (bool): Whether to fuse scaled and masked softmax
    """
    def __init__(self,
                 input_in_fp16: bool = False,
                 input_in_bf16: bool = False,
                 attn_mask_type: AttnMaskType = AttnMaskType.causal,
                 scaled_masked_softmax_fusion: bool = False,
                 mask_func: Callable = None,
                 softmax_in_fp32: bool = True,
                 scale: float = None,
                 config: TransformerConfig = None,
                 softmax_compute_dtype: dtype = None
                 ):
        super(FusedScaleMaskSoftmax, self).__init__()
        if scaled_masked_softmax_fusion:
            raise NotImplementedError("For FusedScaleMaskSoftmax, "
                                      "scaled_masked_softmax_fusion is not supported for now.")
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16

        if self.input_in_fp16 and self.input_in_bf16:
            raise ValueError("Both fp16 and bf16 flags cannot be active at the same time")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.mask_func = mask_func
        self.scale = scale
        self.softmax_in_fp32 = softmax_in_fp32
        self.softmax_compute_dtype = softmax_compute_dtype
        if (self.softmax_in_fp32
                and (self.softmax_compute_dtype is not None and self.softmax_compute_dtype != dtype.float32)):
            raise ValueError("softmax_compute_dtype should be float32 when softmax_in_fp32 is True")
        self.causal_attn_mask_type = attn_mask_type == AttnMaskType.causal
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.triu = P.Triu(1)
        self.ones = P.Ones()

        if self.scale is not None and not self.softmax_in_fp32:
            raise ValueError("softmax should be in fp32 when scaled")

        self.shard(config)

    def construct(self, input_: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass of softmax with masked input.

        Args:
            input_ (Tensor): The input tensor.
            mask (Tensor): The mask tensor.
        """
        if self.input_in_float16 and self.softmax_in_fp32:
            input_ = self.cast(input_, dtype.float32)

        if self.scale is not None:
            input_ = input_ * self.scale

        sq = input_.shape[-2]
        if self.causal_attn_mask_type and mask is None and sq > 1:
            mask = self.triu(self.ones((sq, sq))).bool()

        if mask is not None and self.mask_func:
            input_ = self.mask_func(input_, mask)

        if self.softmax_compute_dtype is not None:
            ori_dtype = input_.dtype
            input_ = self.cast(input_, self.softmax_compute_dtype)
        else:
            ori_dtype = None
        output = self.softmax(input_)
        if ori_dtype is not None:
            output = self.cast(output, ori_dtype)
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                output = self.cast(output, dtype.float16)
            else:
                output = self.cast(output, dtype.bfloat16)
        return output

    def shard(self, config: TransformerConfig):
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        cp = config.context_parallel if config and config.context_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1
        self.softmax.shard(((dp, tp, cp, 1),))
