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
"""DeepSeekV2 Model Layers' APIs."""

from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.parallel.shard import Layout
from mindformers.modules.transformer.moe import MoEInfer
from mindformers.models.llama.llama_layer import LlamaFeedForward, LlamaMoeInferFeedForward


class DeepSeekV2RotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding for DeepSeekV2. This matches official implementation in Hugginface.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **parallel_config** (dict): - Parallel Config.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32, use_rope_slice=True):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.dtype = compute_dtype
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True

        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose()
        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()
        self.mul_inc = P.Mul()
        self.neg = P.Neg()
        self.slice = P.StridedSlice()
        self.concat = P.Concat(axis=-1)
        self.shape = P.Shape()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        x = self.bmm_swap(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = self.shape(x)
        x1 = self.slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = self.slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = self.concat((self.neg(x2), x1))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype

        b, h, s, d = self.shape(xq)
        b2, h2, s2, d2 = self.shape(xk)
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)

        xq = self.transpose(self.reshape(xq, (b, h, s, d // 2, 2)), (0, 1, 2, 4, 3))
        xk = self.transpose(self.reshape(xk, (b2, h2, s2, d2 // 2, 2)), (0, 1, 2, 4, 3))
        xq = self.reshape(xq, (b, h, s, d))
        xk = self.reshape(xk, (b2, h2, s2, d2))
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        freqs_cos = self.cast(freqs_cos, self.dtype)
        freqs_sin = self.cast(freqs_sin, self.dtype)
        swap_mask = self.cast(swap_mask, self.dtype)
        mul = self.mul if self.is_first_iteration else self.mul_inc
        if self.use_rope_slice:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.slice_half(xq), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.rotate_half(xq, swap_mask), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.rotate_half(xk, swap_mask), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out

    def shard(self, parallel_config):
        """sharding for rotary embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        strategy_in = (dp, mp, 1, 1)
        if cp > 1:
            layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
            layout_add = (layout("dp", "mp", "cp", "None"), layout("dp", "mp", "cp", "None"))
            layout_bmm_swap = (layout("dp", "mp", "cp", "None"), layout("None", "None"))
            layout_mul = (layout("dp", "mp", "cp", "None"), layout("None", "None", "cp", "None"))
            self.add.shard(in_strategy=layout_add)
            self.bmm_swap.shard(in_strategy=layout_bmm_swap)
            self.mul.shard(in_strategy=layout_mul)
        else:
            self.add.shard((strategy_in, strategy_in))
            self.bmm_swap.shard((strategy_in, (1, 1)))
            self.mul.shard((strategy_in, (1, 1)))
        self.mul_inc.shard((strategy_in, (strategy_in[0], 1, 1, 1)))
        self.neg.shard((strategy_in,))
        self.slice.shard((strategy_in,))
        self.concat.shard((strategy_in, strategy_in))
        transpose_strategy_in = (dp, mp, 1, 1, 1)
        self.transpose.shard((transpose_strategy_in,))


class DeepSeekV2MoEInfer(Cell):
    r"""
    MoE inferernce inherited from MoEInfer, where shared experts are added.
    """
    def __init__(self, hidden_size, intermediate_size, compute_dtype,
                 param_init_type, is_dynamic, moe_config, parallel_config):
        super(DeepSeekV2MoEInfer, self).__init__()
        ffn = LlamaMoeInferFeedForward(dim=hidden_size,
                                       intermediate_size=intermediate_size,
                                       expert_num=moe_config.expert_num,
                                       compute_dtype=compute_dtype,
                                       param_init_type=param_init_type,
                                       is_dynamic=is_dynamic,
                                       use_gmm=True)
        self.routed_experts = MoEInfer(ffn, hidden_size, moe_config, parallel_config)
        intermediate_size_all = int(moe_config.moe_intermediate_size * moe_config.shared_expert_num)
        self.shared_experts = LlamaFeedForward(dim=hidden_size,
                                               intermediate_size=intermediate_size_all,
                                               expert_num=1,
                                               compute_dtype=compute_dtype,
                                               param_init_type=param_init_type,
                                               is_dynamic=is_dynamic,
                                               parallel_config=parallel_config)
        self.add = P.Add()

    def construct(self, x):
        routed_experts_output = self.routed_experts(x)
        shared_experts_output = self.shared_experts(x)
        output = self.add(routed_experts_output, shared_experts_output)
        return output

    def shard(self, parallel_config):
        r"""set parallel strategy"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.add.shard(((dp, 1, 1), (dp, 1, 1)))

        self.routed_experts.ffn.shard(parallel_config)
        self.shared_experts.shard(parallel_config)
        self.shared_experts.mul.shard(((dp, 1, mp), (dp, 1, mp)))
