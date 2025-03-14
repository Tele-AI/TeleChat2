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
"""LLaMA transformer Layer's APIs."""
from typing import Tuple, Optional
import math


import mindspore as ms
from mindspore import nn, Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.parallel.shard import Layout

from mindformers.tools.logger import logger
from mindformers.models.llama.llama_layer import (
    LlamaFeedForward,
    LlamaRMSNorm,
    LlamaMoeInferFeedForward,
    LlamaFeedForwardWithMoE
)
from mindformers.models.utils import predict_lazy_inline
from mindformers.modules.layers import _check_input_dtype, Linear, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules.transformer.moe import MoEV2, MoEInfer
from mindformers.modules.transformer.moev3 import MoEV3
from mindformers.tools.utils import get_predict_run_mode, divide
from mindformers.version_control import check_seqpp_fa_opt_support


class LLamaAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in LLaMA.

    Args:
            - **seq_length** (int): The sequence length of input.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **qkv_has_bias** (bool): Whether Q/K/V in attention has bias or not.
            - **attn_proj_has_bias** (bool): Whether projection in attention has bias or not.
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
            - **init_method_std** (float): The sigma value when using normal type to initialize Linear. Default `0.01`
            - **rmsnorm_compute_2d** (bool): Whether to use 2D Add in RMS_NORM. Default `False` .
            - **use_3d_tensor_parallel** (bool): Whether enable high dimension tensor parallel.
                Replace model_parallel by three dimensions: tp_x, tp_y, tp_z. The product of tp_x, tp_y and tp_z
                should be equal to model_parallel. Default False.
            - **tp_x** (int): The x value of high tensor parallel way. Default 1.
            - **tp_y** (int): The y value of high tensor parallel way. Default 1.
            - **tp_z** (int): The z value of high tensor parallel way. Default 1.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
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
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 attn_proj_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 use_ring_attention=False,
                 use_attn_mask_compression=False,
                 use_eod_attn_mask_compression=False,
                 rmsnorm_compute_2d=False,
                 batch_size=1,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 parallel_decoding=False,
                 rl_config=None,
                 init_method_std=0.01,
                 chunk_prefill=False,
                 use_3d_tensor_parallel=False,
                 tp_x=1,
                 tp_y=1,
                 tp_z=1
                 ):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        self.chunk_prefill = chunk_prefill

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.use_flash_attention = use_flash_attention
        self.use_ring_attention = use_ring_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.use_eod_attn_mask_compression = use_eod_attn_mask_compression
        self.qkv_concat = qkv_concat

        self.rl_config = rl_config

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        self.data_parallel = dp
        self.model_parallel = mp
        self.context_parallel = cp
        # define ulysses context parallel
        self.cp_ds = parallel_config.get_ulysses_cp_num()
        # define colossal ai context parallel
        self.cp_co = cp // self.cp_ds

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        head_parallel = tp_y * self.cp_ds if use_3d_tensor_parallel else mp * self.cp_ds
        if self.n_head % (head_parallel) != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                             "'parallel_config.model_parallel * ulysses_cp_num' "
                             "(or 'tp_y * ulysses_cp_num' when use_3d_tensor_parallel is True),"
                             " but got the n_head is {}, the parallel_config.model_parallel is {}, "
                             "the config.tp_y is {}, and ulysses_cp_num is {}"
                             .format(self.n_head, mp, tp_y if use_3d_tensor_parallel else 1, self.cp_ds))
        if self.n_kv_head % (head_parallel) != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel * ulysses_cp_num' "
                             "(or 'config.tp_y * ulysses_cp_num' when use_3d_tensor_parallel is True), "
                             "but got the n_kv_head is {}, the parallel_config.model_parallel is {}, "
                             "the config.tp_y is {}, and ulysses_cp_num is {}"
                             .format(self.n_kv_head, mp, tp_y if use_3d_tensor_parallel else 1, self.cp_ds))

        if use_3d_tensor_parallel:
            layout_ndtp = Layout((dp, cp, tp_z, tp_x, tp_y), ("dp", "cp", "z", "x", "y"))
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.seq_pipe = parallel_config.seq_split_num > 1
        self.seq_split_num = parallel_config.seq_split_num
        self.seq_seg_len = seq_length // self.seq_split_num
        if self.seq_pipe:
            if not self.use_flash_attention:
                raise ValueError("Seq pipe must using flash attention")
            kv_shape = (batch_size * dp, self.n_kv_head, seq_length, self.head_dim)
            self.key_cache = Parameter(initializer('zeros', shape=kv_shape, dtype=compute_dtype), name="key_cache",
                                       requires_grad=False, parallel_optimizer=False)
            self.value_cache = Parameter(initializer('zeros', shape=kv_shape, dtype=compute_dtype), name="value_cache",
                                         requires_grad=False, parallel_optimizer=False)
            kv_grad_shape = (batch_size, self.n_kv_head // mp, seq_length // cp, self.head_dim)
            self.key_cache_grad = Parameter(initializer('zeros', shape=kv_grad_shape, dtype=compute_dtype),
                                            name="key_cache_grad", requires_grad=False, parallel_optimizer=False)
            self.value_cache_grad = Parameter(initializer('zeros', shape=kv_grad_shape, dtype=compute_dtype),
                                              name="value_cache_grad", requires_grad=False, parallel_optimizer=False)
            self.select = P.Select().add_prim_attr("self_define_shard", True)
            layout_v = Layout((dp, mp, cp), ("dp", "mp", "cp"))
            in_layout_v = (layout_v("dp", "mp", "cp", "None"), layout_v("dp", "mp", "cp", "None"),
                           layout_v("dp", "mp", "cp", "None"))
            out_layout_v = (layout_v("dp", "mp", "cp", "None"),)
            self.select.shard(in_layout_v, out_layout_v)
            self.add_k = P.Add().shard(((dp, mp, cp, 1), (dp, mp, cp, 1)))
            self.add_v = P.Add().shard(((dp, mp, cp, 1), (dp, mp, cp, 1)))
            self.mul_kv = P.Mul().shard(((dp, mp, cp, 1), (dp, mp, cp, 1)))
            self.assign_kv = P.Assign().shard(((dp, mp, cp, 1), (dp, mp, cp, 1)))
            self.mul_update = P.Mul().shard(((dp, mp, cp, 1), ()))
            self.not_equal_ones = P.NotEqual().shard(((dp, mp, cp, 1), ()))
            self.not_equal_seq = P.NotEqual()
            self.seq_split_size = Tensor(self.seq_split_num - 1, dtype=mstype.int32)
            self.tile_kv = P.Tile().shard(((dp, mp, cp, 1),))

        if self.qkv_concat:
            self.w_qkv = Linear(in_channels=self.hidden_size,
                                out_channels=self.hidden_size + self.kv_dim * 2,
                                init_method_std=init_method_std,
                                has_bias=qkv_has_bias,
                                compute_dtype=compute_dtype,
                                param_init_type=param_init_type)
            if qkv_has_bias:
                if use_3d_tensor_parallel:
                    self.w_qkv.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                     (layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("y",)),
                                     enable_nd_tp=True)
                else:
                    self.w_qkv.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
            else:
                if use_3d_tensor_parallel:
                    self.w_qkv.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))), \
                                     enable_nd_tp=True)
                else:
                    self.w_qkv.shard(((dp * cp, 1), (mp, 1)))
            self.split_qkv = ms.ops.auto_generate.SplitWithSize()
            self.split_qkv.add_prim_attr("skip_redistribution", True)
            if use_3d_tensor_parallel:
                self.split_qkv.shard(((dp, cp * tp_z * tp_x, tp_y, 1),))
            else:
                self.split_qkv.shard(((dp, cp, mp, 1),))
        else:
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             init_method_std=init_method_std,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wk = Linear(self.hidden_size,
                             self.kv_dim,
                             init_method_std=init_method_std,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wv = Linear(self.hidden_size,
                             self.kv_dim,
                             init_method_std=init_method_std,
                             has_bias=qkv_has_bias,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            if qkv_has_bias:
                if use_3d_tensor_parallel:
                    self.wq.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))), \
                                  (layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("y",)), enable_nd_tp=True)
                    self.wk.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))), \
                                  (layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("y",)), enable_nd_tp=True)
                    self.wv.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))), \
                                  (layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("y",)), enable_nd_tp=True)
                else:
                    self.wq.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
                    self.wk.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
                    self.wv.shard(((dp * cp, 1), (mp, 1)), ((dp * cp, mp), (mp,)))
            else:
                if use_3d_tensor_parallel:
                    self.wq.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                  enable_nd_tp=True)
                    self.wk.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                  enable_nd_tp=True)
                    self.wv.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("y", ("x", "z"))),
                                  enable_nd_tp=True)
                else:
                    self.wq.shard(((dp * cp, 1), (mp, 1)))
                    self.wk.shard(((dp * cp, 1), (mp, 1)))
                    self.wv.shard(((dp * cp, 1), (mp, 1)))
        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         init_method_std=init_method_std,
                         has_bias=attn_proj_has_bias,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        if use_3d_tensor_parallel:
            self.wo.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("x", ("y", "z"))), \
                          (layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("x")), enable_nd_tp=True)
        else:
            self.wo.shard(((dp * cp, mp), (1, mp)), ((dp * cp, 1), (1,)), out_strategy_matmul=((dp * cp, 1),))

        if self.use_past:
            self.infer_attention = InferAttention(self.n_head,
                                                  self.head_dim,
                                                  self.n_kv_head,
                                                  seq_length=seq_length,
                                                  pa_n_head_split=self.n_head // mp,
                                                  pa_n_kv_head_split=self.n_kv_head // mp,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  block_size=self.block_size,
                                                  num_blocks=self.num_blocks,
                                                  is_dynamic=is_dynamic,
                                                  use_flash_attention=self.use_flash_attention,
                                                  rotary_cos_format=2,
                                                  compute_dtype=compute_dtype,
                                                  parallel_decoding=parallel_decoding,
                                                  chunk_prefill=chunk_prefill,
                                                  )
            self.infer_attention.shard(parallel_config)
        else:
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            self.batch_matmul = P.BatchMatMul()
            self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
            self.mul = P.Mul()
            self.add = P.Add()
            self.softmax = P.Softmax()
            self.cast_attn = P.Cast()
            self.tile_kv = P.Tile()

            self.apply_rotary_emb = RotaryEmbedding(self.head_dim,
                                                    rotary_dtype,
                                                    use_rope_slice=use_rope_slice,
                                                    use_3d_tensor_parallel=use_3d_tensor_parallel,
                                                    tp_x=tp_x,
                                                    tp_y=tp_y,
                                                    tp_z=tp_z)

            # ulysses context parallel, initial related ops
            if self.cp_ds > 1:
                self._ulysses_initial()

            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.transpose.shard(((dp, cp, mp, 1),))
                if cp > 1:
                    layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
                    layout_merger_head_transpose = (layout("dp", "mp", "cp", "None"),)
                    self.merger_head_transpose.shard(in_strategy=layout_merger_head_transpose)
                else:
                    self.merger_head_transpose.shard(((dp, mp, 1, 1),))
                self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.apply_rotary_emb.shard(parallel_config)
            else:
                if use_3d_tensor_parallel:
                    self.transpose.shard((layout_ndtp("dp", ("cp", "z", "x"), "y", "None"),))
                else:
                    self.transpose.shard(((dp, cp, mp, 1),))
                if cp > 1:
                    if use_3d_tensor_parallel:
                        self.merger_head_transpose.shard((layout_ndtp("dp", "y", ("cp", "z", "x"), "None"),))
                    else:
                        layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
                        layout_merger_head_transpose = (layout("dp", "mp", "cp", "None"),)
                        self.merger_head_transpose.shard(in_strategy=layout_merger_head_transpose)
                else:
                    if use_3d_tensor_parallel:
                        self.merger_head_transpose.shard((layout_ndtp("dp", "y", ("cp", "z", "x"), "None"),))
                    else:
                        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
                self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul.shard(((dp, mp, 1, 1), ()))
                self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
                self.softmax.shard(((dp, mp, 1, 1),))
                self.tile_kv.shard(((dp, mp, 1, 1),))
                self.apply_rotary_emb.shard(parallel_config)

            if parallel_config.use_seq_parallel and self.is_first_iteration:
                if use_3d_tensor_parallel:
                    self.wo.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"), layout_ndtp("x", ("y", "z"))),
                                  (layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("x",)), enable_nd_tp=True)
                else:
                    self.wo.shard(((dp * cp, mp), (1, mp)), ((dp * cp * mp, 1), (1,)),
                                  out_strategy_matmul=((dp * cp * mp, 1),))
            if parallel_config.recompute.select_recompute and not self.use_flash_attention:
                self.apply_rotary_emb.recompute()
                self.tile_kv.recompute()
                self.batch_matmul_q_k.recompute()
                self.mul.recompute()
                self.add.recompute()
                self.cast_attn.recompute()
                self.softmax.recompute()
                self.batch_matmul.recompute()
            self.input_layout = None
            if self.use_flash_attention:
                self.input_layout = "BSH" if cp > 1 else "BNSD"
                if self.use_eod_attn_mask_compression and not self.use_ring_attention:
                    self.sparse_mode = 8
                    self.input_layout = "TND"
                elif self.use_attn_mask_compression and not self.use_ring_attention:
                    self.sparse_mode = 2
                else:
                    self.sparse_mode = 0
                if self.seq_pipe and not check_seqpp_fa_opt_support():
                    next_tokens = seq_length - self.seq_seg_len
                else:
                    next_tokens = 0

                self.flash_attention = FlashAttention(head_num=self.n_head,
                                                      pre_tokens=2147483647,
                                                      next_tokens=next_tokens,
                                                      input_layout=self.input_layout,
                                                      keep_prob=1.,
                                                      scale_value=1. / math.sqrt(self.head_dim),
                                                      sparse_mode=self.sparse_mode,
                                                      use_attention_mask=True,
                                                      use_ring_attention=self.use_ring_attention,
                                                      use_3d_tensor_parallel=use_3d_tensor_parallel,
                                                      use_actual_seqlen=self.use_eod_attn_mask_compression,
                                                      tp_x=tp_x,
                                                      tp_y=tp_y,
                                                      tp_z=tp_z)
                self.flash_attention.shard(parallel_config)

    def _ulysses_initial(self):
        """initial ulysses related ops."""
        self.transpose_back = P.Transpose()
        self.transpose_ulysses = P.Transpose()
        self.transpose_a2a = P.Transpose()
        self.transpose_ulysses_merger_a2a = P.Transpose()
        self.transpose_ulysses_merger = P.Transpose()
        dp = self.data_parallel
        mp = self.model_parallel
        cp = self.context_parallel
        # ulysses shard strategy
        if self.is_first_iteration:
            self.wo.shard(((dp * cp, mp), (1, mp)), out_strategy_matmul=((dp * cp * mp, 1),))
            layout = Layout((dp, cp, mp), ("dp", "cp", "mp"))
            layout_transpose_back = (layout("dp", "mp", "cp", "None"),)
            self.transpose_back.shard(in_strategy=layout_transpose_back)
            self.transpose_ulysses.shard(((dp, cp, mp, 1, 1, 1),))
            self.transpose_a2a.shard(((dp, self.cp_co, self.cp_ds, mp, 1, 1),))
            self.transpose_ulysses_merger_a2a.shard(((dp, self.cp_co, self.cp_ds, mp, 1, 1),))
            self.transpose_ulysses_merger.shard(((dp, cp, 1, mp, 1, 1),))

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None, batch_valid_length=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, q_seq_lens=None, kv_mask=None,
                  seq_chunk=None, actual_seq_len=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim]
        if not self.rmsnorm_compute_2d:
            bs, seq_len, _ = self.shape(x)
        else:
            seq_len = self.seq_length
            bs = self.shape(x)[0] // seq_len
        if self.qkv_concat:
            qkv = self.cast(self.w_qkv(x), self.dtype)
            reshape_qkv = self.reshape(qkv, (bs, seq_len, self.n_kv_head, (self.n_rep + 2) * self.head_dim))
            query, key, value = self.split_qkv(reshape_qkv,
                                               (self.head_dim * self.n_rep,
                                                self.head_dim,
                                                self.head_dim), 3)
            query = self.reshape(query, (bs, seq_len, self.hidden_size))
            key = self.reshape(key, (bs, seq_len, self.kv_dim))
            value = self.reshape(value, (bs, seq_len, self.kv_dim))
        else:
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        # key and value for current token(s)
        if self.use_past:
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 freqs_cis, mask, prefix_keys_values=prefix_keys_values,
                                                 q_seq_lens=q_seq_lens)
        else:
            query = self.transpose(self.reshape(query, (bs, seq_len, self.n_head, self.head_dim)), (0, 2, 1, 3))
            key = self.transpose(self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
            query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, cp, 1
            # with ulysses context parallel, insert all to all before FA
            if self.input_layout == 'TND':
                query = self._merge_seq_len(query)
                key = self._merge_seq_len(key)
                value = self.reshape(value, (bs * seq_len, self.n_kv_head, self.head_dim))
            elif self.context_parallel > 1 and self.cp_ds > 1:
                # for query & key, transpose from BNSD back to BSND
                query = self.transpose_back(query, (0, 2, 1, 3))
                query = self._ulysses_q_a2a(query)
                key = self.transpose_back(key, (0, 2, 1, 3))
                key = self._ulysses_kv_a2a(key)
                # value is BSND, no need for transpose back
                value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))
                value = self._ulysses_kv_a2a(value)
            elif self.context_parallel > 1:
                query = self._merge_heads(query)
                key = self._merge_heads(key)
                if self.rl_config is not None:
                    value = self.reshape(value, (bs, seq_len, -1))
            else:
                value = self.transpose(self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim)), (0, 2, 1, 3))
                key, value = self._cat_prefix(key, value, prefix_keys_values)

            if self.seq_pipe:
                key = self.tile_kv(key, (1, 1, self.seq_split_num, 1))
                value = self.tile_kv(value, (1, 1, self.seq_split_num, 1))
                key = self.mul_kv(key, kv_mask)
                value = self.mul_kv(value, kv_mask)
                seq_zero = self.mul_update(kv_mask, 0)
                ones = self.not_equal_ones(seq_zero, -1)
                kv_equal = self.mul_update(ones, self.not_equal_seq(seq_chunk, self.seq_split_size))
                key_update = self.add_k(key, self.key_cache)
                value_update = self.add_v(value, self.value_cache)
                update_k = self.select(kv_equal, key_update, seq_zero)
                update_v = self.select(kv_equal, value_update, seq_zero)
                update_k = F.stop_gradient(update_k)
                update_v = F.stop_gradient(update_v)
                k_update = self.assign_kv(self.key_cache, update_k)
                v_update = self.assign_kv(self.value_cache, update_v)
                query = F.depend(query, k_update)
                query = F.depend(query, v_update)
                if self.context_parallel > 1:
                    context_layer = self.flash_attention(query, key_update, value_update, mask)
                else:
                    context_layer = self.flash_attention(query, key_update, value_update, mask)
                    context_layer = self._merge_heads(context_layer)
            else:
                if self.use_flash_attention:
                    # with ulysses context parallel, insert all to all after FA
                    if self.use_eod_attn_mask_compression:
                        context_layer = self.flash_attention(
                            query, key, value, mask,
                            actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
                        )
                    else:
                        context_layer = self.flash_attention(query, key, value, mask)
                    if self.input_layout == 'TND':
                        context_layer = self.reshape(context_layer, (bs, seq_len, -1))
                    elif self.context_parallel > 1 and self.cp_ds > 1:
                        context_layer = self._ulysses_context_layer_a2a(context_layer)
                    elif self.context_parallel <= 1:
                        context_layer = self._merge_heads(context_layer)
                else:
                    key = self._repeat_kv(key, self.n_rep)
                    value = self._repeat_kv(value, self.n_rep)
                    context_layer = self._attn(query, key, value, mask)

        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        if self.rmsnorm_compute_2d and self.rl_config is not None:
            context_layer = self.reshape(context_layer, (bs * seq_len, -1))
        output = self.wo(context_layer)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)
        return output

    def _cat_prefix(self, key, value, prefix_keys_values):
        r'''
        concat prefix_keys_values to key and value
        prefix_keys_values: shape(2, bs, pre_len, num_heads * kv_channels)
        '''
        if prefix_keys_values is not None:
            bs, n_kv_head, _, head_dim = key.shape
            past_key = prefix_keys_values[0]
            past_value = prefix_keys_values[1]
            past_key = self.transpose(self.reshape(past_key, (bs, -1, n_kv_head, head_dim)), (0, 2, 1, 3))
            past_value = self.transpose(self.reshape(past_value, (bs, -1, n_kv_head, head_dim)), (0, 2, 1, 3))
            past_key = self.cast(past_key, self.dtype)
            past_value = self.cast(past_value, self.dtype)
            cat = P.Concat(2)
            key = cat((past_key, key))
            value = cat((past_value, value))
        return key, value

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_seq_len(self, x):
        """
        convert a bhsd input to a tnd output

        Inputs:
            x: input tensor

        Output:
            x_merge: the tnd output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs * seq_len, n_head, head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

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
        if self.rmsnorm_compute_2d:
            if self.rl_config is not None:
                new_shape = (bs, seq_len, n_head * head_dim)
            else:
                new_shape = (bs * seq_len, n_head * head_dim)
        else:
            new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _ulysses_q_a2a(self, qkv):
        """Given a qkv tensor with shape of (bs, seq_len, n_head, head_dim),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape of (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all to all commu.
        """
        bs, seq_len, _, _ = F.shape(qkv)
        new_shape = (bs, seq_len, self.model_parallel, self.cp_ds, -1, self.head_dim)
        # [bs, seq_len, n_head, head_dim] -> [bs, seq_len, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [bs, seq_len, n_head/cp_ds, cp_ds, head_dim] -> [bs, seq_len, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4, 5))
        # insert all-to-all (dp, cp, 1, mp, 1) -> (dp, cp_co, cp_ds, mp, 1)
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4, 5))
        # reshape to BSH, here set -1 to H, for kv head could be different from q head
        qkv = F.reshape(qkv, (bs, seq_len, -1))
        return qkv

    def _ulysses_kv_a2a(self, qkv):
        """Given a qkv tensor with shape of (bs, seq_len, n_head, head_dim),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            qkv (Tensor): qkv after rotary embedding and before attention, with shape of (B, S, N, D)

        Returns:
            Tensor: qkv tensor after all to all commu.
        """
        bs, seq_len, _, _ = F.shape(qkv)
        new_shape = (bs, seq_len, self.model_parallel, self.cp_ds, -1, self.head_dim)
        # [bs, seq_len, n_head, head_dim] -> [bs, seq_len, n_head/cp_ds, cp_ds, head_dim]
        qkv = self.reshape(qkv, new_shape)
        # [bs, seq_len, n_head/cp_ds, cp_ds, head_dim] -> [bs, seq_len, cp_ds, n_head/cp_ds, head_dim]
        qkv = self.transpose_ulysses(qkv, (0, 1, 3, 2, 4, 5))
        # insert all-to-all (dp, cp, 1, mp, 1) -> (dp, cp_co, cp_ds, mp, 1)
        qkv = self.transpose_a2a(qkv, (0, 1, 2, 3, 4, 5))
        # reshape to BSH, here set -1 to H, for kv head could be different from q head
        qkv = F.reshape(qkv, (bs, seq_len, -1))
        return qkv

    def _ulysses_context_layer_a2a(self, context_layer):
        """Given the context_layer tensor after fa, with shape of (bs, seq_len, hidden_size),
        insert all to all in right place using transpose with specific shard strategy.
        refers to <https://arxiv.org/abs/2309.14509>

        Args:
            context_layer (Tensor): context layer after attention, with shape of (B, S, H)

        Returns:
            Tensor: context layer tensor after all to all commu.
        """
        bs, seq_len, _ = F.shape(context_layer)
        new_shape = (bs, seq_len, self.cp_ds, self.model_parallel, -1, self.head_dim)
        context_layer = F.reshape(context_layer, new_shape)
        # insert all-to-all back (dp, cp_co, cp_ds, mp, 1) -> (dp, cp, 1, mp, 1)
        context_layer = self.transpose_ulysses_merger_a2a(context_layer, (0, 1, 2, 3, 4, 5))
        context_layer = self.transpose_ulysses_merger(context_layer, (0, 1, 3, 2, 4, 5))
        # reshape back to BSH
        context_layer = F.reshape(context_layer, (bs, seq_len, self.hidden_size))
        return context_layer

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
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


# pylint: disable=C0326
class LLamaDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            seq_length (int): The sequence length of input.
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
            qkv_has_bias(bool): Whether Q/K/V in attention has bias or not.
            attn_proj_has_bias(bool): Whether projection in attention has bias or not.
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
            residual_dtype(str): The residual compute dtype. Default None .
            init_method_std(float): The sigma value when using normal type to initialize Linear. Default 0.01 .
            rmsnorm_compute_2d(bool): Whether to use 2D Add in RMS_NORM. Default: ``False`` .

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.
            - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
            - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, head_dim, seq_length),
              (batch_size, num_heads, seq_length, head_dim)).

    """

    @predict_lazy_inline
    def __init__(self,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 init_method_std: float = 0.01,
                 qkv_concat=False,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 residual_dtype=mstype.float32,
                 qkv_has_bias=False,
                 attn_proj_has_bias=False,
                 use_past=False,
                 is_dynamic=False,
                 use_rope_slice=False,
                 moe_config=None,
                 use_flash_attention=False,
                 use_ring_attention=False,
                 use_attn_mask_compression=False,
                 use_eod_attn_mask_compression=False,
                 rmsnorm_compute_2d=False,
                 batch_size=1,
                 block_size: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 parallel_config=TransformerOpParallelConfig(),
                 parallel_decoding=False,
                 rl_config=None,
                 fused_kernel=True,
                 chunk_prefill=False,
                 use_3d_tensor_parallel=False,
                 tp_x=1,
                 tp_y=1,
                 tp_z=1
                 ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = divide(self.hidden_size, self.n_head)
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.chunk_prefill = chunk_prefill
        self.seq_pipe = parallel_config and parallel_config.seq_split_num > 1
        self.cast = P.Cast()

        self.residual_dtype = residual_dtype
        self.residual_cast_flag = residual_dtype != compute_dtype
        if self.residual_cast_flag:
            logger.info(f"residual cast flag: {self.residual_cast_flag}, residual dtype: {residual_dtype}")
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.add = P.Add()
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype,
                                     fused_kernel=fused_kernel)
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype,
                                           fused_kernel=fused_kernel)
        self.attention = LLamaAttention(seq_length=seq_length,
                                        dim=dim,
                                        n_heads=n_heads,
                                        n_kv_heads=n_kv_heads,
                                        qkv_concat=qkv_concat,
                                        compute_dtype=compute_dtype,
                                        softmax_compute_dtype=softmax_compute_dtype,
                                        rotary_dtype=rotary_dtype,
                                        param_init_type=param_init_type,
                                        qkv_has_bias=qkv_has_bias,
                                        attn_proj_has_bias=attn_proj_has_bias,
                                        use_past=use_past,
                                        is_dynamic=is_dynamic,
                                        use_rope_slice=use_rope_slice,
                                        use_flash_attention=use_flash_attention,
                                        use_ring_attention=use_ring_attention,
                                        use_attn_mask_compression=use_attn_mask_compression,
                                        use_eod_attn_mask_compression=use_eod_attn_mask_compression,
                                        rmsnorm_compute_2d=rmsnorm_compute_2d,
                                        block_size=block_size,
                                        num_blocks=num_blocks,
                                        batch_size=batch_size,
                                        parallel_config=parallel_config,
                                        parallel_decoding=parallel_decoding,
                                        rl_config=rl_config,
                                        init_method_std=init_method_std,
                                        chunk_prefill=chunk_prefill,
                                        use_3d_tensor_parallel=use_3d_tensor_parallel,
                                        tp_x=tp_x,
                                        tp_y=tp_y,
                                        tp_z=tp_z
                                        )

        self.expert_num = 1 if moe_config is None else moe_config.expert_num
        self.shared_expert_num = 0 if moe_config is None else moe_config.shared_expert_num
        # set kbk infer for moe structural models.
        self.use_moe_infer = use_past and (self.expert_num > 1)
        self.use_gmm = moe_config.use_gmm if moe_config else False
        if self.use_moe_infer:
            ffn = LlamaMoeInferFeedForward(dim=self.hidden_size,
                                           intermediate_size=intermediate_size,
                                           hidden_dim=4 * self.hidden_size,
                                           multiple_of=multiple_of,
                                           expert_num=self.expert_num,
                                           ffn_dim_multiplier=ffn_dim_multiplier,
                                           compute_dtype=compute_dtype,
                                           param_init_type=param_init_type,
                                           use_gmm=self.use_moe_infer,
                                           init_method_std=init_method_std)
        else:
            ffn = LlamaFeedForward(dim=self.hidden_size,
                                   intermediate_size=intermediate_size,
                                   hidden_dim=4 * self.hidden_size,
                                   multiple_of=multiple_of,
                                   expert_num=self.expert_num,
                                   ffn_dim_multiplier=ffn_dim_multiplier,
                                   compute_dtype=compute_dtype,
                                   param_init_type=param_init_type,
                                   ffn_concat=qkv_concat,
                                   parallel_config=parallel_config,
                                   moe_config=moe_config,
                                   init_method_std=init_method_std,
                                   rmsnorm_compute_2d=rmsnorm_compute_2d,
                                   use_3d_tensor_parallel=use_3d_tensor_parallel,
                                   tp_x=tp_x,
                                   tp_y=tp_y,
                                   tp_z=tp_z) if (self.shared_expert_num == 0 and not self.use_gmm) else None
        if self.expert_num == 1:
            self.feed_forward = ffn
        else:
            if self.shared_expert_num == 0:
                if self.use_moe_infer:
                    self.feed_forward = MoEInfer(
                        ffn=ffn,
                        dim=self.hidden_size,
                        moe_config=moe_config,
                        parallel_config=parallel_config)
                elif self.use_gmm:
                    self.feed_forward = MoEV3(
                        dim=self.hidden_size,
                        intermediate_size=intermediate_size,
                        compute_dtype=compute_dtype,
                        param_init_type=param_init_type,
                        moe_config=moe_config,
                        parallel_config=parallel_config
                    )
                else:
                    self.feed_forward = MoEV2(
                        ffn=ffn,
                        dim=self.hidden_size,
                        moe_config=moe_config,
                        parallel_config=parallel_config)
            else:
                self.feed_forward = LlamaFeedForwardWithMoE(self.hidden_size,
                                                            intermediate_size=intermediate_size,
                                                            compute_dtype=compute_dtype,
                                                            param_init_type=param_init_type,
                                                            moe_config=moe_config,
                                                            parallel_config=parallel_config,
                                                            use_moe_infer=self.use_moe_infer,
                                                            init_method_std=init_method_std)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        if use_3d_tensor_parallel:
            layout_ndtp = Layout((dp, cp, tp_z, tp_x, tp_y), ("dp", "cp", "z", "x", "y"))
        if self.expert_num == 1:
            self.feed_forward.shard(parallel_config)
        elif self.shared_expert_num == 0:
            self.feed_forward.ffn.shard(parallel_config)
        else:
            self.feed_forward.shard(parallel_config)
        if not rmsnorm_compute_2d:
            self.add.shard(((dp, cp, 1), (dp, cp, 1)))
        else:
            self.add.shard(((dp * cp, 1), (dp * cp, 1)))
        if cp > 1:
            if not rmsnorm_compute_2d:
                self.attention_norm.shard((dp, cp * mp, 1))
                self.ffn_norm.shard((dp, cp * mp, 1))
            else:
                self.attention_norm.shard((dp * cp * mp, 1))
                self.ffn_norm.shard((dp * cp * mp, 1))
        elif rmsnorm_compute_2d:
            self.attention_norm.shard((dp, 1))
            self.ffn_norm.shard((dp, 1))
        else:
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))

        if moe_config is None or not moe_config.expert_num > 1:
            if not rmsnorm_compute_2d:
                if use_3d_tensor_parallel:
                    self.feed_forward.mul.shard((layout_ndtp("dp", ("cp", "z", "x"), "y"),
                                                 layout_ndtp("dp", ("cp", "z", "x"), "y")))
                else:
                    self.feed_forward.mul.shard(((dp, cp, mp), (dp, cp, mp)))
            else:
                if use_3d_tensor_parallel:
                    self.feed_forward.mul.shard((layout_ndtp(("dp", "cp", "z", "x"), "y"),
                                                 layout_ndtp(("dp", "cp", "z", "x"), "y")))
                else:
                    self.feed_forward.mul.shard(((dp * cp, mp), (dp * cp, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            if not rmsnorm_compute_2d:
                if use_3d_tensor_parallel:
                    self.add.shard((layout_ndtp("dp", ("cp", "z", "y"), "x"),
                                    layout_ndtp("dp", ("cp", "z", "y"), "x")))
                    self.attention_norm.shard_layout(layout_ndtp("dp", ("cp", "z", "y"), "x"), layout_ndtp("x", ))
                    self.ffn_norm.shard_layout(layout_ndtp("dp", ("cp", "z", "y"), "x"), layout_ndtp("x", ))
                else:
                    self.add.shard(((dp, cp * mp, 1), (dp, cp * mp, 1)))
                    self.attention_norm.shard((dp, cp * mp, 1))
                    self.ffn_norm.shard((dp, cp * mp, 1))
            else:
                if use_3d_tensor_parallel:
                    self.add.shard((layout_ndtp(("dp", "cp", "z", "y"), "x"),
                                    (("dp", "cp", "z", "y"), "x")))
                    self.attention_norm.shard_layout(layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("x", ))
                    self.ffn_norm.shard_layout(layout_ndtp(("dp", "cp", "z", "y"), "x"), layout_ndtp("x", ))
                else:
                    self.add.shard(((dp * mp * cp, 1), (dp * mp * cp, 1)))
                    self.attention_norm.shard((dp * mp * cp, 1))
                    self.ffn_norm.shard((dp * mp * cp, 1))
            if moe_config is None or not moe_config.expert_num > 1 and not use_3d_tensor_parallel:
                self.feed_forward.w2.shard(((dp * cp, mp), (1, mp)), out_strategy_matmul=((dp * cp * mp, 1),))
        self.predict_run_mode = get_predict_run_mode()
        if self.predict_run_mode:
            self.no_inline = False

    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None, slot_mapping=None,
                  prefix_keys_values=None, q_seq_lens=None, kv_mask=None, seq_chunk=None, actual_seq_len=None):
        """ Forward of transformer block. """
        if not self.use_past:
            self._check_input(x, freqs_cis, mask)
        if actual_seq_len is not None:
            _check_input_dtype(actual_seq_len.dtype, "actual_seq_len",
                               [mstype.int32, mstype.int64], self.cls_name)
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        if self.seq_pipe:
            h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables, slot_mapping,
                               prefix_keys_values, q_seq_lens, kv_mask, seq_chunk, actual_seq_len=actual_seq_len)
        else:
            h = self.attention(input_x, freqs_cis, mask, batch_valid_length, block_tables,
                               slot_mapping, prefix_keys_values, q_seq_lens, actual_seq_len=actual_seq_len)
        if self.residual_cast_flag:
            x = self.cast(x, self.residual_dtype)
            h = self.cast(h, self.residual_dtype)

        h = self.add(x, h)
        if self.residual_cast_flag:
            h = self.cast(h, self.dtype)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)
        if self.residual_cast_flag:
            h = self.cast(h, self.residual_dtype)
            ffn_out = self.cast(ffn_out, self.residual_dtype)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        if self.residual_cast_flag:
            out = self.cast(out, self.dtype)
        return out

    def _check_input(self, x, freqs_cis, mask):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if swap_mask is not None:
            _check_input_dtype(swap_mask.dtype, "swap_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16, mstype.uint8, mstype.bool_],
                               self.cls_name)
        return True
