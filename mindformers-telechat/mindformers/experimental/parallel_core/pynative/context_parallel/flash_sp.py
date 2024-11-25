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
"""Ring Attention APIs."""
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.ops import Send, Receive
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from mindformers.experimental.parallel_core.pynative.parallel_state import get_context_parallel_group, \
    get_context_parallel_world_size, get_context_parallel_rank, get_sp_send_stream


class FlashSP(nn.Cell):
    """Attention implementation with sequence parallelism
    This function contains the ring attention primitives used in FlashSP
    Specifically, it includes an interface for calling FlashSP operation.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D
    Args:
        head_num (int): The head num of query.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        input_layout (str): Specifies the layout of input `query`, key and value. Currently only the value
        "BSH" is supported. Default: "BSH". Currently only input_layout = "BSH" is supported.
        sparse_mode (int): Indicates sparse mode. Default 0. Currently only sparse_mode = 0 is supported.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        use_attention_mask (bool): The value is True if attention_mask is passed. Default: False.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        Currently only use_alibi_mask = False is supported.
        use_mqa (bool): Specifies whether using MQA. Default: False. Currently only use_mqa = False is supported.
        dp (int): Data parallel num.
        mp (int): Model parallel num. Currently only mp = 1 is supported.
        sp (int): Sequence parallel num.


    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)`.
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048). Currently only attn_mask = None is supported. Please use attn_mask_type to indicate the mask.
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization. Currently only alibi_mask = None is supported.
          Input tensor of shape :math: `(B, N1, S1, S2)`, `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`
          or (1024, 1024).
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
          Currently only padding_mask = None is supported.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`. Currently only prefix = None is supported.
        - **attn_mask_type** (str) - The attention mask type. Currently only value of "causal" is supported.
          The value "causal" indicates the causal mask is used.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    """

    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 input_layout="BSH",
                 sparse_mode=0,
                 use_attention_mask=False,
                 use_alibi_mask=False,
                 use_mqa=False,
                 dp=1,
                 mp=1,
                 sp=1
                 ):
        super(FlashSP, self).__init__()
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.use_attention_mask = use_attention_mask
        self.use_alibi_mask = use_alibi_mask
        self.use_mqa = use_mqa
        self.dp = dp
        self.mp = mp
        self.sp = sp

        if sp == 1:
            raise ValueError(f"Only sp > 1 is supported. For sp == 1, please use FlashAttentionScore")

        if sparse_mode != 0:
            raise ValueError(f"Only sparse_mode = 0 is supported")

        if input_layout != "BSH":
            raise ValueError(f"Only input_layout = 'BSH' is supported")

        init_sp = get_context_parallel_world_size()
        if sp != init_sp:
            raise ValueError(f"The sp group is initialized as {init_sp},"
                             f"but got different sp = {sp} in FlashSP parameters")

        if self.use_alibi_mask:
            raise ValueError(f"Only use_alibi_mask = False is supported")

        if self.use_mqa:
            raise ValueError(f"Only use_mqa = False is supported")

        if self.mp != 1:
            raise ValueError(f"Only mp = 1 is supported")

        self.flash_attention = FlashAttentionScore(head_num=self.head_num,
                                                   keep_prob=self.keep_prob,
                                                   scale_value=self.scale_value,
                                                   pre_tokens=self.pre_tokens,
                                                   next_tokens=self.next_tokens,
                                                   input_layout=self.input_layout,
                                                   inner_precise=0,
                                                   sparse_mode=self.sparse_mode)
        self.stream_send_qkv = get_sp_send_stream()
        self.stream_recv_qkv = get_sp_send_stream()
        self.stream_send_oml = get_sp_send_stream()
        self.stream_recv_oml = get_sp_send_stream()

    def p2p_send_communicate(self, send_tensor, send_dst,
                             sp_group, stream, send_ops, sr_tag):
        """Point-to-point communications of QKV and attn_out in Attention with sequence parallelism"""

        send_op = Send(sr_tag, send_dst, group=sp_group)
        send_op.add_prim_attr("dtype", mstype.float16)
        with ms.hal.StreamCtx(stream):
            send_op(send_tensor)
        send_ops.append(stream)
        return send_ops

    def p2p_recv_communicate(self, recv_tensor, recv_src,
                             sp_group, stream, recv_ops, sr_tag):
        """Point-to-point communications of QKV and attn_out in Attention with sequence parallelism"""

        recv_op = Receive(sr_tag, recv_src, shape=recv_tensor.shape, dtype=recv_tensor.dtype, group=sp_group)
        with ms.hal.StreamCtx(stream):
            recv_tensor = recv_op(Tensor(0.0, dtype=mstype.float16))
        recv_ops.append(stream)
        return recv_ops, recv_tensor

    def forward_update(self, prev_attn_out, prev_softmax_max, prev_softmax_sum,
                       cur_attn_out, cur_softmax_max, cur_softmax_sum):
        '''Updata ring attention output'''
        softmax_max = ops.maximum(prev_softmax_max, cur_softmax_max)
        prev_scale = ops.exp(prev_softmax_max - softmax_max)
        cur_scale = ops.exp(cur_softmax_max - softmax_max)

        prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
        cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
        softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

        prev_out_scale = prev_softmax_sum_scaled / softmax_sum
        cur_out_scale = cur_softmax_sum_scaled / softmax_sum

        n = prev_out_scale.shape[1]

        if self.input_layout == "BSH" or self.input_layout == "SBH":
            h = prev_attn_out.shape[-1]
            d = h // n
        elif self.input_layout == "BNSD":
            h = prev_attn_out.shape[1]
            d = prev_attn_out.shape[-1]
        else:
            raise ValueError(f"Only input_layout = 'BSH' or 'SBH' or 'BNSD' is supported")

        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        if self.input_layout == "BSH":
            prev_out_scale = ops.transpose(prev_out_scale, (0, 2, 1, 3))
            prev_out_scale = prev_out_scale.reshape(prev_out_scale.shape[0], prev_out_scale.shape[1], -1)
        elif self.input_layout == "SBH":
            prev_out_scale = ops.transpose(prev_out_scale, (2, 0, 1, 3))
            prev_out_scale = prev_out_scale.reshape(prev_out_scale.shape[0], prev_out_scale.shape[1], -1)
        elif self.input_layout == "BNSD":
            pass
        else:
            raise ValueError(f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        if self.input_layout == "BSH":
            cur_out_scale = ops.transpose(cur_out_scale, (0, 2, 1, 3))
            cur_out_scale = cur_out_scale.reshape(cur_out_scale.shape[0], cur_out_scale.shape[1], -1)
        elif self.input_layout == "SBH":
            cur_out_scale = ops.transpose(cur_out_scale, (2, 0, 1, 3))
            cur_out_scale = cur_out_scale.reshape(cur_out_scale.shape[0], cur_out_scale.shape[1], -1)
        elif self.input_layout == "BNSD":
            pass
        else:
            raise ValueError(f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale

        return attn_out, softmax_max, softmax_sum

    def check_parameter(self, q, k, v, attn_mask, alibi_mask, prefix, padding_mask, attn_mask_type):
        '''check ring attention intput'''
        if attn_mask is not None:
            raise ValueError(f"Only attn_mask = None is supported")
        if attn_mask_type != "causal":
            raise ValueError(f"Only attn_mask_type = 'causal' is supported")
        if alibi_mask is not None:
            raise ValueError(f"Only alibi_mask = None is supported")
        if prefix is not None:
            raise ValueError(f"Only prefix = None is supported")
        if padding_mask is not None:
            raise ValueError(f"Only padding_mask = None is supported")

        s1 = q.shape[1]
        s2 = k.shape[1]
        s3 = v.shape[1]
        if s2 != s3:
            raise ValueError(f"The sequence length of input k and v should be equal, but got {s2} and {s3}")
        if s2 % 2 != 0:
            raise ValueError(f"The sequence length of input k and v must be an even number, but got {s2} and {s3}")

        b1 = q.shape[0]
        b2 = k.shape[0]
        b3 = v.shape[0]
        if b2 != b1:
            raise ValueError(f"The batch size of input q is not equal to k, but got batch size {b1} and {b2}")
        if b3 != b1:
            raise ValueError(f"The batch size of input q is not equal to v, but got batch_size {b1} and {b3}")

        if self.pre_tokens < s1 or self.pre_tokens < s2:
            raise ValueError(f"The pre_tokens should be larger or equal to the sequence of q and k,"
                             f"but got pre_tokens is {self.pre_tokens}, and the sequence length of q is {s1}"
                             f"and sequence length of kv is {s2}")

        if self.next_tokens < s1 or self.next_tokens < s2:
            raise ValueError(f"The next_tokens should be larger or equal to the sequence of q and k,"
                             f"but got next_tokens is {self.next_tokens}, and the sequence length of q is {s1}"
                             f"and sequence length of kv is {s2}")

        if self.next_tokens < 0:
            raise ValueError(f"The next_tokens should be larger or equal to 0, but got {self.next_tokens}")

    def get_loop_steps(self, sp_size):
        return sp_size // 2 + 1

    def get_send_qkv_dst_rank(self, rank, step, sp_size):
        return (rank + step + 1) % sp_size

    def get_recv_qkv_src_rank(self, rank, step, sp_size):
        return (rank + sp_size - step - 1) % sp_size

    def get_send_oml_dst_rank(self, rank, step, sp_size):
        return (rank + sp_size - step) % sp_size

    def get_recv_oml_src_rank(self, rank, step, sp_size):
        return (rank + step) % sp_size

    def get_rank_index(self, rank, step, sp_size):
        rank_list = [i for i in range(sp_size)]
        rank_order = []
        for i, r in enumerate(rank_list):
            if i % (step + 1) == 0:
                rank_order.append(r)
        for i in range(1, step + 1):
            for j in range(i, len(rank_list), step + 1):
                rank_order.append(rank_list[j])
        return rank_order.index(rank)

    def construct_send_oml_tensor(self, softmax_max_, softmax_sum_, attn_out_):
        '''construct oml tensor for send'''
        send_softmax_max = softmax_max_[:, :, :, 0]
        send_softmax_max = ops.transpose(send_softmax_max, (0, 2, 1))
        send_softmax_sum = softmax_sum_[:, :, :, 0]
        send_softmax_sum = ops.transpose(send_softmax_sum, (0, 2, 1))
        send_softmax_max = send_softmax_max.astype(mstype.float16)
        send_softmax_sum = send_softmax_sum.astype(mstype.float16)
        attn_out_ = attn_out_.astype(mstype.float16)
        send_oml_tensor = ops.cat((attn_out_, send_softmax_max), axis=-1)
        send_oml_tensor = ops.cat((send_oml_tensor, send_softmax_sum), axis=-1)
        return send_oml_tensor

    def dismantle_recv_oml_tensor(self, recv_oml_tensor, hidden_dim):
        attn_out_ = recv_oml_tensor[:, :, 0:hidden_dim]
        tmp_softmax_max = recv_oml_tensor[:, :, hidden_dim:hidden_dim + self.head_num]
        tmp_softmax_sum = recv_oml_tensor[:, :, hidden_dim + self.head_num:]
        tmp_softmax_max = ops.transpose(tmp_softmax_max, (0, 2, 1))
        tmp_softmax_sum = ops.transpose(tmp_softmax_sum, (0, 2, 1))
        softmax_max_ = tmp_softmax_max.unsqueeze(-1).repeat(8, axis=-1)
        softmax_sum_ = tmp_softmax_sum.unsqueeze(-1).repeat(8, axis=-1)
        return attn_out_, softmax_max_, softmax_sum_

    def get_recv_oml_tensor_shape(self, q):
        return (q.shape[0], q.shape[1], q.shape[2] + 2 * self.head_num)

    def prepare_comm(self, q):
        '''prepare communication for send/recv'''
        if self.send_qkv_ops:
            for send_qkv_op in self.send_qkv_ops:
                send_qkv_op.synchronize()
            self.send_qkv_ops.clear()

        if self.send_oml_ops:
            for send_oml_op in self.send_oml_ops:
                send_oml_op.synchronize()
            self.send_oml_ops.clear()

        if self.recv_qkv_ops:
            for recv_qkv_op in self.recv_qkv_ops:
                recv_qkv_op.synchronize()
            self.recv_qkv_ops.clear()

        if self.recv_oml_ops:
            for recv_oml_op in self.recv_oml_ops:
                recv_oml_op.synchronize()
            self.recv_oml_ops.clear()

            cur_attn_out, cur_softmax_max, cur_softmax_sum = self.dismantle_recv_oml_tensor(
                self.recv_oml_tensor, q.shape[-1])
            attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                self.attn_out, self.softmax_max, self.softmax_sum,
                cur_attn_out, cur_softmax_max, cur_softmax_sum)
            self.attn_out, self.softmax_max, self.softmax_sum = attn_out_updated, softmax_max_updated, \
                softmax_sum_updated

    def prepare_qkv(self, step, step_, inner_loop_steps, loop_steps, attn_mask_start, attn_mask_end,
                    q, k, v, kv, rank, sp_size, len_k):
        '''prepare qkv content'''
        cur_attn_mask = None
        if inner_loop_steps * step + step_ == 0:
            cur_attn_mask = attn_mask_start
            cur_q = q
            cur_k = k
            cur_v = v
        elif inner_loop_steps * step + step_ == inner_loop_steps * loop_steps - 2:
            cur_attn_mask = attn_mask_end
            cur_q = q
            cur_k = k
            cur_v = v
        else:
            if rank < sp_size // 2:
                if inner_loop_steps * rank < inner_loop_steps * step + step_ < inner_loop_steps * loop_steps - 2:
                    cur_q = self.recv_qkv_tensor
                    cur_kv = kv[(step_ + 1) % inner_loop_steps]
                    cur_k, cur_v = cur_kv[0], cur_kv[1]
                else:
                    cur_q = q
                    cur_k, cur_v = self.recv_qkv_tensor[:, 0:len_k // 2, :], self.recv_qkv_tensor[:, len_k // 2:, :]
            else:
                cur_q = q
                cur_k, cur_v = self.recv_qkv_tensor[:, 0:len_k // 2, :], self.recv_qkv_tensor[:, len_k // 2:, :]
        return cur_q, cur_k, cur_v, cur_attn_mask

    def construct_qkv_comm(self, step, step_, inner_loop_steps, loop_steps, q, send_kv, rank, sp_size, sp_group):
        '''construct send/recv qkv tensors for communication'''
        send_qkv_flag = 0
        receive_qkv_flag = 0
        rank_idx = self.get_rank_index(rank, step, sp_size)
        send_qkv_dst_rank = self.get_send_qkv_dst_rank(rank, step, sp_size)
        if rank + step + 1 >= sp_size:
            if step_ % inner_loop_steps == 0:
                self.send_qkv_tensor = q
                send_qkv_flag = 1
        else:
            if step < loop_steps - 2:
                self.send_qkv_tensor = send_kv[(step_ + 1) % inner_loop_steps]
                send_qkv_flag = 1
            else:
                if step_ % inner_loop_steps == 0:
                    self.send_qkv_tensor = send_kv[(step_ + 1) % inner_loop_steps]
                    send_qkv_flag = 1
        recv_qkv_src_rank = self.get_recv_qkv_src_rank(rank, step, sp_size)
        if rank + sp_size - step - 1 < sp_size:
            if step_ % inner_loop_steps == 0:
                self.recv_qkv_tensor = ms.numpy.empty_like(q)
                receive_qkv_flag = 1
        else:
            if step < loop_steps - 2:
                self.recv_qkv_tensor = ms.numpy.empty_like(send_kv[(step_ + 1) % inner_loop_steps])
                receive_qkv_flag = 1
            else:
                if step_ % inner_loop_steps == 0:
                    self.recv_qkv_tensor = ms.numpy.empty_like(send_kv[(step_ + 1) % inner_loop_steps])
                    receive_qkv_flag = 1

        if rank_idx % 2 == 0:
            if send_qkv_flag == 1:
                self.send_qkv_ops = self.p2p_send_communicate(
                    self.send_qkv_tensor, send_qkv_dst_rank, sp_group, self.stream_send_qkv, self.send_qkv_ops, 0)
            if receive_qkv_flag == 1:
                self.recv_qkv_ops, self.recv_qkv_tensor = self.p2p_recv_communicate(
                    self.recv_qkv_tensor, recv_qkv_src_rank, sp_group, self.stream_recv_qkv, self.recv_qkv_ops, 0)
        else:
            if receive_qkv_flag == 1:
                self.recv_qkv_ops, self.recv_qkv_tensor = self.p2p_recv_communicate(
                    self.recv_qkv_tensor, recv_qkv_src_rank, sp_group, self.stream_recv_qkv, self.recv_qkv_ops, 0)
            if send_qkv_flag == 1:
                self.send_qkv_ops = self.p2p_send_communicate(
                    self.send_qkv_tensor, send_qkv_dst_rank, sp_group, self.stream_send_qkv, self.send_qkv_ops, 0)

    def construct_oml_comm(self, step, step_, inner_loop_steps, loop_steps, q, rank, sp_size, sp_group):
        '''construct send/recv oml tensors for communication'''
        if rank < sp_size // 2:
            send_oml_dst_rank = self.get_send_oml_dst_rank(rank, step, sp_size)
            if rank + sp_size - step < sp_size:
                if step < loop_steps - 1:
                    if step_ % inner_loop_steps == 1:
                        self.send_oml_ops = self.p2p_send_communicate(
                            self.send_oml_tensor, send_oml_dst_rank, sp_group,
                            self.stream_send_oml, self.send_oml_ops, 1)
                else:
                    if step_ % inner_loop_steps == 0:
                        self.send_oml_ops = self.p2p_send_communicate(
                            self.send_oml_tensor, send_oml_dst_rank, sp_group,
                            self.stream_send_oml, self.send_oml_ops, 1)
        else:
            recv_oml_src_rank = self.get_recv_oml_src_rank(rank, step, sp_size)
            if rank + step >= sp_size:
                if step < loop_steps - 1:
                    if step_ % inner_loop_steps == 1:
                        self.recv_oml_tensor = ms.numpy.empty(self.get_recv_oml_tensor_shape(q),
                                                              dtype=mstype.float16)
                        self.recv_oml_ops, self.recv_oml_tensor = self.p2p_recv_communicate(
                            self.recv_oml_tensor, recv_oml_src_rank, sp_group,
                            self.stream_recv_oml, self.recv_oml_ops, 1)
                else:
                    if step_ % inner_loop_steps == 0:
                        self.recv_oml_tensor = ms.numpy.empty(self.get_recv_oml_tensor_shape(q),
                                                              dtype=mstype.float16)
                        self.recv_oml_ops, self.recv_oml_tensor = self.p2p_recv_communicate(
                            self.recv_oml_tensor, recv_oml_src_rank, sp_group,
                            self.stream_recv_oml, self.recv_oml_ops, 1)

    def update_out(self, step, step_, inner_loop_steps, loop_steps, cur_attn_out, cur_softmax_max,
                   cur_softmax_sum, rank, sp_size):
        '''update attention out'''
        if inner_loop_steps * step + step_ == 0:
            self.attn_out = cur_attn_out
            self.softmax_max = cur_softmax_max
            self.softmax_sum = cur_softmax_sum
        elif inner_loop_steps * step + step_ == inner_loop_steps * loop_steps - 2:
            attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                self.attn_out, self.softmax_max, self.softmax_sum,
                cur_attn_out, cur_softmax_max, cur_softmax_sum)
            self.attn_out, self.softmax_max, self.softmax_sum = attn_out_updated, \
                softmax_max_updated, softmax_sum_updated
        else:
            if rank < sp_size // 2:
                if inner_loop_steps * rank < inner_loop_steps * step + step_ < inner_loop_steps * loop_steps - 2:
                    if step_ % inner_loop_steps == 0:
                        self.send_attn_out, self.send_softmax_max, self.send_softmax_sum = self.forward_update(
                            self.send_attn_out, self.send_softmax_max, self.send_softmax_sum,
                            cur_attn_out, cur_softmax_max, cur_softmax_sum)
                    else:
                        self.send_attn_out, self.send_softmax_max, self.send_softmax_sum = cur_attn_out, \
                            cur_softmax_max, cur_softmax_sum
                    self.send_oml_tensor = self.construct_send_oml_tensor(
                        self.send_softmax_max, self.send_softmax_sum, self.send_attn_out)
                else:
                    attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                        self.attn_out, self.softmax_max, self.softmax_sum,
                        cur_attn_out, cur_softmax_max, cur_softmax_sum)
                    self.attn_out, self.softmax_max, self.softmax_sum = attn_out_updated, \
                        softmax_max_updated, softmax_sum_updated
            else:
                attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                    self.attn_out, self.softmax_max, self.softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum)
                self.attn_out, self.softmax_max, self.softmax_sum = attn_out_updated, \
                    softmax_max_updated, softmax_sum_updated

    def construct(self, q, k, v, attn_mask=None, alibi_mask=None, prefix=None, padding_mask=None,
                  attn_mask_type='causal'):
        '''Forward of FlashSP block'''
        self.check_parameter(q, k, v, attn_mask, alibi_mask, prefix, padding_mask, attn_mask_type)
        sp_group = get_context_parallel_group()
        sp_size = get_context_parallel_world_size()
        rank = get_context_parallel_rank()

        self.send_qkv_ops = []
        self.recv_qkv_ops = []
        self.send_oml_ops = []
        self.recv_oml_ops = []

        len_k = k.shape[1]
        len_v = v.shape[1]
        k_a = k[:, 0:len_k // 2, :]
        k_b = k[:, len_k // 2:, :]
        v_a = v[:, 0:len_v // 2, :]
        v_b = v[:, len_v // 2:, :]

        kv_a = [k_a, v_a]
        kv_b = [k_b, v_b]
        kv = [kv_a, kv_b]

        send_kv_a = ops.cat((k_a, v_a), axis=1)
        send_kv_b = ops.cat((k_b, v_b), axis=1)
        send_kv = [send_kv_a, send_kv_b]

        drop_mask = None
        if attn_mask is None:
            one_mask = ops.ones((q.shape[1] // 2, k.shape[1] // 2), dtype=mstype.uint8)
            zero_mask = ops.zeros((q.shape[1] // 2, k.shape[1] // 2), dtype=mstype.uint8)
            triu_mask = ops.triu(one_mask, diagonal=1)
            attn_mask_start = ops.cat((ops.cat((triu_mask, one_mask), axis=1),
                                       ops.cat((one_mask, triu_mask), axis=1)), axis=0)
            attn_mask_end = ops.cat((ops.cat((one_mask, one_mask), axis=1),
                                     ops.cat((zero_mask, one_mask), axis=1)), axis=0)

        self.attn_out, self.softmax_max, self.softmax_sum = None, None, None
        self.send_attn_out, self.send_softmax_max, self.send_softmax_sum = None, None, None
        self.send_oml_tensor = None
        self.recv_oml_tensor = None
        self.send_qkv_tensor = None
        self.recv_qkv_tensor = None

        loop_steps = self.get_loop_steps(sp_size)
        inner_loop_steps = 2
        for step in range(loop_steps):
            for step_ in range(inner_loop_steps):
                self.prepare_comm(q)

                if inner_loop_steps * step + step_ == inner_loop_steps * loop_steps - 1:
                    continue

                cur_q, cur_k, cur_v, cur_attn_mask = self.prepare_qkv(step, step_, inner_loop_steps, loop_steps,
                                                                      attn_mask_start, attn_mask_end,
                                                                      q, k, v, kv, rank, sp_size, len_k)
                if step < loop_steps - 1:
                    self.construct_qkv_comm(step, step_, inner_loop_steps, loop_steps,
                                            q, send_kv, rank, sp_size, sp_group)

                if step > 0:
                    self.construct_oml_comm(step, step_, inner_loop_steps, loop_steps,
                                            q, rank, sp_size, sp_group)

                all_att_outs = self.flash_attention(cur_q,
                                                    cur_k,
                                                    cur_v,
                                                    alibi_mask,
                                                    drop_mask,
                                                    padding_mask,
                                                    cur_attn_mask,
                                                    prefix)

                cur_attn_out = all_att_outs[3]
                cur_softmax_max = all_att_outs[0]
                cur_softmax_sum = all_att_outs[1]

                self.update_out(step, step_, inner_loop_steps, loop_steps, cur_attn_out,
                                cur_softmax_max, cur_softmax_sum, rank, sp_size)
        return self.attn_out
