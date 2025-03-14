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
"""DeepseekV3 models' APIs."""
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from deepseek2_model import DeepseekV2ForCausalLM


class TrainingDeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    r"""
    Provide DeepseekV3 training loss or logits through network.
    Args:
        config (DeepseekV3Config): The config of DeepseekV3 model.

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

    def __init__(self, config):
        super(TrainingDeepseekV3ForCausalLM, self).__init__(config)
        self.mtp_depth = config.mtp_depth
        self.mtp_loss_factor = config.mtp_loss_factor
        self.split = P.Split(axis=1, output_num=1 + self.mtp_depth)
        self.slice = P.StridedSlice()
        self.concat_2d = P.Concat(axis=-1)
        self.zeros_op = P.Zeros()

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.split.shard(((dp, 1, mp),))
        self.slice.shard(((dp, 1),))
        self.concat_2d.shard(((dp, 1), (dp, 1)))
        self.zeros_op.shard(((dp, 1),))

        self.input_sliced_sig = config.input_sliced_sig
        self.seq_split_num = config.parallel_config.seq_split_num
        self.seq_pipe = self.seq_split_num > 1

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None):
        """DeepseekV2ForCausalLM forward.
        """
        bsz, seqlen = self.shape(input_ids)
        if self.input_sliced_sig:
            tokens = input_ids
        else:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))  # (B, S)

        output, extra_loss = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables,
                                        slot_mapping, prefix_keys_values, self.init_extra_loss)
        logits = self.lm_head(output)  # (B, 3S, V)
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)  # (B, S)

        split_logits = self.split(logits)
        if self.seq_pipe:
            numerator, denominator = self.loss(self.reshape(split_logits[0], (-1, split_logits[0].shape[-1])),
                                               self.reshape(labels, (-1,)),
                                               self.reshape(input_mask, (-1,)))
            numerator1 = 0.0
            denominator1 = 0.0
            for i in range(self.mtp_depth):
                labels = self._shift_and_pad(labels)
                input_mask = self._shift_and_pad(input_mask)
                numerator_i, denominator_i = self.loss(self.reshape(split_logits[i + 1],
                                                                    (-1, split_logits[i + 1].shape[-1])),
                                                       self.reshape(labels, (-1,)),
                                                       self.reshape(input_mask, (-1,)))
                numerator_i = numerator_i * self.mtp_loss_factor / self.mtp_depth
                numerator1 += numerator_i
                denominator1 += denominator_i

            return numerator, denominator, numerator1, denominator1, extra_loss

        loss = self.loss(self.reshape(split_logits[0], (-1, split_logits[0].shape[-1])),
                         self.reshape(labels, (-1,)),
                         self.reshape(input_mask, (-1,))) + extra_loss
        for i in range(self.mtp_depth):
            labels = self._shift_and_pad(labels)
            input_mask = self._shift_and_pad(input_mask)
            loss += self.loss(self.reshape(split_logits[i + 1], (-1, split_logits[i + 1].shape[-1])),
                              self.reshape(labels, (-1,)),
                              self.reshape(input_mask, (-1,))) * self.mtp_loss_factor / self.mtp_depth
        return loss

    def _shift_and_pad(self, x):
        """implement roll with shift and pad."""
        bs, seq_len = self.shape(x)
        pad_zeros = self.zeros_op((bs, 1))
        x = self.slice(x, (0, 1), (bs, seq_len), (1, 1))
        x = self.concat_2d((x, self.cast(pad_zeros, x.dtype)))

        return x
