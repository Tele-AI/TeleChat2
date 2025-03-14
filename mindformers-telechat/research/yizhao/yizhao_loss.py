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
"""YiZhao loss"""
import mindspore as ms
from mindspore import nn
from mindspore import ops as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from yizhao_config import YiZhaoConfig

from mindformers.core.loss import CrossEntropyLoss
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class DPOLoss(nn.Cell):
    """ DPO Loss function for Yizhao """
    def __init__(self, config: YiZhaoConfig):
        super(DPOLoss, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.gatherd = P.GatherD()
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.slice_ind = P.StridedSlice().shard(((1,),))
        self.slice_mask = P.StridedSlice().shard(((1, 1),))
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))
        self.log_softmax = P.LogSoftmax()
        self.squeeze = P.Squeeze(-1).shard(((1, 1, 1),))
        self.expand = P.ExpandDims().shard(((1, 1),))
        self.label_pad_token_id = config.pad_token_id
        self.average_log_prob = True
        self.reference_free = False
        self.log_sigmoid = nn.LogSigmoid()
        self.not_equal = P.NotEqual()
        self.beta = 0.2
        self.cast = P.Cast()
        self.enable_force_redistribute = True
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)

    def _get_batch_logps(self, logits, labels, loss_mask=None):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with value of
            label_pad_token_id are ignored. Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of
            the given labels under the given logits.
        """
        if loss_mask is None:
            loss_mask = self.not_equal(labels, self.label_pad_token_id)
        # [bs, seq_len] -> [bs, seq_len]
        labels = self.mul(labels, loss_mask)
        # [bs, seq_len, vocab_size]
        log_probs = self.log_softmax(logits)
        # [bs, seq_len] -> [bs, seq_len, 1]
        index = self.expand(labels, -1)
        index = self.cast(index, ms.int32)
        # [bs, seq_len, 1]
        per_token_logps = self.gatherd(log_probs, -1, index)
        # [bs, seq_len, 1] -> [bs, seq_len]
        per_token_logps = self.squeeze(per_token_logps)
        return self.reduce_sum(per_token_logps * loss_mask, -1)

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_mask):
        """
        Compute dpo loss of the given labels under the given logits.

        Args:
            policy_chosen_logps: Log probabilities of the chosen actions by the policy.
            policy_rejected_logps: Log probabilities of the rejected actions by the policy.
            ref_chosen_logps: Log probabilities of the chosen actions by the reference model.
            ref_rejected_logps: Log probabilities of the rejected actions by the reference model.
            loss_mask: Mask to indicate which elements to consider for loss computation.

        Returns:
            losses: Computed losses.
            chosen_rewards: Rewards for chosen actions.
            rejected_rewards: Rewards for rejected actions.
            policy_chosen_logps_avg: Average log probabilities of chosen actions by the policy.
        """
        bs, seq_len = loss_mask.shape
        chosen_loss_mask = self.slice_mask(loss_mask, (0, 0), (bs // 2, seq_len), (1, 1))
        chosen_valid_len = self.reduce_sum(chosen_loss_mask, -1)
        policy_chosen_logps_avg = policy_chosen_logps / chosen_valid_len
        if self.average_log_prob:
            rejected_loss_mask = self.slice_mask(loss_mask, (bs // 2, 0), (bs, seq_len), (1, 1))
            rejected_valid_len = self.reduce_sum(rejected_loss_mask, -1)
            policy_chosen_logps = policy_chosen_logps / chosen_valid_len
            ref_chosen_logps = ref_chosen_logps / chosen_valid_len
            policy_rejected_logps = policy_rejected_logps / rejected_valid_len
            ref_rejected_logps = ref_rejected_logps / rejected_valid_len

        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps
        if self.reference_free:
            ref_log_ratios = 0
        logits = policy_log_ratios - ref_log_ratios
        losses = -self.log_sigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        return losses, chosen_rewards, rejected_rewards, policy_chosen_logps_avg

    def construct(self, policy_logits, policy_labels, loss_mask, ref_chosen_logps, ref_rejected_logps):
        """ Construct of DPO loss """
        all_logps = self._get_batch_logps(policy_logits, policy_labels, loss_mask)
        bs = all_logps.shape[0] // 2    # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_ind(all_logps, (0,), (bs,), (1,))
        policy_rejected_logps = self.slice_ind(all_logps, (bs,), (2 * bs,), (1,))
        dpo_loss, chosen_rewards, rejected_rewards, policy_chosen_logps_avg = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_mask
        )
        sft_loss = -policy_chosen_logps_avg
        if self.phase == "train":
            return dpo_loss, sft_loss
        return dpo_loss, sft_loss, chosen_rewards, rejected_rewards


class DPOCrossEntropy(CrossEntropyLoss):
    """ DPO CrossEntropy for Yizhao """
    def __init__(self, parallel_config, **kwargs):
        super().__init__(parallel_config, **kwargs)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.slice_3d = P.StridedSlice().shard(((dp, mp, 1),))
        self.slice_2d = P.StridedSlice().shard(((dp, mp),))

    def construct(self, logits, label, input_mask):
        """ Construct of DPO cross entropy """
        bs, seq_len, vocab_size = logits.shape    # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_3d(logits, (0, 0, 0), (bs // 2, seq_len, vocab_size), (1, 1, 1))
        label = self.slice_2d(label, (0, 0), (bs // 2, seq_len), (1, 1))
        input_mask = self.slice_2d(input_mask, (0, 0), (bs // 2, seq_len), (1, 1))
        return super().construct(policy_chosen_logps.reshape((-1, policy_chosen_logps.shape[-1])),
                                 label.reshape((-1,)), input_mask.reshape((-1,)))
