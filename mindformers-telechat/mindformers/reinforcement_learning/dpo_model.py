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
"""DPO model for llm model."""
from typing import Union
import copy
import mindspore as ms
from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.reinforcement_learning.rl_config import DPOConfig
from mindformers.models.utils import lazy_inline


class DPOModel(PreTrainedModel):
    """
    DPOModel define direct preference optimization model for LLM model.
    Args:
        config(DPOConfig): DPO config,define direct preference optimization algorithm.
        base_model(PreTrainedModel): pretrained model for DPO.
    """
    @lazy_inline
    @args_type_check(config=(dict, DPOConfig))
    def __init__(self, config: Union[dict, DPOConfig], base_model: PreTrainedModel, ref_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.add = P.Add()

        config = base_model.config
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.use_data_packing = config.dataset_config.use_data_packing
        self.input_sliced_sig = config.input_sliced_sig
        self.dpo_model = base_model
        self.ref_model = ref_model
        self.ref_model.add_flags_recursive(freeze=True)
        if self.ref_model is not None:
            self.freeze_ref_model()
        self.end_stage_id = 1

        self.use_past = config.use_past
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.dpo_loss = DPOLossV2(config)
        else:
            loss_parallel_config = copy.deepcopy(config)
            # total mp
            loss_parallel_config.parallel_config.model_parallel = dp * mp
            loss_parallel_config.parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0:  # For large scale training
                loss_parallel_config.parallel_config.model_parallel = 8
                loss_parallel_config.parallel_config.data_parallel = dp * mp // 8
            self.dpo_loss = DPOLossV2(loss_parallel_config)

        self.alpha = config.rl_config.dpo_alpha
        self.beta = config.rl_config.dpo_beta


    def freeze_ref_model(self):
        for param in self.ref_model.get_parameters():
            param.requires_grad = False

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.dpo_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.dpo_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return self.dpo_model.prepare_inputs_for_predict_layout(input_ids, **kwargs)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.dpo_model.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        return self.dpo_model.set_dynamic_inputs(**kwargs)

    def add_flags_custom(self, is_first_iteration):
        return self.dpo_model.add_flags_custom(is_first_iteration)

    def to_embeddings(self, tokens):
        return self.dpo_model.to_embeddings(tokens)

    def convert_name(self, weight_name):
        return self.dpo_model.convert_name(weight_name)

    def convert_weight_dict(self, source_dict, **kwargs):
        dpo_weight_dict = self.dpo_model.convert_weight_dict(source_dict, **kwargs)
        ref_weight_dict = self.ref_model.convert_weight_dict(source_dict, **kwargs)
        prefix_ref = "network._backbone.network.ref_model."
        ref_weight_dict = {f"{prefix_ref}{key}": value for key, value in ref_weight_dict.items()}
        merge_dict = {**dpo_weight_dict, **ref_weight_dict}
        return merge_dict

    def convert_map_dict(self, source_dict, **kwargs):
        dpo_weight_dict = self.dpo_model.convert_map_dict(source_dict, **kwargs)
        ref_weight_dict = self.ref_model.convert_map_dict(source_dict, **kwargs)
        prefix_ref = "network._backbone.network.ref_model."
        ref_weight_dict = {f"{prefix_ref}{key}": value for key, value in ref_weight_dict.items()}
        merge_dict = {**dpo_weight_dict, **ref_weight_dict}
        return merge_dict

    def make_attention_mask(self, eod_vec):

        dp = self.config.parallel_config.data_parallel
        # eod_vec shape (bs, seq_len)
        _, seq_length = eod_vec.shape
        get_attention_mask = GetEodResetMask(seq_length, dp)
        attention_mask = get_attention_mask(eod_vec)
        return attention_mask


    def construct(self, chosen_attention_mask=None, chosen_index_packed=None,
                  chosen_input_ids=None, chosen_labels=None, chosen_lens=None,
                  chosen_loss_mask=None, chosen_position_id=None, rejected_attention_mask=None,
                  rejected_index_packed=None, rejected_input_ids=None, rejected_labels=None,
                  rejected_lens=None, rejected_loss_mask=None, rejected_position_id=None,
                  input_position=None, position_ids=None, attention_mask=None, input_embeds=None,
                  init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        """ Construct of DPO model
            Args:
                chosen_attention_mask: A mask indicating which positions in the chosen samples
                                       need to be attended to in the attention mechanism.
                chosen_index_packed: Packed index information of the chosen samples.
                chosen_input_ids: Input identifiers corresponding to the chosen samples,
                                  usually numerical sequences converted from text.
                chosen_labels: True labels corresponding to the chosen samples,
                               used for supervised learning during model training.
                chosen_lens: Actual lengths of the chosen samples, useful for handling variable-length sequences.
                chosen_loss_mask: A mask specifying which elements in the chosen samples
                                  should participate in the calculation.
                rejected_attention_mask: A mask indicating which positions in the rejected samples
                                         need to be attended to in the attention mechanism.
                rejected_index_packed: Packed index information of the rejected samples.
                rejected_input_ids: Input identifiers corresponding to the rejected samples,
                                    usually numerical sequences converted from text.
                rejected_labels: True labels corresponding to the rejected samples,
                                 used for contrastive learning during model training.
                rejected_lens: Actual lengths of the rejected samples, useful for handling variable-length sequences.
                rejected_loss_mask: A mask specifying which elements in the rejected samples
                                    should participate in the calculation.
                input_position: Position information of input elements in the sequence.
                position_ids: Position encoding identifiers, used to provide position information of
                              the input sequence to the model.
                attention_mask: A general attention mask used to control the calculation scope of
                                the attention mechanism.
                input_embeds: Embedding representations of the input,
                              which are the results of converting input data into low-dimensional vectors.
                init_reset: A flag used to control the model initialization or reset operation.
                batch_valid_length: Information about the valid lengths of sequences in the batch.
                batch_index: Index information of samples in the batch.
                zactivate_len: Information about the effective length of a certain activation operation.
                block_tables: Block table information, possibly used for data block processing or storage.
                slot_mapping: Slot mapping information, used to map data to specific slots or positions.
            return:
                A tuple containing the Direct Preference Optimization (DPO) loss and
                the Supervised Fine - Tuning (SFT) loss,
                which represent the respective losses calculated based on the input log probabilities and masks.
        """
        # concat rejected & chosen
        input_ids = ops.concat((chosen_input_ids, rejected_input_ids), axis=0)
        labels = ops.concat((chosen_labels, rejected_labels), axis=0)
        loss_mask = ops.concat((chosen_loss_mask, rejected_loss_mask), axis=0)
        if chosen_attention_mask is not None:
            attention_mask = ops.concat((chosen_attention_mask, rejected_attention_mask), axis=0)
        if chosen_position_id is not None:
            position_ids = ops.concat((chosen_position_id, rejected_position_id), axis=0)
        if self.use_data_packing:
            index_packed = ops.concat((chosen_index_packed, rejected_index_packed), axis=0)
            combined_lens = ops.concat((chosen_lens, rejected_lens), axis=0)
        else:
            combined_lens = None
            index_packed = None
        bsz, ori_seqlen = self.shape(input_ids)
        if not self.input_sliced_sig:
            if attention_mask is not None:
                attention_mask = self.slice(attention_mask, (0, 1), (bsz, ori_seqlen), (1, 1))
            if chosen_position_id is not None:
                position_ids = self.slice(position_ids, (0, 0), (bsz, ori_seqlen - 1), (1, 1))
            if self.use_data_packing:
                index_packed = self.slice(index_packed, (0, 1), (bsz, ori_seqlen), (1, 1))
            loss_mask = self.slice(loss_mask, (0, 1), (bsz, ori_seqlen), (1, 1))

        if self.use_data_packing and attention_mask is not None:
            attention_mask = self.make_attention_mask(attention_mask)  # 运行时制作attention_mask
        else:
            attention_mask = None

        logits = self.dpo_model(
            input_ids=input_ids,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

        ref_logits = self.ref_model(
            input_ids=input_ids,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

        if not self.input_sliced_sig:
            labels = self.slice(labels, (0, 1), (bsz, ori_seqlen), (1, 1))
            tokens = self.slice(input_ids, (0, 0), (bsz, ori_seqlen - 1), (1, 1))

        if logits.ndim <= 2:
            logits = self.reshape(logits, (bsz, tokens.shape[1], logits.shape[-1]))
            ref_logits = self.reshape(ref_logits, (bsz, tokens.shape[1], ref_logits.shape[-1]))

        if self.use_data_packing:
            packed_num = self.shape(rejected_lens)[1]
        else:
            packed_num = None
        dpo_loss, sft_loss = self.dpo_loss(
            policy_logits=logits,
            labels=labels,
            loss_mask=loss_mask,
            ref_logits=ref_logits,
            index_packed=index_packed,
            packed_num=packed_num,
            combined_lens=combined_lens
        )

        return self.alpha * dpo_loss + self.beta * sft_loss


#  eod attention_mask
class GetEodResetMask(nn.Cell):
    """Get Eod Reset Mask"""
    def __init__(self, seq_length, dp):
        super(GetEodResetMask, self).__init__()
        self.seq_length = seq_length
        self.expand_dims = P.ExpandDims().shard(((dp, 1),))
        self.expand_dims_2 = P.ExpandDims().shard(((dp, 1, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.equal = P.Equal().shard(((dp, 1, 1,), (dp, 1, 1)))
        self.tril_op = P.Tril().shard(((dp, 1, 1,),))
        self.sub = P.Sub().shard(((), (dp, 1, 1),))

    def construct(self, eod_vec):
        """construct"""
        eod_vec_row = self.expand_dims(eod_vec, 1)
        eod_vec_column = self.expand_dims(eod_vec, 2)
        eod_matrix_1 = self.tile(eod_vec_row, (1, self.seq_length, 1))
        eod_matrix_2 = self.tile(eod_vec_column, (1, 1, self.seq_length))
        eod_matrix = self.equal(eod_matrix_1, eod_matrix_2)
        eod_matrix = F.cast(eod_matrix, mstype.uint8)
        mask = self.tril_op(eod_matrix)
        mask = self.sub(1, mask)
        return self.expand_dims_2(mask, 1)


class DPOLossV2(nn.Cell):
    """ DPO Loss function """

    def __init__(self, config):
        super(DPOLossV2, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
        self.vocab_size = config.vocab_size
        self.use_data_packing = config.dataset_config.use_data_packing

        self.gatherd = P.GatherD().shard(((dp * mp * cp, 1), (dp * mp * cp, 1)))
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, mp),))
        self.slice = P.StridedSlice()
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))
        self.log_softmax = P.LogSoftmax().shard(((dp * cp * mp, 1),))
        self.expand = P.ExpandDims().shard(((dp, mp),))
        self.label_pad_token_id = config.pad_token_id
        self.average_log_prob = True
        self.reference_free = False
        self.log_sigmoid = nn.LogSigmoid()
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        # for cal reward
        self.beta = 0.1
        self.enable_force_redistribute = True
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)

    def _get_batch_logps(self, logits=None, labels=None, loss_mask=None,
                         index_packed=None, packed_num=None, combined_lens=None):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: The raw, unnormalized predictions from the model, used to calculate log probabilities.
            labels: The ground truth labels corresponding to the model's predictions for
                    computing the log probabilities.
            loss_mask: A mask indicating which elements should be considered in the log probability calculation,
                       typically used to ignore padding.
            index_packed: The packed index information, which might be used to reorganize or
                          access relevant data for log probability computation.
            packed_num: The number of packed elements, providing information about the scale of
                        the packed data for log probability calculation.
            chosen_lens: The lengths of the chosen sequences, which can be used to handle variable - length
                         sequences in log probability calculation.
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log
            probabilities of the given labels under the given logits.
        """
        if loss_mask is None:
            loss_mask = self.not_equal(labels, self.label_pad_token_id)
        # [bs, seq_len] -> [bs, seq_len]
        labels = self.mul(labels, loss_mask)
        bs, seq_len, _ = logits.shape  # [bs, seq_len, vocab_size]
        logits = logits.reshape(bs * seq_len, -1)  # [bs * seq_len, vocab_size]

        log_probs = self.log_softmax(logits)

        # [bs, seq_len] -> [bs * seq_len, 1]
        index = self.expand(labels, -1)
        index = self.cast(index, mstype.int32)
        index = self.reshape(index, (bs * seq_len, -1))

        # [bs * seq_len, 1]
        per_token_logps = self.gatherd(log_probs, -1, index)
        # [bs * seq_len, 1] -> [bs, seq_len]
        per_token_logps = self.reshape(per_token_logps, (bs, seq_len))

        if self.use_data_packing:
            all_logps = self.batch_unsorted_segment_sum(per_token_logps * loss_mask,
                                                        segments_ids=index_packed,
                                                        num_segments=packed_num)

            if self.average_log_prob:
                return all_logps / combined_lens
            return all_logps
        if self.average_log_prob:
            return self.reduce_sum(self.mul(per_token_logps, loss_mask), -1) / self.reduce_sum(loss_mask, -1)
        return self.reduce_sum(self.mul(per_token_logps, loss_mask), -1)


    def batch_unsorted_segment_sum(self, input_ids, segments_ids, num_segments):
        """
        batch unsorted_segment_sum

        Args:
            input_ids: shape (batch_size, seq_len)
            segment_ids: shape (batch_size, seq_len)
            num_segments: int
        Returns:
            shape (batch_size, num_segments)
        """
        bs, seq_len = input_ids.shape
        output = ops.zeros((bs, num_segments), input_ids.dtype)
        for b in range(bs):
            current_input = self.slice(input_ids, (b, 0), (b + 1, seq_len), (1, 1))
            current_segment = self.slice(segments_ids, (b, 0), (b + 1, seq_len), (1, 1))
            seg_sum = ops.unsorted_segment_sum(current_input, current_segment, num_segments)
            output[b] = seg_sum
        return output


    def dpo_loss(self, policy_chosen_logps=None, policy_rejected_logps=None, ref_chosen_logps=None,
                 ref_rejected_logps=None, loss_mask=None, combined_lens=None):
        """
        Compute dpo loss of the given labels under the given logits.

        Args:
            policy_chosen_logps: Log probabilities of the chosen actions by the policy.
            policy_rejected_logps: Log probabilities of the rejected actions by the policy.
            ref_chosen_logps: Log probabilities of the chosen actions by the reference model.
            ref_rejected_logps: Log probabilities of the rejected actions by the reference model.
            loss_mask: Mask for sequences in loss computation.
            combined_lens: Actual lengths of sequences in the batch.
        Returns:
            losses: Computed losses.
            chosen_rewards: Rewards for chosen actions.
            rejected_rewards: Rewards for rejected actions.
            policy_chosen_logps_avg: Average log probabilities of chosen actions by the policy.
        """
        bs, seq_len = loss_mask.shape
        if self.average_log_prob:
            # already computed, just use
            policy_chosen_logps_avg = policy_chosen_logps
        elif self.use_data_packing:
            chosen_lens = self.slice(combined_lens, (0, 0), (bs // 2, combined_lens.shape[1]), (1, 1))
            policy_chosen_logps_avg = policy_chosen_logps / chosen_lens
        else:
            chosen_loss_mask = self.slice(loss_mask, (0, 0), (bs // 2, seq_len), (1, 1))
            chosen_valid_len = self.reduce_sum(chosen_loss_mask, -1)
            policy_chosen_logps_avg = policy_chosen_logps / chosen_valid_len

        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps

        if self.reference_free:
            ref_log_ratios = 0
        logits = policy_log_ratios - ref_log_ratios
        losses = -self.log_sigmoid(self.beta * logits)  # if logits is very large, the losses would be nan
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        if self.use_data_packing:
            losses = losses.mean()
            policy_chosen_logps_avg = policy_chosen_logps_avg.mean()

        return losses, chosen_rewards, rejected_rewards, policy_chosen_logps_avg

    def construct(self, policy_logits=None, labels=None, loss_mask=None,
                  ref_logits=None, ref_chosen_logps=None, ref_rejected_logps=None,
                  index_packed=None, packed_num=None, combined_lens=None):
        """ Construct of DPO loss
            Args:
                policy_logits: The raw, unnormalized scores output by the policy model,
                               used to compute probabilities for actions.
                labels: The ground - truth labels corresponding to the model's outputs,
                        used for supervision.
                loss_mask: A binary mask indicating which elements in the samples
                           should be considered for loss calculation.
                ref_logits: The raw, unnormalized scores output by the reference model,
                            serving as a baseline for comparison.
                index_packed: The packed index information of the samples, which may be used for efficient data access.
                packed_num: The number of packed elements, which is useful for batch processing and data organization.
                combined_lens: The lengths of the sequences, helping to handle variable - length
                               sequences in calculations.
        """

        bs = policy_logits.shape[0] // 2  # a sample has two bs responses (chosen and rejected)

        policy_logps = self._get_batch_logps(policy_logits,
                                             labels,
                                             loss_mask,
                                             index_packed,
                                             packed_num,
                                             combined_lens
                                             )

        policy_chosen_logps = self.slice(policy_logps, (0,), (bs,), (1,))
        policy_rejected_logps = self.slice(policy_logps, (bs,), (2 * bs,), (1,))

        if ref_logits is not None:
            ref_logps = self._get_batch_logps(ref_logits,
                                              labels,
                                              loss_mask,
                                              index_packed,
                                              packed_num,
                                              combined_lens
                                              )

            ref_chosen_logps = self.slice(ref_logps, (0,), (bs,), (1,))
            ref_rejected_logps = self.slice(ref_logps, (bs,), (2 * bs,), (1,))

        dpo_loss, chosen_rewards, rejected_rewards, policy_chosen_logps_avg = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_mask,
            combined_lens
        )
        # pylint: disable=E1130
        sft_loss = -policy_chosen_logps_avg
        dpo_loss = self.cast(dpo_loss, ms.float32)
        sft_loss = self.cast(sft_loss, ms.float32)
        if self.phase == "train":
            return dpo_loss, sft_loss
        return dpo_loss, sft_loss, chosen_rewards, rejected_rewards
