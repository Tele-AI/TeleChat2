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
"""Solver Class"""

import os
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

import pulp as lpSolver

import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.layer import Layer
from toolkit.pipeline_balance.utils.logger import logger

# seqpipe const
tensor_float_16 = 2
tensor_float_32 = 4
const_from_byte_to_mb = 1024 * 1024
# llama intermideate_size
llama_intermideate_size = 11008


@dataclass
class PipelineMemoryConstraint:
    """constraint struct"""
    prob: Any
    variables: Any
    layers_sorted: dict[Any]
    num_of_stage: int
    num_of_interleave: int
    micro_batch: int
    memory_limit: int


class SappSolver:
    """solver for pipeline balance"""

    BIG_M = 1000000

    MEM_OVERHEAD_NAME = "memory_overhead"
    TOTAL_SUM = "var_sum_FPi_BPi"
    CHUNKS_SUM = "chunks_sum"
    PREV_DIFF = "prev_diff"
    NEXT_DIFF = "next_diff"
    MAX_STAGE_TIME = "max_stage_time"
    MAX_LAST_CHUNK = "max_last_chunk"
    LAYER_FRONTIER = "layer_frontier"
    PROP_PHASE = IntEnum("Propagation", ["FW", "BW"], start=0)

    def __init__(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            num_of_micro_batch: int,
            max_memory: int,
            layers: list[Layer],
            layers_sorted: dict[Layer.type_enum, list[Layer]],
            vpp_less_memory: bool = False,
            constant_memory: int = 0,
            optimization_level: int = 1,
            description: str = "Pipeline_execution_time_minimize",
            extracted_training_params: dict[str, int] = None,
            seq_split_num: int = 1,
    ):

        self.num_of_stage_ = num_of_stage
        self.num_of_interleave_ = num_of_interleave
        self.num_of_micro_batch_ = num_of_micro_batch
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        self.constant_memory_ = constant_memory
        self.optimization_level_ = optimization_level
        self.layers_ = layers
        self.layers_sorted_ = layers_sorted

        self.recompute_considered_ = self.find_recompute_considered(
            layers_sorted)
        self.extracted_training_params_ = extracted_training_params
        self.seq_split_num_ = seq_split_num
        self.seq_pipe = self.seq_split_num_ > 1
        if self.seq_pipe:
            # compute seq layer memory based on 1f1b layer memory
            for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                if layer.memory_parameter_ is not None:
                    logger.info(f"Body Layer 1f1b Parameter Memory: {layer.memory_parameter_}")
                    layer.memory_parameter_ = self.compute_seq_mem_parameter(layer.memory_parameter_,
                                                                             self.extracted_training_params_)
                    logger.info(f"Body Layer Seq Parameter Memory: {layer.memory_parameter_}")
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        if rec.name == "FULL":
                            self.recompute_considered_[rec] = False
                            layer.memory_activation_rec_[rec] = None
                            logger.error("Seqpipe doesn't support full recomputation, \
                                         recompute_activation is set as None for seqpp")
                            continue
                        logger.info(f"Body Layer 1f1b {rec} activation Memory: {layer.memory_activation_rec_[rec]}")
                        layer.memory_activation_rec_[rec] = self.compute_seq_mem_activation(
                            layer.memory_activation_rec_[rec],
                            self.extracted_training_params_,
                            self.seq_split_num_
                        )
                        logger.info(f"Body Layer seq {rec} activation Memory: {layer.memory_activation_rec_[rec]}")

            for head in self.layers_sorted_[Layer.type_enum.HEAD]:
                if head.memory_parameter_ is not None:
                    logger.info(f"Head cost 1f1b: {head.memory_parameter_}")
                    head.memory_parameter_ = self.compute_seq_mem_head_cost(head.memory_parameter_,
                                                                            self.extracted_training_params_,
                                                                            self.seq_split_num_)
                    logger.info(f"Head cost Seq: {head.memory_parameter_}")

            for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
                if tail.memory_parameter_ is not None:
                    logger.info(f"Tail cost 1f1b: {tail.memory_parameter_}")
                    tail.memory_parameter_ = self.compute_seq_mem_tail_cost(tail.memory_parameter_,
                                                                            self.extracted_training_params_,
                                                                            self.seq_split_num_)
                    logger.info(f"Tail cost seq: {tail.memory_parameter_}")
            # microbatch * seq chunk
            self.num_of_micro_batch_ = self.num_of_micro_batch_ * self.seq_split_num_
            # time / seq split number  (theoretically)
            for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                logger.info(f"Body Layer 1f1b fp time: {layer.forward_time_}")
                logger.info("Body Layer 1f1b bp time:")
                for key, value in layer.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)
                layer.time_ = layer.time_ / self.seq_split_num_
                layer.forward_time_ = layer.forward_time_ / self.seq_split_num_
                layer.update_internal_time_for_seqpp()
                logger.info(f"Body Layer seq fp time: {layer.forward_time_}")
                logger.info("Body Layer seq bp time:")
                for key, value in layer.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)
            for head in self.layers_sorted_[Layer.type_enum.HEAD]:
                logger.info(f"Head Layer 1f1b fp time: {head.forward_time_}")
                logger.info("Head Layer 1f1b bp time:")
                for key, value in head.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)
                head.time_ = head.time_ / self.seq_split_num_
                head.forward_time_ = head.forward_time_ / self.seq_split_num_
                head.update_internal_time_for_seqpp()
                logger.info(f"Head Layer seq fp time: {head.forward_time_}")
                logger.info("Head Layer seq bp time:")
                for key, value in head.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)
            for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
                logger.info(f"Tail Layer 1f1b fp time: {tail.forward_time_}")
                logger.info("Tail Layer 1f1b bp time:")
                for key, value in tail.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)
                tail.time_ = tail.time_ / self.seq_split_num_
                tail.forward_time_ = tail.forward_time_ / self.seq_split_num_
                tail.update_internal_time_for_seqpp()
                logger.info(f"Tail Layer seq fp time: {tail.forward_time_}")
                logger.info("Tail Layer seq bp time:")
                for key, value in tail.backward_time_rec_.items():
                    logger.output("%s: %s", key, value)

        self.variables_ = self._create_variables_to_solve_(
            num_of_stage, num_of_interleave, layers_sorted)
        self.problem_ = self._create_problem_(description)

    @staticmethod
    def compute_forward_in_backward(num_of_stage: int,
                                    micro_batch: int) -> list[int]:
        """Computes the number of forward propagation happening after a backward"""
        n = num_of_stage - 1
        factors = []
        for _ in range(num_of_stage):
            factors.append(abs(n))
            n -= 2
        if micro_batch < 2 * num_of_stage:
            for i in range(num_of_stage // 2):
                factors[i] = 0
        return factors

    @staticmethod
    def compute_lm_forward_in_backward(num_of_stage: int) -> list[int]:
        """Function compute_forward_in_backward in less_memory schedule"""
        return [s for s in range(num_of_stage)]

    @staticmethod
    def compute_activation_nums(num_of_stage: int, num_of_interleave: int,
                                micro_batch: int) -> list[list[int]]:
        """compute the number of activation"""
        activation_nums = []
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[0][s] += max(0, num_of_stage - 2 * s - 1)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] += min(
                    0, num_of_stage - 2 * s - 1)
            for i in range(num_of_interleave):
                for s in range(num_of_stage):
                    activation_nums[i][s] = min(activation_nums[i][s],
                                                micro_batch)
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)

        return activation_nums

    @staticmethod
    def compute_less_activation_nums(
            num_of_stage: int, num_of_interleave: int) -> list[list[int]]:
        """compute number of less_mem activation"""
        activation_nums = []
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] -= s
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)
        return activation_nums

    #######################################################################
    ##                                                                   ##
    ##                            SeqPipe                                ##
    ##                                                                   ##
    #######################################################################
    @staticmethod
    def compute_activation_seq_nums(num_of_stage: int, num_of_interleave: int,
                                    seq_split_num: int, micro_batch: int, less_memory: False) -> list[list[int]]:
        """compute the number of activation for seq chunks"""
        activation_nums = []
        if less_memory:
            act_gap = 1
        else:
            act_gap = 2
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] = seq_split_num

            loop_index = 1
            for stage_index in range(num_of_stage - 2, -1, -1):
                flag_added = False
                for chunk_index in range(num_of_interleave):
                    condition1 = activation_nums[chunk_index][stage_index + 1] % num_of_stage != 0
                    condition2 = activation_nums[chunk_index][stage_index + 1] // num_of_stage < loop_index
                    if condition1 or condition2:
                        for update in range(stage_index + 1):
                            activation_nums[chunk_index][update] += act_gap
                        flag_added = True
                        break
                if not flag_added:
                    for update in range(stage_index + 1):
                        activation_nums[0][update] += act_gap
                    loop_index += 1
            # microbatch
            for i in range(num_of_interleave):
                for s in range(num_of_stage):
                    activation_nums[i][s] = min(activation_nums[i][s],
                                                micro_batch)
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s + seq_split_num - 1)

        logger.output(f"compute_activation_seq_nums: {activation_nums}")
        return activation_nums

    @staticmethod
    def compute_seq_mem_activation(original_memory_activation: float,
                                   extracted_training_params: dict[str, int],
                                   seq_split_num: int) -> float:
        """compute activation memory for seqpipe"""
        # context parallel? cp?
        batch_size = extracted_training_params['batch_size']
        heads = extracted_training_params['num_heads']
        seq_length = extracted_training_params['seq_length']
        head_dim = extracted_training_params['head_dim']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # 2*Kv add
        # cp?
        kv_update_mem_byte = 2 * ((tensor_float_16 * batch_size * heads * seq_length * head_dim) / (mp))
        kv_update_mem = kv_update_mem_byte / const_from_byte_to_mb
        # Attention Key,Value
        # cp?
        key_mem_byte = (tensor_float_16 * batch_size * heads * seq_length * head_dim) / (mp)
        key_mem = key_mem_byte / const_from_byte_to_mb
        # cp?
        value_mem_byte = (tensor_float_16 * batch_size * heads * seq_length * head_dim) / (mp)
        value_mem = value_mem_byte / const_from_byte_to_mb

        seq_memory_activation = (original_memory_activation - key_mem - value_mem) / seq_split_num + kv_update_mem
        return seq_memory_activation

    @staticmethod
    def compute_seq_mem_parameter(original_memory_parameter: float, extracted_training_params: dict[str, int]) -> float:
        """compute layer parameter memory for seqpipe"""
        # context parallel? cp?
        batch_size = extracted_training_params['batch_size']
        heads = extracted_training_params['num_heads']
        seq_length = extracted_training_params['seq_length']
        head_dim = extracted_training_params['head_dim']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # cp?
        kv_cache_parameter_mem_byte = 4 * (tensor_float_16 * batch_size * heads * seq_length * head_dim / (mp))
        kv_cache_parameter_mem = kv_cache_parameter_mem_byte / const_from_byte_to_mb
        seq_memory_parameter = original_memory_parameter + kv_cache_parameter_mem
        return seq_memory_parameter

    @staticmethod
    def compute_seq_mem_head_cost(original_head_cost: float,
                                  extracted_training_params: dict[str, int],
                                  seq_split_num: int) -> float:
        """compute head stage extra cost for seqpipe"""
        batch_size = extracted_training_params['batch_size']
        seq_length = extracted_training_params['seq_length']
        hidden_size = extracted_training_params['hidden_size']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        if mp > 1:
            # comm operator Mem (recv+reduceScatter)
            # cp?
            comm_operator_mem_byte = 2 * (tensor_float_16 * batch_size * seq_length * hidden_size / (mp))
            comm_operator_mem = comm_operator_mem_byte / const_from_byte_to_mb
            # StridedSliceGrad Operator Mem
            stridslice_operator_mem_byte = tensor_float_16 * batch_size * seq_length * hidden_size
            stridslice_operator_mem = stridslice_operator_mem_byte / const_from_byte_to_mb
            seq_head_cost = original_head_cost - (1 - 1 / seq_split_num) * (comm_operator_mem + stridslice_operator_mem)
        else:
            # comm operator Mem (recv)
            # cp?
            comm_operator_mem_byte = tensor_float_16 * batch_size * seq_length * hidden_size / (mp)
            comm_operator_mem = comm_operator_mem_byte / const_from_byte_to_mb
            # Grad/MatMul // Grad/Mul Operator Mem
            # cp?
            mul_operator_mem_byte = 1 * (tensor_float_16 * batch_size * seq_length * llama_intermideate_size / (mp))
            mul_operator_mem = mul_operator_mem_byte / const_from_byte_to_mb
            seq_head_cost = original_head_cost - (1 - 1 / seq_split_num) * (comm_operator_mem + mul_operator_mem)
        return seq_head_cost

    @staticmethod
    def compute_seq_mem_tail_cost(original_tail_cost: float,
                                  extracted_training_params: dict[str, int],
                                  seq_split_num: int) -> float:
        """compute tail stage extra cost for seqpipe"""
        batch_size = extracted_training_params['batch_size']
        seq_length = extracted_training_params['seq_length']
        vocab_size = extracted_training_params['vocab_size']
        mp = extracted_training_params['model_parallel']
        # cp = extracted_training_params['context_parallel']
        # Memory extra introduced by loss op:
        # cp?
        loss_operator_mem_byte = tensor_float_32 * batch_size * seq_length * vocab_size / (mp)
        loss_operator_mem = loss_operator_mem_byte / const_from_byte_to_mb
        # New tail Cost = Old tail Cost - (3-3/k)M + (k-1)(M/k)
        seq_tail_cost = original_tail_cost - (3 - 3 / seq_split_num) * loss_operator_mem + (
            seq_split_num - 1) * (loss_operator_mem / seq_split_num)
        return seq_tail_cost

    def add_total_nb_layer_constraint(self, prob, variables, sorted_layers):
        """Constraints to respect the total number of layer"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            prob += (lpSolver.lpSum(
                variables[layer.name_][rec] for rec in Recompute.TYPE
                if self.recompute_considered_[rec]) == layer.nb_layer_)
        return prob

    def add_stage_nb_layer_constraint(self, prob, variables, sorted_layers):
        """Constraints to respect the total number of layer"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            for i in range(self.num_of_interleave_):
                for s in range(self.num_of_stage_):
                    prob += (lpSolver.lpSum(variables[layer.name_][rec][i][s]
                                            for rec in Recompute.TYPE
                                            if self.recompute_considered_[rec])
                             >= 1)
        return prob

    def add_multimodal_sequence_constraint(self, prob, variables, sorted_layers):
        """Constraints to enforce a frontier between the types of BODY layers"""
        for frontier in range(1, len(sorted_layers[Layer.type_enum.BODY])):
            layer = sorted_layers[Layer.type_enum.BODY][frontier].name_
            for v in range(0, self.num_of_interleave_):
                for s in range(0, self.num_of_stage_):
                    cur_chunk_sum = lpSolver.lpSum(
                        variables[layer][rec][v][ss] for rec in Recompute.TYPE
                        if self.recompute_considered_[rec] for ss in range(s)
                    )
                    prev_chunk_sum = lpSolver.lpSum(
                        variables[layer][rec][vv][ss]
                        for rec in Recompute.TYPE if self.recompute_considered_[rec]
                        for vv in range(v) for ss in range(self.num_of_stage_)
                    )
                    prob += (
                        variables[self.LAYER_FRONTIER][frontier - 1][v][s]
                        >= (cur_chunk_sum + prev_chunk_sum) / self.BIG_M
                    )

        for frontier in range(1, len(sorted_layers[Layer.type_enum.BODY])):
            layer = sorted_layers[Layer.type_enum.BODY][frontier - 1].name_
            for s in range(0, self.num_of_stage_):
                for v in range(0, self.num_of_interleave_):
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            prob += variables[layer][rec][v][s] <= (
                                1 - variables[self.LAYER_FRONTIER][frontier - 1][v][s]
                                ) * self.BIG_M
        return prob

    @staticmethod
    def find_recompute_considered(layers_sorted):
        """Find the recomputation types considered"""
        return layers_sorted[Layer.type_enum.BODY][0].recompute_considered_

    def max_stage_micro_eq_stage(self, prob, layers_sorted):
        """apply optimizations on vpp when pp=#mb"""
        last_chunk = self.num_of_interleave_ - 1

        for i_stage in range(self.num_of_stage_):
            for inter in range(last_chunk):
                prob += self.variables_[self.MAX_STAGE_TIME] >= (
                    self._max_stage_bound_i_bp(layers_sorted, i_stage, inter) +
                    self._max_stage_bound_head_tail(layers_sorted, i_stage,
                                                    -1, inter))

        if self.vpp_less_memory_:
            factors = self.compute_lm_forward_in_backward(self.num_of_stage_)
        else:
            factors = self.compute_forward_in_backward(
                self.num_of_stage_, self.num_of_micro_batch_)

        for i_stage in range(self.num_of_stage_):
            logger.debug(
                f"v={last_chunk}, s={i_stage}: (BP + HT) + "
                f"({factors[i_stage]} / {self.num_of_micro_batch_} * FP")
            prob += self.variables_[self.MAX_LAST_CHUNK] >= (
                self._max_stage_bound_i_bp(layers_sorted, i_stage, last_chunk) +
                self._max_stage_bound_head_tail(layers_sorted, i_stage, last_chunk, last_chunk) +
                (factors[i_stage] / self.num_of_micro_batch_) *
                self._max_stage_bound_i_fp(layers_sorted, i_stage, last_chunk))

        if self.optimization_level_ >= 2:
            logger.debug("Approach 2a")
            prob += self.variables_[self.MAX_STAGE_TIME] >= (
                self.variables_[self.MAX_LAST_CHUNK])

            return self.variables_[self.MAX_STAGE_TIME]
        logger.debug("Approach 2b")
        prob += self.variables_[self.MAX_LAST_CHUNK] >= (
            self.variables_[self.MAX_STAGE_TIME])

        return (self.variables_[self.MAX_STAGE_TIME] +
                self.variables_[self.MAX_LAST_CHUNK])

    def add_performance_constraint(self, prob, layers_sorted, pipeline_total_time):
        """add performance constraint"""
        max_stage_time = self.variables_[self.MAX_STAGE_TIME]
        max_stage_time = self.add_max_stage_constraint(prob, layers_sorted, max_stage_time)

        total_sum = self.variables_[self.TOTAL_SUM]
        prob += total_sum >= self._total_sum(layers_sorted)

        if self.optimization_level_ >= 2:
            # approach A
            for v in range(self.num_of_interleave_ - 1):
                prob += self.variables_[self.PREV_DIFF][v] >= (
                    self._prev_diff_sum(layers_sorted, prob, v))

                prob += self.variables_[self.CHUNKS_SUM][v] >= (
                    (self.num_of_interleave_ - v) / self.num_of_interleave_ *
                    self._chunks_sum(layers_sorted, v))

            chunks_sum = lpSolver.lpSum(self.variables_[self.CHUNKS_SUM])
            prev_diff = lpSolver.lpSum(self.variables_[self.PREV_DIFF])

            next_diff = self.variables_[self.NEXT_DIFF]
            prob += next_diff >= (
                self._next_diff_sum(layers_sorted, prob))

            prob += pipeline_total_time >= (
                (total_sum + chunks_sum + prev_diff + next_diff)
                / max(1, (self.num_of_interleave_ - 2))
                + max_stage_time * (self.num_of_micro_batch_ - 2)
            )
        else:
            # approach B
            prob += pipeline_total_time >= max_stage_time
        return prob

    def add_max_stage_constraint(self, prob, layers_sorted, max_stage_time):
        """add constrains based on max stage time"""
        if (self.num_of_interleave_ > 1 and self.optimization_level_ >= 1
                and self.num_of_micro_batch_ == self.num_of_stage_):
            max_stage_time = self.max_stage_micro_eq_stage(prob, layers_sorted)
        else:
            # Constraints on sub-main-part of a stage that it may take (for all stage)
            for i_stage in range(self.num_of_stage_):
                for inter_f in range(self.num_of_interleave_):
                    for inter_b in range(self.num_of_interleave_):
                        prob += max_stage_time >= (
                            self._max_stage_bound_i_fp(layers_sorted, i_stage, inter_f)
                            + self._max_stage_bound_i_bp(
                                layers_sorted, i_stage, inter_b
                            )
                            + self._max_stage_bound_head_tail(
                                layers_sorted, i_stage, inter_f, inter_b
                            )
                        )

        return max_stage_time

    ############################################
    #            Memory Constraint             #
    ############################################

    def stage_param_memory(self, variables, layers_sorted, stage_id,
                           num_of_stage, num_of_interleave):
        """memory constraint for parameters"""
        bound = lpSolver.LpAffineExpression()
        for inter_id in range(num_of_interleave):
            for layer in layers_sorted[Layer.type_enum.BODY]:
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        bound += (
                            variables[layer.name_][rec][inter_id][stage_id] *
                            layer.memory_parameter_)
        if stage_id == 0:
            for head in layers_sorted[Layer.type_enum.HEAD]:
                bound += head.memory_parameter_
        if stage_id == num_of_stage - 1:
            for tail in layers_sorted[Layer.type_enum.TAIL]:
                bound += tail.memory_parameter_
        return bound

    def stage_active_memory_per_micro(self, variables, layers_sorted, stage_id,
                                      inter_id):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (variables[layer.name_][rec][inter_id][stage_id] *
                              layer.memory_activation_rec_[rec])
        return bound

    def stage_active_memory(self, variables, layers_sorted, stage_id,
                            num_of_interleave, activation_nums):
        """calculate active memory of a stage"""
        bound = lpSolver.LpAffineExpression()
        for inter_id in range(num_of_interleave):
            for layer in layers_sorted[Layer.type_enum.BODY]:
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        bound += (
                            variables[layer.name_][rec][inter_id][stage_id] *
                            layer.memory_activation_rec_[rec] *
                            activation_nums[inter_id][stage_id])
        return bound

    def init_overhead_variables(self, variables, s):
        """init memory overhead variables"""
        bound = lpSolver.LpAffineExpression()
        vf = self.num_of_interleave_ - 1
        vb = self.num_of_interleave_ - 1
        incr_f = True
        if self.vpp_less_memory_:
            for _ in range(self.num_of_interleave_ - 1):
                if incr_f:
                    vf = (vf + 1) % self.num_of_interleave_
                    factor = abs(self.num_of_stage_ - s)
                else:
                    vb = vb - 1
                    factor = s
                incr_f = not incr_f

                logger.debug(f"{factor} * (act({vf},{s}) - act({vb},{s})")
                bound += factor * (
                    self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                    - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb))
        else:
            for _ in range(self.num_of_interleave_ - 1):
                if incr_f:
                    vf = (vf + 1) % self.num_of_interleave_
                    logger.debug(f"{self.num_of_stage_ - abs(self.num_of_stage_ - 2*s - 1)}"
                                 f" * (act({vf},{s}) - act({vb},{s})")
                    bound += (self.num_of_stage_ - abs(self.num_of_stage_ - 2 * s - 1)) * (
                        self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                        - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb)
                        )
                else:
                    vb = vb - 1
                    logger.debug(f"{ max(self.num_of_stage_ - 2*s - 1, 0)}"
                                 f" * (act({vf+1},{s}) - act({vb+1},{s})")
                    bound += max(self.num_of_stage_ - 2 * s - 1, 0) * (
                        self.stage_active_memory_per_micro(variables,
                                                           self.layers_sorted_, s, vf + 1)
                        - self.stage_active_memory_per_micro(variables,
                                                             self.layers_sorted_, s, vb + 1)
                        )
                    logger.debug(f"{ max(-(self.num_of_stage_ - 2*s - 1), 0)}"
                                 f" * (act({vf},{s}) - act({vb},{s})")
                    bound += max(-(self.num_of_stage_ - 2 * s - 1), 0) * (
                        self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vf)
                        - self.stage_active_memory_per_micro(variables, self.layers_sorted_, s, vb)
                        )
                incr_f = not incr_f

        return bound

    def stage_overhead_memory(self, variables, stage_id):
        bound = lpSolver.LpAffineExpression()
        for v in range(self.num_of_interleave_ - 1):
            bound += variables[self.MEM_OVERHEAD_NAME][stage_id][v]
        return bound

    def add_pipeline_memory_constraint(self,
                                       constraint: PipelineMemoryConstraint):
        """add constrain on the memory of pipeline"""
        prob = constraint.prob
        variables = constraint.variables
        layers_sorted = constraint.layers_sorted
        num_of_stage = constraint.num_of_stage
        num_of_interleave = constraint.num_of_interleave
        micro_batch = constraint.micro_batch
        memory_limit = constraint.memory_limit

        if self.vpp_less_memory_:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    num_of_stage, num_of_interleave, self.seq_split_num_, micro_batch, True)
            else:
                activation_nums = self.compute_less_activation_nums(
                    num_of_stage, num_of_interleave)
        else:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    num_of_stage, num_of_interleave, self.seq_split_num_, micro_batch, False)
            else:
                activation_nums = self.compute_activation_nums(
                    num_of_stage, num_of_interleave, micro_batch)
        logger.info("activation nums = %s", activation_nums)

        if self.num_of_stage_ == self.num_of_micro_batch_:
            for s in range(num_of_stage):
                prob += memory_limit >= (
                    self.stage_param_memory(variables, layers_sorted, s,
                                            num_of_stage, num_of_interleave) +
                    self.stage_active_memory(variables, layers_sorted, s,
                                             num_of_interleave, activation_nums) +
                    self.constant_memory_)
        else:
            for s in range(num_of_stage):
                prob += variables[self.MEM_OVERHEAD_NAME][s] >= (
                    self.init_overhead_variables(variables, s)
                )
                prob += memory_limit >= (
                    self.stage_param_memory(
                        variables, layers_sorted, s, num_of_stage, num_of_interleave
                    )
                    + self.stage_active_memory(
                        variables, layers_sorted, s, num_of_interleave, activation_nums
                    )
                    + variables[self.MEM_OVERHEAD_NAME][s]
                    + self.constant_memory_
                )

    def get_simulator_memory_activation(self) -> list[float]:
        """Give the activation memory per stage for simulator."""

        memory_active = []
        if self.has_some_memory_info():
            for inter in range(self.num_of_interleave_):
                memory_active.append([])
                for stage in range(self.num_of_stage_):
                    memory_active[inter].append(0)
                    memory_active[inter][stage] = sum(
                        self.variables_.get(layer.name_)[rec][inter][stage].varValue
                        * layer.memory_activation_rec_[rec]
                        for rec in Recompute.TYPE
                        if self.recompute_considered_[rec]
                        for layer in self.layers_sorted_[Layer.type_enum.BODY]
                    )
        return memory_active

    def get_simulator_memory_parameter(self) -> list[float]:
        """Give the parameter memory per stage for simulator."""
        memory_param_stage = [0] * self.num_of_stage_
        if self.has_some_memory_info():
            for inter in range(self.num_of_interleave_):
                for stage in range(self.num_of_stage_):
                    memory_param_stage[stage] += sum(
                        self.variables_.get(layer.name_)[rec][inter][stage].varValue
                        * layer.memory_parameter_
                        for rec in Recompute.TYPE if self.recompute_considered_[rec]
                        for layer in self.layers_sorted_[Layer.type_enum.BODY])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param_stage[0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param_stage[self.num_of_stage_ -
                                   1] += tail.memory_parameter_
        memory_param = [memory_param_stage] * self.num_of_interleave_
        return memory_param

    def get_simulator_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_.get(
                                layer.name_)[rec][i][s].varValue * (
                                    layer.forward_time_ +
                                    layer.backward_time_rec_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_
        return time

    def get_simulator_forward_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_[layer.name_][rec][i][
                                s].varValue * (layer.forward_time_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_
        return time

    def get_simulator_recompute_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time_all_rec = []
        time_no_rec = []
        for i in range(self.num_of_interleave_):
            time_all_rec.append([])
            time_no_rec.append([])
            for s in range(self.num_of_stage_):
                time_all_rec[i].append(0)
                time_no_rec[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time_all_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[rec])
                            time_no_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[
                                        Recompute.TYPE.NONE])
        return [[r - n for r, n in zip(ar, nr)]
                for ar, nr in zip(time_all_rec, time_no_rec)]

    def has_some_memory_info(self) -> bool:
        """Check if there is some information for memory constraint."""
        some_info = False
        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                some_info = True
        return some_info

    ############################################
    #            General Constraint            #
    ############################################
    def add_optional_recompute_constraint(self, prob, variables,
                                          sorted_layers):
        """Constraints to put unused recompute variables at 0"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if not self.recompute_considered_[rec]:
                    prob += lpSolver.lpSum(variables[layer.name_][rec]) == 0

    def dump_problem(self, folder=None):
        """
        dump pulp problem inside folder
        """
        dump_name = "problem_" + str(self.layers_[0].model_name_)
        dump_name += "_" + str(self.max_memory_)
        dump_name += "_" + str(self.num_of_interleave_)
        dump_name += "_" + str(self.num_of_stage_)

        logger.info("dump_problem:out folder = %s", folder)
        if folder is not None:
            dump_name = os.path.join(folder, dump_name)
        dump_name += ".lp"
        logger.info("dump problem file: %s", dump_name)
        self.problem_.writeLP(dump_name)

    def print_results(self):
        """Print the detailed results"""
        if self.has_some_memory_info():
            print("For max memory ", self.max_memory_)
            print("==============")
        for body_layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = body_layer.name_
            print("For layer:", layer_name)
            print("=========")
            print("  Forward Prop time: ", body_layer.forward_time_)
            for rec in Recompute.TYPE:
                if body_layer.recompute_considered_[rec]:
                    print(
                        "  Backward Prop",
                        Recompute.YAML_NAME[rec],
                        "time:",
                        body_layer.backward_time_rec_[rec],
                    )
            for inter in range(self.num_of_interleave_):
                for stage in range(self.num_of_stage_):
                    print("    Assign", layer_name, end=": ")
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            value = str(int(self.variables_[layer_name][rec][inter][stage].varValue))
                            print(value if rec is Recompute.TYPE.NONE else f"+ {value} {rec.name}", end=" ")
                    print(
                        (" of chunk " +
                         str(inter) if self.num_of_interleave_ != 1 else ""),
                        " to stage " + str(stage),
                    )
        for s in range(self.num_of_stage_):
            logger.debug(f"{self.MEM_OVERHEAD_NAME}[{s}] ="
                         f"{self.variables_[self.MEM_OVERHEAD_NAME][s].varValue}")

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(f"{self.CHUNKS_SUM}[{v}] = {self.variables_[self.CHUNKS_SUM][v].varValue}")

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(f"{self.PREV_DIFF}[{v}] = {self.variables_[self.PREV_DIFF][v].varValue}")

        logger.debug(f"{self.NEXT_DIFF} = {self.variables_[self.NEXT_DIFF].varValue}")
        logger.debug(f"{self.TOTAL_SUM} = {self.variables_[self.TOTAL_SUM].varValue}")
        logger.debug(f"{self.MAX_STAGE_TIME} = {self.variables_[self.MAX_STAGE_TIME].varValue}")
        logger.debug(f"{self.MAX_LAST_CHUNK} = {self.variables_[self.MAX_LAST_CHUNK].varValue}")

        for body_layer in range(len(self.layers_sorted_[Layer.type_enum.BODY]) - 1):
            for v in range(self.num_of_interleave_):
                for s in range(self.num_of_stage_):
                    logger.info(f"{self.LAYER_FRONTIER}[{body_layer}][{v}][{s}] = "
                                f"{self.variables_[self.LAYER_FRONTIER][body_layer][v][s].varValue}")

    def debug_print_solver_theoretical_memory(self):
        """print theoretical solver memory model"""
        logger.info("%s Solver Theoretical Memory Analysis %s", "=" * 20, "=" * 20)

        if self.vpp_less_memory_:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.seq_split_num_, self.num_of_micro_batch_, True)
            else:
                activation_nums = self.compute_less_activation_nums(
                    self.num_of_stage_, self.num_of_interleave_)
            overhead_factors = self.compute_lm_forward_in_backward(self.num_of_stage_)
        else:
            if self.seq_pipe:
                activation_nums = self.compute_activation_seq_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.seq_split_num_, self.num_of_micro_batch_, False)
            else:
                activation_nums = self.compute_activation_nums(
                    self.num_of_stage_, self.num_of_interleave_, self.num_of_micro_batch_)
            overhead_factors = self.compute_forward_in_backward(self.num_of_stage_, self.num_of_micro_batch_)

        # compute theoretical value for each stage
        for s in range(self.num_of_stage_):
            param_mem = self.stage_param_memory(
                self.variables_,
                self.layers_sorted_,
                s,
                self.num_of_stage_,
                self.num_of_interleave_
            ).value()

            act_mem = self.stage_active_memory(
                self.variables_,
                self.layers_sorted_,
                s,
                self.num_of_interleave_,
                activation_nums
            ).value()

            overhead = self.variables_[self.MEM_OVERHEAD_NAME][s].varValue * overhead_factors[s]

            total = param_mem + act_mem + overhead + self.constant_memory_

            logger.info("Stage %d Solver Memory Analysis:", s)
            logger.info(f"Parameter Memory:     {param_mem:.2f}")
            logger.info(f"Activation Memory:    {act_mem:.2f}")
            logger.info(f"Memory Overhead:      {overhead:.2f}")
            logger.info(f"Constant Memory:      {self.constant_memory_:.2f}")
            logger.info(f"Total Theoretical Memory: {total:.2f}")

    def solve(self, time_limit=90, dump_folder=None):
        """Solve the problem and print the results"""
        logger.info("solve:out folder = %s", dump_folder)
        self.dump_problem(dump_folder)
        solver = lpSolver.getSolver("PULP_CBC_CMD", timeLimit=time_limit)
        self.problem_.solve(solver)

        self.print_results()

        self.debug_print_solver_theoretical_memory()

        for name, result in self.result().items():
            logger.output("%s %s %s", name, result, "\n")

    def result(self) -> dict[str, list[list[str]]]:
        """return schedule distribution for each layer (in the form of a dict)"""
        r = {}
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = layer.name_
            inter = []
            for i in range(self.num_of_interleave_):
                stage = []
                for s in range(self.num_of_stage_):
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            stage.append(
                                str(
                                    self.variables_.get(layer_name)[rec][i]
                                    [s].varValue) + " + ")
                inter.append(stage)
            r[layer_name] = inter
        return r

    def _create_problem_(self, description: str) -> lpSolver.LpProblem:
        """create the problem"""
        prob = lpSolver.LpProblem(description, lpSolver.LpMinimize)
        layers_sorted = self.layers_sorted_
        num_of_stage = self.num_of_stage_
        num_of_interleave = self.num_of_interleave_
        num_of_micro_batch = self.num_of_micro_batch_
        max_memory = self.max_memory_
        # Local variable declaration
        # max time that a "main" stage have to take (var to minimize)
        pipeline_total_time = lpSolver.LpVariable("pipeline_total_time", 0,
                                                  None, lpSolver.LpContinuous)

        # Var to Minimize
        prob += pipeline_total_time

        self.add_total_nb_layer_constraint(prob, self.variables_, layers_sorted)
        self.add_stage_nb_layer_constraint(prob, self.variables_, layers_sorted)
        self.add_multimodal_sequence_constraint(prob, self.variables_, layers_sorted)
        self.add_performance_constraint(prob, layers_sorted, pipeline_total_time)

        constraint = PipelineMemoryConstraint(
            prob=prob,
            variables=self.variables_,
            layers_sorted=layers_sorted,
            num_of_stage=num_of_stage,
            num_of_interleave=num_of_interleave,
            micro_batch=num_of_micro_batch,
            memory_limit=max_memory,
        )
        if self.has_some_memory_info():
            self.add_pipeline_memory_constraint(constraint)
        return prob

    def _create_variables_to_solve_(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            layers: dict[Layer.type_enum, list[Layer]],
    ):
        """create variables to solve"""
        variables = {}

        variables[self.TOTAL_SUM] = lpSolver.LpVariable(
            self.TOTAL_SUM, 0, None, lpSolver.LpContinuous)

        chunks_sum_dict = lpSolver.LpVariable.dicts(
            name=self.CHUNKS_SUM,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous
        )
        chunks_sum_list = list(chunks_sum_dict.values())
        variables[self.CHUNKS_SUM] = chunks_sum_list

        prev_diff_dict = lpSolver.LpVariable.dicts(
            name=self.PREV_DIFF,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous
        )
        prev_diff_list = list(prev_diff_dict.values())
        variables[self.PREV_DIFF] = prev_diff_list

        logger.debug(f" range({self.LAYER_FRONTIER}) = "
                     f"{range(1, len(self.layers_sorted_[Layer.type_enum.BODY]))}")
        layer_frontier_dict = lpSolver.LpVariable.dicts(
            name=self.LAYER_FRONTIER,
            indices=(
                range(1, len(self.layers_sorted_[Layer.type_enum.BODY])),
                range(0, self.num_of_interleave_),
                range(0, self.num_of_stage_)),
            lowBound=0,
            upBound=1,
            cat=lpSolver.LpBinary
        )
        layer_frontier_list = list(layer_frontier_dict.values())
        variables[self.LAYER_FRONTIER] = layer_frontier_list

        variables[self.NEXT_DIFF] = lpSolver.LpVariable(
            self.NEXT_DIFF, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_STAGE_TIME] = lpSolver.LpVariable(
            self.MAX_STAGE_TIME, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_LAST_CHUNK] = lpSolver.LpVariable(
            self.MAX_LAST_CHUNK, 0, None, lpSolver.LpContinuous)

        lp_variable_dict = lpSolver.LpVariable.dicts(
            name=self.MEM_OVERHEAD_NAME,
            indices=(range(0, self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpInteger,
        )
        variables_list = list(lp_variable_dict.values())
        variables[self.MEM_OVERHEAD_NAME] = variables_list

        for layer in layers[Layer.type_enum.BODY]:
            variable_dict = lpSolver.LpVariable.dicts(
                name=layer.name_,
                indices=(
                    range(0, len(Recompute.TYPE)),
                    range(0, num_of_interleave),
                    range(0, num_of_stage),
                ),
                lowBound=0,
                upBound=None,
                cat=lpSolver.LpInteger,
            )
            variable_values = list(variable_dict.values())
            interleave_values = []
            for interleave in variable_values:
                interleave_value = [y for y in interleave.values()]
                interleave_values.append(interleave_value)
            variables[layer.name_] = interleave_values

        return variables

    ############################################
    #             Time Constraint              #
    ############################################
    def _max_stage_bound_i_fp(self, layers_sorted, stage_id, inter_f):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (self.variables_[layer.name_][rec][inter_f][stage_id] *
                              layer.forward_time_)
        return bound

    def _max_stage_bound_i_bp(self, layers_sorted, stage_id, inter_b):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (self.variables_[layer.name_][rec][inter_b][stage_id] *
                              layer.backward_time_rec_[rec])
        return bound

    def _max_stage_bound_head_tail(self, layers_sorted, stage_id, inter_f,
                                   inter_b):
        """maximize the stage bound of head and tail"""
        bound = lpSolver.LpAffineExpression()
        if stage_id == 0:
            if inter_f == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_
            if inter_b == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2
        if stage_id == self.num_of_stage_ - 1:
            if inter_f == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_
            if inter_b == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2
        return bound

    def _total_sum(self, layers_sorted):
        """sum up the layer time"""
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    for inter in range(self.num_of_interleave_):
                        for stage in range(self.num_of_stage_):
                            bound += self.variables_[layer.name_][rec][inter][stage] * (
                                layer.forward_time_ +
                                layer.backward_time_rec_[rec])
        return bound

    def body_layer_time(self, prop, layer, inter, stage):
        """calculate body layer time"""
        if prop == self.PROP_PHASE.FW:
            bound = lpSolver.lpSum(
                self.variables_[layer.name_][rec][inter][stage] * layer.forward_time_
                for rec in Recompute.TYPE if self.recompute_considered_[rec])
        else:
            bound = lpSolver.lpSum(
                self.variables_[layer.name_][rec][inter][stage] * layer.backward_time_rec_[rec]
                for rec in Recompute.TYPE if self.recompute_considered_[rec])

        return bound

    def micro_batch_time(self, prop, layers_sorted, inter, stage):
        """computes the time taken by a given micro batch propagation"""
        bound = lpSolver.LpAffineExpression()
        if prop == self.PROP_PHASE.FW:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_
        else:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2
        return bound

    def _chunks_sum(self, layers_sorted, v):
        """sum up the warm-up and cool-down time of a given chunk"""
        bound = lpSolver.LpAffineExpression()
        for stage in range(self.num_of_stage_):
            bound += self.micro_batch_time(self.PROP_PHASE.FW, layers_sorted, v, stage)
            bound += self.micro_batch_time(self.PROP_PHASE.BW, layers_sorted, v, stage)
        # normalize
        bound = bound / self.num_of_stage_
        return bound

    def _prev_diff_sum(self, layers_sorted, prob, v):
        """models bubble time for the first diagonal (forward, interleave 0)"""
        max_prev_stages = lpSolver.LpVariable.dicts(
            name="max_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_prev_stages = lpSolver.LpVariable.dicts(
            name="diff_with_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        prob += max_prev_stages[0] >= (self.micro_batch_time(
            self.PROP_PHASE.FW, layers_sorted, v, 0))

        for stage in range(1, self.num_of_stage_):
            prob += max_prev_stages[stage] >= max_prev_stages[stage - 1]
            prob += max_prev_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, layers_sorted, v, stage))

            prob += diff_with_prev_stages[stage] >= (
                max_prev_stages[stage - 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, layers_sorted, v, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_prev_stages[s] for s in range(1, self.num_of_stage_))
        return bound

    def _next_diff_sum(self, layers_sorted, prob):
        """models bubble time for the last diagonal (forward, last chunk)"""
        last_chunk = self.num_of_interleave_ - 1
        max_next_stages = lpSolver.LpVariable.dicts(
            name="max_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_next_stages = lpSolver.LpVariable.dicts(
            name="diff_with_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        prob += max_next_stages[self.num_of_stage_ -
                                1] >= (self.micro_batch_time(
                                    self.PROP_PHASE.FW, layers_sorted, last_chunk,
                                    self.num_of_stage_ - 1))

        for stage in reversed(range(0, self.num_of_stage_ - 1)):
            prob += max_next_stages[stage] >= max_next_stages[stage + 1]
            prob += max_next_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, layers_sorted, last_chunk, stage))

            prob += diff_with_next_stages[stage] >= (
                max_next_stages[stage + 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, layers_sorted, last_chunk, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_next_stages[s] for s in range(self.num_of_stage_ - 1))
        return bound
