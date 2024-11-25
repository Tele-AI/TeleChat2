# Copyright 2020-2024 The HuggingFace Inc. team
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
# ============================================================================
"""Logits Processor for generation."""
from importlib import import_module
from dataclasses import dataclass
from threading import Thread
from typing import Union, List
import numpy as np

import mindspore as ms
from mindspore import Tensor, mint
from mindspore.ops.auto_generate import Scatter  # internal api for aclnn op

from mindformers.version_control import get_scatter
from .utils import log_softmax, softmax, topk
from ..tools.logger import logger

__all__ = ["LogitsProcessor", "LogitsProcessorList", "RepetitionPenaltyLogitsProcessor",
           "FrequencyPenaltyLogitsProcessor", "PresencePenaltyLogitsProcessor",
           "LogitNormalization", "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper",
           "MinLengthLogitsProcessor", "MinNewTokensLengthLogitsProcessor", "SamplingLogitsProcessor",
           "GreedySearchLogitsProcessor"]


def run_using_numpy():
    """Set run postprocess using Numpy operators or MindSpore operators."""
    context_module = import_module("mindformers.core.context.build_context")
    context_instance = context_module.CONTEXT_INSTANCE
    return context_module.get_context("postprocess_use_numpy") if context_instance is not None else False


@dataclass
class Threshold:
    """
    Define the Threshold class.
    Args:
        value (`float`): the threshold value.
        check_equal (`bool`): whether check equal to the threshold value or not.
    """
    value: float
    check_equal: bool


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __init__(self):
        self.use_numpy = run_using_numpy()
        self.scatter = get_scatter()

    def selected_scatter(self, input_tensor, dim, index, src):
        if isinstance(self.scatter, Scatter):
            return self.scatter(input_tensor, dim, index, src, reduce=0)
        return self.scatter(input_tensor, dim, index, src)

    def __call__(self, input_ids, scores, **kwargs):
        """Method for processing logits."""
        if self.use_numpy:
            return self.process_np(scores, input_ids)
        return self.process_ms(scores, input_ids, **kwargs)

    def process_ms(self, logits, sequence_ids, **kwargs):
        """
        Do Process in MindSpore.

        Args:
            logits: scores from model
            sequence_ids: input_ids from tokenizer
            **kwargs: other attributes.

        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def process_np(self, logits, sequence_ids):
        """
        Do Process in Numpy.

        Args:
            logits: scores from model
            sequence_ids: input_ids from tokenizer

        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @staticmethod
    def check_params(params, params_name, force_float=False, force_int=False, force_int_tuple=False, low_threshold=None,
                     high_threshold=None):
        """
        Check params validation.
        """
        if params is None:
            raise ValueError(f"`{params_name}` is required, but is None.")
        if isinstance(params, Tensor):
            # For performance, DO NOT run checker for Tensor.
            return params
        params = params.tolist() if isinstance(params, np.ndarray) else params

        if force_int:
            if isinstance(params, list):
                raise ValueError(f"`{params_name}` is required to be a integer, but is {params}.")
            params = int(params)
        elif force_float:
            if isinstance(params, list):
                raise ValueError(f"`{params_name}` is required to be a float, but is {params}.")
            params = float(params)
        elif force_int_tuple:
            params = params if isinstance(params, list) else [params]
            if not all(isinstance(i, int) for i in params):
                raise ValueError(f"`{params_name}` is required to be the list of integers, but is {params}")

        if isinstance(low_threshold, Threshold):
            value = low_threshold.value
            check_equal = low_threshold.check_equal
            if check_equal:
                if isinstance(params, list):
                    if any(i <= value for i in params):
                        raise ValueError(f"`{params_name}` is required to be the list with value more than {value}, "
                                         f"but is {params}.")
                elif params <= value:
                    raise ValueError(f"`{params_name}` is required to be more than {value}, but is {params}.")
            else:
                if isinstance(params, list):
                    if any(i < value for i in params):
                        raise ValueError(f"`{params_name}` is required to be the list with value no less than {value}, "
                                         f"but is {params}.")
                elif params < value:
                    raise ValueError(f"`{params_name}` is required to be no less than {value}, but is {params}.")
        if isinstance(high_threshold, Threshold):
            value = high_threshold.value
            check_equal = high_threshold.check_equal
            if check_equal:
                if isinstance(params, list):
                    if any(i >= value for i in params):
                        raise ValueError(f"`{params_name}` is required to be the list with value less than {value}, "
                                         f"but is {params}.")
                elif params >= value:
                    raise ValueError(f"`{params_name}` is required to be less than {value}, but is {params}.")
            else:
                if isinstance(params, list):
                    if any(i > value for i in params):
                        raise ValueError(f"`{params_name}` is required to be the list with value no more than {value}, "
                                         f"but is {params}.")
                elif params > value:
                    raise ValueError(f"`{params_name}` is required to be no more than {value}, but is {params}.")
        return params


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently
    process a `scores` input tensor. This class inherits from list and adds a specific *__call__* method
    to apply each [`LogitsProcessor`] to the inputs.
    """

    def __init__(self):
        super().__init__()
        self.use_numpy = run_using_numpy()

    def __call__(self, input_ids, scores, is_finished=None, **kwargs):
        if not self.use_numpy:
            return self.process_ms(input_ids, scores, **kwargs)

        all_threads = []
        for i in range(0, input_ids.shape[0]):
            if is_finished and is_finished[i]:
                continue
            thread = Thread(target=self.process, args=(i, input_ids, scores))
            all_threads.append(thread)
            thread.start()
        for thread in all_threads:
            thread.join()
        return scores

    def process_ms(self, input_ids, scores, **kwargs):
        """apply process using mindspore"""
        if not self:
            # logits_process list is empty, return.
            return scores

        input_ids = Tensor.from_numpy(input_ids)
        scores = Tensor.from_numpy(scores)
        for processor in self:
            scores = processor(input_ids, scores, **kwargs)

        return scores.asnumpy()

    def process(self, i, input_ids, scores):
        """apply process"""
        input_ids = input_ids[i: i + 1]
        scores_i = scores[i: i + 1]
        for processor in self:
            scores_i = processor(input_ids, scores_i)
        scores[i] = scores_i


class TemperatureLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float = None):
        super().__init__()
        if temperature is not None:
            temperature = self.check_params(temperature, "temperature", force_float=True,
                                            low_threshold=Threshold(0, True))

        self.temperature = temperature

    def process_ms(self, logits, sequence_ids, **kwargs):
        temperature = kwargs.get("temperature", self.temperature)
        temperature = self.check_params(temperature, "temperature", force_float=True, low_threshold=Threshold(0, True))
        return logits / temperature

    def process_np(self, logits, sequence_ids):
        return logits / self.temperature


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, repetition_penalty: float = None):
        super().__init__()
        if repetition_penalty is not None:
            repetition_penalty = self.check_params(repetition_penalty, "repetition_penalty", force_float=True,
                                                   low_threshold=Threshold(0, True))

        self.penalty = repetition_penalty

    def process_ms(self, logits, sequence_ids, **kwargs):
        repetition_penalty = kwargs.get("repetition_penalty", self.penalty)
        repetition_penalty = self.check_params(repetition_penalty, "repetition_penalty", force_float=True,
                                               low_threshold=Threshold(0, True))
        repetition_logits = mint.gather(logits, 1, sequence_ids)
        repetition_logits = mint.where(repetition_logits < 0, repetition_logits * repetition_penalty,
                                       repetition_logits / repetition_penalty)
        return self.selected_scatter(logits, -1, sequence_ids, repetition_logits.astype(logits.dtype))

    def process_np(self, logits, sequence_ids):
        score = np.take_along_axis(logits, sequence_ids, axis=1)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        negative_index = score < 0
        positive_index = ~negative_index
        score[negative_index] = score[negative_index] * self.penalty
        score[positive_index] = score[positive_index] / self.penalty

        np.put_along_axis(logits, sequence_ids, score, axis=1)
        return logits


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on frequency sequences.

    Args:
        frequency_penalty (`float`):
            The parameter for frequency_penalty. 0.0 means no penalty.
        output_tokens_counts (`np.ndarray`):
            The counts for tokens.
    """

    def __init__(self, frequency_penalty: float = None, output_tokens_counts=None):
        super().__init__()
        if frequency_penalty is not None:
            frequency_penalty = self.check_params(frequency_penalty, "frequency_penalty", force_float=True,
                                                  low_threshold=Threshold(0, True))
        self.penalty = frequency_penalty
        self.output_tokens_counts = output_tokens_counts

    def process_ms(self, logits, sequence_ids, **kwargs):
        frequency_penalty = kwargs.get("frequency_penalty", self.penalty)
        frequency_penalty = self.check_params(frequency_penalty, "frequency_penalty", force_float=True,
                                              low_threshold=Threshold(0, True))
        output_tokens_counts = kwargs.get("output_tokens_counts", Tensor(
            self.output_tokens_counts) if self.output_tokens_counts is not None else None)
        logits -= frequency_penalty * output_tokens_counts
        return logits

    def process_np(self, logits, sequence_ids):
        logits -= self.penalty * self.output_tokens_counts
        return logits


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on presence sequences.

    Args:
        presence_penalty (`float`):
            The parameter for presence_penalty. 0.0 means no penalty.
        output_tokens_mask (`np.ndarray`):
            The mask for output tokens.
    """

    def __init__(self, presence_penalty: float = None, output_tokens_mask=None):
        super().__init__()
        if presence_penalty is not None:
            presence_penalty = self.check_params(presence_penalty, "presence_penalty", force_float=True,
                                                 low_threshold=Threshold(0, True))

        self.penalty = presence_penalty
        self.output_tokens_mask = output_tokens_mask

    def process_ms(self, logits, sequence_ids, **kwargs):
        presence_penalty = kwargs.get("presence_penalty", self.penalty)
        output_tokens_mask = kwargs.get("output_tokens_mask", Tensor(
            self.output_tokens_mask, dtype=ms.bool_) if self.output_tokens_mask is not None else None)
        presence_penalty = self.check_params(presence_penalty, "presence_penalty", force_float=True,
                                             low_threshold=Threshold(0, True))
        logits -= presence_penalty * output_tokens_mask
        return logits

    def process_np(self, logits, sequence_ids):
        logits -= self.penalty * self.output_tokens_mask
        return logits


class SamplingLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs sampling which is valid for MindIE.

    Args:
        do_sample (`np.ndarray`):
            The array marking do sample or not for each batch.
        seed_array (`np.ndarray`):
            The random seed array generated by MindIE.
    """

    def __init__(self, do_sample=None, seed_array=None):
        super().__init__()
        self.do_sample = do_sample
        self.seed_array = seed_array

    def process_ms(self, logits, sequence_ids, **kwargs):
        do_sample = kwargs.get("do_sample", Tensor(self.do_sample) if self.do_sample is not None else None)
        seed_array = kwargs.get("seed_array", self.seed_array)
        do_sample = self.check_params(do_sample, "do_sample")
        seed_array = self.check_params(seed_array, "seed_array")

        indices = mint.nonzero(do_sample > 0).squeeze(1)
        argmax_indices = mint.nonzero(do_sample == 0).squeeze(1)
        filtered_logits = mint.index_select(logits, dim=0, index=indices)
        sampled_probs = mint.nn.functional.softmax(filtered_logits, dim=-1)
        sampled_tokens = self.multinomial_ms(sampled_probs, 1, np.array(seed_array)[indices.asnumpy()]).squeeze(1)
        tokens = Tensor([-1] * len(do_sample), ms.int64)
        tokens = self.selected_scatter(tokens, 0, index=indices, src=sampled_tokens)
        if argmax_indices.numel() == 0:
            return logits, tokens.reshape(-1)
        filtered_logits = mint.index_select(logits, dim=0, index=argmax_indices)
        argmax_tokens = filtered_logits.argmax(axis=-1).astype(ms.int64)
        tokens = self.selected_scatter(tokens, 0, index=argmax_indices, src=argmax_tokens)
        return logits, tokens.reshape(-1)

    @staticmethod
    def multinomial_ms(prob_matrix, num_samples, seeds):
        """Multinomial in MS."""
        random_value = []
        for cur_seed in seeds:
            np.random.seed(cur_seed)
            random_value.append(np.random.rand(num_samples))
        sorted_prob, indices = mint.sort(prob_matrix, descending=True)
        cdf_matrix = mint.cumsum(sorted_prob, dim=-1)
        selected_ids = mint.searchsorted(cdf_matrix, Tensor(random_value))
        selected_ids = mint.clamp(selected_ids, min=0, max=indices.shape[-1] - 1)
        selected_tokens = mint.gather(indices, -1, selected_ids)
        return selected_tokens

    def process_np(self, logits, sequence_ids):
        indices = np.argwhere(self.do_sample > 0).squeeze(1)
        argmax_indices = np.argwhere(self.do_sample == 0).squeeze(1)
        filtered_logits = logits[indices]
        sampled_probs = softmax(filtered_logits, axis=-1)
        sampled_tokens = self.multinomial_np(sampled_probs, 1, self.seed_array[indices]).squeeze(1)
        tokens = np.array([-1] * len(self.do_sample), dtype=np.int64)
        tokens[indices] = sampled_tokens
        if argmax_indices.size == 0:
            return logits, tokens.reshape(-1)
        filtered_logits = logits[argmax_indices]
        argmax_tokens = filtered_logits.argmax(axis=-1).astype(np.int64)
        tokens[argmax_indices] = argmax_tokens
        return logits, tokens.reshape(-1)

    @staticmethod
    def multinomial_np(prob_matrix, num_samples, seeds):
        """Multinomial in Numpy."""
        random_value = []
        for cur_seed in seeds:
            np.random.seed(cur_seed)
            random_value.append(np.random.rand(num_samples))
        sorted_prob = np.sort(prob_matrix)[:, ::-1]
        indices = np.argsort(-prob_matrix, kind="stable")
        cdf_matrix = np.cumsum(sorted_prob, axis=-1)
        selected_ids = np.zeros((cdf_matrix.shape[0], 1))
        for i in range(selected_ids.shape[0]):
            selected_ids[i] = np.searchsorted(cdf_matrix[i], np.array(random_value)[i])
        selected_ids = np.clip(selected_ids, a_min=0, a_max=indices.shape[-1] - 1)
        selected_tokens = np.take_along_axis(indices, selected_ids.astype(np.int64), axis=-1)
        return selected_tokens


class TopPLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-50000`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float = None, filter_value: float = -50000, min_tokens_to_keep: int = 1):
        super().__init__()
        if top_p is not None:
            top_p = self.check_params(top_p, "top_p", force_float=True, low_threshold=Threshold(0, False),
                                      high_threshold=Threshold(1, False))
        min_tokens_to_keep = self.check_params(min_tokens_to_keep, "min_tokens_to_keep", force_int=True,
                                               low_threshold=Threshold(0, False))
        self.top_p = top_p
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = min_tokens_to_keep

    def process_ms(self, logits, sequence_ids, **kwargs):
        top_p = kwargs.get("top_p", self.top_p)
        filter_value = kwargs.get("filter_value", self.filter_value)
        min_tokens_to_keep = kwargs.get("min_tokens_to_keep", self.min_tokens_to_keep)
        top_p = self.check_params(top_p, "top_p", force_float=True, low_threshold=Threshold(0, False),
                                  high_threshold=Threshold(1, False))
        filter_value = self.check_params(filter_value, "filter_value", force_float=True)
        min_tokens_to_keep = self.check_params(min_tokens_to_keep, "min_tokens_to_keep", force_int=True,
                                               low_threshold=Threshold(0, False))

        sorted_logits, sorted_indices = mint.sort(logits, descending=True)
        cumulative_probs = mint.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = mint.cumsum(cumulative_probs, dim=-1)

        # Remove tokens with cumulative top_p above the threshold
        sorted_indices_to_keep = Tensor(cumulative_probs < top_p, ms.int32)
        sorted_indices_to_keep[:, :min_tokens_to_keep] = 1
        indices_to_keep = self.selected_scatter(sorted_indices_to_keep, -1, index=sorted_indices,
                                                src=sorted_indices_to_keep)
        return mint.where(indices_to_keep.astype("bool"), logits, filter_value)

    def process_np(self, logits, sequence_ids):
        candidate_logits = np.sort(logits)[:, ::-1]
        candidate_indices = np.argsort(-logits, kind="stable")
        cumulative_probs = softmax(candidate_logits)
        cumulative_probs = np.cumsum(cumulative_probs, axis=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_keep = cumulative_probs < self.top_p
        # Keep at least min_tokens_to_keep
        sorted_indices_to_keep[:, :self.min_tokens_to_keep] = 1

        # Set remove indices, filter negative value
        indices_to_remove = np.ones_like(logits).astype(np.bool_)
        np.put_along_axis(indices_to_remove, candidate_indices, ~sorted_indices_to_keep, axis=-1)
        logits[indices_to_remove] = self.filter_value
        return logits


class TopKLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int = None, filter_value: float = -50000, min_tokens_to_keep: int = 1):
        super().__init__()
        if top_k is not None:
            top_k = self.check_params(top_k, "top_k", force_int=True, low_threshold=Threshold(0, True))
            top_k = max(top_k, min_tokens_to_keep)

        self.top_k = top_k
        self.filter_value = float(filter_value)

    def process_ms(self, logits, sequence_ids, **kwargs):
        max_top_k = kwargs.get("max_top_k", self.top_k)
        top_k = kwargs.get("top_k", Tensor([[self.top_k - 1]] * logits.shape[0]) if self.top_k is not None else None)
        filter_value = kwargs.get("filter_value", self.filter_value)
        min_tokens_to_keep = kwargs.get("min_tokens_to_keep")
        max_top_k = self.check_params(max_top_k, "max_top_k", force_int=True, low_threshold=Threshold(0, True))
        filter_value = self.check_params(filter_value, "filter_value", force_float=True)
        max_top_k = max(max_top_k, min_tokens_to_keep) if min_tokens_to_keep is not None else max_top_k

        topk_logits, _ = mint.topk(logits, max_top_k, 1)
        kth_logits = mint.gather(topk_logits, 1, top_k).reshape(-1, 1)
        return mint.where(logits < kth_logits, filter_value, logits)

    def process_np(self, logits, sequence_ids):
        top_k = min(self.top_k, logits.shape[-1])  # Safety Check
        # Remove all tokens with a probability less than the last token of the top_k
        indices_to_remove = logits < topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = self.filter_value
        return logits


class LogitNormalization(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """

    def process_ms(self, logits, sequence_ids, **kwargs):
        return mint.log(mint.nn.functional.softmax(logits, dim=-1))

    def process_np(self, logits, sequence_ids):
        return log_softmax(logits, axis=-1)


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, min_length: int = None, eos_token_id: Union[int, List[int]] = None, pad_token_id: int = None):
        super().__init__()
        if min_length is not None:
            min_length = self.check_params(min_length, "min_length", force_int=True, low_threshold=Threshold(0, False))
        if eos_token_id is not None:
            eos_token_id = self.check_params(eos_token_id, "eos_token_id", force_int_tuple=True)
            if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
                logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def process_ms(self, logits, sequence_ids, **kwargs):
        pad_token_id = kwargs.get("pad_token_id", self.pad_token_id)
        min_length = kwargs.get("min_length", self.min_length)
        eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        min_length = self.check_params(min_length, "min_length", force_int=True, low_threshold=Threshold(0, False))
        eos_token_id = self.check_params(eos_token_id, "eos_token_id", force_int_tuple=True)

        valid_length_each_example = mint.max(mint.nonzero(sequence_ids != pad_token_id), dim=-1)[0] + 1
        cur_len = mint.max(valid_length_each_example)
        if cur_len < min_length:
            eos_token_id = Tensor([eos_token_id] * logits.shape[0])
            eos_token_value = Tensor([[-float("inf")] * eos_token_id.shape[1]] * eos_token_id.shape[0],
                                     dtype=logits.dtype)
            logits = self.selected_scatter(logits, -1, index=eos_token_id, src=eos_token_value)
        return logits

    def process_np(self, logits, sequence_ids):
        batch_size = sequence_ids.shape[0]

        valid_length_each_example = []
        for i in range(batch_size):
            valid_length_each_example.append(np.max(np.argwhere(sequence_ids[i] != self.pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)

        cur_len = np.max(valid_length_each_example)
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                logits[:, i] = -float("inf")
        return logits


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Note that for decoder-only models, such as Llama2, `min_length` will compute the length of `prompt + newly
    generated tokens` whereas for other models it will behave as `min_new_tokens`, that is, taking only into account
    the newly generated ones.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, prompt_length_to_skip: int = None, min_new_tokens: int = None,
                 eos_token_id: Union[int, List[int]] = None, pad_token_id: int = None):
        super().__init__()
        if prompt_length_to_skip is not None:
            prompt_length_to_skip = self.check_params(prompt_length_to_skip, "prompt_length_to_skip", force_int=True,
                                                      low_threshold=Threshold(0, False))
        if min_new_tokens is not None:
            min_new_tokens = self.check_params(min_new_tokens, "min_new_tokens", force_int=True,
                                               low_threshold=Threshold(0, False))
        if eos_token_id is not None:
            eos_token_id = self.check_params(eos_token_id, "eos_token_id", force_int_tuple=True,
                                             low_threshold=Threshold(0, False))

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def process_ms(self, logits, sequence_ids, **kwargs):
        pad_token_id = kwargs.get("pad_token_id", self.pad_token_id)
        prompt_length_to_skip = kwargs.get("prompt_length_to_skip", self.prompt_length_to_skip)
        eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        min_new_tokens = kwargs.get("min_new_tokens", self.min_new_tokens)

        prompt_length_to_skip = self.check_params(prompt_length_to_skip, "prompt_length_to_skip", force_int=True,
                                                  low_threshold=Threshold(0, False))
        min_new_tokens = self.check_params(min_new_tokens, "min_new_tokens", force_int=True,
                                           low_threshold=Threshold(0, False))
        eos_token_id = self.check_params(eos_token_id, "eos_token_id", force_int_tuple=True,
                                         low_threshold=Threshold(0, False))

        valid_length_each_example = mint.max(mint.nonzero(sequence_ids != pad_token_id), dim=-1)[0] + 1
        cur_len = mint.max(valid_length_each_example)
        new_tokens_length = cur_len - prompt_length_to_skip
        if new_tokens_length < min_new_tokens:
            eos_token_id = Tensor([eos_token_id] * logits.shape[0])
            eos_token_value = Tensor([[-float("inf")] * eos_token_id.shape[1]] * eos_token_id.shape[0],
                                     dtype=logits.dtype)
            logits = self.selected_scatter(logits, -1, index=eos_token_id, src=eos_token_value)
        return logits

    def process_np(self, logits, sequence_ids):
        batch_size = logits.shape[0]
        valid_length_each_example = []
        for i in range(batch_size):
            valid_length_each_example.append(np.max(np.argwhere(sequence_ids[i] != self.pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)

        cur_len = np.max(valid_length_each_example)
        new_tokens_length = cur_len - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                logits[:, i] = -float("inf")
        return logits


class GreedySearchLogitsProcessor(LogitsProcessor):
    """LogitsProcessor in GreedySearch Mode."""

    def process_ms(self, logits, sequence_ids, **kwargs):
        return mint.argmax(logits, -1)

    def process_np(self, logits, sequence_ids):
        return np.argmax(logits, axis=-1)
