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
"""utils for text generation."""
from collections import UserDict
from dataclasses import dataclass
from threading import Thread
from typing import Optional
import numpy as np


def log_softmax(x, axis=None):
    """numpy implemented log softmax function. refers to https://github.com/scipy/scipy/blob/v1.11.1/scipy/special/_logsumexp.py"""
    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    out = tmp - out
    return out


def softmax(x, axis=None):
    """numpy implemented softmax function. refers to https://github.com/scipy/scipy/blob/v1.11.1/scipy/special/_logsumexp.py"""
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def softmax_single(i, res, x):
    res[i] = softmax(x)


def softmax_with_threads(x, is_finished=None):
    """calculate softmax with threads"""
    res = np.ones_like(x)
    all_threads = []
    for i in range(0, res.shape[0]):
        if is_finished and is_finished[i]:
            continue
        thread = Thread(target=softmax_single,
                        args=(i, res, x[i]))
        all_threads.append(thread)
        thread.start()
    for thread in all_threads:
        thread.join()
    return res


def topk(x, top_k, axis=-1, largest=True, sort=True):
    """numpy implemented topk sample."""
    # safety check
    if x.shape[axis] <= top_k:
        top_k = x.shape[axis] - 1
    if largest:
        topk_index = np.argpartition(-x, top_k, axis=axis)
    else:
        topk_index = np.argpartition(x, top_k, axis=axis)
    topk_index = np.take(topk_index, np.arange(top_k), axis=axis)
    topk_data = np.take_along_axis(x, topk_index, axis=axis)
    if sort:
        sort_index = (
            np.argsort(-topk_data, axis=axis)
            if largest
            else np.argsort(topk_data, axis=axis)
        )
        topk_data = np.take_along_axis(topk_data, sort_index, axis=axis)
        topk_index = np.take_along_axis(topk_index, sort_index, axis=axis)
    return topk_data, topk_index


@dataclass
class GenerateOutput(UserDict):
    """
    Outputs of generate.

    Args:
        sequences (`list` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(np.ndarray)` *optional*, returned when `output_scores=True` is passed or when
            `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token
            before SoftMax) at each generation step. Tuple of `np.ndarray` with up to `max_new_tokens` elements
            (one element for each generated token), with each item of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(np.ndarray)` *optional*, returned when `output_logits=True` is passed or when
            `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token
            before SoftMax) at each generation step. Tuple of `np.ndarray` with up to `max_new_tokens` elements
            (one element for each generated token), with each item of shape `(batch_size, config.vocab_size)`.
    """
    sequences: list = None
    scores: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__init__(
            sequences=self.sequences,
            scores=self.scores,
            logits=self.logits
        )


@dataclass
class InferOutput(UserDict):
    """
    Outputs of infer api.

    Args:
        target_list (`list` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length`
            or shorter if all batches finished early due to the `eos_token_id`.
        probs (`np.ndarray` *optional*, returned when `output_scores=True` is passed or when
            `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at a single infer step, with shape of `(batch_size, config.vocab_size)`.
        logits (`np.ndarray` *optional*, returned when `output_logits=True` is passed or when
            `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token
            before SoftMax) at a single infer step, with shape of `(batch_size, config.vocab_size)`.
    """
    target_list: list = None
    probs: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__init__(
            target_list=self.target_list,
            probs=self.probs,
            logits=self.logits
        )
