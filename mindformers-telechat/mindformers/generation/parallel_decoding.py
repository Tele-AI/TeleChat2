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
"""Parallel Decoding for text generation."""
import numpy as np
import mindspore as ms
from mindspore.common.tensor import Tensor


_support_parallel_decoding = ("la", "memory_decoding", "prefix_cache")
_logits_process = {}
_pre_process = {}


def register_logits_process(name=None):
    def register_fn(fn):
        if name is None:
            _logits_process[fn.__name__] = fn
        else:
            _logits_process[name] = fn
        return fn
    return register_fn


def register_pre_process(name=None):
    def register_fn(fn):
        if name is None:
            _pre_process[fn.__name__] = fn
        else:
            _pre_process[name] = fn
        return fn
    return register_fn


@register_logits_process('la')
def _la_logits_process(logits, q_seq_lens, block_tables, prefill):
    """ lookahead decoding logits process """
    shape = logits.shape
    if len(shape) > 2:
        logits = logits.reshape(-1, shape[-1])
    if q_seq_lens is not None and not isinstance(q_seq_lens, int) and prefill:
        index = []
        batch_size = len(block_tables)
        max_seq_len = logits.shape[0] // batch_size
        for i in range(batch_size):
            index.append(i * max_seq_len + q_seq_lens[i] - 1)
        index = Tensor(index, ms.int32)
        logits = logits[index]
    return logits


def parallel_decoding_logits_process(config, logits, q_seq_lens, block_tables, prefill):
    """ parallel decoding logits process """
    logits_process_fn = None
    if parallel_decoding_control(config):
        logits_process_fn = _logits_process.get(config.parallel_decoding_params.get('parallel_decoding'))
    if logits_process_fn is not None:
        return logits_process_fn(logits, q_seq_lens, block_tables, prefill)
    return logits


def _construct_mask(q_seq_lens):
    token_len = sum(q_seq_lens)
    mask = (np.tril(np.ones((token_len, token_len), dtype=np.float16), k=0) - 1) * -1
    start = 0
    for q_seq_len in q_seq_lens:
        mask[start + q_seq_len:, start: start + q_seq_len] = 1
        start += q_seq_len
    return mask


def _parallel_decoding_pad(inputs, axis, pad_len, value):
    if isinstance(inputs, Tensor):
        inputs = inputs.numpy()
    shape = list(inputs.shape)
    shape[axis] = pad_len - shape[axis]
    if shape[axis] < 0:
        return inputs
    pad = np.full(shape, value, inputs.dtype)
    inputs = np.concatenate((inputs, pad), axis)
    return inputs


def parallel_decoding_process(config, input_ids, model_inputs, **model_kwargs):
    pre_process_fn = None
    if hasattr(config, 'parallel_decoding_params'):
        pre_process_fn = _pre_process.get(config.parallel_decoding_params.get('parallel_decoding'))
    if pre_process_fn is not None:
        return pre_process_fn(config, input_ids, model_inputs, **model_kwargs)
    block_tables = model_kwargs.get('block_tables').astype(np.int32)
    slot_mapping = model_kwargs.get('slot_mapping').astype(np.int32)
    return model_inputs, block_tables, slot_mapping


def _parallel_decoding_pad_2d_tensor(inputs, pad_seq_len, lens, value):
    batch_size = len(lens)
    inputs_pad = np.full((batch_size, pad_seq_len), value, inputs.dtype)
    start = 0
    for i, length in enumerate(lens):
        end = start + length
        inputs_pad[i, :length] = inputs[start: end]
        start = end
    return inputs_pad


@register_pre_process('la')
def _la_pre_process(config, input_ids, model_inputs, **model_kwargs):
    """ parallel decoding process """

    _ = config
    block_tables = model_kwargs.get('block_tables').astype(np.int32)
    slot_mapping = model_kwargs.get('slot_mapping').astype(np.int32)

    # adapt warmup stage, `q_seq_lens` is int dtype
    if model_kwargs.get('q_seq_lens') is not None and isinstance(model_kwargs.get('q_seq_lens'), int):
        q_seq_lens = None
    else:
        q_seq_lens = model_kwargs.get('q_seq_lens')

    if q_seq_lens is not None:
        input_ids = input_ids.reshape(-1)
        sum_q_seq_lens = sum(q_seq_lens)
        max_q_seq_lens = max(q_seq_lens)
        if len(input_ids) != sum_q_seq_lens:
            input_ids = input_ids.tolist()
            input_ids_list = list()
            start = 0
            for q_seq_len in q_seq_lens:
                input_ids_list += input_ids[start: start + q_seq_len]
                start += max_q_seq_lens
            input_ids = np.array(input_ids_list, dtype=np.int32)
        if len(slot_mapping) != sum_q_seq_lens:
            slot_mapping = slot_mapping.tolist()
            slot_mapping_list = list()
            start = 0
            for q_seq_len in q_seq_lens:
                slot_mapping_list += slot_mapping[start: start + q_seq_len]
                start += max_q_seq_lens
            slot_mapping = np.array(slot_mapping_list, dtype=np.int32).reshape(-1)

    input_ids = input_ids.reshape(1, -1)
    seq_len = input_ids.shape[-1]

    attention_mask = model_kwargs.get('spec_mask')
    if attention_mask is None:
        if q_seq_lens is not None:  # prefill stage
            attention_mask = _construct_mask(q_seq_lens)
        else:
            attention_mask = np.zeros((seq_len, seq_len), dtype=np.float16)
    attention_mask = attention_mask.astype(np.bool_).astype(np.float16)

    position_ids = model_kwargs.get('position_ids')
    if position_ids is None:
        position_ids = np.zeros((1, seq_len), dtype=np.int32)
    elif len(position_ids.shape) == 1:
        position_ids = position_ids.reshape(1, -1)

    if q_seq_lens is None:
        q_seq_lens = np.ones((seq_len,), dtype=np.int32)

    model_inputs['input_ids'] = Tensor.from_numpy(input_ids.astype(np.int32)) \
        if isinstance(input_ids, np.ndarray) else Tensor(input_ids, dtype=ms.int32)
    model_inputs['attention_mask'] = Tensor.from_numpy(attention_mask.astype(np.float16)) \
        if isinstance(attention_mask, np.ndarray) else Tensor(attention_mask, dtype=ms.float16)
    model_inputs['position_ids'] = Tensor.from_numpy(position_ids.astype(np.int32)) \
        if isinstance(position_ids, np.ndarray) else Tensor(position_ids, dtype=ms.int32)
    model_inputs['q_seq_lens'] = Tensor.from_numpy(q_seq_lens.astype(np.int32)) \
        if isinstance(q_seq_lens, np.ndarray) else Tensor(q_seq_lens, dtype=ms.int32)
    ms.hal.synchronize()

    return model_inputs, block_tables, slot_mapping


@register_pre_process('memory_decoding')
def _memory_decoding_pre_process(config, input_ids, model_inputs, **model_kwargs):
    """ memory decoding preprocess """
    _ = config

    if model_kwargs.get('q_seq_lens') is not None:
        input_ids = input_ids.reshape((1, -1))

    model_inputs['input_ids'] = Tensor(input_ids, ms.int32)

    block_tables = model_kwargs.get('block_tables').astype(np.int32)
    slot_mapping = model_kwargs.get('slot_mapping').astype(np.int32)

    return model_inputs, block_tables, slot_mapping


@register_pre_process('prefix_cache')
def _prefix_cache_pre_process(config, input_ids, model_inputs, **model_kwargs):
    """ prompt cache pre process """
    _ = config

    model_inputs['input_ids'] = Tensor(input_ids, ms.int32)

    block_tables = model_kwargs.get('block_tables').astype(np.int32)
    slot_mapping = model_kwargs.get('slot_mapping').astype(np.int32)

    return model_inputs, block_tables, slot_mapping


def parallel_decoding_control(config):
    if hasattr(config, "parallel_decoding_params") and config.parallel_decoding_params is not None:
        return config.parallel_decoding_params.get("parallel_decoding") in _support_parallel_decoding
    return False
