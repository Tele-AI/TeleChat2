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
"""Dataset Packing Handler."""
import numpy as np

from mindformers.dataset.handler.base_handler import BaseInstructDataHandler
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


def _init_data(data_names):
    """init dataset columns with empty list"""
    return {
        k: [] if k != 'actual_seq_len' else [0]
        for k in data_names
    }


def _pad_data(data, max_length, pad_value):
    """pad data with pad length"""
    pad_length = max_length - len(data)
    return np.pad(
        np.array(data),
        (0, pad_length),
        mode='constant',
        constant_values=pad_value).tolist()


def _add_dataset(dataset, data, seq_length, **kwargs):
    """add data to dataset"""
    # packing data at max length
    pad_token = kwargs.get('pad_token', 0)
    ignore_token = kwargs.get('ignore_token', -100)
    data['input_ids'] = _pad_data(data.get('input_ids'), seq_length, pad_token)
    data['labels'] = _pad_data(data.get('labels'), seq_length, ignore_token)

    data_names = list(dataset.keys())
    for k in data_names:
        dataset[k].append(data.get(k) if k != 'actual_seq_len' else data.get(k)[1:])
    return dataset


def _process_sample(
        sample: dict,
        data: dict,
        dataset: dict,
        **kwargs
):
    """process single sample in dataset"""
    seq_length = kwargs.get('seq_length')

    cur_seq_length = len(sample.get('input_ids'))
    # skip data out of bounds
    if cur_seq_length > seq_length:
        return dataset, data

    data_names = list(dataset.keys())
    if len(data.get('input_ids')) + cur_seq_length > seq_length:
        # add data to dataset and reset data
        dataset = _add_dataset(dataset, data, **kwargs)
        data = _init_data(data_names)

    # add sample to data
    for k in data_names:
        if k != 'actual_seq_len':
            data[k] += sample[k]
        else:
            data[k] += [data.get(k)[-1] + cur_seq_length]
    return dataset, data


def pack_examples(dataset, **kwargs):
    """packing dataset examples in pack mode"""
    from datasets import Dataset
    from tqdm import tqdm

    data_names = ['input_ids', 'labels', 'actual_seq_len']
    cur_dataset = {k: [] for k in data_names}

    tmp_data = _init_data(data_names)
    for example in tqdm(dataset, desc="Packing"):
        cur_dataset, tmp_data = _process_sample(
            example, tmp_data, cur_dataset, **kwargs
        )

    if sum([len(v) for v in list(cur_dataset.values())]) == 0:
        raise RuntimeError(
            'dataset is empty after packing, maybe all inputs exceed max length or '
            'not packed data do not reach max length.'
        )

    dataset = _add_dataset(cur_dataset, tmp_data, **kwargs)
    return Dataset.from_dict(cur_dataset)


def truncate_examples(examples, seq_length, pad_token, ignore_token):
    """packing dataset examples in truncate mode"""
    # add actual_seq_len into column
    per_seq_len = [len(x) for x in examples[list(examples.keys())[0]]]
    actual_seq_len = []
    cur_seq_len = []
    for seq in per_seq_len:
        if sum(cur_seq_len) + seq > seq_length:
            sub_seq = seq_length - sum(cur_seq_len)
            actual_seq_len.append(cur_seq_len + [sub_seq])
            cur_seq_len = [seq - sub_seq]
        else:
            cur_seq_len.append(seq)
    actual_seq_len.append(cur_seq_len)

    # Join all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {k: [v[i: i + seq_length] for i in range(0, len(v), seq_length)] for k, v in examples.items()}

    # pad tail sample
    for k, v in examples.items():
        if k == 'input_ids':
            v[-1] = _pad_data(v[-1], seq_length, pad_token)
        else:
            v[-1] = _pad_data(v[-1], seq_length, ignore_token)
    examples['actual_seq_len'] = actual_seq_len
    return examples


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class PackingHandler(BaseInstructDataHandler):
    """Dataset Packing Handler"""

    support_pack_mode = ['pack', 'truncate']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pad_token = config.get('pad_token', 0)
        self.ignore_token = config.get('ignore_token', -100)

        if self.packing not in self.support_pack_mode:
            raise ValueError(
                f"Unsupported packing mode {self.packing}, please set packing in {self.support_pack_mode}.")

    def handle(self, dataset):
        _check_input_columns(dataset)

        map_args = dict(batched=True, desc="Packing")
        fn_kwargs = dict(
            seq_length=self.seq_length,
            pad_token=self.pad_token,
            ignore_token=self.ignore_token
        )
        if self.packing == 'truncate':
            return dataset.map(truncate_examples,
                               fn_kwargs=fn_kwargs,
                               **map_args)
        if self.packing == 'pack':
            return pack_examples(dataset, **fn_kwargs)

        if self.output_columns:
            dataset = self._post_process(dataset)
        return dataset

    def format_func(self, example):
        """format_func is not necessary in packing handler"""
        raise NotImplementedError


def _check_input_columns(dataset):
    """check column names in dataset"""
    columns = dataset.column_names
    if 'input_ids' not in columns or 'labels' not in columns:
        raise ValueError(
            f"'input_ids' or 'labels' not in dataset columns.")
