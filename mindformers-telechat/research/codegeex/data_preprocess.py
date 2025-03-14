# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Dataset utils for training
"""

import os
from abc import ABC, abstractmethod
import pickle
import numpy as np
from mindspore.mindrecord import FileWriter


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class LMDBDataset(Dataset):
    """Read data from lmdb"""
    def __init__(self, path, process_fn=None):
        import lmdb

        self.path = path
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(f"Get {self.path}: {idx}")
        with self.env.begin(write=False) as txn:
            key = str(idx).encode("utf-8")
            try:
                row = pickle.loads(txn.get(key))
            except TypeError as e:
                raise IndexError("Index out of range") from e
            if self.process_fn:
                return self.process_fn(row)
            return row


class PadDataset(Dataset):
    """Pad data"""
    def __init__(self, dataset, seq_len, eod_id):
        self.dataset = dataset
        self.seq_len = seq_len + 1
        self.eod_id = eod_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx][0]
        return (item[:self.seq_len],) if self.seq_len <= len(item) else (
            np.concatenate((item, np.ones(self.seq_len - len(item)) * self.eod_id), axis=0),)


def get_code_data_train(code_data_path, args, process_fn=None):
    """Get train data"""
    if os.path.exists(os.path.join(code_data_path, 'data.mdb')):
        full_path = os.path.join(code_data_path)
    print(f"Loading code data {full_path}")
    data = LMDBDataset(
        full_path,
        process_fn=process_fn,
    )
    data = PadDataset(
        data,
        args.seq_length,
        args.eod_id,
    )
    return data


def generate_mindrecord(args, file_name="codegeex.mindrecord"):
    """Generate mindrecord format data."""
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]}}
    writer = FileWriter(file_name, shard_num=1, overwrite=True)
    writer.add_schema(data_schema, "it is a code dataset")

    data = []
    train_data = get_code_data_train(args.code_data, args)
    for i, input_id in enumerate(train_data):
        print(i)
        sample = {"input_ids": np.array(input_id).squeeze().astype(np.int32)}
        data.append(sample)
        if i > 100:
            writer.write_raw_data(data)
            data = []

    if data:
        print(data)
        writer.write_raw_data(data)

    writer.commit()


if __name__ == "__main__":
    import argparse
    args_opt = argparse.ArgumentParser(description="PanguAlpha training")
    args_opt.add_argument("--seq_length",
                          type=int,
                          default=2048,
                          help="sequence length, default is 2048.")
    args_opt.add_argument("--eod_id",
                          type=int, default=50256,
                          help="The id of end of document")
    args_opt.add_argument("--eod_reset",
                          type=int,
                          default=1,
                          help="Enable eod mask, default is 1.")
    args_opt.add_argument('--code_data',
                          type=str,
                          help='Location of code data.')
    args_opt = args_opt.parse_args()
    print(args_opt)
    generate_mindrecord(args_opt)
