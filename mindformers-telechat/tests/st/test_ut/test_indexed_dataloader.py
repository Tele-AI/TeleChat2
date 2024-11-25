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
"""test indexed dataloader."""
import os
import pytest
import numpy as np

from mindformers import MindFormerConfig
from mindformers.dataset import build_dataset
from mindformers.dataset.dataloader import indexed_dataset


def save_data():
    """save data into bin file"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file_path = os.path.join(cur_dir, 'test_data.bin')
    idx_file_path = os.path.join(cur_dir, 'test_data.idx')

    builder = indexed_dataset.IndexedDataBuilder(
        bin_file_path,
        dtype=np.int32,
    )

    source_data = []

    for _ in range(30):
        random_data = np.random.randint(0, 32000, size=4097, dtype=np.int32)
        source_data.append(random_data)

    for data in source_data:
        builder.add_document(data, [len(data)])

    builder.finalize(idx_file_path)

    return bin_file_path, idx_file_path, source_data


def make_dataset(path_prefix: str):
    """generate dataset"""
    data_loader = {
        "type": "IndexedDataLoader",
        "path_prefix": path_prefix,
        "shuffle": False,
    }

    train_dataset = {
        "data_loader": data_loader,
        "input_columns": ["input_ids"],
        "python_multiprocessing": False,
        "drop_remainder": True,
        "batch_size": 1,
        "repeat": 1,
        "numa_enable": False,
        "prefetch_size": 1,
        "seed": 0,
    }

    train_dataset_task = {
        "type": "CausalLanguageModelDataset",
        "dataset_config": train_dataset,
    }

    config = MindFormerConfig(train_dataset=train_dataset, train_dataset_task=train_dataset_task)
    return build_dataset(config.train_dataset_task)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_indexed_dataloader():
    """
    Feature: IndexedDataLoader.
    Description: Test IndexedDataLoader functional
    Expectation: No Exception
    """
    bin_file_path, _, source_data = save_data()
    path_prefix, _ = os.path.splitext(bin_file_path)

    datasets = make_dataset(path_prefix=path_prefix)

    assert len(datasets) == len(source_data)

    for item in datasets.create_dict_iterator():
        assert source_data[0].all() == item['input_ids'].value().asnumpy()[0].all()
        break
