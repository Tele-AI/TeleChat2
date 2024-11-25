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
"""test general dataset."""
import numpy as np

from mindformers import MindFormerConfig
from mindformers.dataset import GeneralDataset

DATASET_CONFIG = MindFormerConfig()
DATASET_CONFIG.seed = 0
DATASET_CONFIG.prefetch_size = 1
DATASET_CONFIG.numa_enable = False
DATASET_CONFIG.batch_size = 2
DATASET_CONFIG.drop_remainder = True


def generate_multi_dimension():
    """generate multi-dimension data"""
    for i in range(64):
        yield (np.array([[i, i + 1], [i + 2, i + 3]]),)


def generate_multi_column():
    """generate multi-column data"""
    for i in range(64):
        yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])


class MyIterable:
    """Iterable Dataset"""

    def __init__(self):
        self._index = 0
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        item = (self._data[self._index], self._label[self._index])
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)


class MyAccessible:
    """Accessible Dataset"""

    def __init__(self):
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


class TestGeneralDataset:
    """A test class for testing general dataset."""

    def test_general_dataset_from_multi_dimension(self):
        """test general dataset from multi-dimension."""
        DATASET_CONFIG.input_columns = ["multi_dimensional_data"]
        DATASET_CONFIG.output_columns = DATASET_CONFIG.input_columns

        def data_collator(multi_dimensional_data):
            return multi_dimensional_data

        GeneralDataset(dataset_config=DATASET_CONFIG,
                       dataset=generate_multi_dimension,
                       data_collator=data_collator)

    def test_general_dataset_from_multi_column(self):
        """test general dataset from multi-column."""
        DATASET_CONFIG.input_columns = ["col1", "col2"]
        DATASET_CONFIG.output_columns = DATASET_CONFIG.input_columns

        def data_collator(col1, col2):
            return col1, col2

        GeneralDataset(dataset_config=DATASET_CONFIG,
                       dataset=generate_multi_column,
                       data_collator=data_collator)

    def test_general_dataset_my_iterable(self):
        """test general dataset with my iterable dataset."""
        DATASET_CONFIG.input_columns = ["data", "label"]
        DATASET_CONFIG.output_columns = DATASET_CONFIG.input_columns

        def data_collator(data, label):
            return data, label

        GeneralDataset(dataset_config=DATASET_CONFIG,
                       dataset=MyIterable(),
                       data_collator=data_collator)

    def test_general_dataset_my_accessible(self):
        """test general dataset with my accessible dataset."""
        DATASET_CONFIG.input_columns = ["data", "label"]
        DATASET_CONFIG.output_columns = DATASET_CONFIG.input_columns

        def data_collator(data, label):
            return data, label

        GeneralDataset(dataset_config=DATASET_CONFIG,
                       dataset=MyAccessible(),
                       data_collator=data_collator)
