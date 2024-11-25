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
"""test multisource dataloader"""
import os
import time
from unittest import mock

import numpy as np
import pytest

from mindspore.dataset import GeneratorDataset

from mindformers.dataset import MultiSourceDataLoader


class RandomAccessDataset:
    """Random Access Dataset"""
    def __init__(self, data_size: int, seq_length: int = 10) -> None:
        self.data = np.random.sample((data_size, seq_length)).astype(np.float16)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def make_dataset(data_size: list, seq_length: int, data_source_type: str, shuffle: bool = False):
    dataloader = RandomAccessDataset(data_size=data_size, seq_length=seq_length)
    if data_source_type == "iterator":
        return GeneratorDataset(dataloader, column_names=["input_ids"], shuffle=shuffle)
    return dataloader


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
class TestMuitiSourceDataLoader:
    """test multisource dataloader."""
    @classmethod
    def setup_class(cls):
        """set up test conditions for test class."""
        cls.tmp_npz_path = "./tmp_indices.npz"
        cls.shuffle_strategy = 'files'

        if isinstance(cls.shuffle_strategy, bool):
            cls.shuffle_file = cls.shuffle_strategy
        elif cls.shuffle_strategy == 'infile':
            cls.shuffle_file = True
        elif cls.shuffle_strategy == 'files':
            cls.shuffle_file = False
        else:
            cls.shuffle_file = True

    @pytest.mark.run(order=1)
    @mock.patch('mindformers.dataset.dataloader.multi_source_dataloader.build_dataset_loader')
    def test_build_iter_dataset(self, mock_build_dataset: mock.MagicMock):
        """test build a iterator multi source dataloader."""
        # set input arguments for MultiSourceDataLoader
        data_size = 10
        seq_length = 10
        data_num = 5
        sub_data_loader = [dict(type="GeneratorDataset") for _ in range(data_num)]
        sub_data_loader_args = dict(column_names=["input_ids"])
        nums_per_dataset = [data_size] * data_num
        data_source_type = "iterator"
        # mock build_dataset_loader
        mock_build_dataset.return_value = make_dataset(data_size, seq_length, data_source_type,
                                                       self.__class__.shuffle_file)
        np.random.seed(0)
        multi_iter_dataset = MultiSourceDataLoader(sub_data_loader,
                                                   sub_data_loader_args,
                                                   nums_per_dataset=nums_per_dataset,
                                                   data_source_type=data_source_type,
                                                   shuffle=self.__class__.shuffle_strategy)
        assert multi_iter_dataset.get_dataset_size() == data_size * data_num

    @pytest.mark.run(order=2)
    @mock.patch('mindformers.dataset.dataloader.multi_source_dataloader.build_dataset_loader')
    def test_build_random_access_dataset(self, mock_build_dataset: mock.MagicMock):
        """test build a random access multi source dataloader."""
        # set input arguments for MultiSourceDataLoader
        data_size = 10
        seq_length = 10
        data_num = 5
        sub_data_loader = [dict(type="GeneratorDataset") for _ in range(data_num)]
        sub_data_loader_args = dict(column_names=["input_ids"])
        nums_per_dataset = [data_size] * data_num
        data_source_type = "random_access"
        # mock build_dataset_loader
        mock_build_dataset.return_value = make_dataset(data_size, seq_length, data_source_type)
        np.random.seed(0)
        multi_access_dataset = MultiSourceDataLoader(sub_data_loader,
                                                     sub_data_loader_args,
                                                     nums_per_dataset=nums_per_dataset,
                                                     data_source_type=data_source_type,
                                                     shuffle=self.__class__.shuffle_strategy)
        assert multi_access_dataset.get_dataset_size() == data_size * data_num
        for item in multi_access_dataset.create_dict_iterator():
            assert item["input_ids"].shape == (seq_length,)

    @pytest.mark.run(order=3)
    @mock.patch('mindformers.dataset.dataloader.multi_source_dataloader.build_dataset_loader')
    def test_save_indices_for_random_access_dataset(self, mock_build_dataset: mock.MagicMock):
        """test save & load indices to disk."""
        data_size = 10000
        seq_length = 10
        data_num = 5
        sub_data_loader = [dict(type="GeneratorDataset") for _ in range(data_num)]
        sub_data_loader_args = dict(column_names=["input_ids"])
        nums_per_dataset = [data_size] * data_num
        data_source_type = "random_access"
        npz_path = self.__class__.tmp_npz_path
        # mock build_dataset_loader
        mock_build_dataset.return_value = make_dataset(data_size, seq_length, data_source_type)

        # build dataset and save indices
        multi_access_dataset1 = MultiSourceDataLoader(sub_data_loader,
                                                      sub_data_loader_args,
                                                      nums_per_dataset=nums_per_dataset,
                                                      data_source_type=data_source_type,
                                                      shuffle=self.__class__.shuffle_strategy,
                                                      save_indices_npz_path=npz_path)
        assert multi_access_dataset1.get_dataset_size() == data_size * data_num

        # rebuild dataset with loading dataset
        sub_data_loader = [dict(type="GeneratorDataset") for _ in range(data_num)]
        start_time = time.time()
        multi_access_dataset2 = MultiSourceDataLoader(sub_data_loader,
                                                      sub_data_loader_args,
                                                      nums_per_dataset=nums_per_dataset,
                                                      data_source_type=data_source_type,
                                                      shuffle=self.__class__.shuffle_strategy,
                                                      load_indices_npz_path=npz_path)
        build_time = time.time() - start_time
        assert build_time < 2       # less than 2 seconds.
        assert multi_access_dataset2.get_dataset_size() == data_size * data_num

        # clean up tmp file.
        if os.path.exists(self.__class__.tmp_npz_path):
            os.remove(self.__class__.tmp_npz_path)

    @pytest.mark.run(order=4)
    @mock.patch('mindformers.dataset.dataloader.multi_source_dataloader.build_dataset_loader')
    def test_skip_time_for_random_access_dataset(self, mock_build_dataset: mock.MagicMock):
        """test skip efficience for random access dataset."""
        data_size = 1000000
        seq_length = 10
        data_num = 5
        sub_data_loader = [dict(type="GeneratorDataset") for _ in range(data_num)]
        sub_data_loader_args = dict(column_names=["input_ids"])
        nums_per_dataset = [data_size] * data_num
        data_source_type = "random_access"
        # mock build_dataset_loader
        mock_build_dataset.return_value = make_dataset(data_size, seq_length, data_source_type)

        # build dataset
        multi_access_dataset = MultiSourceDataLoader(sub_data_loader,
                                                     sub_data_loader_args,
                                                     nums_per_dataset=nums_per_dataset,
                                                     data_source_type=data_source_type,
                                                     shuffle=self.__class__.shuffle_strategy)
        assert multi_access_dataset.get_dataset_size() == data_size * data_num

        start_time = time.time()
        # skip 80% of the whole dataset
        multi_access_dataset.set_init_step(int(data_size * data_num * 0.8))
        iter_dataset = multi_access_dataset.create_dict_iterator(num_epochs=1)
        next(iter_dataset)
        skip_time = time.time() - start_time
        assert skip_time < 2        # less than 2 seconds.
