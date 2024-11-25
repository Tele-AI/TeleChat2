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
"""MultiSource DataLoader."""
import inspect
import random
from itertools import accumulate
from typing import Optional, List, Union

import numpy as np
from tqdm import tqdm

from mindspore.dataset import Dataset, GeneratorDataset, Shuffle

from .build_dataloader import build_dataset_loader
from ...tools.logger import logger
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class MultiSourceDataLoader:
    """MultiSource Dataloader with class name as text column"""

    def __new__(cls, sub_data_loader, sub_data_loader_args=None,
                dataset_ratios: Optional[List[float]] = None,
                samples_count: Optional[int] = None,
                nums_per_dataset: Optional[List[int]] = None,
                shuffle: Optional[Union[bool, Shuffle]] = True,
                shuffle_buffer_size: Optional[int] = 320,
                data_source_type: Optional[str] = "iterator",
                load_indices_npz_path: Optional[str] = None,
                save_indices_npz_path: Optional[str] = None,
                **kwargs):
        if dataset_ratios is not None:
            if any([ratios < 0 for ratios in dataset_ratios]):
                raise ValueError(
                    f"the dataset_ratios should be a list of positive value, but got {dataset_ratios}")

            if abs(sum(dataset_ratios) - 1) > 1e-5:
                raise ValueError("the sum of ratios is not equals to 1")

            if samples_count is None or samples_count <= 0:
                raise ValueError(f"the samples_count should be a positive int when dataset_ratios is not None, "
                                 f"but got {samples_count}")

            if not isinstance(dataset_ratios, list) or len(dataset_ratios) != len(sub_data_loader):
                raise ValueError(
                    "the dataset_ratios should be a list and the length should equal to sub_data_loader")

            nums_per_dataset = [int(ratio * samples_count) for ratio in dataset_ratios]
            need_sample = True
        else:
            if nums_per_dataset is None:
                nums_per_dataset = []
                need_sample = False
            else:
                if not isinstance(nums_per_dataset, list) or len(nums_per_dataset) != len(sub_data_loader):
                    raise ValueError(
                        "the nums_per_dataset should be a list and the length should equal to sub_data_loader")

                if any([num < 0 for num in nums_per_dataset]):
                    raise ValueError(
                        f"the nums_per_dataset should be a list of positive value, but got {nums_per_dataset}")

                need_sample = True

        if sub_data_loader_args is None:
            sub_data_loader_args = dict()

        if not isinstance(shuffle, bool) and shuffle.lower() not in ["global", "files", "infile"]:
            raise ValueError(
                f"shuffle should be a bool or a str and the value must be one of ['global', 'files', 'infile']")

        if isinstance(shuffle, bool):
            shuffle_dataset = shuffle
            shuffle_file = shuffle
        elif shuffle == Shuffle.INFILE:
            shuffle_dataset = False
            shuffle_file = True
        elif shuffle == Shuffle.FILES:
            shuffle_dataset = True
            shuffle_file = False
        else:
            shuffle_dataset = True
            shuffle_file = True
        sub_data_loader_args["shuffle"] = shuffle_file

        dataset_loaders = []
        for sub_data_loader_config in sub_data_loader:
            class_name = sub_data_loader_config["type"]
            sub_data_loader_config.pop("type")
            sub_data_loader_config.update(sub_data_loader_args)
            sub_data_loader_config.update(kwargs)
            data_loader_args = prepare_generator_sub_dataloader_args(class_name, sub_data_loader_config)
            sub_dataset_loader = build_dataset_loader(
                class_name=class_name,
                **data_loader_args
            )
            dataset_loaders.append(sub_dataset_loader)

        for index, sub_data_loader_item in enumerate(dataset_loaders):
            if isinstance(sub_data_loader_item, Dataset):
                actual_nums_over_this_dataset = sub_data_loader_item.get_dataset_size()
            else:
                actual_nums_over_this_dataset = len(sub_data_loader_item)

            if not need_sample:
                nums_per_dataset.append(actual_nums_over_this_dataset)
            else:
                if nums_per_dataset[index] > actual_nums_over_this_dataset:
                    specific_size = nums_per_dataset[index]
                    nums_per_dataset[index] = actual_nums_over_this_dataset

                    if dataset_ratios:
                        logger.warning("The size of %s-th dataloader is less then specific size, "
                                       "specific size is ratio*samples_count=%s*%s=%s. "
                                       "The actual size will reset to dataset_size=%s.",
                                       index + 1, dataset_ratios[index], samples_count,
                                       specific_size, actual_nums_over_this_dataset)
                    else:
                        logger.warning("The size of %s-th dataloader is less then specific size, "
                                       "specific size is nums_per_dataset[%s]=%s. "
                                       "The actual size will reset to dataset_size=%s.",
                                       index + 1, index, specific_size, actual_nums_over_this_dataset)

        logger.info("MultiSourceDataloader will be created! Actual nums_per_dataset is %s according to the dataset "
                    "ratios or actual sub dataset size. The total dataset size is %s if drop_remainder=False.",
                    nums_per_dataset, sum(nums_per_dataset))

        if data_source_type == "iterator":
            if load_indices_npz_path is not None or save_indices_npz_path is not None:
                logger.warning("For MultiSourceDataLoader class, arguments `load_indices_npz_path` and "
                               "`save_indices_npz_path` only take effect when `data_source_type` is random_access.")
            data_source = MultiSourceIterDataSet(dataset_loaders, nums_per_dataset, shuffle_dataset=shuffle_dataset,
                                                 shuffle_buffer_size=shuffle_buffer_size)
        elif data_source_type == "random_access":
            # for random access dataset, we control shuffle with indices
            sub_data_loader_args["shuffle"] = False
            shuffle_random_access = shuffle
            if shuffle_dataset and shuffle_file:
                # when both dataset & sample shuffle is True, we can simply use generator shuffle
                sub_data_loader_args["shuffle"] = True
                shuffle_random_access = False
            data_source = MultiSourceRandomAccessDataset(dataset_loaders, nums_per_dataset,
                                                         shuffle=shuffle_random_access,
                                                         shuffle_buffer_size=shuffle_buffer_size,
                                                         load_indices_npz_path=load_indices_npz_path,
                                                         save_indices_npz_path=save_indices_npz_path)
        else:
            raise ValueError("For MultiSourceDataLoader class, argument `data_source_type` should be one of: "
                             f"1. iterator; 2. random_access. But got {data_source_type}, please check your args.")

        return GeneratorDataset(data_source, **prepare_generator_dataset_args(sub_data_loader_args))


def prepare_generator_sub_dataloader_args(class_name, full_args):
    cls_obj = MindFormerRegister.get_cls(module_type="dataset_loader", class_name=class_name)

    arg_keys = inspect.signature(cls_obj).parameters.keys()

    if "kwargs" in arg_keys:
        return full_args
    return {
        key: full_args.get(key)
        for key in arg_keys if key in full_args
    }


def prepare_generator_dataset_args(full_args):
    arg_keys = inspect.signature(GeneratorDataset).parameters.keys()

    return {
        key: full_args.get(key)
        for key in arg_keys if key != "source" and key in full_args
    }


class MultiSourceIterDataSet:
    """MultiSource Dataloader with class name as text column"""

    def __init__(self, dataset_list, sample_nums, shuffle_buffer_size=320,
                 shuffle_dataset: Optional[bool] = True):
        self.sample_nums = sample_nums
        self.lasted_samples = sample_nums
        self.size = sum(sample_nums)
        self.created = 0
        self.shuffle = shuffle_dataset

        self.dataset_list = dataset_list

        self.acc_sample_nums = list(accumulate(self.sample_nums))
        self.cur_dataset_index = 0

        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_buffer = []
        self.shuffle_num_list_base = [int(self.shuffle_buffer_size * sample_num / self.size)
                                      for sample_num in self.sample_nums]
        self.shuffle_buffer_base_size = sum(self.shuffle_num_list_base)
        self.shuffle_diff = self.shuffle_buffer_size - self.shuffle_buffer_base_size

    def _reset_parameters(self):
        """reset parameters when return a new iter object"""
        # global setting
        self.created = 0
        self.size = sum(self.sample_nums)
        self.dataset_iter_list = [dataset.create_tuple_iterator() for dataset in self.dataset_list]

        # seq setting
        self.cur_dataset_index = 0
        self.acc_sample_nums = list(accumulate(self.sample_nums))

        # shuffle setting
        self.shuffle_buffer = []
        self.lasted_samples = self.sample_nums

    def _get_dataset_shuffle_index(self):
        buffer_index = self.created % self.shuffle_buffer_size
        if buffer_index == 0:
            self._push_data_to_shuffle_buffer()

        return self.shuffle_buffer[buffer_index]

    def _get_dataset_seq_index(self):
        if self.created < self.acc_sample_nums[self.cur_dataset_index]:
            return self.cur_dataset_index
        self.cur_dataset_index += 1
        return self.cur_dataset_index

    def _get_random_valid_dataset_index(self):
        cur_index = random.randint(0, len(self.lasted_samples) - 1)
        init_index = cur_index
        while True:
            cur_index = cur_index + 1 if cur_index + 1 != len(self.lasted_samples) else 0
            if self.lasted_samples[cur_index] > 0:
                return cur_index
            if cur_index == init_index:
                logger.warning(self.lasted_samples)
                return init_index

    def _push_data_to_shuffle_buffer(self):
        """push data to shuffle buffer when shuffle buffer is empty"""
        self.shuffle_buffer = []
        if sum(self.lasted_samples) > self.shuffle_buffer_base_size:
            diff = self.shuffle_diff
            shuffle_num_list = self.shuffle_num_list_base.copy()
            self.lasted_samples = [self.lasted_samples[index] - shuffle_num_list[index]
                                   for index in range(len(self.lasted_samples))]
            while diff != 0:
                cur_index = self._get_random_valid_dataset_index()
                shuffle_num_list[cur_index] += 1
                self.lasted_samples[cur_index] -= 1
                diff -= 1
            for index, sample_num in enumerate(shuffle_num_list):
                self.shuffle_buffer.extend([index] * sample_num)
        else:
            for index, sample_num in enumerate(self.lasted_samples):
                self.shuffle_buffer.extend([index] * sample_num)
        random.shuffle(self.shuffle_buffer)

    def get_next_dataset_index(self):
        if self.shuffle:
            dataset_index = self._get_dataset_shuffle_index()
        else:
            dataset_index = self._get_dataset_seq_index()
        return dataset_index

    def __next__(self):
        if self.created == self.size:
            raise StopIteration

        dataset_index = self.get_next_dataset_index()
        item = next(self.dataset_iter_list[dataset_index])
        self.created += 1
        return item

    def __iter__(self):
        self._reset_parameters()
        return self

    def __len__(self):
        return self.size


class MultiSourceRandomAccessDataset:
    """MultiSource Dataloader with random access implement."""
    def __init__(self, dataset_list, sample_nums, shuffle_buffer_size=320,
                 shuffle: Optional[bool] = True,
                 load_indices_npz_path: str = None,
                 save_indices_npz_path: str = None):
        self.sample_nums = sample_nums
        self.lasted_samples = sample_nums
        self.size = sum(sample_nums)

        if isinstance(shuffle, bool):
            self.shuffle_dataset = shuffle
            self.shuffle_file = shuffle
        elif shuffle == Shuffle.INFILE:
            self.shuffle_dataset = False
            self.shuffle_file = True
        elif shuffle == Shuffle.FILES:
            self.shuffle_dataset = True
            self.shuffle_file = False
        else:
            self.shuffle_dataset = True
            self.shuffle_file = True

        self.dataset_list = dataset_list
        self.acc_sample_nums = list(accumulate(self.sample_nums))

        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_buffer = []
        self.shuffle_num_list_base = [int(self.shuffle_buffer_size * sample_num / self.size)
                                      for sample_num in self.sample_nums]
        self.shuffle_buffer_base_size = sum(self.shuffle_num_list_base)
        self.shuffle_diff = self.shuffle_buffer_size - self.shuffle_buffer_base_size

        logger.info(".......... Building indices for multi-source dataset ..........")
        if load_indices_npz_path is None:
            self.dataloader_index, self.data_sample_index = self.build_indices()
        else:
            logger.info(f".......... load indices from npz file: {load_indices_npz_path} ..........")
            load_dict = np.load(load_indices_npz_path)
            self.dataloader_index = load_dict["dataloader_index"]
            self.data_sample_index = load_dict["data_sample_index"]
        if save_indices_npz_path is not None:
            logger.info(f".......... save indices to npz file: {save_indices_npz_path} ..........")
            np.savez_compressed(save_indices_npz_path, dataloader_index=self.dataloader_index,
                                data_sample_index=self.data_sample_index)
        logger.info(".......... Build indices DONE for multi-source dataset ..........")

    def build_indices(self):
        dataloader_index = self._build_dataloader_indices()
        dataloader_index = np.array(dataloader_index, dtype=np.uint8)
        data_sample_index = self._build_data_sample_indices(dataloader_index)
        return dataloader_index, data_sample_index

    def _build_dataloader_indices(self):
        if self.shuffle_dataset:
            return self._build_shuffle_dataloader_indices()
        return self._build_seq_dataloader_indices()

    def _build_shuffle_dataloader_indices(self):
        dataloader_index = []
        with tqdm(total=self.size, desc="Building dataloader index") as pbar:
            while len(dataloader_index) < self.size:
                self._push_data_to_shuffle_buffer()
                dataloader_index.extend(self.shuffle_buffer)
                pbar.update(len(self.shuffle_buffer))
        return dataloader_index

    def _build_seq_dataloader_indices(self):
        cur_dataset_index = 0
        dataloader_index = []
        for sample_num in self.sample_nums:
            seq_index = np.ones(shape=(sample_num,), dtype=np.int32) * cur_dataset_index
            cur_dataset_index += 1
            dataloader_index.extend(seq_index)
        return dataloader_index

    def _build_data_sample_indices(self, dataloader_index):
        data_sample_index = np.zeros(self.size, dtype=np.int32)
        for i, sample_num in enumerate(self.sample_nums):
            seq_sample_index = np.arange(sample_num, dtype=np.int32)
            if self.shuffle_file:
                random.shuffle(seq_sample_index)
            tmp_indices = np.where(dataloader_index == i)[0]
            data_sample_index[tmp_indices] = seq_sample_index
        return data_sample_index

    def _push_data_to_shuffle_buffer(self):
        """push data to shuffle buffer when shuffle buffer is empty"""
        self.shuffle_buffer = []
        if sum(self.lasted_samples) > self.shuffle_buffer_base_size:
            diff = self.shuffle_diff
            shuffle_num_list = self.shuffle_num_list_base.copy()
            self.lasted_samples = [self.lasted_samples[index] - shuffle_num_list[index]
                                   for index in range(len(self.lasted_samples))]
            while diff != 0:
                cur_index = self._get_random_valid_dataset_index()
                shuffle_num_list[cur_index] += 1
                self.lasted_samples[cur_index] -= 1
                diff -= 1
            for index, sample_num in enumerate(shuffle_num_list):
                self.shuffle_buffer.extend([index] * sample_num)
        else:
            for index, sample_num in enumerate(self.lasted_samples):
                self.shuffle_buffer.extend([index] * sample_num)
        random.shuffle(self.shuffle_buffer)

    def _get_random_valid_dataset_index(self):
        cur_index = random.randint(0, len(self.lasted_samples) - 1)
        init_index = cur_index
        while True:
            cur_index = cur_index + 1 if cur_index + 1 != len(self.lasted_samples) else 0
            if self.lasted_samples[cur_index] > 0:
                return cur_index
            if cur_index == init_index:
                logger.warning(self.lasted_samples)
                return init_index

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataloader_index[idx]
        sample_idx = self.data_sample_index[idx]
        return self.dataset_list[(int)(dataset_idx)][(int)(sample_idx)]
