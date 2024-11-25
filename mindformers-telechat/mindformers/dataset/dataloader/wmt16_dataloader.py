# Copyright 2022 Huawei Technologies Co., Ltd
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
"""WMT16 DataLoader."""
import os

from mindspore.dataset import GeneratorDataset

from mindformers.tools.logger import logger
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class WMT16DataLoader:
    """WMT16 Dataloader"""
    _default_column_names = ["source", "target"]

    def __new__(cls, dataset_dir, column_names=None, stage="train", num_shards=1, shard_id=0):
        """
        WMT16 Dataloader API

        Args:
            dataset_dir: the directory to dataset
            column_names: the output column names, a tuple or a list of string with length 2
            stage: the supported `option` are in ["train"、"test"、"del"、"all"]

        Return:
            a GeneratorDataset for WMT16 dataset
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names
            logger.info("The column_names to the WMT16DataLoader is None, so assign it with default_column_names %s",
                        cls._default_column_names)

        if column_names and not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 2, but got {type(column_names)}")

        if len(column_names) != 2:
            raise ValueError(f"the length of column_names should be 2,"
                             f" but got {len(column_names)}")

        if not isinstance(column_names[0], str) or not isinstance(column_names[1], str):
            raise ValueError(f"the item type of column_names should be string,"
                             f" but got {type(column_names[0])} and {type(column_names[1])}")

        dataset = WMT16DataSet(dataset_dir, stage)
        return GeneratorDataset(dataset, column_names, num_shards=num_shards, shard_id=shard_id)


def read_text(train_file):
    """Read the text files and return a list."""
    with open(train_file) as fp:
        data = []
        for line in fp:
            line = line.strip()
            if line:
                data.append(line)
    return data


class WMT16DataSet:
    """WMT16 DataSet"""

    def __init__(self, dataset_dir, stage="train"):
        """
        WMT16DataSet Dataset

        Args:
            dataset_dir: the directory to wmt16 dataset
            stage: the supported key word are in ["train"、"test"、"del"、"all"]

        Return:
            an iterable dataset for wmt16 dataset
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")
        self.dataset_dir = dataset_dir
        self.stage = stage

        dataset_dict = dict()

        def read_and_add_to_stage(cur_stage):
            source_path = os.path.join(dataset_dir, f'{cur_stage}.source')
            src = read_text(source_path)
            target_source = os.path.join(dataset_dir, f'{cur_stage}.target')
            tgt = read_text(target_source)
            src_tgt_pair = list(zip(src, tgt))
            return src_tgt_pair

        logger.info("Start to read the raw data from the disk %s.", dataset_dir)
        if 'stage' != 'all':
            dataset_dict[stage] = read_and_add_to_stage(stage)
        else:
            for item in ['train', 'dev', 'test']:
                dataset_dict[stage] = read_and_add_to_stage(item)

        self.dataset_dict = dataset_dict

    def __getitem__(self, item):
        return self.dataset_dict.get(self.stage)[item]

    def __len__(self):
        return len(self.dataset_dict.get(self.stage))
