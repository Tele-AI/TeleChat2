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
# This file was refer to project:
# https://gitee.com/mindspore/models/blob/master/official/nlp/Bert/src/generate_mindrecord/generate_cluener_mindrecord.py
# ============================================================================
"""CLUENER DataLoader."""
import os
import json
from typing import Optional, Union, List, Tuple
from mindspore.dataset import GeneratorDataset

from ...tools.register import MindFormerRegister, MindFormerModuleType
from ..labels import cluener_labels


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class CLUENERDataLoader:
    """CLUENER Dataloader"""
    _default_column_names = ["text", "label_id"]

    def __new__(cls, dataset_dir: str,
                column_names: Optional[Union[List[str], Tuple[str]]] = None,
                stage: Optional[str] = "train", **kwargs):
        r"""
        CLUENER Dataloader API.

        Args:
            dataset_dir (str): The directory to cluener dataset.
            column_names (Optional[Union[List[str], Tuple[str]]]): The output column names,
                a tuple or a list of string with length 2
            stage (Optional[str]): The supported keywords are in ["train", "test", "del", "all"]

        Return:
            A GeneratorDataset for cluener dataset

        Raises:
            ValueError: Error input for dataset_dir, and column_names.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import CLUENERDataLoader
            >>> data_loader = CLUENERDataLoader("./cluener/")
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
                [Tensor(shape=[], dtype=String, value= '虚幻引擎3动作游戏《黑光》新作公布'),
                 Tensor(shape=[49], dtype=Int64, value= [ 0,  0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0,  0,  5, 15, 15,  0,  0,  0,  0,  0,
                        5, 15, 15, 15, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0,  0, 0])]
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names

        if not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 2, but got {type(column_names)}")

        if len(column_names) != 2:
            raise ValueError(f"the length of column_names should be 2,"
                             f" but got {len(column_names)}")

        if not isinstance(column_names[0], str) or not isinstance(column_names[1], str):
            raise ValueError(f"the item type of column_names should be string,"
                             f" but got {type(column_names[0])} and {type(column_names[1])}")

        kwargs.pop("None", None)
        cluener_dataset = CLUENERDataSet(dataset_dir, stage)
        return GeneratorDataset(cluener_dataset, column_names)


class CLUENERDataSet:
    """CLUENER DataSet"""
    def __init__(self, dataset_dir, stage="train"):
        r"""
        CLUENER Dataset

        Args:
            dataset_dir (str): The directory to cluener dataset.
            stage (str): The supported keywords are in ["train", "dev", "test"]

        Return:
            A iterable dataset for cluener dataset

        Raises:
            ValueError: Error input for dataset_dir, stage.
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        self.dataset_dir = dataset_dir
        self.label2id = {label: label_id for label_id, label in enumerate(cluener_labels)}
        self.texts = []
        self.label_ids = []

        if stage == "train":
            train_data_path = os.path.join(dataset_dir, "train.json")
            self._get_train_examples(train_data_path)

        elif stage == "dev":
            dev_data_path = os.path.join(dataset_dir, "dev.json")
            self._get_dev_examples(dev_data_path)

        elif stage == "test":
            test_data_path = os.path.join(dataset_dir, "test.json")
            self._get_test_examples(test_data_path)

        else:
            raise ValueError("unsupported stage.")

    def __getitem__(self, item):
        text = self.texts[item]
        label_id = self.label_ids[item]

        return text, label_id

    def __len__(self):
        return len(self.texts)

    def _get_train_examples(self, train_data_path):
        """Get train examples."""
        return self._create_examples(self._read_json(train_data_path))

    def _get_dev_examples(self, dev_data_path):
        """Get dev examples."""
        return self._create_examples(self._read_json(dev_data_path))

    def _get_test_examples(self, test_data_path):
        """Get test examples."""
        return self._create_examples(self._read_json(test_data_path))

    def _read_json(self, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
            return lines

    def generate_label(self, line, label):
        """Generate label"""
        for label_name, words in line['label'].items():
            for _, indices in words.items():
                for index in indices:
                    if index[0] == index[1]:
                        label[index[0]] = 'S-' + label_name
                    else:
                        label[index[0]] = 'B-' + label_name
                        label[index[1]] = 'I-' + label_name
                        for j in range(index[0] + 1, index[1]):
                            label[j] = 'I-' + label_name
        return label

    def _create_examples(self, lines):
        """Create Example."""
        for line in lines:
            text = line['text']
            label = ['O'] * len(text)
            if 'label' in line:
                label = self.generate_label(line, label)

            label_id = []
            for token_label in label:
                label_id.append(self.label2id[token_label])

            self.texts.append(text)
            self.label_ids.append(label_id)
