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
"""Cifar100 DataLoader."""
import os
import pickle
from typing import Optional, Union, List, Tuple
import numpy as np

from mindspore.dataset import GeneratorDataset

from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class Cifar100DataLoader:
    """Cifar100 Dataloader with class name as text column"""
    _default_column_names = ["image", "text", "label"]

    def __new__(cls, dataset_dir: str, column_names: Optional[Union[List[str], Tuple[str]]] = None,
                stage: Optional[str] = "train", fine_label: Optional[bool] = True,
                shuffle: Optional[bool] = False,
                hypothesis_template: Optional[str] = "This is a photo of {}.", **kwargs):
        r"""
        Cifar100 Dataloader API.

        Args:
            dataset_dir (str): The dataset directory, such as "/home/desktop/cifar-100-python"
            column_names (Optional[Union[List[str], Tuple[str]]]): The output column names,
                a tuple or a list of string with length 3.
            stage (Optional[str]): The supported key words are in ["train","test", "all"]
            fine_label (Optional[bool]): True for 100 classes, False for 10 classes.
            shuffle (Optional[bool]): Shuffle the samples.
            hypothesis_template (Optional[str]): Prompt template for class label.

        Return:
            A GeneratorDataset for Cifar100 dataset

        Raises:
            ValueError: Error input for dataset_dir and column_names.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import Cifar100DataLoader
            >>> dataloader = Cifar100DataLoader("./cifar-100-python")
            >>> dataloader = dataloader.batch(1)
            >>> for item in dataloader:
            >>>     print(item)
            >>>     break
                [Tensor(shape=[1, 32, 32, 3], dtype=UInt8, value=
                [[[[255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                ...
                [195, 205, 193],
                [212, 224, 204],
                [182, 194, 167]]]]]), Tensor(shape=[1, 100], dtype=String, value=
                [['This is a photo of apple.', 'This is a photo of aquarium_fish.',
                'This is a photo of baby.', 'This is a photo of bear.',
                ...
                'This is a photo of woman.', 'This is a photo of worm.']]),
                 Tensor(shape=[1], dtype=Int32, value= [19])]
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names

        if not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 3, but got {type(column_names)}")

        if len(column_names) != 3:
            raise ValueError(f"the length of column_names should be 3,"
                             f" but got {len(column_names)}")

        for index, _ in enumerate(column_names):
            if not isinstance(column_names[index], str):
                raise ValueError(f"the item type of column_names should be string,"
                                 f" but got {type(column_names[index])}")

        kwargs.pop("None", None)
        cifar100_dataset = Cifar100DataSet(dataset_dir, stage, fine_label, hypothesis_template)
        cifar100_dataloader = GeneratorDataset(cifar100_dataset, column_names, shuffle=shuffle)
        setattr(cifar100_dataloader, "label_names", cifar100_dataset.label_names)
        return cifar100_dataloader


class Cifar100DataSet:
    """Cifar100 dataSet with class name"""
    def __init__(self, dataset_dir: str, stage: Optional[str] = "train",
                 fine_label: Optional[bool] = True,
                 hypothesis_template: Optional[str] = "This is a photo of {}."):
        r"""
        Cifar100 Dataset

        Args:
            dataset_dir (str): The dataset directory, such as "/home/desktop/cifar-100-python".
            stage (Optional[str]): The supported keywords are in ["train", "test", "all"].
            fine_label (Optional[bool]): True for 100 classes, False for 10 classes.
            hypothesis_template (Optional[str]): Prompt template for class label.

        Return:
             A iterable dataset for Cifar100 dataset.

        Raises:
            ValueError: Unsupported values for stage.
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if stage not in ["train", "test", "all"]:
            raise ValueError("unsupported stage,"
                             " stage should be in [\"train\", \"test\", \"all\"]")

        meta_file = os.path.join(dataset_dir, "meta")
        with open(meta_file, 'rb') as fo:
            meta_dict = pickle.load(fo, encoding="bytes")

        if fine_label:
            label_names = meta_dict[b'fine_label_names']
        else:
            label_names = meta_dict[b'coarse_label_names']
        for index, _ in enumerate(label_names):
            label_names[index] = label_names[index].decode()

        self.label_names = label_names
        hypothesis = []
        for label_name in label_names:
            hypothesis.append(hypothesis_template.format(label_name))

        if stage in ["train", "all"]:
            train_file = os.path.join(dataset_dir, "train")
            with open(train_file, 'rb') as fo:
                train_dict = pickle.load(fo, encoding="bytes")

            if fine_label:
                train_label = train_dict[b"fine_labels"]
            else:
                train_label = train_dict[b"coarse_labels"]
            train_image = train_dict[b"data"]
            train_num = train_image.shape[0]
            train_image = train_image.reshape((train_num, 3, 32, 32))
            train_image = train_image.transpose(0, 2, 3, 1)
            self.image = train_image
            self.label = train_label
            self.text = hypothesis

        if stage in ["test", "all"]:
            test_file = os.path.join(dataset_dir, "test")
            with open(test_file, 'rb') as fo:
                test_dict = pickle.load(fo, encoding="bytes")

            if fine_label:
                test_label = test_dict[b"fine_labels"]
            else:
                test_label = test_dict[b"coarse_labels"]
            test_image = test_dict[b"data"]
            test_num = test_image.shape[0]
            test_image = test_image.reshape((test_num, 3, 32, 32))
            test_image = test_image.transpose(0, 2, 3, 1)
            self.image = test_image
            self.label = test_label
            self.text = hypothesis

        if stage == "all":
            self.image = np.row_stack([train_image, test_image])
            self.label = train_label + test_label
            self.text = hypothesis

    def __getitem__(self, item):
        """getitem"""
        return self.image[item], self.text, self.label[item]

    def __len__(self):
        """len"""
        return self.image.shape[0]
