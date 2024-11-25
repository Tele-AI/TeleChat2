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
"""MultiImageCapPairs DataLoader."""

import os
from collections import OrderedDict
from typing import Optional, Union, List, Tuple
import json
import copy

from mindspore.dataset import GeneratorDataset
from PIL import Image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class MultiImgCapDataLoader:
    """Multiple Image-Caption Dataloader"""
    _default_column_names = ["image", "text"]

    def __new__(cls,
                dataset_dir: str,
                annotation_files: List[str],
                image_dirs: List[str],
                column_names: Optional[Union[List[str], Tuple[str]]] = None,
                stage: Optional[str] = "train",
                repeat_images: Optional[bool] = False,
                shuffle: Optional[bool] = True,
                **kwargs):
        r"""
        MultiImgCapDataLoader Dataloader API.

        Args:
            dataset_dir (str): The directory which is the parent dir of these datasets.
            annotation_files list[str]: the list of files contains annotations.
            image_dirs list[str]: the list of dir contains images.
                                  (one-to-one matching to ' annotation_files')
            column_names (Optional[Union[List[str], Tuple[str]]]): The output column names,
                a tuple or a list of string with length 2
            stage (Optional[str]): The supported keywords are in ["train", "eval"]
            repeat_images (Optional[bool]): whether repeat image when it has multiple
                                            corresponding captions.
            shuffle (Optional[bool]): whether to shuffle the dataset.

        Return:
            A GeneratorDataset for loading multiple image-caption datasets

        Raises:
            ValueError: Error input for dataset_dir, and column_names.
            TypeError: Type error for column_names.
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if len(image_dirs) != len(annotation_files):
            raise ValueError(
                "the number of image_dirs should be equal to annotation_files!"
            )

        for i, _ in enumerate(annotation_files):
            annotation_files[i] = os.path.join(dataset_dir,
                                               annotation_files[i])
            if not os.path.isfile(annotation_files[i]):
                raise ValueError(f"{annotation_files[i]} is not existed.")

        for i, _ in enumerate(image_dirs):
            image_dirs[i] = os.path.join(dataset_dir, image_dirs[i])
            if not os.path.isdir(image_dirs[i]):
                raise ValueError(f"{image_dirs[i]} is not existed.")

        if column_names is None:
            column_names = cls._default_column_names

        kwargs.pop("None", None)
        multicap_dataset = MultiImgCapDataSet(image_dirs, annotation_files,
                                              stage, repeat_images)

        return GeneratorDataset(multicap_dataset,
                                column_names,
                                shuffle=shuffle,
                                **kwargs)


class MultiImgCapDataSet:
    """MultiImgCapDataSet API.

        Args:
            image_dirs (str): The directory which contains images.
            annotation_files list[str]: the list of files contains annotations.
            stage (Optional[str]): The supported keywords are in ["train", "eval"]
            repeat_images (Optional[bool]): whether repeat image when it has multiple corresponding captions.

        Return:
            A Dataset for loading multiple image-caption datasets

    """

    def __init__(self,
                 image_dirs,
                 annotation_files,
                 stage="train",
                 repeat_images=False):
        image_dirs = os.path.realpath(image_dirs)
        self.annotation = []
        if stage in ("train", "eval"):
            for i, annotation_file in enumerate(annotation_files):
                with open(annotation_file, 'r', encoding='utf-8') as file:
                    new_annotation = json.load(file)
                for new_ann in new_annotation:
                    new_ann["image"] = os.path.join(image_dirs[i],
                                                    new_ann["image"])
                self.annotation.extend(new_annotation)
            if stage == "eval":
                self.txt2img = {}
                self.img2txt = {}

                if repeat_images:
                    new_annotation = []

                txt_id = 0
                for img_id, ann in enumerate(self.annotation):
                    self.img2txt[img_id] = []
                    for i, caption in enumerate(ann["caption"]):
                        self.img2txt[img_id].append(txt_id)
                        self.txt2img[txt_id] = img_id
                        if repeat_images:
                            temp = copy.deepcopy(ann)
                            temp.update({"caption": caption})
                            new_annotation.append(temp)
                        txt_id += 1
                if repeat_images:
                    self.annotation = new_annotation
        else:
            raise ValueError("unsupported stage.")

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = Image.open(ann["image"]).convert("RGB")

        return image, ann["caption"]

    def __len__(self):
        return len(self.annotation)

    def display_item(self, index):
        """display item

        Args:
            index (int): index

        Returns:
            out (OrderedDict): item info
        """
        sample, ann = self[index], self.annotation[index]

        return OrderedDict({
            "file": ann["image"],
            "caption": ann["caption"],
            "image": sample[0],
        })
