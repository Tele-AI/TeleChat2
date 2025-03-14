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
"""VisualGLM DataLoader"""

import random
from typing import Callable, Union

import numpy as np
from mindspore.dataset import GeneratorDataset

from mindformers.dataset.dataloader.sft_dataloader import SFTDataSet
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


def custom_map_func(row_dict, **kwargs):
    """Default data parsing function.Returns the first three values of `row_dict`."""
    kwargs.clear()
    values = list(row_dict.values())
    if len(values) == 1:
        return dict(img=values[0], prompt="", label="")
    if len(values) == 2:
        return dict(img=values[0], prompt=values[1], label="")
    return dict(img=values[0], prompt=values[1], label=values[2])


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class VisualGLMDataLoader:
    """VisualGLM DataLoader"""
    def __new__(cls,
                dataset_dir: str,
                tokenizer: Union[str, dict, Callable],
                column_names: str,
                dataset_name: str = "",
                file_format: str = None,
                customized_reader: Callable = None,
                customized_parser: Callable = None,
                shuffle: bool = False,
                scale: int = 1,
                random_mapping: bool = False,
                **kwargs):
        r"""
        VisualGLM DataLoader implementation.
        Args:
            dataset_dir (str): The directory path to parquet text with hdfs.
            dataset_name (str): Dataset name. Currently, ["wikitext"] is supported.
            file_format (str): Retrieves the end character of the desired file name.
            customized_reader (Callable): User-defined functions for reading data.
                The input parameter is the path of the dataset file.
                The return value is a list of many sentences.
            customized_parser (Callable): User-defined function for parsing data.
                The input parameter is a dictionary that contains a single line of data.
                There are three return values: prompt, answerh and label. If a value is not required,
                an empty string is returned.
            shuffle (Optional[bool]): Whether or not to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.

        Return:
            A GeneratorDataset object.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from visualglm_dataloader import VisualGLMDataLoader
            >>> data_loader = VisualGLMDataLoader(dataset_dir="The required task dataset path",
            ...                                    dataset_name="alpaca",
            ...                                    file_format="json",
            ...                                    shuffle=True)
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        del customized_parser
        if random_mapping:
            dataset = SFTRandomMappingDataSet(dataset_dir, column_names, tokenizer, dataset_name, file_format,
                                              customized_reader, map_function=custom_map_func, scale=scale)
        else:
            dataset = SFTDataSet(dataset_dir, column_names=column_names, tokenizer=tokenizer, dataset_name=dataset_name,
                                 file_format=file_format, read_function=customized_reader,
                                 map_function=custom_map_func)
        return GeneratorDataset(dataset, column_names=column_names, shuffle=shuffle, **kwargs)


class SFTRandomMappingDataSet(SFTDataSet):
    """
    sftdataset with random mapping
    """
    def __init__(self, dataset_dir, column_names, tokenizer, dataset_name=None, file_format=None,
                 customized_reader=None, map_function=custom_map_func, scale=1):
        super().__init__(dataset_dir=dataset_dir,
                         column_names=column_names,
                         tokenizer=tokenizer,
                         dataset_name=dataset_name,
                         file_format=file_format,
                         read_function=customized_reader,
                         map_function=map_function
                         )

        self.scale = scale

    def __len__(self):
        return (self.table.shape[0]) * self.scale

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        i = rng.randint(self.table.shape[0])
        return super().__getitem__(i)
