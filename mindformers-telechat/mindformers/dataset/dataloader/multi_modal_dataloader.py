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
"""Multimodal DataLoader."""

import json
import os
from typing import Callable
from typing import Optional

from mindspore.dataset import GeneratorDataset
from pyarrow import RecordBatch

from mindformers.dataset.dataloader.sft_dataloader import SFTDataSet
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class BaseMultiModalDataLoader:
    """multimodal dataloader for conversation-format dataset"""

    def __new__(cls,
                annotation_file: str,
                shuffle: Optional[bool] = True,
                extra_kwargs: Optional[dict] = None,
                **kwargs):
        if not os.path.exists(annotation_file):
            raise FileExistsError(f"The annotation_file {annotation_file} is not exist.")

        if extra_kwargs is None:
            extra_kwargs = {}

        dataset = MultiModalSFTDataLoader(annotation_file, **extra_kwargs)
        return GeneratorDataset(dataset, ["conversations"], shuffle=shuffle, **kwargs)


class MultiModalSFTDataLoader(SFTDataSet):
    """multimodal sft dataset for conversation-format dataset"""

    def __init__(self,
                 annotation_file: str,
                 max_length: int = 1025,
                 read_function: Callable = None,
                 map_function_kwargs: dict = None,
                 max_chunk_size: int = 1024):
        self.max_chunk_size = max_chunk_size

        super().__init__(annotation_file, None, None, file_format="json", max_length=max_length,
                         read_function=read_function, map_function=self._simple_map,
                         map_function_kwargs=map_function_kwargs)
        self.num_samples = sum(len(batch) for batch in self.table)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        chunk_idx, item_idx = divmod(i, self.max_chunk_size)
        example = self.table[chunk_idx].take([item_idx]).to_pylist()[0]
        example = self.map_function(example, **self.map_function_kwargs)
        return (example,)

    @staticmethod
    def _simple_map(example, **kwargs):
        """map function to convert sample"""
        data_field = kwargs.get("data_field", "conversations")
        from_keyword = kwargs.get("from_keyword", "from")
        value_keyword = kwargs.get("value_keyword", "value")

        conversation_data = []
        for message in example[data_field]:
            from_ = message[from_keyword]
            value = message[value_keyword]
            conversation_data.append([from_, value])
        return conversation_data

    def _read_json(self, path):
        """Reads JSON format with pyarrow and return list of RecordBatch"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [RecordBatch.from_pylist(data[i:i + self.max_chunk_size])
                for i in range(0, len(data), self.max_chunk_size)]
