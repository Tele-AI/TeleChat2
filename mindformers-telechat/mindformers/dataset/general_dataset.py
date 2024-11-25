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
"""General Dataset."""
from collections.abc import Iterable
from typing import Optional, Union, Callable

from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class GeneralDataset(BaseDataset):
    """
    General Dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_collator (Optional[Callable]):
            Batch data processing function.
        dataset (Union[Iterable, Callable]):
            Dataset object for creating dataloader.

    Returns:
        A dataset for GeneratorDataset.
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                dataset: Union[Iterable, Callable] = None,
                data_collator: Optional[Callable] = None,
                **kwargs):
        """new method"""
        logger.info("Now Create Generator Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        shard_able = hasattr(dataset, '__getitem__')
        dataset = GeneratorDataset(dataset,
                                   column_names=dataset_config.input_columns,
                                   shuffle=dataset_config.shuffle,
                                   num_shards=device_num if shard_able else None,
                                   shard_id=rank_id if shard_able else None,
                                   python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.output_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        if data_collator is not None:
            dataset = dataset.map(data_collator,
                                  input_columns=dataset_config.output_columns)
        dataset = dataset.repeat(dataset_config.repeat)

        return dataset
