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
"""Question Answering Dataset."""
from typing import Optional, Union, Callable

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from ..models.build_tokenizer import build_tokenizer
from .sampler import build_sampler
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class QuestionAnsweringDataset(BaseDataset):
    """
    Question Answering Dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        tokenizer (Union[dict, list]):
            Tokenizer configuration or object.
        sampler (Union[dict, list]):
            Sampler configuration or object.
        input_columns (list):
            Column name before the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained
            in the last batch is smaller than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            ndicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for QuestionAnsweringDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import QuestionAnsweringDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['question_answering']['qa_bert_base_uncased']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/question_answering.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = QuestionAnsweringDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindformers import AutoTokenizer
        >>> from mindformers.dataset import QuestionAnsweringDataset, SQuADDataLoader
        >>> tokenizer = AutoTokenizer.from_pretrained('qa_bert_base_uncased_squad')
        >>> data_loader = SQuADDataLoader(dataset_dir="The required task dataset path", tokenizer=tokenizer,
        ...                               column_names=['input_ids', 'input_mask', 'token_type_id', 'start_position',
        ...                                             'end_position', 'unique_id'])
        >>> dataset_from_param = QuestionAnsweringDataset(data_loader=data_loader, tokenizer=tokenizer, batch_size=12,
        ...                                               input_columns=['input_ids', 'input_mask', 'token_type_id',
        ...                                                              'start_position', 'end_position',
        ...                                                              'unique_id'])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                tokenizer: Union[dict, Callable] = None,
                sampler: Union[dict, Callable] = None,
                input_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        """new method"""
        logger.info("Now Create Question Answering Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        if isinstance(dataset_config.tokenizer, dict):
            tokenizer = build_tokenizer(dataset_config.tokenizer)
        else:
            tokenizer = dataset_config.tokenizer

        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'tokenizer': tokenizer,
                                                         'num_parallel_workers': dataset_config.num_parallel_workers,
                                                         'num_shards': device_num,
                                                         'shard_id': rank_id
                                                         })
        else:
            dataset = dataset_config.data_loader

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.input_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)

        type_cast_op = TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = get_dataset_map(dataset, type_cast_op, input_columns=input_arg)
        return dataset
