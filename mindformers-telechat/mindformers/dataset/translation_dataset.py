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
"""T5 Dataset."""
import os
import copy
from typing import Optional, Union, Callable
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset import MindDataset

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset
from ..models.auto.tokenization_auto import AutoTokenizer


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TranslationDataset(BaseDataset):
    """Translation dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        tokenizer (Union[dict, list]):
            Tokenizer configuration or object.
        input_columns (list):
            Column name before the map function.
        output_columns (list):
            Column name after the map function.
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
        src_max_length (int):
            Maximum length of the source sequence.
        tgt_max_length (int):
            Maximum length of the target sequence.
        prefix (str):
            Prefix of prompt.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for TranslationDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import TranslationDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['translation']['t5_small']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/t5.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = TranslationDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindformers.dataset import TranslationDataset, WMT16DataLoader
        >>> from mindformers import AutoTokenizer
        >>> data_loader = WMT16DataLoader(dataset_dir="The required task dataset path")
        >>> tokenizer = AutoTokenizer.from_pretrained("t5_small")
        >>> dataset_from_param = TranslationDataset(data_loader=data_loader, tokenizer=tokenizer,
        ...                                         input_columns=['input_ids', 'attention_mask', 'labels'],
        ...                                         output_columns=['input_ids', 'attention_mask', 'labels'],
        ...                                         src_max_length=1024, tgt_max_length=128,
        ...                                         prefix='translate the English to Romanian:')
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                tokenizer: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                repeat: int = 1,
                src_max_length: int = None,
                tgt_max_length: int = None,
                prefix: str = "",
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Translation Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        cls.src_max_length = src_max_length
        cls.tgt_max_length = tgt_max_length
        cls.prefix = prefix

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != 'MindDataset':
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)
        elif isinstance(dataset_config.data_loader, MindDataset):
            dataset = dataset_config.data_loader
        else:
            dataset = cls._tokenizer_map(dataset_config.data_loader, dataset_config.tokenizer)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = get_dataset_map(dataset, operations=type_cast_op, input_columns=input_arg)
        return dataset

    @classmethod
    def _tokenizer_map(cls, dataset, tokenizer_config):
        """Maps the tokenizer on the source and the output"""
        if isinstance(tokenizer_config, dict):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.type)
            prefix = tokenizer_config.prefix if tokenizer_config.prefix else cls.prefix
            src_max_length = tokenizer_config.src_max_length if tokenizer_config.src_max_length else cls.src_max_length
            tgt_max_length = tokenizer_config.tgt_max_length if tokenizer_config.tgt_max_length else cls.tgt_max_length
        else:
            tokenizer = tokenizer_config
            prefix = cls.prefix
            src_max_length = cls.src_max_length
            tgt_max_length = cls.tgt_max_length

        logger.info("Start tokenize on the dataset using tokenizer: %s", tokenizer_config)

        def pad_max_function(src, tgt):
            src = src.tolist()
            if isinstance(src, bytes):
                src = src.decode()
            output = tokenizer(prefix + src, padding='max_length', max_length=src_max_length, truncation=True)

            tgt = tgt.tolist()
            if isinstance(tgt, bytes):
                tgt = tgt.decode()
            tgt_output = tokenizer(tgt, padding='max_length', max_length=tgt_max_length, truncation=True)

            input_ids = np.array(output['input_ids'], np.int32)
            attention_mask = np.array(output['attention_mask'], np.float32)
            labels = np.array(tgt_output['input_ids'], np.int32)
            return input_ids, attention_mask, labels

        dataset = get_dataset_map(dataset, pad_max_function,
                                  input_columns=['source', 'target'],
                                  output_columns=['input_ids', 'attention_mask', 'labels'])
        dataset = dataset.project(columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        rank_id, device_num = cls._generate_shard_info()

        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': device_num, 'shard_id': rank_id})

        dataset = cls._tokenizer_map(dataset, dataset_config.tokenizer)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        rank_id, device_num = cls._generate_shard_info()

        dataset_config = copy.deepcopy(dataset_config)

        dataset_files = []
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".mindrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if data_dir.endswith(".mindrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        logger.info("Using args %s to instance the dataset.", dataset_config.data_loader)
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': device_num, 'shard_id': rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
