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
"""Causal Image Modeling Dataset."""
import os
import copy
import re
from functools import partial
from typing import Union, Optional, Callable

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.dataset.base_dataset import BaseDataset

__all__ = ['DPODataset']


def get_input_data_batch_slice_map(chosen_input_ids, chosen_labels,
                                   chosen_attention_mask, chosen_loss_mask, chosen_ref_logps,
                                   rejected_input_ids, rejected_labels,
                                   rejected_attention_mask, rejected_loss_mask, rejected_ref_logps,
                                   dis, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset
    """
    rank = int(rank_id)
    chosen_input_ids = chosen_input_ids[rank * dis: (rank + 1) * dis]
    rejected_input_ids = rejected_input_ids[rank * dis: (rank + 1) * dis]
    chosen_labels = chosen_labels[rank * dis: (rank + 1) * dis]
    rejected_labels = rejected_labels[rank * dis: (rank + 1) * dis]
    chosen_attention_mask = chosen_attention_mask[rank * dis: (rank + 1) * dis]
    rejected_attention_mask = rejected_attention_mask[rank * dis: (rank + 1) * dis]
    chosen_loss_mask = chosen_loss_mask[rank * dis: (rank + 1) * dis]
    rejected_loss_mask = rejected_loss_mask[rank * dis: (rank + 1) * dis]
    chosen_ref_logps = chosen_ref_logps[rank * dis: (rank + 1) * dis]
    rejected_ref_logps = rejected_ref_logps[rank * dis: (rank + 1) * dis]

    return chosen_input_ids, chosen_labels, chosen_attention_mask, chosen_loss_mask, chosen_ref_logps, \
        rejected_input_ids, rejected_labels, rejected_attention_mask, rejected_loss_mask, rejected_ref_logps


def get_input_packing_data_batch_slice_map(chosen_attention_mask, chosen_index_packed, chosen_input_ids, chosen_labels,
                                           chosen_lens, chosen_loss_mask, chosen_position_id, rejected_attention_mask,
                                           rejected_index_packed, rejected_input_ids, rejected_labels, rejected_lens,
                                           rejected_loss_mask, rejected_position_id, dis: int = 0, rank_id: int = 0):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset
    """
    rank = int(rank_id)
    chosen_input_ids = chosen_input_ids[rank * dis: (rank + 1) * dis]
    rejected_input_ids = rejected_input_ids[rank * dis: (rank + 1) * dis]
    chosen_labels = chosen_labels[rank * dis: (rank + 1) * dis]
    rejected_labels = rejected_labels[rank * dis: (rank + 1) * dis]
    chosen_attention_mask = chosen_attention_mask[rank * dis: (rank + 1) * dis]
    rejected_attention_mask = rejected_attention_mask[rank * dis: (rank + 1) * dis]
    chosen_index_packed = chosen_index_packed[rank * dis: (rank + 1) * dis]
    chosen_lens = chosen_lens[rank * dis: (rank + 1) * dis]
    chosen_loss_mask = chosen_loss_mask[rank * dis: (rank + 1) * dis]
    chosen_position_id = chosen_position_id[rank * dis: (rank + 1) * dis]
    rejected_index_packed = rejected_index_packed[rank * dis: (rank + 1) * dis]
    rejected_lens = rejected_lens[rank * dis: (rank + 1) * dis]
    rejected_loss_mask = rejected_loss_mask[rank * dis: (rank + 1) * dis]
    rejected_position_id = rejected_position_id[rank * dis: (rank + 1) * dis]

    return chosen_attention_mask, chosen_index_packed, chosen_input_ids, chosen_labels, chosen_lens, \
            chosen_loss_mask, chosen_position_id, rejected_attention_mask, rejected_index_packed, \
            rejected_input_ids, rejected_labels, rejected_lens, rejected_loss_mask, rejected_position_id


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class DPODataset(BaseDataset):
    """
    Causal Language Model pretrain dataset.
    output input_ids columns

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        input_columns (list):
            Column name before the map function.
        output_columns (list):
            Column name after the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained in the last batch is smaller
            than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        python_multiprocessing (bool):
            Enabling the Python Multi-Process Mode to Accelerate Map Operations. Default: False.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        eod_reset (bool):
            Specifies whether to reset the EOD. Default: False.
        eod_token_id (int):
            Indicates the token id of the EOD.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for CausalLanguageModelDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['text_generation']['gpt2']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = CausalLanguageModelDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindspore.dataset import MindDataset
        >>> from mindformers.dataset import CausalLanguageModelDataset
        >>> data_loader = MindDataset(dataset_files="The required task dataset path", shuffle=True)
        >>> dataset_from_param = CausalLanguageModelDataset(data_loader=data_loader,
        ...                                                 input_columns=["input_ids", "attention_mask"])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                eod_reset: bool = False,
                eod_token_id: Optional[int] = None,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Causal Language Model Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        print('dataset_config', dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        use_data_packing = dataset_config.use_data_packing

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)
        else:
            dataset = dataset_config.data_loader
        type_cast_op = TypeCast(mstype.int32)
        if dataset_config.eod_reset:
            if cls._is_semi_full_batch() or cls._is_data_parallel():
                rank_id = 0
                dis = dataset_config.batch_size
            else:
                # Each card slice a small batch from the full batch
                dis = dataset_config.batch_size // device_num
                if dataset_config.batch_size % device_num != 0:
                    raise ValueError(
                        f"batch size {dataset_config.batch_size} should be a multiple of device number {device_num}."
                        " You should change the args: per_batch_size.")

            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns)
            if use_data_packing:
                my_get_input_data_batch_slice_map = partial(get_input_packing_data_batch_slice_map,
                                                            dis=dis, rank_id=rank_id)
            else:
                my_get_input_data_batch_slice_map = partial(get_input_data_batch_slice_map, dis=dis, rank_id=rank_id)

            dataset = get_dataset_map(dataset, my_get_input_data_batch_slice_map,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns)
            dataset = dataset.project(columns=dataset_config.output_columns)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    output_columns=dataset_config.input_columns,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
            dataset = dataset.project(columns=dataset_config.output_columns)
        for input_arg in ['chosen_input_ids', 'rejected_input_ids', 'chosen_labels', 'rejected_labels']:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id})
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_files = []
        mind_compile = re.compile(r"mindrecord\d*$")
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if re.findall(mind_compile, file) or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if re.findall(mind_compile, data_dir) or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
