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
"""YiZhao Dataset."""
import copy
import os
import re
from typing import Union, Optional, Callable

import numpy as np
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import TypeCast

from mindformers.dataset.base_dataset import BaseDataset
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.version_control import get_dataset_map


def _process_raw_text_data(dataset_config):
    """Process the text data"""
    dataset_dir = dataset_config.data_loader.pop("dataset_dir")
    dataset = build_dataset_loader(
        dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                  'num_shards': dataset_config.device_num,
                                                  'shard_id': dataset_config.rank_id})
    return dataset


def _process_mindrecord_data(dataset_config):
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


def get_input_data_batch_slice_map(chosen_input_ids, chosen_labels,
                                   chosen_attention_mask, chosen_loss_mask, chosen_ref_logps, rejected_input_ids,
                                   rejected_labels,
                                   rejected_attention_mask, rejected_loss_mask, rejected_ref_logps,
                                   dis, rank_id: int = 0, micro_batch: int = 1):
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

    # Full batch for pipeline parallel
    bs = chosen_input_ids.shape[0]
    input_ids = []
    labels = []
    attention_mask = []
    loss_mask = []
    size_per_stage = bs // micro_batch
    for stage in range(micro_batch):
        input_ids.append(chosen_input_ids[stage * size_per_stage: (stage + 1) * size_per_stage])
        input_ids.append(rejected_input_ids[stage * size_per_stage: (stage + 1) * size_per_stage])
        labels.append(chosen_labels[stage * size_per_stage: (stage + 1) * size_per_stage])
        labels.append(rejected_labels[stage * size_per_stage: (stage + 1) * size_per_stage])
        attention_mask.append(chosen_attention_mask[stage * size_per_stage: (stage + 1) * size_per_stage])
        attention_mask.append(rejected_attention_mask[stage * size_per_stage: (stage + 1) * size_per_stage])
        loss_mask.append(chosen_loss_mask[stage * size_per_stage: (stage + 1) * size_per_stage])
        loss_mask.append(rejected_loss_mask[stage * size_per_stage: (stage + 1) * size_per_stage])

    input_ids = np.concatenate(input_ids)
    labels = np.concatenate(labels)
    attention_mask = np.concatenate(attention_mask)
    loss_mask = np.concatenate(loss_mask)

    return input_ids, labels, attention_mask, loss_mask, chosen_ref_logps, rejected_ref_logps


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class YiZhaoDPODataset(BaseDataset):
    """ YiZhaoDPODataset """

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
        logger.info("Now Create DPO Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        micro_batch = dataset_config.micro_batch

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = _process_raw_text_data(dataset_config)
            else:
                dataset = _process_mindrecord_data(dataset_config)
        else:
            dataset = dataset_config.data_loader

        type_cast_op = TypeCast(mstype.int32)
        float_type_cast_op = TypeCast(mstype.float32)
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

        def map_func(chosen_input_ids, chosen_labels,
                     chosen_attention_mask, chosen_loss_mask, chosen_ref_logps,
                     rejected_input_ids, rejected_labels,
                     rejected_attention_mask, rejected_loss_mask, rejected_ref_logps):
            return get_input_data_batch_slice_map(chosen_input_ids=chosen_input_ids,
                                                  chosen_labels=chosen_labels,
                                                  chosen_attention_mask=chosen_attention_mask,
                                                  chosen_loss_mask=chosen_loss_mask,
                                                  chosen_ref_logps=chosen_ref_logps,
                                                  rejected_input_ids=rejected_input_ids,
                                                  rejected_labels=rejected_labels,
                                                  rejected_attention_mask=rejected_attention_mask,
                                                  rejected_loss_mask=rejected_loss_mask,
                                                  rejected_ref_logps=rejected_ref_logps,
                                                  rank_id=rank_id,
                                                  dis=dis,
                                                  micro_batch=micro_batch)

        dataset = get_dataset_map(dataset, map_func,
                                  input_columns=dataset_config.input_columns,
                                  output_columns=dataset_config.output_columns)
        dataset = dataset.project(columns=dataset_config.output_columns)

        for input_arg in ['input_ids', 'labels']:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        for input_arg in ['ref_chosen_logps', 'ref_rejected_logps']:
            dataset = dataset.map(operations=float_type_cast_op, input_columns=input_arg)

        dataset = dataset.repeat(dataset_config.repeat)
        return dataset


def generate_labels_and_mask(input_ids, pad_id=151329):
    """

    Args:
        input_ids:
        pad_id:

    Returns:

    """
    input_ids = np.array(input_ids)

    tokens = input_ids.copy()
    tokens = tokens[:, :-1]
    tokens[tokens == -100] = pad_id
    labels = input_ids[:, 1:].copy()
    attention_mask = np.ones(tokens.shape)
    loss_mask = (labels != -100)

    tokens = np.array(tokens, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    attention_mask = np.array(attention_mask, dtype=np.int32)
    loss_mask = np.array(loss_mask, dtype=np.float32)

    return tokens, labels, attention_mask, loss_mask


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class YiZhaoPretrainDataset(BaseDataset):
    """ YiZhaoPretrainDataset """

    # pylint: disable=W0613, W0612
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

        logger.info("Now Create DPO Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        dataset_config = copy.deepcopy(dataset_config)
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        micro_batch = dataset_config.micro_batch

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != "MindDataset" and \
                    dataset_config.data_loader.type != "TFRecordDataset":
                dataset = _process_raw_text_data(dataset_config)
            else:
                dataset = _process_mindrecord_data(dataset_config)
        else:
            dataset = dataset_config.data_loader

        type_cast_op = TypeCast(mstype.int32)
        float_type_cast_op = TypeCast(mstype.float32)
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

        def map_func(input_ids):
            return generate_labels_and_mask(input_ids=input_ids)

        dataset = get_dataset_map(dataset, map_func,
                                  input_columns=dataset_config.input_columns,
                                  output_columns=dataset_config.output_columns)
        dataset = dataset.project(columns=dataset_config.output_columns)

        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
