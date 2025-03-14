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
"""Dataset for multi-modal sft task"""
from typing import Optional, Union, Callable
import os
import numpy as np

import mindspore as ms
from mindspore.dataset import DistributedSampler

from mindformers.dataset.base_dataset import BaseDataset
from mindformers.dataset.dataloader.build_dataloader import build_dataset_loader
from mindformers.dataset.transforms import build_transforms
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.version_control import get_dataset_map


# pylint: disable=W0613
def batch_add(col, batch_info):
    """batch multi modal data"""
    output = col.copy()
    batch_size = len(col)
    full_batch = ms.get_auto_parallel_context("full_batch")
    ds_stra = ms.get_auto_parallel_context("dataset_strategy")
    dynamic_batch = os.environ.get("IMG_DYNAMIC_BATCH")
    use_dynamic_batch = dynamic_batch and (dynamic_batch == '1' or dynamic_batch.lower() == 'true')
    if full_batch or not isinstance(ds_stra, (list, tuple)) or use_dynamic_batch:
        adder = np.array([[i, 0] for i in range(batch_size)], dtype=np.int32).reshape((batch_size, 1, 1, 2))
    else:
        shard_id = get_real_rank()
        num_shards = get_real_group_size()
        pp = ms.get_auto_parallel_context("pipeline_stages")
        dp = ds_stra[0][0]
        mp = num_shards // pp // dp
        shard_id = shard_id % (num_shards // pp) // mp
        adder = [[i + shard_id * batch_size, 0] for i in range(batch_size)]
        adder = np.array(adder, dtype=np.int32).reshape((batch_size, 1, 1, 2))
    output = output + adder
    return (output,)


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ModalToTextSFTDataset(BaseDataset):
    """Dataset for MultiModal"""

    def __new__(cls,
                dataset_config: Optional[dict] = None,
                tokenizer: Union[dict, Callable] = None,
                sampler: Union[dict, Callable] = None,
                **kwargs):
        """new method"""
        logger.info("Now Create ModalToText SFT Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        shard_id, num_shards = cls._generate_shard_info()

        # build dataloader
        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader)
        else:
            dataset = dataset_config.data_loader
        src_dataset_size = dataset.get_dataset_size()
        if num_shards is not None and shard_id is not None:
            # if full_batch=False, num_shards and shard_id are not None
            # shuffle has been set in build_dataset_loader process
            dataset.add_sampler(DistributedSampler(
                num_shards=num_shards, shard_id=shard_id, shuffle=False))

        # build tokenizer
        if isinstance(dataset_config.tokenizer, dict):
            tokenizer = build_tokenizer(dataset_config.tokenizer)
        else:
            tokenizer = dataset_config.tokenizer

        if not hasattr(dataset_config, "modal_to_text_transform"):
            raise ValueError("ModalToTextSFTDataset should have a `modal_to_text_transform` to transform raw text to "
                             "multi_modal data")

        # build transforms
        if isinstance(dataset_config.modal_to_text_transform, dict):
            max_length = dataset_config.modal_to_text_transform.get("max_length")
            if max_length is None:
                raise ValueError("`modal_to_text_transform` should set max_length")
            modal_to_text_transform = build_transforms(dataset_config.modal_to_text_transform,
                                                       default_args={"tokenizer": tokenizer, "max_length": max_length})
        else:
            modal_to_text_transform = dataset_config.modal_to_text_transform

        # build modal content transforms
        if (isinstance(dataset_config.modal_content_transforms, list)
                and isinstance(dataset_config.modal_content_transforms[0], dict)) \
                or isinstance(dataset_config.modal_content_transforms, dict):
            modal_content_transforms = build_transforms(dataset_config.modal_content_transforms)
        else:
            modal_content_transforms = dataset_config.modal_content_transforms

        output_columns = modal_to_text_transform.model_transform_template.output_columns
        dataset = get_dataset_map(dataset, modal_to_text_transform,
                                  input_columns=["conversations"],
                                  output_columns=output_columns,
                                  num_parallel_workers=dataset_config.num_parallel_workers,
                                  python_multiprocessing=dataset_config.python_multiprocessing)

        if modal_content_transforms is not None:
            modal_content_input_columns = dataset_config.get("modal_content_input_columns")
            if modal_content_input_columns is None:
                raise ValueError("`modal_content_input_columns` needs to be set when modal_content_transforms "
                                 "is specified.")
            modal_content_output_columns = dataset_config.get("modal_content_output_columns",
                                                              modal_content_input_columns)
            dataset = get_dataset_map(dataset, modal_content_transforms,
                                      input_columns=modal_content_input_columns,
                                      output_columns=modal_content_output_columns,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        net_input_columns = dataset_config.get("net_input_columns", output_columns)
        if net_input_columns is not None:
            dataset = dataset.project(columns=net_input_columns)
        if dataset_config.img_dynamic_batch:
            os.environ["IMG_DYNAMIC_BATCH"] = "1"
        batch_input_columns = [col for col in output_columns if col.endswith("_context_pos")]
        if batch_input_columns:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    num_parallel_workers=dataset_config.num_parallel_workers,
                                    input_columns=batch_input_columns,
                                    per_batch_map=batch_add)
        else:
            dataset = dataset.batch(dataset_config.batch_size,
                                    drop_remainder=dataset_config.drop_remainder,
                                    num_parallel_workers=dataset_config.num_parallel_workers)
        if not ms.get_auto_parallel_context("full_batch"):
            # reset dataset size for full_batch=False
            take_num = src_dataset_size // (dataset_config.batch_size * num_shards)
            dataset = dataset.take(take_num)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
