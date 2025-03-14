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
"""Base Dataset."""
import os

import mindspore as ms
import mindspore.dataset as ds

from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.logger import logger


class BaseDataset:
    """
    Base Dataset.

    Args:
        dataset_config (dict): Config for dataset.

    """

    def __init__(self, dataset_config: dict = None):
        self.dataset_config = dataset_config

    @classmethod
    def check_dataset_config(cls, dataset_config, params):
        """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
        if not dataset_config:
            params.pop("dataset_config")
            kwargs = params.pop("kwargs") if params.get("kwargs") else {}
            params.update(kwargs)
            dataset_config = MindFormerConfig(**params)
        return dataset_config

    @classmethod
    def init_dataset_config(cls, dataset_config):
        """Init dataset config."""
        ds.config.set_seed(dataset_config.seed)
        ds.config.set_prefetch_size(dataset_config.prefetch_size)
        ds.config.set_numa_enable(dataset_config.numa_enable)

        if dataset_config.auto_tune:
            if dataset_config.profile:
                raise EnvironmentError(
                    "MindSpore's AutoTune is enabled, so Profile cannot be enabled,"
                    "now Profile's flag is True, please set to False!")
            os.makedirs(dataset_config.filepath_prefix, exist_ok=True)
            dataset_config.filepath_prefix = os.path.join(dataset_config.filepath_prefix, "autotune")
            ds.config.set_enable_autotune(True, filepath_prefix=dataset_config.filepath_prefix)
            ds.config.set_autotune_interval(dataset_config.autotune_per_step)

    @classmethod
    def _generate_shard_info(cls):
        """Generate shard info for dataset."""
        shard_id = get_real_rank()
        num_shards = get_real_group_size()

        ds_stra = ms.context.get_auto_parallel_context("dataset_strategy")
        if cls._is_semi_full_batch():
            shard_id = None
            num_shards = None
        elif cls._is_semi() and not cls._is_full_batch():
            pp = ms.context.get_auto_parallel_context("pipeline_stages")
            first_input_stra = ds_stra[0]
            dp = first_input_stra[0]
            mp = num_shards // pp // dp
            shard_id = shard_id % (num_shards // pp) // mp
            num_shards = dp

        logger.info(f"Now dataset_strategy is {ds_stra}, shard_id: {shard_id}, num_shards: {num_shards}")
        return shard_id, num_shards

    @classmethod
    def _check_device_rank_for_parallel(cls, shard_id, num_shards):
        """Check num shards and shard id in auto parallel mode."""
        if cls._is_semi_full_batch():
            shard_id = None
            num_shards = None
        return shard_id, num_shards

    @classmethod
    def _is_semi(cls):
        return ms.context.get_auto_parallel_context("parallel_mode") in ['semi_auto_parallel', 'auto_parallel']

    @classmethod
    def _is_full_batch(cls):
        return ms.context.get_auto_parallel_context("full_batch")

    @classmethod
    def _is_semi_full_batch(cls):
        return cls._is_semi() and cls._is_full_batch()

    @classmethod
    def _is_data_parallel(cls):
        return ms.context.get_auto_parallel_context("parallel_mode") == ms.context.ParallelMode.DATA_PARALLEL
