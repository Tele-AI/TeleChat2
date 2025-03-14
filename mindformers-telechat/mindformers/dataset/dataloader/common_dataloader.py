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
"""Common DataLoader"""
import importlib

from typing import Optional
from packaging import version
from mindformers.dataset.handler import build_data_handler
from mindformers.tools.logger import logger
from ...tools.register import MindFormerRegister, MindFormerModuleType
from .base_dataloader import BaseDataLoader


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class CommonDataLoader(BaseDataLoader):
    """Common Dataloader"""
    _support_parameters = ["path", "name", "data_dir", "data_files", "split", "cache_dir", "features",
                           "download_config", "download_mode", "verification_mode", "ignore_verifications",
                           "keep_in_memory", "save_infos", "revision", "token", "use_auth_token", "task",
                           "streaming", "num_proc", "storage_options", "trust_remote_code"]

    # pylint: disable=W0102
    def __new__(cls,
                shard_id: Optional[int] = None,
                num_shards: Optional[int] = None,
                column_names: list = ["input_ids", "labels"],
                shuffle: bool = False,
                path: Optional[str] = None,
                load_func: str = 'load_dataset',
                handler: Optional[list] = None,
                packing: str = None,
                adaptor_config: dict = None,
                **kwargs):
        if path is None or path.strip() == "":
            raise ValueError(f"path should not be empty.")

        if "split" not in kwargs:
            kwargs["split"] = "train"

        kwargs = cls._filter_params(kwargs=kwargs)
        ms_adaptor_execution()
        dataset = cls.load_dataset(path=path, load_func=load_func, **kwargs)

        if handler:  # data preprocess
            if not isinstance(handler, list):
                raise ValueError(f"handler in config should be set as 'list', but got {type(handler)}.")
            for per_handler in handler:
                data_handler = build_data_handler(per_handler, packing=packing)
                dataset = data_handler.handle(dataset)

        dataset = dataset.to_ms_dataset(columns=column_names,
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        shuffle=shuffle,
                                        packing=packing,
                                        adaptor_config=adaptor_config)

        return dataset

    @classmethod
    def _filter_params(cls, kwargs):
        result = {}
        for key in kwargs:
            if key not in cls._support_parameters:
                logger.info(f"dataset load_dataset not support params: {key}")
                continue
            result[key] = kwargs[key]
        return result


def ms_adaptor_execution():
    """ms adaptor execution"""
    try:
        from datasets import config
        ms_version = version.parse(importlib.metadata.version("mindspore"))
        config.MS_VERSION = ms_version
        logger.info(f"Mindspore version {ms_version} available.")

        from datasets import Dataset
        from .ms_ds_convertor import to_ms_dataset

        setattr(Dataset, "to_ms_dataset", to_ms_dataset)
    except importlib.metadata.PackageNotFoundError:
        logger.error(f"to_ms_dataset adaptor failed.")
