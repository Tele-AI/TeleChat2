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
"""Base DataLoader"""
import os

from mindformers.tools.logger import logger


class BaseDataLoader:
    """Base Dataloader"""

    support_load_func = ['load_dataset', 'load_from_disk']

    @classmethod
    def load_dataset(cls, path: str, load_func: str = 'load_dataset', **kwargs):
        """load dataset"""
        import datasets
        from datasets import Dataset
        try:
            logger.info(f"USE_OM: {os.environ.get('USE_OM', False)}")
            # pylint: disable=W0611
            import openmind_datasets
            logger.info("connect openmind")

        except (ModuleNotFoundError, KeyError):
            logger.info("connect huggingFace")

        if load_func == 'load_dataset':
            dataset = datasets.load_dataset(path, **kwargs)
        elif load_func == 'load_from_disk':
            dataset = Dataset.load_from_disk(path)
        else:
            raise ValueError(
                f"Unsupported load_func {load_func}, please set load_func in {cls.support_load_func}.")

        return dataset
