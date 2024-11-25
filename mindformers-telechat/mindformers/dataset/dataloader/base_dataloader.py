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
    @classmethod
    def load_dataset(cls, path: str, **kwargs):
        """load dataset"""
        import datasets
        try:
            logger.info(f"USE_OM : {os.environ['USE_OM']}")
            # pylint: disable=W0611
            import openmind_datasets
            logger.info("connect openmind")

        except ModuleNotFoundError:
            logger.info("connect huggingFace")

        dataset = datasets.load_dataset(path, **kwargs)
        return dataset
