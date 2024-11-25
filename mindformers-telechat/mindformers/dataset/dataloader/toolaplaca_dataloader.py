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
"""ToolAplaca DataLoader"""
import json
import os
import copy

from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class ToolAlpacaDataLoader:
    """ToolAlpaca Dataloader"""
    def __new__(cls, dataset_dir, shuffle=False, **kwargs):
        r"""
        ToolAlpacaDataLoader Dataloader API.

        Args:
            dataset_dir: The directory to ToolAlpaca dataset.
            shuffle: Whether to shuffle

        Return:
            A GeneratorDataset for ToolAlpaca dataset

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import ToolAlpacaDataLoader
            >>> data_loader = ToolAlpacaDataLoader("./tool_alpaca.jsonl")
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        multiturn_dataset = ToolAlpacaDataset(dataset_dir)

        logger.info("shuffle status is %s", shuffle)
        dataset = GeneratorDataset(multiturn_dataset, column_names=['data'], shuffle=shuffle, **kwargs)

        return dataset


class ToolAlpacaDataset:
    """ToolAlpaca Dataset"""

    def __init__(self, dataset_dir):
        r"""
        ToolAlpaca Dataset

        Args:
            dataset_dir (str): The directory to ToolAlpaca dataset.

        Return:
            A iterable dataset for ToolAlpaca dataset

        a valid input data should be like this:
            {'tools': ['tools'],
             'conversations': [{'role': 'user', 'content': 'content'},
                               {'role': 'assistant', 'content': 'content'},
                               {'role': 'tool', 'name': 'name', 'parameters': {}, 'observation': 'observation'},
                               {'role': 'assistant', 'content': 'content'}]}
             middle 'assistant' and "tool" may occur multiple times
        """
        dataset_dir = os.path.realpath(dataset_dir)
        self.dataset_dir = dataset_dir
        with open(dataset_dir, "r", encoding="utf-8") as fp:
            if dataset_dir.endswith(".json"):
                try:
                    data = json.load(fp)
                except json.JSONDecodeError as e:
                    logger.error("loading data failed, please check your data file.")
            elif dataset_dir.endswith(".jsonl"):
                data = []
                for i, line in enumerate(fp):
                    # avoid empty line
                    if line.strip() == "":
                        logger.info("Drop %s:%d due to empty line.", dataset_dir, i)
                        continue
                    # avoid json loading error
                    try:
                        line = json.loads(line, strict=False)
                        if self._is_data_valid(line, i + 1):
                            data.append(line)
                    except json.JSONDecodeError as e:
                        logger.info("Drop %s:%d due to '%s', line is:\n%s", dataset_dir, i, e, line)
                        continue
        self.data = data
        if self.__len__() <= 0:
            raise ValueError(f"valid data less then 1, loading data failed, please check your data file.")
        logger.info("loading %d data success!", self.__len__())

    def _is_data_valid(self, line, i):
        '''check data validity'''
        # should have keys 'tools' and 'conversations'
        if 'tools' not in line or 'conversations' not in line:
            logger.info("Drop %s:%d due to missed keys 'tools' or 'conversations', line is:\n%s", \
                            self.dataset_dir, i, line)
            return False

        # 'conversations' is a dict with listed format
        conversations = copy.deepcopy(line['conversations'])

        # conversations have at least 4 members
        if not isinstance(conversations, list) or len(conversations) < 4:
            logger.info("Drop %s:%d due to invalid conversations, line is:\n%s", \
                            self.dataset_dir, i, line)
            return False

        conv1 = conversations.pop(0)
        res1 = self._is_1st_conv_valid(conv1, i)

        conv2 = conversations.pop(-1)
        res2 = self._is_last_conv_valid(conv2, i)

        # remain conversations should be "assistant" and 'tool' pair , so len(conversations)%2 == 0
        if conversations and len(conversations) % 2 != 0:
            logger.info("Drop %s:%d, remain conversations should be 'assistant' and 'tool' pair, "
                        "but got %s", self.dataset_dir, i, conversations)
            return False

        while conversations:
            conv1 = conversations.pop(0)
            conv2 = conversations.pop(0)

            res3 = self._is_pair_conv_valid(conv1, conv2, i)

        return res1 and res2 and res3

    def _is_1st_conv_valid(self, conv, i):
        '''1st conversation should be like {'role': 'user', 'content': 'content'}'''
        role, content = conv.get("role"), conv.get("content")
        is_role_valid = not role or role != "user"
        is_content_valid = not content or not isinstance(content, str)
        if is_role_valid or is_content_valid:
            logger.info("Drop %s:%d, expect 1st conv like {'role': 'user', 'content': 'content'}, "
                        "but got %s", self.dataset_dir, i, conv)
            return False
        return True

    def _is_last_conv_valid(self, conv, i):
        '''last conversation should be like {"role": "assistant", "content": "content"}'''
        role, content = conv.get("role"), conv.get("content")
        is_role_valid = not role or role != "assistant"
        is_content_valid = not content or not isinstance(content, str)
        if is_role_valid or is_content_valid:
            logger.info("Drop %s:%d, expect last conv like {'role': 'assistant', 'content': 'content'}, "
                        "but got %s", self.dataset_dir, i, conv)
            return False
        return True

    def _is_pair_conv_valid(self, conv1, conv2, i):
        ''' remain conversations should be "assistant" and 'tool' pair'''
        role1, content1 = conv1.get("role"), conv1.get("content")
        role2, name2 = conv2.get("role"), conv2.get("name")
        parameters2, observation2 = conv2.get("parameters", "#"), conv2.get("observation")

        if not role1 or role1 != "assistant" or not content1 or not isinstance(content1, str) or \
           not role2 or role2 != "tool" or not name2 or not isinstance(name2, str) or \
           parameters2 == "#" or not isinstance(parameters2, dict) or \
           not observation2 or not isinstance(observation2, str):

            logger.info("Drop %s:%d, expect pair conv like {'role': 'assistant', 'content': 'content'},"
                        "{'role': 'tool', 'name': 'name', 'parameters': {}, 'observation': 'observation'}, "
                        "but got %s, %s", self.dataset_dir, i, conv1, conv2)
            return False
        return True

    def __len__(self):
        """Get the size of dataset"""
        return len(self.data)

    def __getitem__(self, i):
        """Return input data"""
        return self.data[i]
