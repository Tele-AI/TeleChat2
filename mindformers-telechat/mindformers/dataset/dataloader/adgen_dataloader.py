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
"""ADGen DataLoader"""
import json
import os

from mindspore.dataset import GeneratorDataset
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class ADGenDataLoader:
    """ADGen Dataloader"""
    def __new__(cls, dataset_dir, phase, shuffle, origin_columns, **kwargs):
        r"""
        ADGen Dataloader API.

        Args:
            dataset_dir: The directory to ADGen dataset.
            column_names (Optional[Union[List[str], Tuple[str]]]): The output column names,
                                                                   a tuple or a list of string with length 6
            phase: The supported keywords are in ["train", "dev"]

        Return:
            A GeneratorDataset for ADGen dataset

        Raises:
            ValueError: Error input for dataset_dir, and column_names.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import ADGenDataLoader
            >>> from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
            >>> data_loader = ADGenDataLoader("./ADGen/")
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        column_names = ["prompt", "answer"]
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if phase not in ["train", "eval"]:
            raise ValueError(f"phase should be in train or eval.")

        # verify dataset column names
        if not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 2, but got {type(column_names)}, length {len(column_names)}")

        for name in column_names:
            if not isinstance(name, str):
                raise ValueError(f"the item type of column_names should be string,"
                                 f" but got {type(name)}")

        # verify origin column names
        if not isinstance(origin_columns, (tuple, list)) or len(origin_columns) != 2:
            raise TypeError(f"origin_columns should be a tuple or a list of string with length 2, "
                            f" but got {type(origin_columns)}, length {len(origin_columns)}")
        adgen_dataset = ADGenDataset(dataset_dir, origin_columns, phase)

        kwargs.pop("None", None)
        info = f"[DATASET] shuffle status is {shuffle}, phase is {phase}."
        logger.info(info)
        kwargs.pop("version", None)
        return GeneratorDataset(adgen_dataset, column_names, shuffle=shuffle, **kwargs)


class ADGenDataset:
    """ADGen Dataset"""

    def __init__(self, dataset_dir, origin_columns, phase="train"):
        r"""
        ADGen Dataset

        Args:
            dataset_dir (str): The directory to ADGen dataset.
            phase (str): The supported keywords are in ["train", "dev"]

        Return:
            A iterable dataset for ADGen dataset

        Raises:
            ValueError: Error input for dataset_dir, phase.
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        self.dataset_dir = dataset_dir
        self.phase = phase

        examples = {}
        content_list = []
        summary_list = []
        self.prompt_column = origin_columns[0]
        self.response_column = origin_columns[1]
        i = 0
        with open(self.dataset_dir) as fp:
            for i, line in enumerate(fp):
                # avoid empty line
                if line.strip() == "":
                    logger.info("Drop %s:%d due to empty line.", self.dataset_dir, i)
                    continue
                # avoid json loading error
                try:
                    row = json.loads(line, strict=False)
                except json.JSONDecodeError as e:
                    logger.info("Drop %s:%d due to '%s', line is:\n%s", self.dataset_dir, i, e, line)
                    continue

                # avoid incompelete keys
                if self.prompt_column not in row or self.response_column not in row:
                    logger.info("Drop %s:%d due to invalid keys, line is:\n%s", self.dataset_dir, i, line)
                    continue
                prompt = row[self.prompt_column]
                response = row[self.response_column]

                # avoid null value
                if prompt.strip() != "" and response.strip() != "":
                    content_list.append(prompt)
                    summary_list.append(response)
                else:
                    logger.info("Drop %s:%d due to null value, line is:\n%s", self.dataset_dir, i, line)

            examples[self.prompt_column] = content_list
            examples[self.response_column] = summary_list
        self.examples = examples
        if self.__len__() <= 0:
            raise ValueError(f"valid data less then 1, loading data failed, please check your data file.")
        logger.info("Loading %d data success.", self.__len__())

    def __len__(self):
        """Get the size of dataset"""
        return len(self.examples.get(self.prompt_column))

    def __getitem__(self, i):
        """Return input data for model"""
        prompt, answer = self.examples.get(self.prompt_column)[i], self.examples.get(self.response_column)[i]

        return prompt, answer


class ADGenFromRemoteDataset:
    """ADGen Dataset"""

    def __init__(self, dataset_path, origin_columns, data_files, phase="train", token=None):
        r"""
        ADGen Dataset

        Args:
            dataset_dir (str): The directory to ADGen dataset.
            phase (str): The supported keywords are in ["train", "dev"]

        Return:
            A iterable dataset for ADGen dataset

        Raises:
            ValueError: Error input for dataset_dir, phase.
        """
        self.phase = phase
        examples = {}
        content_list = []
        summary_list = []
        self.prompt_column = origin_columns[0]
        self.response_column = origin_columns[1]
        dataset = BaseDataLoader.load_dataset(dataset_path, data_files=data_files, split=self.phase, token=token)

        def deal(example):
            if example == "":
                logger.info("Drop %s due to empty line.", example)
                return

            # avoid incompelete keys
            if self.prompt_column not in example or self.response_column not in example:
                logger.info("Drop due to invalid keys, line is:\n%s", example)
                return
            prompt = example[self.prompt_column]
            response = example[self.response_column]

            # avoid null value
            if prompt.strip() != "" and response.strip() != "":
                content_list.append(prompt)
                summary_list.append(response)
            else:
                logger.info("Drop due to null value, line is:\n%s", example)

        dataset.map(deal)

        examples[self.prompt_column] = content_list
        examples[self.response_column] = summary_list
        self.examples = examples
        if self.__len__() <= 0:
            raise ValueError(f"valid data less then 1, loading data failed, please check your data file.")
        logger.info("Loading %d data success.", self.__len__())

    def __len__(self):
        """Get the size of dataset"""
        return len(self.examples.get(self.prompt_column))

    def __getitem__(self, i):
        """Return input data for model"""
        prompt, answer = self.examples.get(self.prompt_column)[i], self.examples.get(self.response_column)[i]

        return prompt, answer
