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
"""SFT DataLoader"""
import os
import json
from json.decoder import JSONDecodeError
from typing import Union, Callable
from pyarrow.json import read_json
from pyarrow.csv import read_csv, ParseOptions
from pyarrow.parquet import ParquetDataset
from pyarrow.lib import Table

from mindspore.dataset import GeneratorDataset
from mindspore._checkparam import args_type_check
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.dataset.dataloader.datareaders import _DATA_READER_MAP
from mindformers.dataset.dataloader.sft_map_functions import _SFT_MAP_FUNCTIONS


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class SFTDataLoader:
    """SFT DataLoader"""
    @args_type_check(dataset_dir=str, column_names=list, tokenizer=(str, dict, Callable), dataset_name=str,
                     file_format=str, max_length=int, read_function=Callable, map_function=Callable,
                     map_function_kwargs=dict, shuffle=bool)
    def __new__(cls,
                dataset_dir: str,
                column_names: list,
                tokenizer: Union[str, dict, Callable],
                dataset_name: str = "",
                file_format: str = None,
                max_length: int = 1025,
                read_function: Callable = None,
                map_function: Callable = None,
                map_function_kwargs: dict = None,
                shuffle: bool = False,
                **kwargs):
        r"""
        SFT DataLoader API.

        Args:
            dataset_dir (str): The directory path to parquet text with hdfs.
            column_names(list): Column names contained in the created dataset.
            tokenizer(Union[str, dict, Callable]): Tokenizer configuration.
            dataset_name (str): Dataset name. Currently, ["alpaca, "advertisegen", "cola", "imdb", "sst-2", "ag-news",
                "tnews", "squad", "cmrc2018", "ag-news", "multi-round-chat"] is supported. If this parameter is set to
                "multi-round-chat", the data of multiple rounds of dialogs is processed.
            file_format (str): Retrieves the end character of the desired file name.
            max_length(int): Maximum length of a token.
            read_function (Callable): User-defined functions for reading data.
                The input parameter is the path of the dataset file.
                The return value is a dictionary. Key indicates the column name,
                    and value indicates the value of the column.
            map_function (Callable): User-defined function for parsing data.
                The input parameter is a dictionary that contains a row of data.
                The return value is a dictionary containing a new row of data, and its keys contain column_names.
            map_function_kwargs(dict): kwargs of `map_function`. Parameters other than `tokenizer` and `max_length`
                used by `map_function` can be transferred through `map_function_kwargs`.
                When `dataset_name` is set to "multi-round-chat", map_function_kwargs supports the following kwargs:
                    data_field (str): Name of the field where dialog data is located. Default: "conversations".
                    from_keyword (str): Keyword representing the source of the conversation statement. Default: "from".
                    value_keyword (str): Keyword representing the content of a conversation statement. Default: "value"
                    user_role_name (str): Conversation initiator.  Default: "human".
                    assistant_role_name (str): Conversation collaborator. Default: "gpt".
                    user_prompt (str): The prompt of the conversation initiator is used to add in front of the statement
                        of the conversation initiator. If not specified, it will not be added.
                    assistant_prompt (str): Conversation collaborator prompt, used to precede the conversation
                        collaborator's statement. If not specified, it will not be added.
                    ignore_token_id (int): Used when calculating label, used to mask the conversation initiator or
                        questioner's statement, Default: -100.
            shuffle (bool): Whether to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.

        Return:
            A GeneratorDataset object.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import SFTDataLoader
            >>> data_loader = SFTDataLoader(dataset_dir="The required task dataset path",
            ...                             column_names=["input_ids"],
            ...                             tokenizer={"type": "GPT2Tokenizer", "max_length": 1025},
            ...                             dataset_name="alpaca",
            ...                             file_format="json",
            ...                             shuffle=True)
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        if max_length <= 0:
            raise TypeError(f"max_length should be an integer greater than 0.")
        dataset_dir = os.path.realpath(dataset_dir)

        dataset = SFTDataSet(dataset_dir, column_names=column_names, tokenizer=tokenizer, dataset_name=dataset_name,
                             file_format=file_format, max_length=max_length, read_function=read_function,
                             map_function=map_function, map_function_kwargs=map_function_kwargs)
        return GeneratorDataset(dataset, column_names=column_names, shuffle=shuffle, **kwargs)


class SFTDataSet:
    r"""
    SFT DataSet API.

    Args:
        dataset_dir (str): The directory path to parquet text with hdfs.
        column_names(list): Column names contained in the created dataset.
        tokenizer(Union[str, dict, Callable]): Tokenizer configuration.
        dataset_name (str): Dataset name. Currently, ["alpaca, "advertisegen", "cola", "imdb", "sst-2", "ag-news",
            "tnews", "squad", "cmrc2018", "ag-news"] is supported.
        file_format (str): Retrieves the end character of the desired file name.
        max_length(int): Maximum length of a token.
        read_function (Callable): User-defined functions for reading data.
            The input parameter is the path of the dataset file.
            The return value is a dictionary. Key indicates the column name,
                and value indicates the value of the column.
        map_function (Callable): User-defined function for parsing data.
            The input parameter is a dictionary that contains a row of data.
            The return value is a dictionary containing a new row of data, and its keys contain column_names.
        map_function_kwargs(dict): kwargs of `map_function`. Parameters other than `tokenizer` and `max_length`
            used by `map_function` can be transferred through `map_function_kwargs`.

    Return:
        A GeneratorDataset object.

    Raises:
        ValueError: Error input for dataset_dir.
        TypeError: Type error for column_names.
    """
    def __init__(self,
                 dataset_dir: str,
                 column_names: list,
                 tokenizer: Union[str, dict, Callable],
                 dataset_name: str = "",
                 file_format: str = None,
                 max_length: int = 1025,
                 read_function: Callable = None,
                 map_function: Callable = None,
                 map_function_kwargs: dict = None):
        dataset_dir = os.path.realpath(dataset_dir)
        self._general_reader_map = {
            "json": self._read_json,
            "jsonl": read_json,
            "csv": read_csv,
            "tsv": self._read_tsv,
            "parquet": self._read_parquet,
        }
        self.tokenizer = self._check_tokenizer(tokenizer)
        self.max_length = max_length
        self.column_names = column_names
        self.map_function_kwargs = self._check_map_function_kwargs(map_function_kwargs)
        file_format = self._check_format(dataset_dir, file_format)
        dataset_name = dataset_name.lower() if dataset_name else "default"
        if read_function:
            self.table = Table.from_pydict(read_function(dataset_dir))
        elif dataset_name in _DATA_READER_MAP:
            self.table = Table.from_pydict(_DATA_READER_MAP[dataset_name](dataset_dir))
        else:
            self.table = self._general_reader_map.get(file_format)(dataset_dir)
        self.map_function = map_function \
            if map_function else _SFT_MAP_FUNCTIONS.get(dataset_name, _SFT_MAP_FUNCTIONS["default"])

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, i):
        example = self.table.take([i]).to_pylist()[0]
        result = self.map_function(example, **self.map_function_kwargs)
        return tuple([result[col] for col in self.column_names])

    def _check_format(self, dataset_dir, file_format):
        """Check and correct the `format`."""
        if not file_format:
            if os.path.isfile(dataset_dir):
                file_format = os.path.splitext(dataset_dir)[1].strip('.')
            elif os.path.isdir(dataset_dir):
                # For a directory, the suffix of the first file is used. Only the parquet format is supported.
                file_format = os.path.splitext(next(os.scandir(dataset_dir)))[1].strip('.')
                if file_format != "parquet":
                    raise ValueError("The dataset directory supports only the parquet format.")
            else:
                raise FileNotFoundError(rf"No such file or directory: {dataset_dir}")

        file_format = file_format.strip(".").lower()
        if file_format in ("json", "jsonl"):
            try:
                with open(dataset_dir, 'r', encoding='UTF-8') as f:
                    json.load(f)
                file_format = "json"
            except JSONDecodeError:
                file_format = "jsonl"

        if file_format in self._general_reader_map:
            return file_format
        raise ValueError("The dataset file format can only be json, jsonl, csv, tsv, and parquet.")

    def _check_map_function_kwargs(self, kwargs):
        """Check kwargs of the `map_function`"""
        if not kwargs:
            kwargs = {}
        if not kwargs.get("tokenizer"):
            kwargs["tokenizer"] = self.tokenizer
        else:
            kwargs["tokenizer"] = self._check_tokenizer(kwargs["tokenizer"])
        if not kwargs.get("max_length"):
            kwargs["max_length"] = self.max_length
        return kwargs

    @staticmethod
    def _check_tokenizer(tokenizer):
        """Check and create the `tokenizer`."""
        if isinstance(tokenizer, str):
            return build_tokenizer({"type": tokenizer})
        if isinstance(tokenizer, dict):
            return build_tokenizer(tokenizer)
        return tokenizer

    @staticmethod
    def _read_parquet(path):
        """Reads data in parquet format."""
        ds = ParquetDataset(path, memory_map=True)
        return ds.read()

    @staticmethod
    def _read_tsv(path):
        """Reads data in TSV format."""
        return read_csv(path, parse_options=ParseOptions(delimiter="\t"))

    @staticmethod
    def _read_json(path):
        """Reads data in JSON format."""
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return Table.from_pydict(data)
        if isinstance(data, list):
            # For pyarrow 12.0.1, pyarrow.lib.Table does not support from_pylist.
            pydict = {k: [i[k] for i in data] for k in data[0]}
            return Table.from_pydict(pydict)
        raise NotImplementedError
