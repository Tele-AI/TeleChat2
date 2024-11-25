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
"""Training DataLoader."""
import os
import random
import subprocess
import json
from json import JSONDecodeError
from typing import Union, Callable
from multiprocessing import Pool
import numpy as np
from pyarrow.json import read_json
from pyarrow.csv import read_csv, ParseOptions
from pyarrow.parquet import ParquetDataset
from pyarrow.lib import Table

from mindspore.dataset import GeneratorDataset
from mindspore._checkparam import args_type_check
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.dataset.dataloader.datareaders import _DATA_READER_MAP
from mindformers.tools.logger import logger


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class TrainingDataLoader:
    """Training DataLoader."""
    @args_type_check(dataset_dir=str, column_names=list, tokenizer=(str, dict, Callable), dataset_name=str,
                     is_align=bool, max_length=int, text_col=str, file_format=str, read_function=Callable,
                     shuffle=bool, samples_num=int, skip_num=int, file_limit=int)
    def __new__(cls,
                dataset_dir: str,
                column_names: list,
                tokenizer: Union[str, dict, Callable],
                dataset_name: str = "",
                is_align: bool = True,
                max_length: int = 1025,
                text_col: str = "",
                file_format: str = None,
                read_function: Callable = None,
                shuffle: bool = False,
                samples_num: int = 10000,
                skip_num: int = 0,
                file_limit: int = 1,
                process_num: int = 64,
                **kwargs):
        r"""
        Training DataLoader API.

        Args:
            dataset_dir (str): The directory path to parquet text with hdfs.
            column_names(list): Column names contained in the created dataset.
            tokenizer (Union[str, dict, Callable]): Tokenizer configuration.
            dataset_name (str): Dataset name. Currently, ["wikitext"] is supported.
            is_align (bool): Indicates whether to align input_ids to `max_length`.
            max_length (int): Maximum length of a token.
            text_col (str): Column name of the dataset to be trained.
            file_format (str): Retrieves the end character of the desired file name.
            read_function (Callable): User-defined functions for reading data.
                The input parameter is the path of the dataset file.
                The return value is a dictionary. Key indicates the column name,
                    and value indicates the value of the column.
            shuffle (bool): Whether to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.
            samples_num(int): Specifies the number of samples to be trained.
            skip_num(int): Skip the first N elements of the dataset.
            file_limit(int): Limit on the number of files read at a time.

        Return:
            A GeneratorDataset object.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import TrainingDataLoader
            >>> data_loader = TrainingDataLoader(dataset_dir="The required task dataset path",
            ...                                  column_names=["input_ids", "attention_mask"],
            ...                                  tokenizer={"type": "GPT2Tokenizer", "max_length": 1025},
            ...                                  dataset_name="wikitext",
            ...                                  file_format="tokens",
            ...                                  shuffle=True)
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if max_length <= 0:
            raise TypeError(f"max_length should be an integer greater than 0.")

        logger.info("dataset_dir: %s, samples_num: %s", dataset_dir, samples_num)
        training_dataset = TrainingDataset(dataset_dir, column_names=column_names, tokenizer=tokenizer,
                                           dataset_name=dataset_name, is_align=is_align, max_length=max_length,
                                           text_col=text_col, file_format=file_format, read_function=read_function,
                                           shuffle=shuffle, samples_num=samples_num, file_limit=file_limit,
                                           process_num=process_num)

        kwargs["num_shards"] = None
        kwargs["shard_id"] = None
        gen_dataset = GeneratorDataset(training_dataset, column_names=column_names, shuffle=shuffle, **kwargs)
        logger.info("NOTE: The sample of Dataset will skip %s", skip_num)
        gen_dataset = gen_dataset.skip(skip_num)
        return gen_dataset


def run_cmd(command, pipeline=None):
    """Run the shell command."""
    ret = subprocess.run(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if pipeline is not None:
        ret = subprocess.run(pipeline, input=ret.stdout, shell=False,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if "No such file" in ret.stderr:
        return False, ret.stderr.strip()
    if "Files exists" in ret.stderr:
        return False, ret.stderr.strip()
    if "permission denied" in ret.stderr:
        return False, ret.stderr.strip()
    if ret.returncode == 0:
        return True, ret.stdout.strip()
    return False, ret.stderr.strip()


class TrainingDataset:
    r"""
    Training DataLoader API.

    Args:
        dataset_dir (str): The directory path to parquet text with hdfs.
        column_names(list): Column names contained in the created dataset.
        tokenizer (Union[str, dict, Callable]): Tokenizer configuration.
        dataset_name (str): Dataset name. Currently, ["wikitext"] is supported.
        is_align (bool): Indicates whether to align input_ids to `max_length`.
        max_length (int): Maximum length of a token.
        text_col (str): Column name of the dataset to be trained.
        file_format (str): Retrieves the end character of the desired file name.
        read_function (Callable): User-defined functions for reading data.
            The input parameter is the path of the dataset file.
            The return value is a dictionary. Key indicates the column name,
                and value indicates the value of the column.
        shuffle (bool): Whether or not to perform shuffle on the dataset.
            Random accessible input is required.
            Default: True, expected order behavior shown in the table below.
        samples_num(int): Specifies the number of samples to be trained.
        file_limit(int): Limit on the number of files read at a time.

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
                 is_align: bool = True,
                 max_length: int = 1025,
                 text_col: str = "",
                 file_format: str = None,
                 read_function: Callable = None,
                 shuffle: bool = True,
                 samples_num: int = 10000,
                 file_limit: int = 1,
                 process_num: int = 64):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.format = file_format
        self.column_names = column_names
        self.tokenizer = self._check_tokenizer(tokenizer)
        self.text_col = text_col
        self.read_function = read_function
        self.shuffle = shuffle
        self.sample_number = samples_num
        self.file_limit = file_limit
        self.process_num = process_num if process_num != -1 else os.cpu_count()
        self.current_index = self.file_limit
        self.start_index = 0
        self.files = []
        self.current_files = []
        self.current_samples_number = 0
        self.current_samples = []
        self.iter_index = 0
        self.global_index = 0
        self.is_align = is_align
        self.max_length = max_length
        self.download_path = "./output/hdfs_dataset"
        self._general_reader_map = {
            "json": self._read_json,
            "jsonl": read_json,
            "csv": read_csv,
            "tsv": self._read_tsv,
            "parquet": self._read_parquet,
        }

        logger.info("Attention: The maximum data sample size currently set is: %s. "
                    "If exceeding the maximum number of samples, the training will be exit ", self.sample_number)

        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path, exist_ok=True)
        self._walk_files(self.dataset_dir, file_format=self.format)
        self._update_current_files()

        logger.info("Training Dataset config: shuffle: %s, is_align: %s, max_length: %s",
                    self.shuffle, self.is_align, self.max_length)

    def __iter__(self):
        self.global_index = 0
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= self.current_samples_number and self.iter_index != 0:
            self._reset_iter_index()

        if self.global_index >= self.sample_number:
            logger.info("global index: %s reach to steps: %s", self.global_index, self.sample_number)
            raise StopIteration

        data_item = self.current_samples[self.iter_index]
        self.global_index += 1
        self.iter_index += 1

        result = self.tokenizer.prepare_for_model(ids=data_item.tolist(), pair_ids=None, add_special_tokens=True,
                                                  max_length=self.max_length, padding='max_length',
                                                  truncation=True, truncate_direction="LEFT",
                                                  return_attention_mask=True)
        return tuple([result[col] for col in self.column_names])

    def __len__(self):
        return self.sample_number

    def _check_format(self, dataset_dir, file_format):
        """Check and correct the `format`."""
        if not file_format:
            if os.path.isfile(dataset_dir):
                file_format = os.path.splitext(dataset_dir)[1].strip('.')
            elif os.path.isdir(dataset_dir):
                file_format = os.path.splitext(next(os.scandir(dataset_dir)))[1].strip('.')
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

    def _reset_iter_index(self):
        """Reset index and update files."""
        self.current_samples = []
        self.iter_index = 0
        self._update_current_files()

    def _update_current_files(self):
        """Update the files to be read."""
        self.current_files = self.files[self.start_index:self.current_index]
        if not self.current_files:
            self.start_index = 0
            self.current_index = self.file_limit
            self.current_files = self.files[self.start_index:self.current_index]
            logger.info("All data files have been read, but the number of samples is still not met. "
                        "Now start reading again from the beginning.")
        self._read_files()
        self.start_index += self.file_limit
        self.current_index += self.file_limit
        if self.shuffle:
            random.shuffle(self.current_samples)

    def _read_files(self):
        """Read data from multiple files"""
        for file in self.current_files:
            logger.info("Current read file: %s, Current files: %s", file, self.current_files)
            data_items = self._read_file(file)
            self.current_samples.extend(data_items)
        self.current_samples_number = len(self.current_samples)
        logger.info("Samples length: %s", self.current_samples_number)

    def _tokenizer_func(self, input_data):
        """The functions for tokenizer"""
        input_ids = self.tokenizer.encode(input_data, add_special_tokens=False)
        return input_ids

    def _parallel_map(self, iterable):
        """Multi-process acceleration tokenizer."""
        def pad_func(input_list):
            """Align the length of inpud_ids."""
            token_list = []
            token_item = []
            for token in input_list:
                token_item.extend(token)
                if len(token_item) >= self.max_length:
                    stop_idx = 0
                    for idx in range(0, len(token_item) - self.max_length, self.max_length):
                        token_list.append(np.array(token_item[idx:idx + self.max_length]))
                        stop_idx = idx
                    token_item = token_item[stop_idx + self.max_length:]
            return token_list

        logger.info("Start Tokenizer sample")
        pool = Pool(processes=self.process_num)
        encoded_sentences = pool.map(self._tokenizer_func, iterable)
        pool.close()
        pool.join()
        del pool
        logger.info("Tokenizer sample completed")
        if not self.is_align:
            return encoded_sentences
        logger.info("Start Padding sample")
        aligned_token = pad_func(encoded_sentences)
        logger.info("Padding sample completed")
        return aligned_token

    @staticmethod
    def _get_sentences_list(table, text_col):
        """Obtains a list of sentences based on col. If col is empty, the first column of the table is used."""
        pydict = table.to_pydict()
        if text_col:
            return pydict[text_col]
        return pydict[list(pydict.keys())[0]]

    def _read_dataset(self, local_path):
        """Read data in various formats."""
        if self.read_function:
            table = Table.from_pydict(self.read_function(local_path))
            sentences = self._get_sentences_list(table, self.text_col)
        elif self.dataset_name in _DATA_READER_MAP:
            table = Table.from_pydict(_DATA_READER_MAP[self.dataset_name](local_path))
            sentences = self._get_sentences_list(table, self.text_col)
        elif self.format and self.format in self._general_reader_map:
            table = self._general_reader_map[self.format](local_path)
            table_context = self._get_sentences_list(table, self.text_col)
            sentences = []
            for text in table_context:
                lines = text.strip().split("\n")
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line:
                        sentences.append(stripped_line.rstrip())
            del table_context
        else:
            raise ValueError("This dataset is not supported.")

        del table
        encoded_sentences = self._parallel_map(sentences)
        return encoded_sentences

    def _read_file(self, file_name, retry: int = 5):
        """Read data from a single file."""
        if os.path.exists(file_name):
            logger.info("Load local dataset: %s dataset completed", file_name)
            return self._read_dataset(file_name)

        filename = os.path.split(file_name)[-1]
        local_path = os.path.abspath(f"{self.download_path}/{filename}")
        for th in range(retry):
            logger.info("Download hdfs path %s time: %s, filename: %s, local_path: %s",
                        th, file_name, filename, local_path)
            if os.path.exists(local_path):
                os.remove(local_path)
            status, out_info = run_cmd(["hdfs", "dfs", "-get", file_name, local_path])
            if not status:
                logger.error("%s", out_info)
                logger.info("Download hdfs_file: %s failed, will retry the %s times.", file_name, th)
                continue
            break
        logger.info("Download hdfs dataset: from %s to %s completed.", file_name, local_path)
        hdfs_file_gen = self._read_dataset(local_path)
        logger.info("Load downloaded local dataset: %s completed", local_path)
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info("Local file: %s delete completed", local_path)
        return hdfs_file_gen

    def _walk_files(self, dataset_dir, file_format):
        """Obtaining files in `format`."""
        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            # Obtaining local files
            logger.info("Detect local dataset directory: %s. Traverse local directory", dataset_dir)
            current_list = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)
                            if file.endswith(file_format)]
        elif os.path.exists(dataset_dir) and os.path.isfile(dataset_dir):
            current_list = [dataset_dir]
        else:
            # Obtaining files in HDFS
            logger.info("Can not find local dataset. Traverse HDFS directory: %s", dataset_dir)
            status, current_str = run_cmd(['hdfs', 'dfs', '-ls', f"{dataset_dir}/*{file_format}"],
                                          pipeline=['awk', '{print $NF}'])
            if not status:
                raise ValueError(f"Get dataset file failed, {current_str}")
            if not current_str:
                raise ValueError(f"No dataset file in {dataset_dir}")
            current_list = current_str.strip().split("\n")
            if not current_list[0]:
                raise ValueError(f"No dataset file in {dataset_dir}")

        current_list = sorted(current_list)
        for file in current_list:
            if file not in self.files:
                self.files.append(file)
        logger.info("Current get all files: %s", self.files)

    def _get_all_samples_number(self):
        """Get the number of all samples."""
        num_samples = 0
        files_num = len(self.files)
        for i, file in enumerate(self.files):
            logger.info("%s / %s hdfs_file: %s", i, files_num, file)
            hdfs_file_gen = self._read_file(file)
            num_samples += len(hdfs_file_gen)
        return num_samples
