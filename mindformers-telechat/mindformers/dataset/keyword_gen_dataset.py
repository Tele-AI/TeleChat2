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
"""Keyword Generation Dataset."""
import copy
import os
from typing import Optional, Union, Callable, List

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset import MindDataset

from mindformers.models.tokenization_utils_base import PreTrainedTokenizerBase
from mindformers.dataset.base_dataset import BaseDataset
from mindformers.dataset.dataloader import build_dataset_loader
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.version_control import get_dataset_map


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class KeyWordGenDataset(BaseDataset):
    """
    Keyword generation dataset.

    The columns of the generated dataset depend on the value of `phase` .

    - When `phase` is ``'train'``, the columns are :py:obj:`[input_ids, labels, position_ids, attention_mask]` .
    - When `phase` is ``'eval'``, the columns are :py:obj:`[input_ids, labels]` .

    The tensor of each column will be cast to int32 type.

    Args:
        dataset_config (dict, optional):
            Config for dataset. When `dataset_config` is an empty dict or is None, all arguments below
            will build a non-empty `dataset_config`. Otherwise, they will be ignored. Default: None.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
            when `data_loader` is a `dict`, this string "type", "dataset_dir", "dataset_files", "phase", "shuffle",
            "origin_columns" and "version" are the keys can be parsed.

            - type: Required. Indicates the type of dataset. The value must be string or class type.
              When the value is "MindDataset", one of `dataset_dir` and `dataset_files` is required,
              where `dataset_dir` takes effect first; otherwise `dataset_dir` is required.

            - dataset_dir: The path or directory of dataset. When `type` is "MindDataset" and `dataset_dir` is
              a directory, search for files in `mindrecord` format recursively in the directory.

            - dataset_files: The path of files in `mindrecord` format.
              Take effect when `type` is "MindDataset", otherwise this key is ignored. Must be `list` or `tuple`.

            - phase: Required. The dataset subset to be loaded. The value can be 'train' and "eval".

            - shuffle: Required. Whether to perform shuffle on the dataset. Must be `bool`.

            - origin_columns: Required. The column names corresponding to py:obj:`[prompt, answer]`
              in the origin dataset files. Must be a list of two strings.

            - version: Optional. Version of the map function. The value can be 1 or 2. Default when missing: 1.

        tokenizer (Union[dict, Callable]):
            Tokenizer configuration or object.
        input_columns (list[str]):
            Column name before the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained
            in the last batch is smaller than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        ignore_pad_token_for_loss (bool):
            Whether ignore pad token for loss. Default: True.
        max_source_length (int):
            Maximum length of the source sequence.
        max_target_length (int):
            Maximum length of the target sequence.
        phase (int):
            Phase of a task, which can be 'train' or 'eval'. Ignored when `data_loader` is `dict`. Default: 'train'.
        version (int):
            Version of the map function, which can be 1 or 2. Ignored when `data_loader` is `dict`. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        Instance of KeyWordGenDataset.

    Raises:
        ValueError: If `dataset_config` doesn't contain "dataset_dir" or "dataset_files" as its key.

    Examples:
        >>> from mindformers.dataset import KeyWordGenDataset, ADGenDataLoader
        >>> from mindformers import AutoTokenizer
        >>> # Note:
        >>> #     `"/path/to/train.json"` should be replaced with the real path of the dataset file.
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md
        >>> data_loader = ADGenDataLoader(dataset_dir="/path/to/train.json", shuffle=True, phase='train',
        ...                               origin_columns=['content', 'summary'])
        >>> tokenizer = AutoTokenizer.from_pretrained('glm_6b')
        >>> dataset_from_param = KeyWordGenDataset(data_loader=data_loader, tokenizer=tokenizer,
        ...                                        input_columns=['input_ids', 'labels',
        ...                                                       'position_ids', 'attention_mask'],
        ...                                        max_source_length=64, max_target_length=64,
        ...                                        ignore_pad_token_for_loss=True, phase='train', batch_size=1)
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                tokenizer: Union[dict, Callable] = None,
                input_columns: List[str] = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                repeat: int = 1,
                ignore_pad_token_for_loss: bool = True,
                max_source_length: int = None,
                max_target_length: int = None,
                phase: str = 'train',
                version: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Keyword Generation Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num

        if isinstance(dataset_config.data_loader, dict):
            if dataset_config.data_loader.type != 'MindDataset':
                dataset = cls._process_raw_text_data(dataset_config)
            else:
                dataset = cls._process_mindrecord_data(dataset_config)
        elif isinstance(dataset_config.data_loader, MindDataset):
            dataset = dataset_config.data_loader
        else:
            dataset = cls._tokenizer_map(dataset_config.data_loader, dataset_config)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = get_dataset_map(dataset, type_cast_op, input_columns=input_arg)
        return dataset

    @classmethod
    def _tokenizer_map(cls, dataset, dataset_config):
        """Maps the tokenizer on the source and the output"""
        if isinstance(dataset_config.data_loader, dict):
            phase = dataset_config.data_loader.phase
            version = dataset_config.data_loader.version if dataset_config.data_loader.version else 1
        else:
            phase = dataset_config.phase
            version = dataset_config.version

        if isinstance(dataset_config.tokenizer, PreTrainedTokenizerBase):
            tokenizer = dataset_config.tokenizer
        else:
            tokenizer = build_tokenizer(dataset_config.tokenizer)

        if version == 2:
            train_dataset_function = cls._train_dataset_functionv2
            train_output_columns = ["input_ids", "labels"]
            eval_dataset_function = cls._eval_dataset_functionv2
        elif version == 3:
            train_dataset_function = cls._train_dataset_functionv3
            train_output_columns = ["input_ids", "labels"]
        else:
            train_dataset_function = cls._train_dataset_function
            train_output_columns = ["input_ids", "labels", "position_ids", "attention_mask"]
            eval_dataset_function = cls._eval_dataset_function
        input_columns = ["prompt", "answer"]
        eval_output_columns = ["input_ids", "labels"]

        # Avoid to_json error when summary monitor is opened
        def train_dataset_func(prompt, answer):
            return train_dataset_function(prompt, answer, dataset_config, tokenizer)

        def eval_dataset_func(prompt, answer):
            return eval_dataset_function(prompt, answer, dataset_config, tokenizer)

        if phase == "train":
            dataset = get_dataset_map(dataset,
                                      train_dataset_func,
                                      input_columns=input_columns,
                                      output_columns=train_output_columns)
            dataset = dataset.project(columns=train_output_columns)
        if phase == "eval":
            dataset = get_dataset_map(dataset,
                                      eval_dataset_func,
                                      input_columns=input_columns,
                                      output_columns=eval_output_columns)
            dataset = dataset.project(columns=eval_output_columns)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir", None)
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id})

        dataset = cls._tokenizer_map(dataset, dataset_config)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        dataset_config = copy.deepcopy(dataset_config)

        dataset_files = []
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".mindrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if data_dir.endswith(".mindrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")

        logger.info("Using args %s to instance the dataset.", dataset_config.data_loader)

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset

    @classmethod
    def _train_dataset_function(cls, prompt, answer, dataset_config, tokenizer):
        """generates train dataset"""
        max_source_length = dataset_config.max_source_length
        max_target_length = dataset_config.max_target_length
        max_seq_length = max_source_length + max_target_length + 1
        ignore_pad_token_for_loss = dataset_config.ignore_pad_token_for_loss

        prompt, answer = prompt.tolist(), answer.tolist()
        prompt_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(prompt_ids) > max_source_length - 1:
            prompt_ids = prompt_ids[: max_source_length - 1]

        if len(answer_ids) > max_target_length - 2:
            answer_ids = answer_ids[: max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, answer_ids)
        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        label = [-100] * context_length + input_ids[mask_position + 2:]  # +1 for logits shift

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        label = label + [tokenizer.pad_token_id] * (pad_len + 1)  # +1 for logits shift
        if ignore_pad_token_for_loss:
            label = [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in label]

        position_ids = cls._create_position_ids(np.array(input_ids))
        attention_mask = cls._get_masks(np.array(input_ids))

        return input_ids, label, position_ids, attention_mask

    @classmethod
    def _train_dataset_functionv2(cls, prompt, answer, dataset_config, tokenizer):
        """generates train dataset"""
        max_source_length = dataset_config.max_source_length
        max_target_length = dataset_config.max_target_length
        max_seq_length = max_source_length + max_target_length + 1
        ignore_pad_token_for_loss = dataset_config.ignore_pad_token_for_loss

        prompt, answer = prompt.tolist(), answer.tolist()
        history = None
        prompt = tokenizer.build_prompt(prompt, history)
        prompt_ids = tokenizer.encode(text=prompt, add_special_tokens=True, max_length=max_source_length)
        answer_ids = tokenizer.encode(text=answer, add_special_tokens=False, max_length=max_target_length)

        context_length = len(prompt_ids)
        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.pad_token_id] * (context_length - 1) + answer_ids + [tokenizer.eos_token_id]

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * (pad_len + 1)  # +1 for logits shift

        if ignore_pad_token_for_loss:
            labels = [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in labels]
        return input_ids, labels

    @classmethod
    def _train_dataset_functionv3(cls, prompt, answer, dataset_config, tokenizer):
        """generates train dataset"""
        max_seq_length = dataset_config.max_source_length + dataset_config.max_target_length + 1
        prompt, answer = prompt.tolist(), answer.tolist()
        prompt_ids = tokenizer.encode(text=prompt,
                                      add_special_tokens=True,
                                      truncation=True,
                                      max_length=dataset_config.max_source_length)
        answer_ids = tokenizer.encode(text=answer,
                                      add_special_tokens=False,
                                      truncation=True,
                                      max_length=dataset_config.max_target_length)

        context_length = len(prompt_ids)
        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.pad_token_id] * (context_length - 1) + answer_ids + [tokenizer.eos_token_id]

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * (pad_len + 1)  # +1 for logits shift

        if dataset_config.ignore_pad_token_for_loss:
            labels = [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in labels]
        return input_ids, labels

    @classmethod
    def _eval_dataset_function(cls, prompt, answer, dataset_config, tokenizer):
        """generates eval dataset"""
        max_source_length = dataset_config.max_source_length
        max_target_length = dataset_config.max_target_length

        prompt, answer = prompt.tolist(), answer.tolist()
        if len(prompt) > max_source_length - 2:
            prompt = prompt[: max_source_length - 2]

        if len(answer) > max_target_length - 2:
            answer = answer[: max_target_length - 2]

        input_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
        label = tokenizer.encode(text=answer, add_special_tokens=True)

        pad_len = max_source_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        pad_len = max_target_length - len(label)
        label = label + [tokenizer.pad_token_id] * pad_len

        return input_ids, label

    @classmethod
    def _eval_dataset_functionv2(cls, prompt, answer, dataset_config, tokenizer):
        """generates eval dataset"""
        max_source_length = dataset_config.max_source_length
        max_target_length = dataset_config.max_target_length

        prompt, answer = prompt.tolist(), answer.tolist()
        history = None
        prompt = tokenizer.build_prompt(prompt, history)

        if len(prompt) > max_source_length - 1:
            prompt = prompt[: max_source_length - 1]

        if len(answer) > max_target_length - 1:
            answer = answer[: max_target_length - 1]

        input_ids = tokenizer.encode(text=prompt, add_special_tokens=True, max_length=max_source_length)
        label = tokenizer.encode(text=answer, add_special_tokens=True, max_length=max_target_length)

        pad_len = max_source_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        pad_len = max_target_length - len(label)
        label = label + [tokenizer.pad_token_id] * pad_len

        return input_ids, label

    @classmethod
    def _get_masks(cls, input_ids, bos_token_id=130004):
        """generate mask from input id"""

        seq_length = input_ids.shape[0]

        mask = bos_token_id * np.ones(shape=(seq_length), dtype=np.int32)
        mask = np.equal(input_ids, mask)
        # 要求input_ids中有且仅有一个bos_token_id
        context_lengths = np.argwhere(mask)[:, -1]

        attention_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
        for context_length in context_lengths:
            attention_mask[:, :context_length] = 1

        attention_mask = np.logical_not(attention_mask.astype(np.bool_))
        attention_mask = attention_mask.astype(np.float32)
        attention_mask = np.expand_dims(attention_mask, 0)
        return attention_mask

    @classmethod
    def _get_position_ids(cls, input_ids, mask_positions, use_gmasks=None,
                          bos_token_id=130004, position_encoding_2d=True):
        """generate position ids from input id and mask positions"""

        seq_length = input_ids.shape[0]
        if use_gmasks is None:
            use_gmasks = [False]
        mask = bos_token_id * np.ones(shape=(seq_length), dtype=np.int32)
        mask = np.equal(input_ids, mask)
        # 要求input_ids中有且仅有一个bos_token_id
        context_lengths = np.argwhere(mask)[:, -1]
        if position_encoding_2d:
            position_ids = np.arange(seq_length, dtype=np.int64)
            for i, context_length in enumerate(context_lengths):
                position_ids[context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, dtype=np.int64),
                np.arange(seq_length - context_length, dtype=np.int64) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0).squeeze()
            position_ids = np.stack((position_ids, block_position_ids), axis=0)
        else:
            position_ids = np.arange(seq_length, dtype=np.int64)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    @classmethod
    def _create_position_ids(cls, input_ids, gmask_token_id=130001):
        """generate position ids from input id"""

        seq_length = input_ids.shape[0]
        seqs = input_ids
        # 要求input_ids中, 每行有且仅有一个gMASK
        use_gmasks = gmask_token_id * np.ones(shape=(seq_length), dtype=np.int32)
        mask = np.equal(seqs, use_gmasks)
        mask_positions = np.argwhere(mask)[:, -1]

        position_ids = cls._get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)
        return position_ids
