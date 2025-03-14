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
"""DPO process"""

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Generator, List, Union

import numpy as np
from tqdm import tqdm

from mindspore.mindrecord import FileWriter
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools import logger
from mindformers.tools.register import MindFormerConfig

ROLE_MAPPING = {"human": "<|user|>", "gpt": "<|assistant|>", "system": "<|system|>"}
SCHEMA = {
    "chosen_input_ids": {"type": "int32", "shape": [-1]},
    "chosen_labels": {"type": "int32", "shape": [-1]},
    "chosen_lens": {"type": "int32", "shape": [-1]},
    "chosen_position_id": {"type": "int32", "shape": [-1]},
    "chosen_loss_mask": {"type": "int32", "shape": [-1]},
    "chosen_attention_mask": {"type": "int32", "shape": [-1]},
    "chosen_index_packed": {"type": "int32", "shape": [-1]},
    "rejected_input_ids": {"type": "int32", "shape": [-1]},
    "rejected_labels": {"type": "int32", "shape": [-1]},
    "rejected_lens": {"type": "int32", "shape": [-1]},
    "rejected_position_id": {"type": "int32", "shape": [-1]},
    "rejected_loss_mask": {"type": "int32", "shape": [-1]},
    "rejected_attention_mask": {"type": "int32", "shape": [-1]},
    "rejected_index_packed": {"type": "int32", "shape": [-1]},
}


class PreprocessConfig:
    """Configuration for DPO preprocessing."""

    def __init__(
            self,
            data_path: str,
            dst_file: str,
            tokenizer,
            input_sliced_sig: bool = False,
            seq_len: int = 1024,
            dataset_type: str = "dpo",
            file_partition: int = 1,
            num_processes: int = None,
            mp: int = 1,
            pack_num: int = None,
        ):
        self.data_path = data_path
        self.dst_file = dst_file
        self.tokenizer = tokenizer
        self.input_sliced_sig = input_sliced_sig
        self.seq_len = seq_len
        self.dataset_type = dataset_type
        self.file_partition = file_partition
        self.num_processes = num_processes
        self.mp = mp
        self.pack_num = pack_num


def build_message(tokenizer, messages, metadata=""):
    """Build message"""
    encoded_messages = []
    for i, msg in enumerate(messages):
        role = ROLE_MAPPING.get(msg["from"], "")
        if not role:
            raise ValueError(f"Unsupported role {msg['from']}")
        message = f"{role}{metadata}\n{msg['value']}"
        tokens = tokenizer.encode(message)
        if i != 0:
            tokens = tokens[2:]  # remove prefix
        encoded_messages.append(tokens)
    prompt_ids = []
    for encoded_ids in encoded_messages[:-1]:
        prompt_ids += encoded_ids
    answer_ids = encoded_messages[-1]
    return prompt_ids, answer_ids


def build_message_cvalues(tokenizer, prompt, ans):
    """Build message cvalues"""
    msg = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
    msg += "<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(msg)
    msg = f"{ans}<|im_end|>"
    answer_ids = tokenizer.encode(msg)
    return prompt_ids, answer_ids


def _process_sample(pair, tokenizer, input_sliced_sig, seq_len, dataset_type):
    """处理单个样本，生成 chosen 和 rejected 的 input_ids, labels 等"""
    if dataset_type == "dpo":
        chosen_messages = pair["conversations"] + [pair["chosen"]]
        rejected_messages = pair["conversations"] + [pair["rejected"]]
        prompt_ids, chosen_ids = build_message(tokenizer, chosen_messages)
        _, rejected_ids = build_message(tokenizer, rejected_messages)
    elif dataset_type == "cvalues":
        prompt_ids, chosen_ids = build_message_cvalues(
            tokenizer, pair["prompt"], pair["pos_resp"]
        )
        _, rejected_ids = build_message_cvalues(
            tokenizer, pair["prompt"], pair["neg_resp"]
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def _build_sample(prompt_ids, resp_ids):
        input_ids = prompt_ids + resp_ids
        attention_mask = [1] * len(input_ids)
        if input_sliced_sig:
            labels = input_ids[1:] + [tokenizer.pad_token_id]
            loss_mask = [0] * len(prompt_ids) + [1] * (len(resp_ids) - 1) + [0]
        else:
            labels = input_ids[:]
            loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

        input_len = len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * (seq_len - input_len)
        labels = labels + [tokenizer.pad_token_id] * (seq_len - input_len)
        attention_mask = attention_mask + [0] * (seq_len - input_len)
        loss_mask = loss_mask + [0] * (seq_len - input_len)

        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            attention_mask = attention_mask[:seq_len]
            loss_mask = loss_mask[:seq_len]

        input_ids = np.array(input_ids, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        attention_mask = np.array(attention_mask, dtype=np.int32)
        loss_mask = np.array(loss_mask, dtype=np.int32)
        return input_ids, labels, attention_mask, loss_mask

    chosen_input_ids, chosen_labels, chosen_attention_mask, chosen_loss_mask = (
        _build_sample(prompt_ids, chosen_ids)
    )
    rejected_input_ids, rejected_labels, rejected_attention_mask, rejected_loss_mask = (
        _build_sample(prompt_ids, rejected_ids)
    )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_loss_mask": chosen_loss_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_loss_mask": rejected_loss_mask,
    }


def preprocess(config: PreprocessConfig):
    """Preprocess data using multiprocessing and write to a MindRecord file."""
    # Default to all CPU cores if num_processes is not specified
    if config.num_processes is None:
        config.num_processes = cpu_count()

    # Load data
    if config.dataset_type == "dpo":
        with open(config.data_path, "r", encoding="utf-8") as file:
            pairs = json.load(file)
    elif config.dataset_type == "cvalues":
        pairs = []
        with open(config.data_path, "r", encoding="utf-8") as file:
            for line in file:
                pairs.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset_type}")

    # Define the processing function with fixed arguments
    process_func = partial(
        _process_sample,
        tokenizer=config.tokenizer,
        input_sliced_sig=config.input_sliced_sig,
        seq_len=config.seq_len,
        dataset_type=config.dataset_type,
    )

    # Process data with multiprocessing
    with Pool(processes=config.num_processes) as pool:
        samples = list(
            tqdm(
                pool.map(process_func, pairs),
                total=len(pairs),
                desc="Processing samples",
            )
        )

    # Ensure output directory exists
    dst_file_path = os.path.dirname(config.dst_file)
    if not os.path.exists(dst_file_path):
        os.makedirs(dst_file_path)

    # Define MindRecord schema
    schema = {
        "chosen_input_ids": {"type": "int32", "shape": [-1]},
        "chosen_labels": {"type": "int32", "shape": [-1]},
        "chosen_attention_mask": {"type": "int32", "shape": [-1]},
        "chosen_loss_mask": {"type": "int32", "shape": [-1]},
        "rejected_input_ids": {"type": "int32", "shape": [-1]},
        "rejected_labels": {"type": "int32", "shape": [-1]},
        "rejected_attention_mask": {"type": "int32", "shape": [-1]},
        "rejected_loss_mask": {"type": "int32", "shape": [-1]},
    }

    # Write to MindRecord file
    writer = FileWriter(file_name=config.dst_file, shard_num=config.file_partition)
    writer.add_schema(schema)
    writer.write_raw_data(samples)
    writer.commit()

    logger.info(f"Data preprocessing completed and saved to {config.dst_file}")


def _build(prompt_ids, resp_ids, input_sliced_sig=False, pad_token_id=None):
    """Build input, labels, attention mask, and loss mask arrays."""
    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    if input_sliced_sig:
        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided for input_sliced_sig=True")
        labels = input_ids[1:] + [pad_token_id]
        loss_mask = [0] * len(prompt_ids) + [1] * (len(resp_ids) - 1) + [0]
    else:
        labels = input_ids[:]
        loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

    input_ids = np.array(input_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    attention_mask = np.array(attention_mask, dtype=np.int32)
    loss_mask = np.array(loss_mask, dtype=np.int32)
    return input_ids, labels, attention_mask, loss_mask, len(resp_ids)


def pad_sequence_to_length(sequence, target_length, pad_value):
    """Pad sequence to target length with specified pad value."""
    current_length = len(sequence)
    if current_length < target_length:
        return np.pad(
            sequence,
            (0, target_length - current_length),
            mode="constant",
            constant_values=pad_value,
        )
    return sequence[:target_length]


def tokenize_sample(args):
    """Tokenize a batch of samples (prompt, pos_resp, neg_resp) with chunk index."""
    chunk_idx, sample_chunk, tokenizer = args
    prompt_prefix = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    )
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

    prompts = [
        f"{prompt_prefix}{sample['prompt']}{prompt_suffix}" for sample in sample_chunk
    ]
    pos_resps = [f"{sample['pos_resp']}<|im_end|>" for sample in sample_chunk]
    neg_resps = [f"{sample['neg_resp']}<|im_end|>" for sample in sample_chunk]

    prompt_ids = tokenizer(prompts)["input_ids"]
    pos_resp_ids = tokenizer(pos_resps)["input_ids"]
    neg_resp_ids = tokenizer(neg_resps)["input_ids"]

    return (chunk_idx, prompt_ids, pos_resp_ids, neg_resp_ids)


def _initialize_transformer_lists():
    """Initialize empty lists for transformer function results."""
    return {
        "chosen_input_ids_lst": [],
        "chosen_labels_lst": [],
        "rejected_input_ids_lst": [],
        "rejected_labels_ids_lst": [],
        "chosen_attention_masks_lst": [],
        "rejected_attention_masks_lst": [],
        "chosen_loss_masks_lst": [],
        "rejected_loss_masks_lst": [],
        "sample_lens": [],
        "chosen_lens": [],
        "rejected_lens": [],
    }


def _prepare_tokenization_tasks(sources, num_cpus, tokenizer):
    """Prepare tasks for parallel tokenization."""

    def split_list(lst, n):
        avg = len(lst) // n
        remainder = len(lst) % n
        result = []
        start = 0
        for i in range(n):
            end = start + avg + (1 if i < remainder else 0)
            result.append(lst[start:end])
            start = end
        return result

    source_chunks = split_list(sources, num_cpus)
    return [
        (i, chunk, tokenizer) for i, chunk in enumerate(source_chunks) if len(chunk) > 0
    ]


def _process_tokenization_results(results):
    """Process and consolidate tokenization results."""
    # Sort results by chunk index to ensure correct order
    results.sort(key=lambda x: x[0])

    prompt_ids_list = [item for chunk in results for item in chunk[1]]
    pos_resp_ids_list = [item for chunk in results for item in chunk[2]]
    neg_resp_ids_list = [item for chunk in results for item in chunk[3]]

    return prompt_ids_list, pos_resp_ids_list, neg_resp_ids_list


def _process_transformer_samples(
        sources,
        token_ids_data,
        config,
        result_lists,
    ):
    """Process individual samples for transformer function."""
    # 从token_ids_data中解包数据
    prompt_ids_list = token_ids_data["prompt_ids"]
    pos_resp_ids_list = token_ids_data["pos_resp_ids"]
    neg_resp_ids_list = token_ids_data["neg_resp_ids"]

    # 从config中获取配置
    pad_token_id = config["pad_token_id"]
    input_sliced_sig = config["input_sliced_sig"]

    for i in range(len(sources)):
        prompt_ids = prompt_ids_list[i]
        chosen_ids = pos_resp_ids_list[i]
        rejected_ids = neg_resp_ids_list[i]

        # 构建序列
        (
            chosen_input_ids,
            chosen_labels,
            chosen_attention_mask,
            chosen_loss_mask,
            chosen_length,
        ) = _build(prompt_ids, chosen_ids, input_sliced_sig, pad_token_id)
        (
            rejected_input_ids,
            rejected_labels,
            rejected_attention_mask,
            rejected_loss_mask,
            rejected_length,
        ) = _build(prompt_ids, rejected_ids, input_sliced_sig, pad_token_id)

        # Pad to max length
        max_length = max(len(chosen_input_ids), len(rejected_input_ids))
        result_lists["chosen_lens"].append(chosen_length)
        result_lists["rejected_lens"].append(rejected_length)

        chosen_input_ids = pad_sequence_to_length(
            chosen_input_ids, max_length, pad_token_id
        )
        rejected_input_ids = pad_sequence_to_length(
            rejected_input_ids, max_length, pad_token_id
        )
        chosen_labels = pad_sequence_to_length(chosen_labels, max_length, pad_token_id)
        rejected_labels = pad_sequence_to_length(
            rejected_labels, max_length, pad_token_id
        )
        chosen_attention_mask = pad_sequence_to_length(
            chosen_attention_mask, max_length, 0
        )
        rejected_attention_mask = pad_sequence_to_length(
            rejected_attention_mask, max_length, 0
        )
        chosen_loss_mask = pad_sequence_to_length(chosen_loss_mask, max_length, 0)
        rejected_loss_mask = pad_sequence_to_length(rejected_loss_mask, max_length, 0)

        # Collect results
        result_lists["chosen_input_ids_lst"].append(chosen_input_ids)
        result_lists["chosen_labels_lst"].append(chosen_labels)
        result_lists["rejected_input_ids_lst"].append(rejected_input_ids)
        result_lists["rejected_labels_ids_lst"].append(rejected_labels)
        result_lists["chosen_attention_masks_lst"].append(chosen_attention_mask)
        result_lists["rejected_attention_masks_lst"].append(rejected_attention_mask)
        result_lists["chosen_loss_masks_lst"].append(chosen_loss_mask)
        result_lists["rejected_loss_masks_lst"].append(rejected_loss_mask)
        result_lists["sample_lens"].append(max_length)


def transformer(sources, tokenizer, input_sliced_sig=False):
    """Transform alpaca dataset to tokenized MindRecord format using batch encoding and multiprocessing."""
    # Initialize result lists
    result_lists = _initialize_transformer_lists()
    pad_token_id = tokenizer.pad_token_id

    # Set up multiprocessing
    num_cpus = cpu_count()
    logger.info(f"Using {num_cpus} CPU cores for tokenization.")
    tasks = _prepare_tokenization_tasks(sources, num_cpus, tokenizer)

    # Run parallel tokenization with progress tracking
    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(tokenize_sample, tasks),
                total=len(tasks),
                desc="Tokenizing samples",
            )
        )

    # Process results
    prompt_ids_list, pos_resp_ids_list, neg_resp_ids_list = (
        _process_tokenization_results(results)
    )

    # Process individual samples
    _process_transformer_samples(
        sources,
        {
            "prompt_ids": prompt_ids_list,
            "pos_resp_ids": pos_resp_ids_list,
            "neg_resp_ids": neg_resp_ids_list,
        },
        {"pad_token_id": pad_token_id, "input_sliced_sig": input_sliced_sig},
        result_lists,
    )

    return result_lists


def read_json_or_jsonl(
        file_path: Union[str, Path],
    ) -> Union[List[Dict], Generator[Dict, None, None]]:
    """Read JSON or JSON Lines file and parse into a list of dictionaries or a generator."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    if file_path.suffix not in (".json", ".jsonl"):
        raise ValueError("File must be in .json or .jsonl format")

    try:
        if file_path.suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list) or not all(
                        isinstance(item, dict) for item in data
                    ):
                    raise ValueError("JSON file content must be a list of dictionaries")
                return data
        else:  # .jsonl
            ret = []
            with file_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ret.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Line {line_num} parsing failed: {e}\nContent: {line}"
                        )
                        continue
            return ret

    except Exception as e:
        raise ValueError("Error reading file") from e


def group_and_pack(token_lengths, pack_num):
    """Group and pack token lengths into batches."""
    pack_group, each_group = [], []

    for token_length in token_lengths:
        each_group.append(token_lengths[token_length])
        if len(each_group) >= pack_num:
            pack_group.append(each_group)
            each_group = [token_lengths[token_length]]
    if each_group:
        pack_group.append(each_group)
    return pack_group


def _analyze_data_lengths(metas, config):
    """Analyze data lengths and calculate packing parameters."""
    len_data = len(metas["chosen_input_ids_lst"])

    logger.info("Constructing token lengths...")
    lengths_dict = {idx: (idx, metas["sample_lens"][idx]) for idx in range(len_data)}
    mean_length = sum(metas["sample_lens"]) / len_data
    logger.info(f"Mean length: {mean_length}")

    pack_num = config.pack_num
    if pack_num is None:
        pack_num = max(int(config.seq_len / mean_length), 1)
    logger.info(f"Pack number: {pack_num}")

    one_data_max_length = int(config.seq_len / pack_num)
    logger.info(f"One data max length: {one_data_max_length}")
    logger.info(f"Actual sample length: {one_data_max_length * pack_num}")

    if one_data_max_length % config.mp != 0:
        one_data_max_length = (one_data_max_length // config.mp) * config.mp
        logger.warning(
            f"one_data_max_length adjusted to: {one_data_max_length}, since it should be divisible by mp: {config.mp}"
        )

    return lengths_dict, pack_num, one_data_max_length


def _process_data_group(group, metas, config, one_data_max_length, pack_num):
    """Process a single data group and prepare arrays for packing."""
    tokenizer = config.tokenizer

    # Initialize arrays for this group
    chosen_input_ids_lst, chosen_labels_lst = [], []
    chosen_loss_mask, chosen_attention_masks_lst = [], []
    rejected_input_ids_lst, rejected_labels_ids_lst = [], []
    rejected_loss_mask, rejected_attention_masks_lst = [], []
    chosen_lens, rejected_lens = [], []
    chosen_position_id, rejected_position_id = [], []
    chosen_index_packed, rejected_index_packed = [], []

    # Process each item in the group
    for idx, g in enumerate(group):
        index, length = g
        if metas["sample_lens"][index] != length:
            logger.warning(f"Length mismatch for index {index}")
            continue

        # Add chosen data
        chosen_input_ids_lst.append(
            pad_sequence_to_length(metas["chosen_input_ids_lst"][index], one_data_max_length, tokenizer.pad_token_id))
        chosen_labels_lst.append(
            pad_sequence_to_length(metas["chosen_labels_lst"][index], one_data_max_length, tokenizer.pad_token_id))
        chosen_loss_mask.append(pad_sequence_to_length(metas["chosen_loss_masks_lst"][index], one_data_max_length, 0))
        chosen_attention_masks_lst.append(
            pad_sequence_to_length(metas["chosen_attention_masks_lst"][index] * (idx + 1), one_data_max_length, 0))
        chosen_index_packed.append([idx] * one_data_max_length)
        chosen_lens.append(metas["chosen_lens"][index])
        chosen_position_id.append(
            pad_sequence_to_length(np.arange(metas["chosen_lens"][index]), one_data_max_length, -1))

      # Add rejected data
        rejected_input_ids_lst.append(
            pad_sequence_to_length(metas["rejected_input_ids_lst"][index], one_data_max_length, tokenizer.pad_token_id))
        rejected_labels_ids_lst.append(
            pad_sequence_to_length(
                metas["rejected_labels_ids_lst"][index], one_data_max_length, tokenizer.pad_token_id))
        rejected_loss_mask.append(
            pad_sequence_to_length(metas["rejected_loss_masks_lst"][index], one_data_max_length, 0))
        rejected_attention_masks_lst.append(
            pad_sequence_to_length(metas["rejected_attention_masks_lst"][index] * (idx + 1), one_data_max_length, 0))
        rejected_index_packed.append([idx] * one_data_max_length)
        rejected_lens.append(metas["rejected_lens"][index])
        rejected_position_id.append(
            pad_sequence_to_length(np.arange(metas["rejected_lens"][index]), one_data_max_length, -1))

    # Concatenate all arrays in this group
    result = {
        "chosen_input_ids": np.concatenate(chosen_input_ids_lst),
        "chosen_labels": np.concatenate(chosen_labels_lst),
        "chosen_loss_mask": np.concatenate(chosen_loss_mask),
        "chosen_attention_mask": np.concatenate(chosen_attention_masks_lst),
        "chosen_index_packed": np.concatenate(chosen_index_packed),
        "chosen_position_id": np.concatenate(chosen_position_id),
        "rejected_input_ids": np.concatenate(rejected_input_ids_lst),
        "rejected_labels": np.concatenate(rejected_labels_ids_lst),
        "rejected_loss_mask": np.concatenate(rejected_loss_mask),
        "rejected_attention_mask": np.concatenate(rejected_attention_masks_lst),
        "rejected_index_packed": np.concatenate(rejected_index_packed),
        "rejected_position_id": np.concatenate(rejected_position_id),
    }

    # Pad lens if needed
    if len(chosen_lens) < pack_num:
        pad_length = pack_num - len(chosen_lens)
        result["chosen_lens"] = np.pad(
            chosen_lens, (0, pad_length), mode="constant", constant_values=0
        )
        result["rejected_lens"] = np.pad(
            rejected_lens, (0, pad_length), mode="constant", constant_values=0
        )
    else:
        result["chosen_lens"] = np.array(chosen_lens)
        result["rejected_lens"] = np.array(rejected_lens)

    return result


def _write_to_mindrecord(packed_data, config):
    """Write packed data to MindRecord file."""
    tokenizer = config.tokenizer
    seq_length = config.seq_len
    pack_num = config.pack_num

    logger.info("Writing data to MindRecord file...")
    out_dir, _ = os.path.split(os.path.abspath(config.dst_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    writer = FileWriter(file_name=config.dst_file, shard_num=config.file_partition)
    writer.set_page_size(256 * 1024 * 1024)
    writer.add_schema(SCHEMA)

    write_data = []
    for item in packed_data:
        x = {
            "chosen_input_ids": pad_sequence_to_length(
                item["chosen_input_ids"], seq_length, tokenizer.pad_token_id
            ),
            "chosen_labels": pad_sequence_to_length(
                item["chosen_labels"], seq_length, tokenizer.pad_token_id
            ),
            "chosen_loss_mask": pad_sequence_to_length(
                item["chosen_loss_mask"], seq_length, 0
            ),
            "chosen_attention_mask": pad_sequence_to_length(
                item["chosen_attention_mask"], seq_length, 0
            ),
            "chosen_lens": np.array(item["chosen_lens"], dtype=np.int32),
            "chosen_position_id": pad_sequence_to_length(
                item["chosen_position_id"], seq_length, -1
            ).astype(np.int32),
            "chosen_index_packed": pad_sequence_to_length(
                item["chosen_index_packed"], seq_length, pack_num - 1
            ).astype(np.int32),
            "rejected_input_ids": pad_sequence_to_length(
                item["rejected_input_ids"], seq_length, tokenizer.pad_token_id
            ),
            "rejected_labels": pad_sequence_to_length(
                item["rejected_labels"], seq_length, tokenizer.pad_token_id
            ),
            "rejected_loss_mask": pad_sequence_to_length(
                item["rejected_loss_mask"], seq_length, 0
            ),
            "rejected_attention_mask": pad_sequence_to_length(
                item["rejected_attention_mask"], seq_length, 0
            ),
            "rejected_lens": np.array(item["rejected_lens"], dtype=np.int32),
            "rejected_position_id": pad_sequence_to_length(
                item["rejected_position_id"], seq_length, -1
            ).astype(np.int32),
            "rejected_index_packed": pad_sequence_to_length(
                item["rejected_index_packed"], seq_length, pack_num - 1
            ).astype(np.int32),
        }
        write_data.append(x)

    writer.write_raw_data(write_data, parallel_writer=True)
    writer.commit()


def pack_data(metas, config: PreprocessConfig):
    """Pack data into MindRecord file."""
    # Analyze data and determine packing parameters
    lengths_dict, pack_num, one_data_max_length = _analyze_data_lengths(metas, config)

    # Update pack_num in config for later use
    config.pack_num = pack_num

    # Group and pack the data
    pack_group = group_and_pack(lengths_dict, pack_num)

    # Process each group
    packed_data = []
    for group in tqdm(pack_group, desc="Packing data groups"):
        group_result = _process_data_group(
            group, metas, config, one_data_max_length, pack_num
        )
        packed_data.append(group_result)

    # Write to MindRecord file
    _write_to_mindrecord(packed_data, config)

    logger.info("Finished")


def _parse_args():
    """Parse command line arguments for DPO preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Path to source JSON or JSONL file.")
    parser.add_argument(
        "--dst", type=str, help="Path to save the output MindRecord file."
    )
    parser.add_argument("--config", type=str, help="Path to model configuration file.")
    parser.add_argument(
        "--seq_len",
        default=1024,
        type=int,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="dpo",
        help="Dataset type: 'dpo' or 'cvalues'.",
    )
    parser.add_argument(
        "--file_partition",
        type=int,
        default=1,
        help="Number of shards for the MindRecord file.",
    )
    parser.add_argument(
        "--register_path",
        default=None,
        type=str,
        help="Path to register external modules or APIs.",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Whether to pack multiple samples into one sequence.",
    )
    parser.add_argument(
        "--mp", type=int, default=1, help="Model parallel size for sequence alignment."
    )
    parser.add_argument(
        "--pack_num",
        type=int,
        default=None,
        help="Number of samples to pack in each sequence.",
    )

    parser.add_argument(
        "--input_sliced_sig",
        action="store_true",
        help="Whether to use sliced inputs for label generation.",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="./tokenizer.model",
        help="Path to tokenizer vocabulary file.",
    )
    parser.add_argument(
        "--pad_token",
        type=str,
        default="<|reserved_special_token_0|>",
        help="Token used for padding sequences.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="Llama3Tokenizer",
        help="Type of tokenizer to use for text processing.",
    )
    parser.add_argument(
        "--processor_type",
        type=str,
        default="LlamaProcessor",
        help="Type of processor for tokenization pipeline.",
    )
    parser.add_argument(
        "--auto_register",
        type=str,
        default="llama3_1.Llama3Tokenizer",
        help="Component to auto-register with MindFormer.",
    )

    return parser.parse_args()


def _setup_environment(args):
    """Setup environment and configuration for processing."""

    work_path = os.path.dirname(os.path.abspath(__file__))

    # Setup config path
    if args.config is not None and not os.path.isabs(args.config):
        args.config = os.path.join(work_path, args.config)

    # Setup register path and environment
    if args.register_path is not None:
        if not os.path.isabs(args.register_path):
            args.register_path = os.path.join(work_path, args.register_path)
        # Setting Environment Variables
        os.environ["REGISTER_PATH"] = args.register_path
        if args.register_path not in sys.path:
            sys.path.append(args.register_path)

    # Create tokenizer config
    dict_config = {
        "return_tensors": "ms",
        "tokenizer": {
            "model_max_length": args.seq_len,
            "vocab_file": args.vocab_file,
            "pad_token": args.pad_token,
            "type": args.tokenizer_type,
        },
        "type": args.processor_type,
        "auto_register": args.auto_register,
    }
    config = MindFormerConfig(processor=dict_config)
    tokenizer = build_tokenizer(config.processor.tokenizer)
    return tokenizer


def main():
    """Main function to preprocess data for DPO training."""
    # Parse arguments and setup environment
    args = _parse_args()
    tokenizer = _setup_environment(args)

    # Create configuration object
    config = PreprocessConfig(
        data_path=args.src,
        dst_file=args.dst,
        tokenizer=tokenizer,
        input_sliced_sig=args.input_sliced_sig,
        seq_len=args.seq_len,
        dataset_type=args.dataset_type,
        file_partition=args.file_partition,
        mp=args.mp,
        pack_num=args.pack_num,
    )

    # Choose processing path based on args
    if args.pack:
        logger.info("Reading original JSON...")
        metas = read_json_or_jsonl(args.src)

        logger.info("Tokenizing JSON...")
        metas = transformer(metas, tokenizer, config.input_sliced_sig)
        pack_data(metas, config)
    else:
        preprocess(config)


if __name__ == "__main__":
    main()
