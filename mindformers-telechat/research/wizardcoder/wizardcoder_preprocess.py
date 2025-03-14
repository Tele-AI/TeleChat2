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
"""
transform wizardcoder-format dataset to mindrecord.
"""
import os
import argparse
import json
import copy
import numpy as np

from mindspore.mindrecord import FileWriter

from mindformers.tools import logger
from wizardcoder_tokenizer import WizardCoderTokenizer


IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='np',
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids for tokenized in tokenized_list]

    input_ids_lens = labels_lens = [
        np.not_equal(tokenized.input_ids, tokenizer.pad_token_id).sum() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer, max_length):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    final_input_ids, final_labels = [], []
    for input_id_tensor, label_tensor in zip(input_ids, labels):
        input_id = input_id_tensor.tolist()
        label = label_tensor.tolist()
        if len(input_id) > max_length:
            input_id = input_id[: max_length]
            label = label[: max_length]
        else:
            input_id += [tokenizer.pad_token_id] * (max_length - len(input_id))
            label += [IGNORE_INDEX] * (max_length - len(label))
        final_input_ids.append(np.array(input_id).astype(np.int32))
        final_labels.append(np.array(label).astype(np.int32))

    return dict(input_ids=final_input_ids, labels=final_labels)


def data_tokenize_function(raw_datas, tokenizer, max_length):
    """Preprocess the data by formatting and preprocessing."""
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources, targets = [], []
    for example in raw_datas:
        if 'input' in example:
            instruction, input_query = example['instruction'], example['input']
            source = prompt_input.format_map(dict(instruction=instruction, input=input_query)) if input_query != "" \
                    else prompt_no_input.format_map(dict(instruction=instruction))

        else:
            instruction = example['instruction']
            source = prompt_no_input.format_map(dict(instruction=instruction))
        target = f"{example['output']}{tokenizer.eos_token}"
        sources.append(source)
        targets.append(target)

    data_dict = preprocess(sources, targets, tokenizer, max_length)
    return data_dict


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, max_length):
        super(SupervisedDataset, self).__init__()

        data_dict = data_tokenize_function(raw_data, tokenizer, max_length)

        self.input_ids = data_dict.get("input_ids")
        self.labels = data_dict.get("labels")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )


# pylint: disable=W0703
def tokenize_qa(tokenizer, file_path, max_length, if_jsonl=True):
    """json or jsonl Dataset handling function"""
    file = None
    raw_data = None
    try:
        if not if_jsonl:
            file = open(file_path, "r")
            raw_data = json.load(file)
        else:
            raw_data = []
            for line in open(file_path, 'r'):
                raw_data.append(json.loads(line))
    except FileNotFoundError as file_not_found_error:
        logger.error(file_not_found_error)
    except UnicodeDecodeError as decode_error:
        logger.error(decode_error)
    except IOError as io_error:
        logger.error(io_error)
    except Exception as exception:
        logger.error(exception)
    finally:
        if file is not None:
            file.close()
    dataset_cls = SupervisedDataset(raw_data, tokenizer, max_length)
    for i, _ in enumerate(dataset_cls):
        yield dataset_cls[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindrecord_schema", type=str, default="wizardcoder")
    parser.add_argument("--input_glob", type=str, default="EvolInstruct-Code-80k_1.json")
    parser.add_argument("--output_file", type=str, default="EvolInstruct.mindrecord")
    parser.add_argument("--vocab_file", type=str, default="vocab.json")
    parser.add_argument("--merge_file", type=str, default="merges.txt")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'labels': {"type": "int32", "shape": [-1]}}

    writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
    writer.add_schema(schema, args.mindrecord_schema)

    # Start to load tokenizer
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"file {args.vocab_file} do not exists.")
    if not os.path.exists(args.merge_file):
        raise FileNotFoundError(f"file {args.merge_file} do not exists.")

    transforms_count = 0

    word_tokenizer = WizardCoderTokenizer(vocab_file=args.vocab_file, merge_file=args.merge_file,
                                          model_max_length=args.seq_length + 1)
    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
