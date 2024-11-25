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
"""Data process for SQuAD Dataset"""
import argparse
import json
import collections
import copy
import logging
import pathlib

import numpy as np
from mindspore.mindrecord import FileWriter
from mindformers import AutoTokenizer

IGNORE_TOKEN_ID = -100


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    label = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(label).astype(np.int32)

    writer.write_raw_data([features])
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./squad/train-v1.1.json", required=False,
                        help='Input raw json file. ')
    parser.add_argument("--output_file", type=str, default="./squad/recode_train.mindrecord", required=False,
                        help='Output MindRecord file. ')
    parser.add_argument("--mode", type=str, default="train",
                        help='Set Data for train or eval.')
    parser.add_argument("--max_length", type=int, default=2048, help='Maximum sequence length. ')
    parser.add_argument("--tokenizer_type", type=str, default="llama_7b",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")

    args = parser.parse_args()

    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", args.input_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)

    input_file = pathlib.Path(args.input_file)

    with input_file.open() as f:
        file = json.load(f)

    sources = []
    targets = []
    for data in file["data"]:
        for paragraph in data["paragraphs"]:
            passage = paragraph["context"]
            query = paragraph["qas"][0]["question"]
            answer = paragraph["qas"][0]["answers"][0]["text"]

            input_str = f"Read the passage and answer the question below.\n\n### Instruction:\n{passage}\n\n### Input:\n{query}\n\n### Response:"
            sources.append(input_str)
            targets.append(answer)

    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", args.output_file)

    writer = FileWriter(args.output_file, 1)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}
                   }
    writer.add_schema(data_schema, "lm-schema")

    total_written = 0
    # for eval
    if args.mode == "eval":
        if hasattr(tokenizer, 'add_bos_token'):
            tokenizer.add_bos_token = True
        if hasattr(tokenizer, 'add_eos_token'):
            tokenizer.add_eos_token = False
        for prompt, answer in zip(sources, targets):
            total_written += 1
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(input_ids) >= args.max_length:
                input_ids = input_ids[:args.max_length]
            else:
                input_ids = np.pad(input_ids, (0, args.max_length - len(input_ids)), 'constant',
                                   constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))

            input_ids = np.array(input_ids).reshape(1, -1)

            label_id = tokenizer.encode(answer, add_special_tokens=False)
            label_id = np.pad(label_id, (0, args.max_length - len(label_id)), 'constant',
                              constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
            label_id = np.array(label_id).reshape(1, -1)
            instance = {"input_ids": input_ids, "labels": label_id}

            write_instance_to_file(writer, instance=instance)
    # for train/finetune
    elif args.mode == "train":
        if hasattr(tokenizer, 'add_bos_token'):
            tokenizer.add_bos_token = True
        if hasattr(tokenizer, 'add_eos_token'):
            tokenizer.add_eos_token = True
        for prompt, answer in zip(sources, targets):
            total_written += 1
            concated_qa = prompt + answer
            input_ids = tokenizer.encode(concated_qa, add_special_tokens=True)
            input_ids = np.array(input_ids)

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids = np.array(prompt_ids)
            prompt_length = len(prompt_ids)
            concat_length = len(input_ids)

            pad_length = args.max_length + 1 - concat_length
            input_ids_new = np.pad(input_ids, (0, pad_length), 'constant',
                                   constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
            label_id_new = copy.deepcopy(input_ids_new)
            label_id_new[:prompt_length] = IGNORE_TOKEN_ID
            label_id_new[-pad_length:] = IGNORE_TOKEN_ID
            instance = {"input_ids": input_ids_new, "labels": label_id_new}
            write_instance_to_file(writer, instance=instance)
    else:
        logging.error("No mode named %s, please set mode as train/eval.", args.mode)

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
