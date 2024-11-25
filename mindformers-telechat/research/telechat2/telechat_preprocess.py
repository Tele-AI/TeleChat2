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
"""generate mindrecord script"""
import os
import argparse
import random
from random import shuffle
import collections
import jsonlines
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from telechat_tokenizer import TelechatTokenizer
from mindformers.tools import logger

IGNORE_TOKEN_ID = -100


def sft_preprocess(datas, vocab_file, max_length):
    tokens = []
    sentence_ids = []
    labels = []
    tokenizer = TelechatTokenizer(vocab_file, fast_tokenizer=True, trust_remote_code=True)
    system_id = tokenizer("<_system>")["input_ids"]
    user_id = tokenizer("<_user>")["input_ids"]
    bot_id = tokenizer("<_bot>")["input_ids"]
    end_id = tokenizer("<_end>")["input_ids"]
    pad_id = tokenizer("<_pad>")["input_ids"]
    for dialog in datas:
        sentence_id = []
        label = []
        # add system prompt
        sys_token = tokenizer(dialog["system"])["input_ids"]
        sentence_id += system_id + sys_token
        label += [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(sys_token)

        # add dialog
        for sentence in dialog["dialog"]:
            role = sentence["role"]
            content = sentence["content"]
            if role == "user":
                user_token = tokenizer(content)["input_ids"]
                sentence_id += user_id + user_token
                label += [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(user_token)
            elif role == "bot":
                bot_token = tokenizer(content)["input_ids"]
                sentence_id += bot_id + bot_token + end_id + [126136, 29]  # [1266136,29] -> '\n'
                label += [IGNORE_TOKEN_ID] + bot_token + end_id + [IGNORE_TOKEN_ID, IGNORE_TOKEN_ID]
        # packing datas
        if len(sentence_id) + len(sentence_ids) <= max_length:
            sentence_ids += sentence_id
            labels += label
            continue
        elif len(sentence_id) > max_length:
            continue
        else:
            sentence_ids = sentence_ids + (max_length - len(sentence_ids)) * pad_id
            labels = labels + (max_length - len(labels)) * [IGNORE_TOKEN_ID]
            tokens.append({"input_ids": sentence_ids, "labels": labels})
            sentence_ids = sentence_id
            labels = label
    return tokens


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]
    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    writer.write_raw_data([features])
    return features


def make_dataset():
    """make dataset."""
    random.seed(args.seed)
    # read jsonl and shuffle
    raw_dataset = jsonlines.open(args.input_dataset_file, "r")
    datas = [data for data in raw_dataset]
    shuffle(datas)
    raw_dataset.close()
    # preprocess
    train_tokens = sft_preprocess(datas, args.vocab_file_path, args.max_length)
    # write mindrecord
    logger.info("***** Writing to output files *****")
    writer = FileWriter(args.output_dataset_file, 1)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "lm-schema")
    for train_token in tqdm(train_tokens):
        write_instance_to_file(writer, instance=train_token)
    writer.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--vocab_file_path", default="", type=str, help='which model to use.')
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    random.seed(args.seed)
    args.max_length = args.max_length + 1
    if args.output_path:
        if not args.output_path.endswith(".mindrecord"):
            os.makedirs(args.output_path, exist_ok=True)
            args.output_dataset_file = os.path.join(args.output_path, "dataset.mindrecord")
        else:
            args.output_dataset_file = args.output_path
    else:
        raise ValueError("output_path is needed.")
    make_dataset()
