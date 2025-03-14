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
import collections
import random
from random import shuffle
import datasets
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from telechat_tokenizer import TelechatTokenizer
from mindformers.tools import logger

IGNORE_TOKEN_ID = -100


class TelechatDataset:
    """TelechatDataset"""
    def __init__(self, output_path, seed, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.raw_datasets = datasets.load_dataset(path="json", data_files=dataset_name)

    def get_train_data(self):
        """get train data"""
        if isinstance(self.raw_datasets, dict):
            dataset = self.raw_datasets["train"]
        elif isinstance(self.raw_datasets, list):
            dataset = self.raw_datasets
        else:
            raise ValueError("dataset type error")
        return dataset

    def get_prompt(self, sample):
        return args.user_token + sample['input'] + args.bot_token

    def get_prompt_and_answer(self, sample):
        return args.user_token + sample['input'] + args.bot_token + sample['output'] + args.end_token


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    writer.write_raw_data([features])
    return features


def tokenizer_with_special_tokens(text, tokenizer):
    """special tokens"""
    if text.count(args.user_token) != text.count(args.bot_token) and \
        text.count(args.bot_token) != text.count(args.end_token):
        logger.error(f"{text.count(args.user_token)} should be equal to {text.count(args.bot_token)}"
                     f" and {text.count(args.bot_token)} should be equal to {text.count(args.end_token)}")
    start_token_id = tokenizer.convert_tokens_to_ids(args.start_token) if args.start_token else None
    user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(args.pad_token)

    def truncate_leading_pad(s, pad_token=args.pad_token):
        count = 0
        while s.startswith(pad_token, count):
            count += len(pad_token)
            if count >= len(s):
                break
        result_string = s[count:]
        return result_string, count // len(pad_token)
    text, pad_len = truncate_leading_pad(text, args.pad_token)
    res = {}
    input_ids = []
    label_ids = []
    if args.start_token and args.start_token in text:
        conversations = text.split(args.start_token)[1:]
        add_start_token = True
    else:
        conversations = [text]
        add_start_token = False
    for conversation in conversations:
        questions = [t.split(args.bot_token)[0] for t in conversation.split(args.user_token)[1:]]
        answers = [t.split(args.bot_token)[-1] for t in conversation.split(args.end_token)[:-1]]
        for question, answer in zip(questions, answers):
            question_ids = tokenizer(question)["input_ids"]
            answer_ids = tokenizer(answer)["input_ids"]
            inputs_tmp = [user_token_id] + question_ids + [bot_token_id] + answer_ids + [end_token_id]
            labels_tmp = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(question_ids) + [IGNORE_TOKEN_ID] + \
                answer_ids + [end_token_id]
            input_ids.extend(inputs_tmp)
            label_ids.extend(labels_tmp)
        if add_start_token:
            input_ids = [start_token_id] + input_ids
            label_ids = [IGNORE_TOKEN_ID] + label_ids
    res["input_ids"] = [pad_token_id] * pad_len + input_ids
    res["labels"] = [IGNORE_TOKEN_ID] * pad_len + label_ids
    return res


def process_dataset(current_dataset, tokenizer, max_seq_len):
    """process dataset."""
    dataset = []
    all_lines = []
    for _, tmp_data in enumerate(current_dataset):
        input_data = tmp_data['input']
        if '<_user>' in input_data and args.user_token != '<_user>':
            input_data = input_data.replace('<_user>', args.user_token)
        if '<_bot>' in input_data and args.bot_token != '<_bot>':
            input_data = input_data.replace('<_bot>', args.bot_token)
        if not input_data.startswith(args.user_token):
            input_data = args.user_token + input_data
        output = tmp_data['output']
        if args.bot_token in input_data:
            concat_line = ""
            input_turns = input_data.split(args.user_token)[1:]
            for item in input_turns:
                if args.bot_token in item:
                    concat_line += args.user_token + item + args.end_token
                else:
                    concat_line += args.user_token + item + args.bot_token
            concat_line += output + args.end_token
        else:
            concat_line = str(input_data) + args.bot_token + str(output) + args.end_token
        if concat_line.count(args.user_token) != concat_line.count(args.bot_token) and \
            concat_line.count(args.user_token) != concat_line.count(args.end_token):
            logger.error(f"{concat_line.count(args.user_token)} should be equal to "
                         f"{concat_line.count(args.bot_token)}"
                         f" and {concat_line.count(args.user_token)} should be equal to "
                         f"{concat_line.count(args.end_token)}")
        if args.start_token:
            concat_line = args.start_token + concat_line
        all_lines.append(concat_line)
    shuffle(all_lines)
    previous_corpus_token_cnt = 0
    shard = []
    padding_out = []
    for corpus in tqdm(all_lines):
        corpus_ids = tokenizer_with_special_tokens(corpus, tokenizer)
        inputs = corpus_ids.get("input_ids")
        if previous_corpus_token_cnt + len(inputs) < max_seq_len:
            shard.append(corpus)
            previous_corpus_token_cnt += len(inputs)
        else:
            shard_output = "".join(shard)
            shard_output = (args.max_length - previous_corpus_token_cnt) * args.pad_token + shard_output
            if len(tokenizer_with_special_tokens(shard_output, tokenizer).get("input_ids")) != max_seq_len:
                logger.error(f"input_ids length should be equal to {max_seq_len}")
            if shard_output.count(args.user_token) >= 1:
                padding_out.append(shard_output)
            if len(inputs) < max_seq_len:
                shard = [corpus]
                previous_corpus_token_cnt = len(inputs)
            else:
                shard = []
                previous_corpus_token_cnt = 0
    logger.info(f'prompt length: {len(padding_out)}')
    for dt in padding_out:
        tokens = tokenizer_with_special_tokens(dt, tokenizer)
        dataset.append(tokens)
    return dataset


def make_dataset():
    """make dataset."""
    raw_dataset = TelechatDataset(args.output_path, args.seed, args.input_dataset_file)
    train_dataset = raw_dataset.get_train_data()
    tokenizer = TelechatTokenizer(args.vocab_file_path, fast_tokenizer=True,
                                  trust_remote_code=True, padding_side="left")
    train_dataset = process_dataset(train_dataset, tokenizer, args.max_length)
    logger.info("***** Writing to output files *****")
    writer = FileWriter(args.output_dataset_file, 1)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "lm-schema")
    for dataset in tqdm(train_dataset):
        instance = {"input_ids": dataset["input_ids"], "labels": dataset["labels"]}
        write_instance_to_file(writer, instance=instance)
    writer.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--vocab_file_path", default="", type=str, help='which model to use.')
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--start_token", type=str, default="<_start>", help="start token")
    parser.add_argument("--user_token", type=str, default="<_user>", help="user token")
    parser.add_argument("--bot_token", type=str, default="<_bot>", help="bot token")
    parser.add_argument("--end_token", type=str, default="<_end>", help="end token")
    parser.add_argument("--pad_token", type=str, default="<_pad>", help="pad token")
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
