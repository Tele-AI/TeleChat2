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
transform dataset to mindrecord.
"""
import argparse
import json
import os

import numpy as np

from mindspore.mindrecord import FileWriter

from mindformers.tools import logger
from research.qwen.qwen_tokenizer import QwenTokenizer

IGNORE_TOKEN_ID = -100


def preprocess(sources, tokenizer, seq_length):
    """conversation preprocess."""
    system_message = "You are a helpful assistant."
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n')['input_ids']
    system_base = tokenizer('system')['input_ids'] + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for _, source in enumerate(sources):
        if roles.get(source[0].get("from")) != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + system_base + tokenizer(system_message)['input_ids'] + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        if len(input_id) != len(target):
            raise ValueError("The len(input_id) should equal to the len(target), but len(input_id) got"
                             f"{len(input_id)}, len(target) got {len(target)}.")
        for _, sentence in enumerate(source):
            role = roles.get(sentence.get("from"))
            input_id_part = tokenizer(role)['input_ids'] + nl_tokens + tokenizer(sentence["value"])['input_ids'] + [
                im_end] + nl_tokens
            input_id += input_id_part
            if role == '<|im_start|>user':
                target_part = [im_start] + [IGNORE_TOKEN_ID] * (len(input_id_part) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                target_part = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role)['input_ids']) + \
                              input_id_part[len(tokenizer(role)['input_ids']) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += target_part
        if len(input_id) != len(target):
            raise ValueError("The len(input_id) should equal to the len(target), but len(input_id) got"
                             f"{len(input_id)}, len(target) got {len(target)}.")
        input_id += [tokenizer.pad_token_id] * (seq_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (seq_length - len(target))
        input_ids.append(input_id[:seq_length])
        targets.append(target[:seq_length])

    input_ids = np.array(input_ids, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    attention_mask = np.where(input_ids == tokenizer.pad_token_id, 0, 1).astype(np.int32)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask
    )


# pylint: disable=C0111
# pylint: disable=W0703
def tokenize_qa(tokenizer, file_path, seq_length):
    file = None
    raw_data = None
    try:
        file = open(file_path, "r")
        raw_data = json.load(file)
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
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length)
    for i, _ in enumerate(dataset_cls):
        yield dataset_cls[i]


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, seq_length)

        self.input_ids = data_dict.get("input_ids")
        self.labels = data_dict.get("labels")
        self.attention_mask = data_dict.get("attention_mask")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='./alpaca-fastchat-qwen.mindrecord')
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=2048)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'labels': {"type": "int32", "shape": [-1]},
              "attention_mask": {"type": "int32", "shape": [-1]}
              }
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema)

    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0
    word_tokenizer = QwenTokenizer(vocab_file=args.model_file)

    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
