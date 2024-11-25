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
transform belle_chat_ramdon_10k dataset to mindrecord.
"""
import argparse
import json
import os
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.tools import logger

from baichuan2_tokenizer import Baichuan2Tokenizer

IGNORE_TOKEN_ID = -100


def preprocess(sources, tokenizer, seq_length, user_tokens=195, assistant_tokens=196):
    """conversation preprocess."""
    input_ids = []
    labels = []
    for example in sources:
        input_id = []
        label = []
        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = tokenizer.encode(value, add_special_tokens=False)

            if from_ == "human":
                input_id += [user_tokens] + value_ids
                label += [tokenizer.eos_token_id] + [IGNORE_TOKEN_ID] * len(value_ids)
            else:
                input_id += [assistant_tokens] + value_ids
                label += [IGNORE_TOKEN_ID] + value_ids

        input_id.append(tokenizer.eos_token_id)
        label.append(tokenizer.eos_token_id)
        if len(input_id) > seq_length:
            input_id = input_id[: seq_length]
            label = label[: seq_length]
        else:
            input_id += [tokenizer.pad_token_id] * (seq_length - len(input_id))
            label += [IGNORE_TOKEN_ID] * (seq_length - len(label))

        input_ids.append(np.array(input_id).astype(np.int32))
        labels.append(np.array(label).astype(np.int32))

    return dict(
        input_ids=input_ids,
        labels=labels
    )


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        data_dict = preprocess(raw_data, tokenizer, seq_length)

        self.input_ids = data_dict.get("input_ids")
        self.labels = data_dict.get("labels")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindrecord_schema", type=str, default="belle_baichuan2")
    parser.add_argument("--input_glob", type=str, default="./belle_chat_ramdon_10k.json")
    parser.add_argument("--output_file", type=str, default="./belle512.mindrecord")
    parser.add_argument("--model_file", type=str, default="./tokenizer.model")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'labels': {"type": "int32", "shape": [-1]}}

    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.mindrecord_schema)

    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0

    word_tokenizer = Baichuan2Tokenizer(vocab_file=args.model_file)
    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
