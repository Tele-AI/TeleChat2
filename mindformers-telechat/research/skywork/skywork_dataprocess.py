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
transform skywork text dataset to mindrecord.
"""

import math
import argparse
import json
import os
import collections
import pathlib
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer


def create_instance(tokenizer, ids, max_length=None):
    """A single sample instance for LM task."""
    pair_ids = None

    output = tokenizer.prepare_for_model(ids=ids,
                                         pair_ids=pair_ids,
                                         add_special_tokens=False,
                                         max_length=max_length,
                                         padding='max_length',
                                         truncate_direction="LEFT",
                                         return_overflowing_tokens=False,
                                         return_attention_mask=True)
    return output


def write_instance_to_file(instance_writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["input_ids"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    instance_writer.write_raw_data([features])

    return features


def tokenize_text(tokenizer, text_list, seq_length, instance_writer, batch_size):
    """tokenize text dataset"""
    dataset_all = []
    for data in text_list:
        dataset_all.append(data['text'])

    batch_num = math.ceil(len(dataset_all) / batch_size)
    print("dataset size ", len(dataset_all))
    print("batch_size ", batch_size)
    total_written = 0
    for i in range(batch_num):
        dataset_valid = dataset_all[i * batch_size:(i + 1) * batch_size]
        data_tokens = tokenizer(dataset_valid)
        input_ids = data_tokens["input_ids"]
        total_ids = [item for sublist in input_ids for item in sublist]

        block_size = seq_length + 1
        total_length = len(total_ids)
        total_length = (total_length // seq_length) * seq_length
        for j in range(total_length // seq_length):
            ids = total_ids[seq_length * j:seq_length * (j + 1)]
            ids.append(tokenizer.pad_token_id)

            output = create_instance(tokenizer, ids, block_size)

            write_instance_to_file(instance_writer, instance=output)
            total_written += 1

    print("Wrote {} total instances".format(total_written))


def get_text(args_param):
    data_path = pathlib.Path(args_param.input_file_path)

    text_list = []

    with open(data_path, 'r', encoding="utf-8") as input_file:
        for line in input_file:
            data = json.loads(line)
            text_list.append({"text": data["content"] + data["summary"]})
    return text_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="AdvertiseGenTrain.jsonl")
    parser.add_argument('--dataset_type', type=str, default='text')
    parser.add_argument('--output_file', type=str, default='AdvertiseGenTrain_text.mindrecord')
    parser.add_argument('--tokenizer', type=str, default='llama', choices=['llama'])
    parser.add_argument('--model_file', type=str, default='./tokenizer.model')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=1000)
    args = parser.parse_args()

    text_dataset = get_text(args)

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    schema = {'input_ids': {"type": "int32", "shape": [-1]}, 'labels': {"type": "int32", "shape": [-1]}}
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.dataset_type)
    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0
    word_tokenizer = LlamaTokenizer(vocab_file=args.model_file)
    word_tokenizer.add_bos_token = True
    word_tokenizer.add_eos_token = False
    tokenize_text(word_tokenizer, text_dataset, args.seq_length, writer, args.batch_size)
    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
