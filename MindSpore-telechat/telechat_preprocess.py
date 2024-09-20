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
from multiprocessing import Pool
import numpy as np
import jsonlines

from mindspore.mindrecord import FileWriter
from telechat_tokenizer import TelechatTokenizer

def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    writer.write_raw_data([features])
    return features

def preprocess_concat_datas(datasets):
    """Preprocess dataset"""
    tokens = []
    tokenizer = TelechatTokenizer(args.tokenizer_file, trust_remote_code=True)
    user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(args.pad_token)
    for data in datasets:
        data = data["text"]
        input_ids = []
        labels = []
        data = data.replace(args.pad_token, "").replace("<pad>", "")
        dialogs = data.split(args.end_token)[:-1]
        for dialog in dialogs:
            dialog = dialog.split(args.bot_token)
            question = dialog[0].replace(args.user_token, "")
            answer = dialog[1]
            input_token = tokenizer(question)["input_ids"]
            output_token = tokenizer(answer)["input_ids"]
            concat_tokens = [user_token_id] + input_token + [bot_token_id] + output_token + [end_token_id]
            concat_labels = [1] + len(input_token) * [0] + [1] + len(output_token) * [1] + [1]
            if len(input_ids) <= args.max_length and len(input_ids) + len(concat_tokens) > args.max_length:
                break
            input_ids = input_ids + concat_tokens
            labels = labels + concat_labels
        input_ids = input_ids + (args.max_length - len(input_ids)) * [pad_token_id]
        labels = labels + (args.max_length - len(labels)) * [0]
        tokens.append({"input_ids": input_ids, "labels": labels})
    return tokens

def process(file_list):
    """Multi-process processing"""
    f_in = jsonlines.open(os.path.join(args.input_dataset_dir, file_list), "r")
    dataset = [data for data in f_in]
    f_in.close()
    tokens = preprocess_concat_datas(dataset)
    print(len(tokens))

    writer = FileWriter(os.path.join(args.output_path, file_list) + ".mindrecord", 1)

    data_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "labels": {"type": "int32", "shape": [-1]}
    }

    writer.add_schema(data_schema, "lm-schema")
    for token in tokens:
        write_instance_to_file(writer, token)
    writer.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument('--tokenizer_file', default='', type=str, help='which model to use.')
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1233)
    parser.add_argument("--user_token", type=str, default="<_user>", help="user token")
    parser.add_argument("--bot_token", type=str, default="<_bot>", help="bot token")
    parser.add_argument("--end_token", type=str, default="<_end>", help="end token")
    parser.add_argument("--pad_token", type=str, default="<_pad>", help="pad token")
    parser.add_argument("--pool_num", type=int, default=32, help="num of pool")
    args = parser.parse_args()

    args.max_length += 1
    file_lists = [i for i in os.listdir(args.input_dataset_dir)]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pool = Pool(args.pool_num)
    results = []
    for single_file in file_lists:
        results.append(pool.apply_async(process, args=(single_file,)))
    pool.close()
    pool.join()
