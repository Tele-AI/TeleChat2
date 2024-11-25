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
transform alpaca dataset to mindrecord.
"""
import argparse
import json
import os
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.tools import logger

from internlm_tokenizer import InternLMTokenizer

IGNORE_TOKEN_ID = -100


def get_chat_format_data(ori_data):
    """Format original data

    Args:
        ori_data (dict): input data sample.

    Returns:
        dict: data sample with chat format.
    """
    input_str = ori_data["input"]
    instruction_str = ori_data["instruction"]
    output_str = ori_data["output"]
    data = dict()
    if input_str != "":
        data["user"] = f"<|User|>:{instruction_str}\n{input_str}"
    else:
        data["user"] = f"<|User|>:{instruction_str}"
    data["bot"] = f"<|Bot|>:{output_str}"
    return data


# pylint: disable=C0326
def preprocess(sources, tokenizer, seq_length, bos_token="<s>", eos_token="</s>"):
    """conversation preprocess."""
    input_ids = []
    labels = []
    for source in sources:
        data = get_chat_format_data(source)
        special_tokens_map = {"<eoh>": 103167, "<eoa>": 103166, "nl_id": 13}
        token_ids = tokenizer.encode(bos_token, add_special_tokens=False)
        human_s = None
        ass_s = None
        human_value = data.get("user")
        ass_value = data.get("bot")
        if human_value:
            human_s = human_value
        else:
            raise ValueError(f"user is not in data:{data}.")
        if ass_value:
            ass_s = ass_value
        else:
            raise ValueError(f"bot is not in data:{data}.")

        human_ids = tokenizer.encode(human_s, add_special_tokens=False) + \
                    [special_tokens_map["<eoh>"], special_tokens_map["nl_id"]]

        ass_template_ids = tokenizer.encode("<|Bot|>:", add_special_tokens=False)

        ignore_len = len(human_ids) + len(ass_template_ids)

        ass_ids = (ass_template_ids + tokenizer.encode(ass_s[8:], add_special_tokens=False) + \
                   [special_tokens_map["<eoa>"], special_tokens_map["nl_id"]])

        targets = np.ones([seq_length, ])
        token_ids += human_ids + ass_ids

        if len(token_ids) > seq_length:
            token_ids = token_ids[:seq_length]
            token_ids += tokenizer.encode(eos_token, add_special_tokens=False)
            targets[:] = IGNORE_TOKEN_ID
        else:
            token_ids += tokenizer.encode(eos_token, add_special_tokens=False)
            ignore_len_end = seq_length - len(token_ids)
            token_ids = np.pad(token_ids, (0, ignore_len_end), 'constant', constant_values=(0, 0))
            targets = np.array(token_ids)
            targets[:ignore_len + 1] = IGNORE_TOKEN_ID
            targets[-ignore_len_end:] = IGNORE_TOKEN_ID

        input_ids.append(np.array(token_ids).astype(np.int32))
        labels.append(np.array(targets).astype(np.int32))

    return dict(
        input_ids=input_ids,
        labels=labels
    )


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        sources = []
        for example in raw_data:
            sources.append(example)
        data_dict = preprocess(sources, tokenizer, seq_length)

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
    parser.add_argument("--mindrecord_schema", type=str, default="internlm_alpaca")
    parser.add_argument("--input_glob", type=str, default="./alpaca_data.json")
    parser.add_argument("--output_file", type=str, default="./alpaca_processed/alpaca.mindrecord")
    parser.add_argument("--model_file", type=str, default="./tokenizer.model")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)
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

    word_tokenizer = InternLMTokenizer(vocab_file=args.model_file)
    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
