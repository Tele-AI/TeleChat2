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

"""
transform alpaca dataset to mindrecord.
"""
import argparse
import json
import os
import numpy as np
from mindspore.mindrecord import FileWriter
from mindformers.tools import logger
from internlm2_tokenizer import InternLM2Tokenizer

np.set_printoptions(threshold=np.inf)
IGNORE_TOKEN_ID = -100


def preprocess(sources, tokenizer, seq_length):
    """From alpaca to mindrecord."""
    input_ids = []
    labels = []
    special_tokenize = {"<s>": tokenizer.encode('')[0],
                        "</s>": tokenizer.encode('</s>', add_special_tokens=False)[0],
                        "newline": tokenizer.encode('\n', add_special_tokens=False)[0]}

    # 对话格式为
    # <|im_start|>system\n
    # source['instruction']<|im_end|>\n
    # <|im_start|>user\n
    # source['input']<|im_end|>\n
    # <|im_start|>assistant\n
    # source['output']<|im_end|>\n

    start_usr_token = tokenizer.encode('<|im_start|>user', add_special_tokens=False) + [special_tokenize["newline"]]
    start_ass_token = tokenizer.encode('<|im_start|>assistant', add_special_tokens=False) \
                      + [special_tokenize["newline"]]
    end_token = tokenizer.encode('<|im_end|>', add_special_tokens=False) + [special_tokenize["newline"]]

    for source in sources:
        ins_token = tokenizer.encode(source['instruction'], add_special_tokens=False)
        inp_token = tokenizer.encode(source['input'], add_special_tokens=False) + [special_tokenize["newline"]]
        out_token = tokenizer.encode(source['output'], add_special_tokens=False)

        # 构建 input_id
        if source['input']:
            input_id = [1,
                        *start_usr_token,
                        *ins_token,
                        special_tokenize["newline"],
                        *inp_token,
                        *end_token,
                        *start_ass_token]
        else:
            input_id = [1, *start_usr_token, *ins_token, *end_token, *start_ass_token]

        # 构建 label
        label = [*out_token, *end_token, 2]

        # 调整 input_id 和 label 的形状
        if len(input_id) + len(label) > seq_length:
            # 如果 input 和 label 的总长度大于 seq_length，就把 label 全设为 IGNORE_TOKEN_ID
            input_id = (input_id + label)[:seq_length - 1] + [2]
            label = np.full_like(input_id, fill_value=IGNORE_TOKEN_ID)
        else:
            input_pad_length = seq_length - len(input_id)
            label_pad_length_head = len(input_id)
            label_pad_length_tail = input_pad_length - len(label)

            input_id = np.pad(input_id, (0, input_pad_length), 'constant', constant_values=(0, 0))
            label = np.pad(label,
                           (label_pad_length_head, label_pad_length_tail),
                           'constant',
                           constant_values=(IGNORE_TOKEN_ID, IGNORE_TOKEN_ID))

        input_ids.append(np.array(input_id).astype(np.int32))
        labels.append(np.array(label).astype(np.int32))

    return dict(input_ids=input_ids, labels=labels)


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
    parser.add_argument("--mindrecord_schema", type=str, default="internlm2_alpaca")
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

    word_tokenizer = InternLM2Tokenizer(vocab_file=args.model_file)
    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print(f"Transformed {transforms_count} records.")
    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print(f"Transform finished, output files refer: {out_file}.")
