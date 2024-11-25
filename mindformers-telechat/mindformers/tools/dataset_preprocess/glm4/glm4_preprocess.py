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
transform dataset to mindrecord.
"""
import argparse
import json
import os
import numpy as np
from mindspore.mindrecord import FileWriter
from mindformers.models.glm2.glm4_tokenizer import ChatGLM4Tokenizer

IGNORE_TOKEN_ID = -100


def process_message(message):
    if 'tools' in message and message['role'] == 'system':
        for tool in message['tools']:
            parameters = tool['function']['parameters']['properties']
            tool['function']['parameters']['properties'] = \
                {k: v for k, v in parameters.items() if
                 v is not None}
    elif 'tools' in message:
        del message['tools']
    return message


def preprocess(messages, tokenizer, seq_length):
    """Preprocesses the data for supervised fine-tuning."""

    ret_input_ids = []
    ret_labels = []
    ret_attention_mask = []
    ret_position_ids = []
    for conv in messages:
        input_ids = [151331, 151333]  # [gMASK] + <sop>
        loss_masks = [False, False]
        for message in conv:
            message = process_message(message)
            loss_mask_val = message['role'] not in ('system', 'user', 'observation')
            if not message:
                continue
            new_input_ids = tokenizer.apply_chat_template([message], tokenize=True)[2:-1]
            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            input_ids += new_input_ids
            loss_masks += new_loss_masks
        input_ids.append(151336)  # EOS token '<|user|>'
        loss_masks = [False, *loss_masks]  # <|assistant|> True -> <|assistant|> False <|user|> True
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)

        input_ids = input_ids[:seq_length]
        labels = labels[:seq_length]
        attention_mask = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))

        input_ids = input_ids + [151329] * (seq_length - len(input_ids))
        labels = labels + [-100] * (seq_length - len(labels))
        attention_mask = attention_mask + [0] * (seq_length - len(attention_mask))
        position_ids = position_ids + [0] * (seq_length - len(position_ids))

        ret_input_ids.append(input_ids[:seq_length - 1])
        ret_labels.append(labels[1:seq_length])
        ret_attention_mask.append(attention_mask[:seq_length - 1])
        ret_position_ids.append(position_ids[:seq_length - 1])

    return {
        'input_ids': np.array(ret_input_ids).astype(np.int32),
        'labels': np.array(ret_labels).astype(np.int32),
        'attention_mask': np.array(ret_attention_mask).astype(np.int32),
        'position_ids': np.array(ret_position_ids).astype(np.int32),
    }


def tokenize_qa(tokenizer, file_path, seq_length):
    raw_data = []
    with open(file_path, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length)
    for i, _ in enumerate(dataset_cls):
        yield dataset_cls[i]


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, seq_length)

        self.input_ids = data_dict.get("input_ids", None)
        self.labels = data_dict.get('labels', None)
        self.attention_mask = data_dict.get('attention_mask', None)
        self.position_ids = data_dict.get('position_ids', None)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str,
                        default='./alpaca_glm4_data.jsonl')
    parser.add_argument('--output_file', type=str,
                        default='./alpaca-fastchat-glm4.mindrecord')
    parser.add_argument('--vocab_file', default=r'./tokenizer.model',
                        type=str,
                        help='vocab_file path')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=8192)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'labels': {"type": "int32", "shape": [-1]},
              }
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema)
    try:
        writer.open_and_set_header()
    except AttributeError:
        pass

    transforms_count = 0
    word_tokenizer = ChatGLM4Tokenizer(
        args.vocab_file, add_bos_token=False, add_eos_token=False)

    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
