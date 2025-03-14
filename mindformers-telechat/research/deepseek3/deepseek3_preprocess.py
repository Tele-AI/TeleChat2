# Copyright 2025 Huawei Technologies Co., Ltd
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
import re
import pathlib

import numpy as np
from mindspore.mindrecord import FileWriter

from deepseek3_conversation import get_default_conv_template
from mindformers.models.llama import LlamaTokenizerFast

IGNORE_TOKEN_ID = -100


def alpaca_converter(data_path):
    """
    fastchat stanford code_alpaca data convert tools.
    """

    # Prompt from stanford code_alpaca's training script

    prompt_dict = {
        "prompt": (
            'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, '
            'and you only answer questions related to computer science. '
            'For politically sensitive questions, security and privacy issues,'
            ' and other non-computer science questions,'
            ' you will refuse to answer.\n'
            '### Instruction:\n'
            '{instruction}\n'
            '### Response:\n'
        )
    }

    data_path = pathlib.Path(data_path)
    with data_path.open(encoding='utf-8') as f:
        data = json.load(f)
    prompt = prompt_dict["prompt"]

    sources = [
        prompt.format_map(example)
        for example in data
    ]

    targets = [example["output"] for example in data]

    new_data = []

    cnt = 1
    max_source_length = 0
    for s, t in zip(sources, targets):
        max_source_length = max(max_source_length, len(s))
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s,
                    },
                    {
                        "from": "gpt",
                        "value": t,
                    },
                ],
            }
        )

        cnt += 1
    print(f"max_source_length={max_source_length},  sample_num={len(new_data)}")
    return new_data


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_wikitext(string):
    """ cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def preprocess(sources, tokenizer, seq_length):
    """conversation preprocess."""
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for _, source in enumerate(sources):
        try:
            if roles.get(source[0].get("from")) != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles.get(sentence.get("from"))
                if role != conv.roles[j % 2]:
                    raise ValueError(f"expect role {role}, but get role {conv.roles[j % 2]}")
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        except ValueError as e:
            print(e)
            continue

    sep = conv.sep + conv.roles[1] + ": "
    # Tokenize conversations
    input_ids = []
    targets = []
    # attention_mask = []
    for conversation in conversations:
        rounds = conversation.split(conv.sep2)
        ids = [tokenizer.bos_token_id]
        mask = [1]
        for _, rou in enumerate(rounds):
            if rou == "":
                break
            conv_out = tokenizer(rou)
            ids.extend(conv_out['input_ids'][1:])
            mask.extend(conv_out['attention_mask'][1:])
        d = {'input_ids': ids, 'attention_mask': mask}

        # pylint: disable=W0212
        d = tokenizer._pad(d, max_length=seq_length, padding_strategy='max_length')

        input_ids.append(d['input_ids'][:seq_length])

        target = np.array(d['input_ids'])
        total_len = int(np.not_equal(target, tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for _, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou)['input_ids']) - 1
            instruction_len = len(tokenizer(parts[0])['input_ids']) - 3

            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < seq_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
        else:
            target = target[:seq_length]
        targets.append(target.tolist())

    input_ids = np.array(input_ids, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def tokenize_wiki(tokenizer, file_path, seq_length, repeat):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith('=') is False:
                content += tokenizer(para)['input_ids']
    content_out = []
    for _ in range(repeat):
        content_out.extend(content)
    content = content_out
    for chunk in chunks(content, seq_length):
        sample = {}
        if len(chunk) == seq_length:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_qa(tokenizer, file_path, seq_length):
    raw_data = alpaca_converter(file_path)
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length)
    for i in dataset_cls:
        yield i


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, seq_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='qa', choices=['wiki', 'qa'])
    parser.add_argument('--input_glob', type=str, required=True)
    parser.add_argument('--output_file', type=str,
                        default='./alpaca-fastchat-deepseek3.mindrecord')
    parser.add_argument('--tokenizer_file', default='./tokenizer.json', type=str,
                        help='tokenizer_file path')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=2048)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if args.dataset_type == 'wiki':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}}
    elif args.dataset_type == 'qa':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}, 'labels': {"type": "int32", "shape": [-1]}}
    else:
        raise ValueError("Not support dataset type: {}".format(args.dataset_type))
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema)

    transforms_count = 0
    word_tokenizer = LlamaTokenizerFast(
        tokenizer_file=args.tokenizer_file, add_bos_token=False, add_eos_token=False)
    word_tokenizer.pad_token_id = 100001

    if args.dataset_type == 'wiki':
        for x in tokenize_wiki(word_tokenizer, args.input_glob, args.seq_length + 1, args.repeat):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'qa':
        for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
