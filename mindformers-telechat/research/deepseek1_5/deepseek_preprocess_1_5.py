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
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import json
import os
import re
import numpy as np

from mindspore.mindrecord import FileWriter

from mindformers.models import build_tokenizer


IGNORE_TOKEN_ID = -100


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch


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


def preprocess(sources, tokenizer, seq_length, pad_token_id):
    """conversation preprocess."""

    # Apply prompt templates
    conversations = []
    for _, source in enumerate(sources):
        input_s = source[0]["value"].lstrip(
            '\n').rstrip(' ') + source[1]["value"] + '\n'
        q_len = len(tokenizer(source[0]["value"].lstrip(
            '\n').rstrip(' '))['input_ids']) - 1
        conversations.append([input_s, q_len])

    # Tokenize conversations
    input_ids = []
    targets = []
    for conversation in conversations:
        ids = tokenizer(conversation[0])['input_ids']
        mask = tokenizer(conversation[0])['attention_mask']
        d = {'input_ids': ids, 'attention_mask': mask}
        target = np.array(d['input_ids'])
        len_inputid = len(d['input_ids'])
        l_target = len(target)
        if l_target < seq_length:
            d['input_ids'] = np.pad(d['input_ids'], ((0), (seq_length - len_inputid)),
                                    mode='constant', constant_values=pad_token_id)
            target = np.pad(target, ((0), (seq_length - l_target)),
                            mode='constant', constant_values=IGNORE_TOKEN_ID)

        target[:conversation[1]] = IGNORE_TOKEN_ID
        targets.append(target[:seq_length].tolist())
        input_ids.append(d['input_ids'][:seq_length])

    input_ids = np.array(input_ids, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length, pad_token_id):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, seq_length, pad_token_id)

        self.input_ids = data_dict.get("input_ids")
        self.labels = data_dict.get("labels")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
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


def tokenize_qa(tokenizer, file_path, seq_length, pad_token_id):
    raw_data = None
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(file_path, flags_, 0o750), 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length, pad_token_id)
    for i, _ in enumerate(dataset_cls):
        yield dataset_cls[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='wiki')
    parser.add_argument('--input_glob', type=str,
                        default='/mnt/luolan/wikitext-2/wiki.train.tokens')
    parser.add_argument('--output_file', type=str,
                        default='./dataset/wiki2048/wiki2048')
    parser.add_argument('--tokenizer', type=str,
                        default='llama', choices=['llama'])
    parser.add_argument('--model_file', type=str,
                        default='/mnt/luolan/llama/tokenizer.model')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=2048)
    parser.add_argument('--pad_token_id', type=int, default=32014)
    args = parser.parse_args()
    # pylint: disable=C0326
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if args.dataset_type == 'wiki':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}, }
    elif args.dataset_type == 'qa':
        schema = {'input_ids': {"type": "int32",
                                "shape": [-1]}, 'labels': {"type": "int32", "shape": [-1]}}
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.dataset_type)

    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0
    tokenizer_dict = {'unk_token': 'None', 'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<|EOT|>',
                      'pad_token': '<｜end▁of▁sentence｜>', 'vocab_file': 'None', 'tokenizer_file': args.model_file,
                      'type': 'LlamaTokenizerFast'}
    word_tokenizer = build_tokenizer(tokenizer_dict)
    if hasattr(word_tokenizer, 'add_bos_token'):
        word_tokenizer.add_bos_token = True
    if hasattr(word_tokenizer, 'add_eos_token'):
        word_tokenizer.add_eos_token = True
    if args.dataset_type == 'wiki':
        for x in tokenize_wiki(word_tokenizer, args.input_glob, args.seq_length + 1, args.repeat):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'qa':
        for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1, args.pad_token_id):
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
