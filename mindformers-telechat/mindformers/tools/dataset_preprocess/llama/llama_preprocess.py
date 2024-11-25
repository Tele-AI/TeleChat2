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
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import json
import os
import re
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.dataset.dataloader.training_dataloader import TrainingDataset
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers.tools import logger

from mindformers.tools.dataset_preprocess.llama.conversation import get_default_conv_template

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


def preprocess(sources, tokenizer, seq_length):
    """conversation preprocess."""
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles.get(source[0].get("from")) != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence.get("from"))
            if role != conv.roles[j % 2]:
                raise ValueError(f"sources[{i}] is wrong.")
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    sep = conv.sep + conv.roles[1] + ": "
    # Tokenize conversations
    input_ids = []
    targets = []
    # attention_mask = []
    for conversation in conversations:
        rounds = conversation.split(conv.sep2)
        ids = [tokenizer.bos_token_id]
        mask = [1]
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            conv_out = tokenizer(rou)
            ids.extend(conv_out['input_ids'][1:])
            mask.extend(conv_out['attention_mask'][1:])
        d = {'input_ids': ids, 'attention_mask': mask}
        # pylint: disable=W0212
        d = tokenizer._pad(d, max_length=seq_length, padding_strategy='max_length')
        input_ids.append(d['input_ids'][:seq_length])
        # attention_mask.append(d['attention_mask'])

        target = np.array(d['input_ids'])
        total_len = int(np.not_equal(target, tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou)['input_ids']) - 1
            instruction_len = len(tokenizer(parts[0])['input_ids']) - 3

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

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


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, seq_length)

        self.input_ids = data_dict.get("input_ids", None)
        self.labels = data_dict.get("labels", None)

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


def tokenize_wikipedia(tokenizer, dataset_dir, seq_length, samples_num):
    """tokenize wikipedia dataset with parquet format"""
    dataset = TrainingDataset(dataset_dir=dataset_dir,
                              column_names=["input_ids"],
                              max_length=seq_length,
                              file_format="parquet",
                              tokenizer=tokenizer,
                              is_align=True,
                              samples_num=samples_num,
                              shuffle=False)
    for data in dataset:
        input_id_list = data[0]
        if len(input_id_list) == seq_length:
            sample = {
                'input_ids': np.array(input_id_list, dtype=np.int32)
            }
            yield sample


# pylint: disable=W0703
def tokenize_qa(tokenizer, file_path, seq_length):
    """tokenize qa dataset"""
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
    parser.add_argument('--dataset_type', type=str, default='wiki')
    parser.add_argument('--input_glob', type=str, default='/mnt/luolan/wikitext-2/wiki.train.tokens')
    parser.add_argument('--output_file', type=str, default='./dataset/wiki2048/wiki2048')
    parser.add_argument('--tokenizer', type=str, default='llama', choices=['llama'])
    parser.add_argument('--model_file', type=str, default='/mnt/luolan/llama/tokenizer.model')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--samples_num', type=int, default=10000)
    parser.add_argument('--seq_length', type=int, default=2048)
    args = parser.parse_args()
    # pylint: disable=C0326
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if args.dataset_type == 'wiki':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}, }
    if args.dataset_type == 'wikipedia':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}, }
    elif args.dataset_type == 'qa':
        schema = {'input_ids': {"type": "int32", "shape": [-1]}, 'labels': {"type": "int32", "shape": [-1]}}
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.dataset_type)

    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0
    word_tokenizer = LlamaTokenizer(vocab_file=args.model_file)
    if hasattr(word_tokenizer, 'add_bos_token'):
        word_tokenizer.add_bos_token = True
    if hasattr(word_tokenizer, 'add_eos_token'):
        word_tokenizer.add_eos_token = True
    if args.dataset_type == 'wiki':
        for x in tokenize_wiki(word_tokenizer, args.input_glob, args.seq_length + 1, args.repeat):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'wikipedia':
        for x in tokenize_wikipedia(word_tokenizer, args.input_glob, args.seq_length + 1, samples_num=args.samples_num):
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
