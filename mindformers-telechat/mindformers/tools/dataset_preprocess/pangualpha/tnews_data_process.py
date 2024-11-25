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
Data process for TNEWS Dataset.

The generated dataset has output columns: ["input_ids", "labels", "attention_mask"],
and the data type of three columns is int32.

Columns：
    input_ids: the tokenized inputs, shape is [num_labels, seq_length]
    labels: the index of the true label, shape is [1,]
    attention_mask: the mask indicating whether each position is a valid input and is not the added prompt,
                    shape is [num_labels, seq_length]

About TNEWS Dataset:
    TNEWS is a 15-class(finance, technology, sports, etc.) short news text classification dataset.
    Each piece of data has three attributes, which are classification ID, classification name,
    and news string (only including the title).

    You can unzip the dataset files into the following structure:

    .. code-block::

        .
        └── tnews_public
             ├── train,json
             ├── test.json
             ├── test1.0.json
             ├── dev.json
             ├── labels.json
             └── README.txt

"""

import itertools
import json
import argparse
import collections
import logging
import numpy as np

from mindspore.mindrecord import FileWriter

from mindformers import AutoTokenizer


EN2ZH_LABEL_MAP = {"news_story": "故事",
                   "news_culture": "文化",
                   "news_entertainment": "娱乐",
                   "news_sports": "体育",
                   "news_finance": "财经",
                   "news_house": "房产",
                   "news_car": "汽车",
                   "news_edu": "教育",
                   "news_tech": "科技",
                   "news_military": "军事",
                   "news_travel": "旅行",
                   "news_world": "世界",
                   "news_stock": "股票",
                   "news_agriculture": "农业",
                   "news_game": "游戏"}
NUM_LABELS = 4


def get_en2zh_label(label_path):
    """get label mapping"""
    en2zh_labels = {}
    print('loading label file ')
    # labels_ch.json is the results that translated from original labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line_json = json.loads(line)
            if "label_desc_ch" in line_json:
                en_label = line_json['label_desc']
                zh_label = line_json['label_desc_ch']
                en2zh_labels[en_label] = zh_label
            else:
                break
    if en2zh_labels == {}:
        en2zh_labels = EN2ZH_LABEL_MAP
    return en2zh_labels


def load_tnews_example_for_shot(data_path, en2zh_labels, num_sample=6, np_rng=None, max_len=150, prompt="这是关于{}的文章：{}"):
    """get prompt for n shot"""
    with open(data_path, 'r', encoding="utf-8") as fid:
        data = [json.loads(x) for x in fid.readlines()]

    if np_rng is None:
        np_rng = np.random.default_rng(0)
    # select sample with balanced labels
    label_key = lambda x: x[1]
    sample_groupbyed = itertools.groupby(
        sorted([(x, en2zh_labels[y['label_desc']]) for x, y in enumerate(data)], key=label_key), key=label_key)
    group_index = [np.array([z[0] for z in y]) for x, y in sample_groupbyed]
    for x in group_index:
        np_rng.shuffle(x)  # in-place
    nums = (num_sample - 1) // len(group_index) + 1
    group_index_concated = np.concatenate([x[:nums] for x in group_index])
    np_rng.shuffle(group_index_concated)
    selected_index = group_index_concated[:num_sample]

    examples = []
    for x in selected_index:
        sentence = data[x]['sentence']
        example_formated = prompt.format(en2zh_labels[data[x]['label_desc']], sentence)[:max_len]
        examples.append(example_formated)
    ret = {
        'zero_shot': '',
        'one_shot': examples[0] + '\n',
        'few_shot': ('\n'.join(examples)) + '\n',
    }
    return ret


def get_data_generate(data_path, en2zh_labels, tokenizer, shot_examples, pad_token="<pad>", seq_length=1024,
                      np_rng=None, task="one_shot", prompt="这是关于{}的文章：{}"):
    """gen data instances"""
    with open(data_path, 'r', encoding="utf-8") as fid:
        data = [json.loads(x) for x in fid.readlines()]

    label_list = sorted({en2zh_labels[x['label_desc']] for x in data})
    id_to_label = dict(enumerate(label_list))
    print('All test case num ', len(data))

    example = shot_examples[task]

    pad_id = tokenizer.encoder[pad_token]

    for instance in data:
        true_label = en2zh_labels[instance['label_desc']]
        tmp0 = sorted(list(set(id_to_label.values()) - {true_label}))
        fake_label = [tmp0[x] for x in np_rng.permutation(len(tmp0))[:NUM_LABELS - 1]]  # [:3]
        pre_instance_tf_label = [true_label] + fake_label
        instance_tf_label = []
        shuffle_idx = np_rng.permutation(len(instance_tf_label))
        for x in shuffle_idx:
            instance_tf_label.append(pre_instance_tf_label[x])
        input_ids_list = []
        mask_list = []
        label_list = []
        instance_res = {}
        for label_i in instance_tf_label:
            prompt_with_shot_example = "{}" + prompt
            tmp0 = tokenizer.tokenize(prompt_with_shot_example.format(example, label_i, ""))
            tmp1 = prompt_with_shot_example.format(example, label_i, instance['sentence'])
            input_ids = tokenizer.encode(tmp1)[:seq_length]
            mask = np.zeros(seq_length)
            mask[len(tmp0):len(input_ids)] = 1
            input_ids = np.pad(input_ids, ((0, seq_length - len(input_ids)),), 'constant', constant_values=(0, pad_id))
            input_ids_list.append(input_ids)
            mask_list.append(mask)
            label_list.append(label_i)

        true_label = [x for x, y in enumerate(label_list) if y == true_label][0]
        instance_res["input_ids"] = input_ids_list
        instance_res["labels"] = true_label
        instance_res["attention_mask"] = mask_list

        yield instance_res


def write_instance_to_file(writer, instance, need_del_keys):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]
    attention_mask = instance["attention_mask"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    features["attention_mask"] = np.asarray(attention_mask).astype(np.int32)

    for need_del_key in need_del_keys:
        del features[need_del_key]

    writer.write_raw_data([features])
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./tnews_public/dev.json", required=True,
                        help="Input raw text file. ")
    parser.add_argument("--label_file", type=str, default="./tnews_public/labels.json", required=True,
                        help="Input label file. ")
    parser.add_argument("--output_file", type=str, default="./tnews.mindrecord", required=True,
                        help="Output MindRecord file which ends with '.mindrecord' ")
    parser.add_argument("--num_splits", type=int, default=1,
                        help="The MindRecord file will be split into the number of partition. ")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length. ")
    parser.add_argument("--tokenizer_type", type=str, default="pangualpha_2_6b",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")
    parser.add_argument("--mindrecord_schema", type=str, default="pangualpha_tnews",
                        help="The name of mindrecord_schema. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "labels", "attention_mask"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")
    parser.add_argument("--n_shot", type=str, default="one_shot", choices=['zero_shot', 'one_shot', 'five_shot'],
                        help="N shot learning of t-news. ")
    parser.add_argument("--prompt_text", type=str, default="这是关于{}的文章：{}",
                        help="Input String format. ")
    args = parser.parse_args()

    np_rng = np.random.default_rng(seed=2)

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)
    en2zh_labels = get_en2zh_label(args.label_file)
    shot_to_example = load_tnews_example_for_shot(args.input_file, en2zh_labels, np_rng=np_rng, prompt=args.prompt_text)
    data_generate = get_data_generate(args.input_file, en2zh_labels=en2zh_labels, tokenizer=tokenizer,
                                      shot_examples=shot_to_example, seq_length=args.max_length,
                                      np_rng=np_rng, task=args.n_shot, prompt=args.prompt_text)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", args.output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]}
                   }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema, args.mindrecord_schema)

    total_written = 0

    for instance in data_generate:
        write_instance_to_file(writer, instance=instance, need_del_keys=need_del_keys)
        total_written += 1

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    main()
