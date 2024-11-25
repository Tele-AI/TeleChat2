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

"""convert dataset to mindrecord"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.tools.logger import logger
from mindformers import AutoTokenizer, MindFormerBook


def create_instance(ds_name, tokenizer, text, max_length):
    """A single sample instance for LM task."""
    if ds_name == 'cola':
        _, label, _, sentence = text.strip().split("\t")
    elif ds_name == 'ag_news':
        label, title, description = text.strip().split("\",\"")
        sentence = title + ". " + description
        label = label.strip("\"")
        sentence = sentence.strip("\"")
        label = int(label) - 1
    elif ds_name == 'imdb':
        sentence = text.strip()[:-9]
        label = text.strip()[-8:]
        sentence = sentence.strip("\"")
        label = 1 if label == 'positive' else 0
    else:
        sentence, label = text.strip().split("\t")

    output = tokenizer(sentence, padding='max_length', max_length=max_length)
    output['labels'] = label
    return output


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    attention_mask = instance["attention_mask"]
    label = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["attention_mask"] = np.asarray(attention_mask).astype(np.int32)
    features["labels"] = np.asarray(label).astype(np.int32)

    writer.write_raw_data([features])
    return features


def main():
    dataset_support_list = ['cola', 'sst_2', 'ag_news', 'imdb']
    tokenizer_support_list = list(MindFormerBook.get_tokenizer_url_support_list().keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help='Dataset name. Now only supports [cola, sst_2, ag_news, imdb]. ',
                        choices=dataset_support_list)
    parser.add_argument("--tokenizer_type",
                        type=str,
                        default="gpt2",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ",
                        choices=tokenizer_support_list)
    parser.add_argument("--data_columns", type=list, default=["input_ids", "labels", "attention_mask"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")
    parser.add_argument("--input_file", type=str, required=True, help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, required=True, help='Output MindRecord file. ')
    parser.add_argument("--max_length", type=int, default=1024, help='Maximum sequence length. ')
    parser.add_argument("--header", type=bool, default=True, help='Has header or not. ')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)

    input_file = args.input_file
    logger.info("***** Reading from input files *****")
    logger.info("Input File: %s", input_file)

    output_file = args.output_file
    logger.info("***** Writing to output files *****")
    logger.info("Output File: %s", output_file)

    writer = FileWriter(output_file)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}
                   }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema)

    total_written = 0
    total_read = 0

    logger.info("***** Reading from  %s *****", input_file)
    with open(input_file, "r") as f:
        if args.header:
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            total_read += 1
            if total_read % 500 == 0:
                logger.info("%d ...", total_read)

            output = create_instance(args.dataset_name, tokenizer, line, args.max_length)
            features = write_instance_to_file(writer, instance=output)
            total_written += 1

            if total_written <= 20:
                logger.info("***** Example *****")
                logger.info("input tokens: %s", tokenizer.decode(output["input_ids"]))
                logger.info("label: %s", output["labels"])

                for feature_name in features.keys():
                    feature = features[feature_name]
                    logger.info("%s: %s", feature_name, feature)

    writer.commit()
    logger.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    main()
