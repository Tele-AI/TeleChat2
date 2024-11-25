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
"""Data process for CMRC2018 Dataset"""
import re
import json
import argparse
import collections
import logging
import numpy as np

from mindspore.mindrecord import FileWriter

from mindformers import AutoTokenizer


np.random.seed(666)


def gen_prompt(prompts_, n_shot, len_ori, max_length=1024):
    if n_shot == 0:
        res = ""
    else:
        ids = np.random.choice(len(prompts_), n_shot).tolist()
        res = ""
        for i in ids:
            if len(res) + len(prompts_[i]) < max_length - len_ori - 10:
                res = res + prompts_[i]
    return res


def cut_sent(para):
    para = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # del extra \n
    return para.split("\n")


class Cmrc2018Dataset:
    """dataset process"""
    def __init__(self, cmrc2018_train_json, cmrc2018_dev_json):
        with open(cmrc2018_train_json, "r", encoding="utf-8") as f:
            self.train_data = json.load(f)["data"]
        with open(cmrc2018_dev_json, "r", encoding="utf-8") as f:
            self.dev_data = json.load(f)["data"]

    def gen_prompt_from_train(self, seg=False):
        """gen prompt from train dataset"""
        data_list = self.train_data
        index = 0
        prompts_ = []
        if seg:
            for data in data_list:
                context = data["paragraphs"][0]["context"]
                context_splits = cut_sent(context)
                qas = data["paragraphs"][0]["qas"]
                for qa in qas:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    for sent in context_splits:
                        if a in sent:
                            prompt = f"阅读文章：{sent}\n问：{q}\n答：{a}\n"
                            prompts_.append(prompt)
            prompt_max_length = 80
            prompts_ = [x for x in prompts_ if len(x) < prompt_max_length]
        else:
            for data in data_list:
                context = data["paragraphs"][0]["context"]
                qas = data["paragraphs"][0]["qas"]
                for qa in qas:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    prompt = f"阅读文章：{context}\n问：{q}\n答：{a}\n"
                    prompts_.append(prompt)
            prompt_max_length = 400
            prompts_ = [x for x in prompts_ if len(x) < prompt_max_length]
        return prompts_

    def get_data_allshot(self, n_content=50, n_q=1, max_length=1024):
        """gen data for all shots"""
        prompts_seg = self.gen_prompt_from_train(seg=True)
        prompts_full = self.gen_prompt_from_train(seg=False)
        input_prompts = []
        answers = []
        data_list = self.dev_data
        for data in data_list[:n_content]:
            context_ = data["paragraphs"][0]["context"]
            qas = data["paragraphs"][0]["qas"]
            for qa in qas[:n_q]:
                q = qa["question"]
                a = qa["answers"][0]["text"]
                input_str0 = f"阅读文章：{context_}\n问：{q}\n答："
                demo0 = ""
                demo1 = gen_prompt(prompts_full, n_shot=1, len_ori=len(input_str0), max_length=max_length)
                demo2 = gen_prompt(prompts_seg, n_shot=5, len_ori=len(input_str0), max_length=max_length)
                input_prompts.append([demo0 + input_str0,
                                      demo1 + input_str0,
                                      demo2 + input_str0])
                answers.append(a)
        return input_prompts, answers

    def get_data(self, n_content=50, n_q=1, n_shot=0):
        """get finally data"""
        prompts_seg = self.gen_prompt_from_train(seg=True)
        prompts_full = self.gen_prompt_from_train(seg=False)
        input_prompts = []
        answers = []

        data_list = self.dev_data
        for data in data_list[:n_content]:
            context_ = data["paragraphs"][0]["context"]
            qas = data["paragraphs"][0]["qas"]
            for qa in qas[:n_q]:
                q = qa["question"]
                a = qa["answers"][0]["text"]
                input_str0 = f"阅读文章：{context_}\n问：{q}\n答："
                if n_shot == 0:
                    demo = ""
                elif n_shot == 1:
                    demo = gen_prompt(prompts_full, n_shot=1, len_ori=len(input_str0))
                else:
                    demo = gen_prompt(prompts_seg, n_shot=n_shot, len_ori=len(input_str0))
                input_prompts.append(demo + input_str0)
                answers.append(a)
        return input_prompts, answers


def write_instance_to_file(writer, instance, need_del_keys):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    attention_mask = instance["attention_mask"]
    label = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["attention_mask"] = np.asarray(attention_mask).astype(np.int32)
    features["labels"] = np.asarray(label).astype(np.int32)

    for need_del_key in need_del_keys:
        del features[need_del_key]

    writer.write_raw_data([features])
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./cmrc2018_public/train.json", required=True,
                        help='Input raw train file. ')
    parser.add_argument("--dev_file", type=str, default="./cmrc2018_public/dev.json", required=True,
                        help='Input raw dev file. ')
    parser.add_argument("--output_file", type=str, default="./cmrc2018.mindrecord", required=True,
                        help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help='The MindRecord file will be split into the number of partition. ')
    parser.add_argument("--max_length", type=int, default=1024, help='Maximum sequence length. ')
    parser.add_argument("--n_shot", type=str, default="zero_shot", choices=['zero_shot', 'one_shot', 'five_shot'],
                        help="N shot learning of cmrc2018. ")
    parser.add_argument("--tokenizer_type", type=str, default="pangualpha_2_6b",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")
    parser.add_argument("--n_content", type=int, default=500, help="How many contents are sampled. ")
    parser.add_argument("--n_q", type=int, default=100, help="How many questions per content as most are sampled. ")
    parser.add_argument("--mindrecord_schema", type=str, default="pangualpha_cmrc2018",
                        help="The name of mindrecord_schema. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "labels"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. "
                             "Please note that 'labels' does input model.construct, it only used at train.evaluate. ")

    args = parser.parse_args()

    dev_file = args.dev_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", dev_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)

    shot_dict = {'zero_shot': 0, 'one_shot': 1, 'five_shot': 2}

    dataset_cmrc2018 = Cmrc2018Dataset(cmrc2018_train_json=args.train_file, cmrc2018_dev_json=args.dev_file)
    prompts, answers = dataset_cmrc2018.get_data_allshot(
        n_content=args.n_content, n_q=args.n_q, max_length=args.max_length)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", args.output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}
                   }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema, args.mindrecord_schema)

    total_written = 0

    for prompt, answer in zip(prompts, answers):
        prompt = prompt.get(shot_dict.get(args.n_shot))

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # this logic refers the open source
        # https://gitee.com/foundation-models/tk-models/blob/master/models/pangu_alpha/cmrc2018/src/evaluate_main.py
        if len(input_ids) >= args.max_length - 4:
            truncated_length = (args.max_length // 100 - 1) * 100
            input_ids = input_ids[-truncated_length:]

        attention_mask = [1] * len(input_ids)
        pad_len = args.max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        input_ids = np.array(input_ids).reshape(1, -1)
        attention_mask = np.array(attention_mask).reshape(1, -1)

        label_id = tokenizer.encode(answer, add_special_tokens=False)
        label_id = np.array(label_id).reshape(1, -1)

        instance = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_id}

        features = write_instance_to_file(writer, instance=instance, need_del_keys=need_del_keys)
        total_written += 1

        display_num = 20  # how many example to display
        if total_written <= display_num:
            logging.info("***** Example *****")
            logging.info("input tokens: %s", tokenizer.decode(instance["input_ids"]))
            logging.info("label tokens: %s", tokenizer.decode(instance["labels"]))

            for feature_name in features.keys():
                feature = features[feature_name]
                logging.info("%s: %s", feature_name, feature)

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    main()
