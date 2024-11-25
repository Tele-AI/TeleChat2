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
"""DataReaders."""
import os
import re
import json


def squad_reader(path):
    """Reading the SQUAD dataset."""
    path = os.path.realpath(path)
    with open(path) as f:
        file = json.load(f)
    sources = []
    targets = []
    for data in file["data"]:
        for paragraph in data["paragraphs"]:
            passage = paragraph["context"]
            query = paragraph["qas"][0]["question"]
            answer = paragraph["qas"][0]["answers"][0]["text"]
            input_str = f"Read the passage and answer the question below.\n\n" \
                        f"### Instruction:\n{passage}\n\n### Input:\n{query}\n\n### Response:"
            sources.append(input_str)
            targets.append(answer)
    return dict(sources=sources, targets=targets)


def cmrc2018_reader(path):
    """Reading the CMRC2018 dataset."""
    path = os.path.realpath(path)
    with open(path) as f:
        file = json.load(f)
    prompts = []
    answers = []
    for data in file["data"]:
        for paragraph in data["paragraphs"]:
            context_ = paragraph["context"]
            qa = paragraph["qas"][0]
            query = qa["question"]
            answer = qa["answers"][0]["text"]
            input_str = f"阅读文章：{context_}\n问：{query}\n答："
            prompts.append(input_str)
            answers.append(answer)
    return dict(prompts=prompts, answers=answers)


def agnews_reader(path):
    """Reading the AG-News dataset."""
    path = os.path.realpath(path)
    sentences = []
    labels = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            label, title, description = line.strip().split("\",\"")
            sentence = title + ". " + description
            label = label.strip("\"")
            sentences.append(sentence.strip("\""))
            labels.append(int(label) - 1)
    return dict(sentence=sentences, label=labels)


def wikitext_reader(path):
    """Reading wikitext datasets. Returns a list of many sentences."""
    path = os.path.realpath(path)

    def wikitext_clean(string):
        """ string clean """
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
        string = string.replace(" .", ".")
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

    def preprocess_data(input_file):
        """preprocess data."""
        dataset_valid = []
        passage = []
        count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if line.startswith('=') and line.endswith('=') and passage:
                        dataset_valid.append(passage)
                        count += 1
                        passage = []
                    elif line.startswith('=') and line.endswith('='):
                        continue
                    else:
                        passage.append(line)
        res = []
        for line in dataset_valid:
            text = ""
            for sentence in line:
                sentence = wikitext_clean(sentence)
                text = text + " " + sentence
            text = text.strip()
            res.append(text)
        return res

    dataset_valid = preprocess_data(path)
    text_total = "\n".join(dataset_valid)
    sentence = text_total.strip().split("\t")
    return dict(sentence=sentence)


_DATA_READER_MAP = {
    "squad": squad_reader,
    "cmrc2018": cmrc2018_reader,
    "ag-news": agnews_reader,
    "wikitext": wikitext_reader,
}
