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
transform common_voice dataset to mindrecord.
"""

import argparse
import os
import csv
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from mindformers.models.whisper.processing_whisper import WhisperFeatureExtractor
from mindformers.models.whisper.tokenization_whisper import WhisperTokenizer


feature_extractor = WhisperFeatureExtractor()


def get_writer(output_file):
    schema = {'input_features': {"type": "float32", "shape": [-1, 3000]},
              'decoder_input_ids': {"type": "int32", "shape": [-1]},
              'encoder_dropout_probability': {"type": "float32", "shape": [-1]},
              'decoder_dropout_probability': {"type": "float32", "shape": [-1]},
              }
    writer1 = FileWriter(file_name=output_file)
    writer1.add_schema(schema, "common_voice")
    return writer1


def get_tokenizer(path):
    return WhisperTokenizer.from_pretrained(path)


def pad(input_ids, seq_length, tokenizer1):
    dic = tokenizer1.pad({"input_ids": input_ids}, max_length=seq_length, padding="max_length", return_tensors="np")
    input_ids = dic.input_ids
    attention_mask = np.logical_not(dic.attention_mask.astype(np.bool_))
    input_ids[attention_mask] = 50256
    return input_ids


def read_datasets(tsv_file, mp3_dir, tokenizer1):
    """read datasets"""
    with open(tsv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in tqdm(reader):
            mp3_file = os.path.join(mp3_dir, row["path"])
            text = row["sentence"]

            input_features = feature_extractor(mp3_file)
            input_ids = tokenizer1(text).input_ids

            yield input_features, input_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir', type=str)
    parser.add_argument('--tsv_file', type=str)
    parser.add_argument('--tokenizer_dir', type=str)
    parser.add_argument('--output_file', type=str, default='./dataset/common_voice')
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--language', type=str, default="Hindi")
    parser.add_argument('--task', type=str, default="transcribe")
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    writer = get_writer(args.output_file)
    tokenizer = WhisperTokenizer.from_pretrained(args.tokenizer_dir, language=args.language, task=args.task)

    for i, data in enumerate(read_datasets(args.tsv_file, args.mp3_dir, tokenizer)):
        minddata = {
            "input_features": data[0],
            "decoder_input_ids": pad(data[1], args.seq_length + 1, tokenizer).astype(np.int32),
            "encoder_dropout_probability": np.array(np.random.rand(32), dtype=np.float32),
            "decoder_dropout_probability": np.array(np.random.rand(32), dtype=np.float32)
        }
        writer.write_raw_data([minddata])

    writer.commit()
