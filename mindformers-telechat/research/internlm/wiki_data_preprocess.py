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
transform wikitext-2 dataset to mindrecord.
"""
import argparse
import os
import numpy as np

from mindspore.mindrecord import FileWriter

from internlm_tokenizer import InternLMTokenizer


def tokenize_wiki(tokenizer, file_path, seq_length, min_length):
    """tokenize wikitext-2 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                line_ids = tokenizer(stripped_line)["input_ids"]
                if len(line_ids) >= min_length:
                    content.append(line_ids)

    for ids in content:
        sample = {}
        if len(ids) < seq_length:
            ids = np.pad(ids, (0, seq_length - len(ids)), 'constant', constant_values=(2, 2))
        else:
            ids = ids[:seq_length]
        sample['input_ids'] = np.array(ids, dtype=np.int32)
        yield sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindrecord_schema", type=str, default="internlm_wiki")
    parser.add_argument("--input_glob", type=str, default="./wikitext-2/wiki.train.tokens")
    parser.add_argument("--output_file", type=str, default="./wiki_processed/wiki.mindrecord")
    parser.add_argument("--model_file", type=str, default="./tokenizer.model")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--min_length", type=int, default=50)
    args = parser.parse_args()
    # pylint: disable=C0326
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]}, }
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.set_page_size(256 * 1024 * 1024)
    writer.add_schema(schema, args.mindrecord_schema)

    # Start to load tokenizer
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"file {args.model_file} do not exists.")

    transforms_count = 0
    word_tokenizer = InternLMTokenizer(vocab_file=args.model_file)

    for x in tokenize_wiki(word_tokenizer, args.input_glob, args.seq_length + 1, args.min_length):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
