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
"""generate rwf train data."""
import os
import json
import random
import argparse


def create_label_data(query, response, video_path):
    """create data with label."""
    label_data = {
        "query": query,
        "response": response,
        "videos": [video_path]
    }
    return label_data


def process_folder(input_folder):
    """generate data from data directory."""
    samples = []
    for subset in ['train', 'val']:
        for label in ['Fight', 'NonFight']:
            query = "判断视频中是否包含暴力内容，输出1为包含暴力内容，输出0为不包含暴力内容。"
            response = "1" if label == 'Fight' else "0"
            folder_path = os.path.join(input_folder, subset, label)

            for video_file in os.listdir(folder_path):
                if not video_file.endswith('.avi'):
                    continue
                video_path = os.path.join(folder_path, video_file)
                label_data = create_label_data(query, response, video_path)
                # samples.append(json.dumps(label_data, ensure_ascii=False))
                samples.append(label_data)
    return samples


def convert_rwf2000_json(dataset_dir, output_file, video_pos_tag=None):
    """generate json files."""
    if video_pos_tag is None:
        video_pos_tag = ('<|reserved_special_token_3|>', '<|reserved_special_token_4|>')

    random.seed(64)
    samples = process_folder(dataset_dir)
    random.shuffle(samples)

    generate_data = []
    for line in samples:
        target_file = line['videos'][0]
        question = line['query']
        answer = line['response']
        if len(target_file) > 70:
            continue
        if 'val' in target_file:
            continue

        target_file = target_file.split('/')
        target_file = os.path.join(dataset_dir, '/'.join(target_file[-3:]))
        target_info = f"{video_pos_tag[0]}{target_file}{video_pos_tag[1]}{question}"
        conversations = [{'from': 'user', 'value': target_info},
                         {'from': 'assistant', 'value': answer}]
        generate_data.append({
            'id': os.path.basename(target_file),
            'conversations': conversations
        })

    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(output_file, flags_, 0o750), 'w', encoding='utf-8') as f:
        json.dump(generate_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="RWF-2000/")
    parser.add_argument("--output_file", type=str, default="RWF-2000/tran.json")
    args = parser.parse_args()

    convert_rwf2000_json(args.data_dir, args.output_file)
