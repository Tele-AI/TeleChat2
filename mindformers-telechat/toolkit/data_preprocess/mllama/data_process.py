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
import string
import stat
from datasets import load_from_disk


def generate_random_code(length=6):
    """
    generate random id。
    """
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return code


def load_data(dataset_dir):
    with open(dataset_dir, 'r') as f:
        data = json.load(f)
    return data


def convert_data_json(dataset_dir, output_file, image_pos_tag=None):
    """generate json files."""
    if image_pos_tag is None:
        image_pos_tag = ('<|reserved_special_token_3|>', '<|reserved_special_token_4|>')
    text_pos_tag = "<|text|>"
    data_set = load_from_disk(dataset_dir)
    result = []
    images_dir = os.path.join(output_file, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for data in data_set:
        conversations = []
        data_id = generate_random_code(8)
        image_list, sample_list = data["images"], data["texts"]
        if len(image_list) > 1:
            raise ValueError("Only support one image per sample")
        image = image_list[0].convert("RGB")
        image_path = f"{images_dir}/{data_id}.png"
        image.save(image_path, lossless=True)
        for sample_dict in sample_list:
            if not conversations:
                # only append image to the first sentence
                image_info = f"{image_pos_tag[0]}{os.path.join(output_file, image_path)}{image_pos_tag[1]}"

                conversations += [{"from": "user", "value": image_info + "<|image|>" + sample_dict["user"].strip()},
                                  {"from": "assistant", "value": text_pos_tag + sample_dict["assistant"].strip()}]
            else:
                conversations += [{"from": "user", "value": text_pos_tag + sample_dict["user"].strip()},
                                  {"from": "assistant", "value": text_pos_tag + sample_dict["assistant"].strip()}]
        result.append({"id": data_id, "conversations": conversations})

    print(len(result))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR
    output_file_path = os.path.join(output_file, "train_data.json")
    with os.fdopen(os.open(output_file_path, flags, mode), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="path/dataset/vision/ocrvqa/")
    parser.add_argument("--output_file", type=str, default="path/dataset/vision/ocrvqa_ms/")
    args = parser.parse_args()

    convert_data_json(args.data_dir, args.output_file)
