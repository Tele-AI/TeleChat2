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
import argparse
import json
import math
import os
import random
import stat

import yaml
import cv2
import numpy as np
from tqdm import tqdm
from mindformers.tools.check_rules import check_yaml_depth_before_loading


# pylint: disable=C0111
def check_video_valid(video_file_path, image_file):
    video_file = os.path.join(video_file_path, image_file)
    cap = cv2.VideoCapture(video_file)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = np.linspace(0, frame_num - 1, 2, dtype=np.int32)
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, _ = cap.read()
        if not ret:
            cap.release()
            print(video_file)
            return False
    cap.release()
    return True


# pylint: disable=C0111
def check_data_file(image_list, video_path_list):
    check_ = False
    if not isinstance(image_list, list):
        image_list = [image_list]
    no_exist_file_list = []
    for image in image_list:
        for video_path in video_path_list:
            if os.path.isfile(os.path.join(video_path, image)):
                check_ = True
                return check_, video_path
            no_exist_file_list.append(os.path.join(video_path, image))
    no_exist_file_str = ", ".join(no_exist_file_list)
    print(f"Image files: {no_exist_file_str} are/is not exists, skip!!!")
    return check_, None


# pylint: disable=C0111
def load_data(dataset_dir):
    with open(dataset_dir, 'r') as f:
        data = json.load(f)
    return data


def added_text_tag(conversation, video_pos_tag, text_tag="<text>"):
    """added text tag for imageprocessor to recognize"""
    for conv in conversation:
        if video_pos_tag[0] in conv.get("value"):
            continue
        else:
            conv["value"] = text_tag + conv.get("value")


# pylint: disable=C0111
def get_origin_dict(dataset_dir):
    list_data_dict = []
    video_or_image_file_path = []
    with open(dataset_dir, "r") as file:
        check_yaml_depth_before_loading(file)
        file.seek(0)
        yaml_data = yaml.safe_load(file)
        datasets = yaml_data.get("datasets")
        for dataset in datasets:
            json_path = dataset.get("json_path")
            sampling_strategy = dataset.get("sampling_strategy", "all")
            sampling_number = None
            print(f"Loading {json_path} with {sampling_strategy} sampling strategy")
            video_or_image_file_path.append(json_path[:json_path.rfind("/")])
            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(float(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                else:
                    sampling_number = int(sampling_number)

            # Apply the sampling strategy
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-int(sampling_number):]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            list_data_dict.extend(cur_data_dict)
    return list_data_dict, video_or_image_file_path


def convert_data_json(dataset_dir, output_file, video_pos_tag=None):
    """generate json files."""
    if video_pos_tag is None:
        video_pos_tag = ('<|reserved_special_token_3|>', '<|reserved_special_token_4|>')

    list_data_dict, video_or_image_file_path = get_origin_dict(dataset_dir)

    new_samples = []
    max_frames = 0
    for line in tqdm(list_data_dict):
        conversation = line.get("conversations")
        question = conversation[0].get("value")
        target_info = ""
        whether_add = True
        if "image" in line or "video" in line:
            image_list = line.get("image") if line.get("image", None) else line.get("video")
            if not isinstance(image_list, list):
                image_list = [image_list]
            check_, file_path = check_data_file(image_list, video_or_image_file_path)
            if check_:
                max_frames = max(len(image_list), max_frames)
                for image in image_list:
                    if "video" in line:
                        res = check_video_valid(file_path, image)
                    target_info += f"{video_pos_tag[0]}{os.path.join(file_path, image)}{video_pos_tag[1]}"
                target_info += question
            else:
                continue
        else:
            target_info = f"{video_pos_tag[0]}{video_pos_tag[1]}{question}"
        conversation[0]["value"] = target_info
        added_text_tag(conversation, video_pos_tag)
        if "video" in line:
            whether_add = res
        if whether_add:
            new_samples.append({"conversations": conversation})
    print(len(new_samples))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(output_file, flags, mode), 'w') as f:
        json.dump(new_samples, f, indent=2, ensure_ascii=False)
    print(f"max frames is {max_frames}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", type=str, default="data_process.yaml", help="data input yaml")
    parser.add_argument("--output_file", type=str, default="XX.json")
    args = parser.parse_args()
    convert_data_json(args.data_yaml, args.output_file)
