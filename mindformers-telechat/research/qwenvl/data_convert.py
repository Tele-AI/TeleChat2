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
convert data to finetune Qwen-VL
"""

import argparse
import json
import os


def convert_conversations(data, image_location, image, user_role_name, assistant_role_name):
    """convert conversations in a training sample"""
    relative_img_path = os.path.join("train2014", f"COCO_train2014_{image}")
    abs_img_path = os.path.join(image_location, relative_img_path)

    if not os.path.exists(abs_img_path):
        return False

    for conversation in data:
        if conversation.get("from") == "human":
            conversation["from"] = user_role_name
        elif conversation.get("from") == "gpt":
            conversation["from"] = assistant_role_name

        if "<image>\n" in conversation.get("value"):
            conversation["value"] = conversation["value"].replace("<image>\n",
                                                                  f"Picture 1: <img>{relative_img_path}</img>\n")
        elif "\n<image>" in conversation.get("value"):
            conversation["value"] = conversation["value"].replace("\n<image>",
                                                                  f"Picture 1: <img>{relative_img_path}</img>\n")
    return True


def main(data_path, image_location, output_path, user_role_name, assistant_role_name):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for data_item in data:
        conversation = data_item.get("conversations")
        image = data_item.pop("image")

        if convert_conversations(conversation, image_location, image, user_role_name, assistant_role_name):
            new_data.append(data_item)
        else:
            print(f"{image} in conversation is not found! id={data_item.get('id')}, this data will be discarded.")

    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(output_path, flags_, 0o750), "w", encoding="utf-8") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="detail_23k.json")
    parser.add_argument("--image_location", type=str, default="/data/coco2014/coco/images")
    parser.add_argument("--output_path", type=str, default="detail_23k_qwenvl_format.json")
    parser.add_argument("--user_role_name", type=str, default="user")
    parser.add_argument("--assistant_role_name", type=str, default="assistant")
    args = parser.parse_args()
    main(args.data_path, args.image_location, args.output_path, args.user_role_name, args.assistant_role_name)
