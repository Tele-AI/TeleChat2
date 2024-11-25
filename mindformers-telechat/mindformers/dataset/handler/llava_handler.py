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
"""Llava Dataset Handler."""
import os
import json

from mindformers.tools import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.dataset.handler.base_handler import BaseInstructDataHandler


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class LlavaInstructDataHandler(BaseInstructDataHandler):
    """llava data handler"""
    def handle(self, dataset):
        """data handler"""
        image_dir = self.config.image_dir

        def convert_row_data(example):
            conversation = example.get("conversations")
            image = example.pop("image")
            if convert_conversations(conversation, image_dir, image, "user", "assistant"):
                example["conversations"] = json.dumps(example["conversations"])
                return example

            logger.info(
                f"{image} in conversation is not found! id={example.get('id')}, this data will be discarded.")
            return None

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
                    conversation["value"] = \
                        conversation["value"].replace("<image>\n", f"Picture 1: <img>{relative_img_path}</img>\n")
                elif "\n<image>" in conversation.get("value"):
                    conversation["value"] = \
                        conversation["value"].replace("\n<image>", f"Picture 1: <img>{relative_img_path}</img>\n")
            return True

        dataset = dataset.map(convert_row_data)

        if self.output_columns:
            remove_col_names = list(set(dataset.column_names) - set(self.output_columns))
            dataset = dataset.remove_columns(remove_col_names)
        return dataset

    def format_func(self, example):
        logger.info(f"nothing to do")
