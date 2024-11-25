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
"""test multi-modal sft dataset."""
import json
import os
import random

import pytest
import numpy as np
from PIL import Image

from mindformers import MindFormerConfig, ModalContentTransformTemplate
from mindformers.dataset import build_dataset
from mindformers.models.multi_modal.modal_content import BaseTextContentBuilder, BaseImageContentBuilder
from mindformers.models.multi_modal.utils import DataRecord
from mindformers.tools import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TestContentTransformTemplate(ModalContentTransformTemplate):
    """modal transform template for testing"""
    def __init__(self, output_columns, tokenizer, image_pad_token, start_token, end_token, image_size=448,
                 num_queries=256, dataset_dir="", mode="predict", modal_content_padding_size=1, max_length=128,
                 **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode,
                         modal_content_padding_size=modal_content_padding_size, max_length=max_length, **kwargs)
        self.dataset_dir = dataset_dir

        self.modal_builders = {
            "image": BaseImageContentBuilder(image_pad_token, num_queries, use_custom_token=True,
                                             start_token=start_token, end_token=end_token, image_location=dataset_dir,
                                             modal_content_max_size=modal_content_padding_size, mode=mode,
                                             max_length=max_length, image_size=image_size),
            "text": BaseTextContentBuilder()
        }

    def build_conversation_input_text(self, raw_inputs, result_recorder: DataRecord):
        return [f"{item[0]}:{item[1]}" for item in raw_inputs]

    def build_labels(self, text_id_list, result_recorder, **kwargs):
        return np.concatenate(text_id_list)

    def get_need_update_output_items(self, result: DataRecord):
        update_items = {"images": self.modal_builders["image"].padding_images_to_max_content_size(result.get("images"))}
        return update_items


def split(x, n):
    """split x to sum of n parts"""
    if x < n:
        parts = [x]
        parts += [0] * (n - 1)
        return parts

    if x % n == 0:
        return [x // n] * n

    parts = []
    zp = n - (x % n)
    pp = x // n
    for i in range(n):
        if i >= zp:
            parts.append(pp + 1)
        else:
            parts.append(pp)
    return parts


def make_image_dataset(dataset_root, start_token, end_token,
                       image_dir_name="multi_modal_images",
                       annotation_filename="multi_modal_conversation.json",
                       sample_nums=32, image_size=224, max_image_num=5):
    """generate image conversation dataset"""
    image_dir = os.path.join(dataset_root, image_dir_name)
    annotation_files = os.path.join(dataset_root, annotation_filename)

    os.makedirs(image_dir, exist_ok=True)

    image_num_per_item = []
    annotation_content = []
    for index in range(sample_nums):
        conversations = []
        image_num = random.randint(0, max_image_num)
        conversation_user_num = random.randint(1, 3)
        add_images = 0
        images_per_conversation = split(image_num, conversation_user_num)
        for image_per in images_per_conversation:
            if image_per == 0:
                conversation = [{
                    "from": "user",
                    "value": "请问你是谁？"
                }, {
                    "from": "assistant",
                    "value": "我是一棵树"
                }]
            else:
                image_placeholder = [
                    f"{start_token}multi_modal_images/{index}_{add_images + i}.jpg{end_token}"
                    for i in range(image_per)]

                conversation = [{
                    "from": "user",
                    "value": f"{''.join(image_placeholder)}这几张图描述了什么内容？"
                }, {
                    "from": "assistant",
                    "value": "每张图上都有一棵树"
                }]
                add_images += image_per
            conversations += conversation

        annotation_content.append(
            {
                "id": f"identity_{index}",
                "conversations": conversations
            }
        )
        for image_num_ in range(image_num):
            image = Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8))
            image.save(os.path.join(image_dir, f"{index}_{image_num_}.jpg"))

        image_num_per_item.append(image_num)

    with open(annotation_files, "w", encoding="utf-8") as f:
        json.dump(annotation_content, f, ensure_ascii=False)
    return image_num_per_item


def make_dataset(batch_size, modal_content_max_size, image_size, max_length, num_queries, start_token, end_token,
                 img_pad_token="<unk>"):
    """generate dataset"""
    dataset_root = "./checkpoint_download/MultiModalDataset"
    image_dir_name = "multi_modal_images"
    annotation_filename = "multi_modal_conversation.json"
    os.makedirs(dataset_root, exist_ok=True)

    image_num_per_item = make_image_dataset(dataset_root, image_dir_name=image_dir_name, image_size=image_size,
                                            annotation_filename=annotation_filename,
                                            max_image_num=modal_content_max_size,
                                            end_token=end_token, start_token=start_token)
    data_loader = {
        "type": "BaseMultiModalDataLoader",
        "annotation_file": os.path.join(dataset_root, annotation_filename),
        "shuffle": False
    }

    train_dataset = {
        "data_loader": data_loader,
        "num_parallel_workers": 1,
        "python_multiprocessing": False,
        "drop_remainder": True,
        "batch_size": batch_size,
        "repeat": 1,
        "numa_enable": False,
        "prefetch_size": 1,
        "seed": 2022,
        "tokenizer": {
            "type": "LlamaTokenizer",
            "pad_token": "<pad>"
        },
        "modal_to_text_transform": {
            "type": "BaseXModalToTextTransform",
            "model_transform_template": {
                "type": "TestContentTransformTemplate",
                "output_columns": ["input_ids", "images", "image_context_pos", "labels"],
                "mode": "train",
                "dataset_dir": dataset_root,
                "modal_content_padding_size": modal_content_max_size,
                "image_size": image_size,
                "num_queries": num_queries,
                "image_pad_token": img_pad_token,
                "start_token": start_token,
                "end_token": end_token
            },
            "max_length": max_length
        },

        "modal_content_input_columns": ["images"],
        "modal_content_output_columns": ["images"],
        "modal_content_transforms":
            [{
                "type": "BatchToTensor",
            }, {
                "type": "BatchNormalize",
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
                "is_hwc": False
            }],
        "net_input_columns": ["input_ids", "images", "image_context_pos", "labels"]
    }
    train_dataset_task = {
        "type": "ModalToTextSFTDataset",
        "dataset_config": train_dataset
    }
    config = MindFormerConfig(train_dataset=train_dataset, train_dataset_task=train_dataset_task)
    return image_num_per_item, build_dataset(config.train_dataset_task)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_modal_dataloader():
    """
    Feature: Multi-modal Dataloader
    Description: Create Multi-modal dataloader and iter it
    Expectation: No Exception
    """
    batch_size = 4
    max_length = 256
    image_size = 224
    num_queries = 4
    modal_content_max_size = 3
    actual_image_num_per_item, dataset = make_dataset(batch_size, modal_content_max_size, image_size, max_length,
                                                      num_queries, start_token="<img>", end_token="</img>")

    img_pad_token_id = 0
    dataset_image_num_list = []
    image_num_in_pos_list = []
    for input_ids, images, image_context_pos, labels in dataset:
        # assert shape of data is correct
        assert input_ids.shape == (batch_size, max_length + 1)
        assert images.shape == (batch_size, modal_content_max_size, 3, image_size, image_size)
        assert image_context_pos.shape == (batch_size, modal_content_max_size, num_queries, 2)
        assert labels.shape == (batch_size, max_length + 1)

        input_ids = input_ids.asnumpy()
        dataset_image_num = ((input_ids == img_pad_token_id).sum(axis=1) / num_queries).astype(np.int32)
        dataset_image_num_list.append(dataset_image_num)

        image_context_pos = image_context_pos.asnumpy()
        image_context_pos = np.delete(image_context_pos, 0, 3).reshape((batch_size,
                                                                        num_queries * modal_content_max_size))

        image_num_in_pos = ((image_context_pos < max_length - num_queries).sum(axis=1) / num_queries).astype(np.int32)
        image_num_in_pos_list.append(image_num_in_pos)

    dataset_image_num_list = np.hstack(dataset_image_num_list).tolist()
    image_num_in_pos_list = np.hstack(image_num_in_pos_list).tolist()

    # assert num of images in the input_ids and context pos are correct
    assert dataset_image_num_list == actual_image_num_per_item
    assert image_num_in_pos_list == actual_image_num_per_item
