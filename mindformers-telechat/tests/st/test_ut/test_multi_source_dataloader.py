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
"""test multi-source dataloader."""
import os
import time

import numpy as np
# import pytest
from PIL import Image
from mindspore import context, nn, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P

from mindformers import MindFormerConfig
from mindformers.dataset import build_dataset
from mindformers.tools.logger import logger


def make_flickr_formate_dataset(dataset_root, dataset_index, sample_nums):
    """generate a fake Flickr8k dataset"""
    dataset_dir = os.path.join(dataset_root, f"Flickr8k_{dataset_index}")
    annotation_dir = os.path.join(dataset_dir, "Flickr8k_text")
    image_dir = os.path.join(dataset_dir, "Flickr8k_Dataset", "Flickr8k_Dataset")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    for index in range(sample_nums):
        image = Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8))
        image.save(os.path.join(image_dir, f"test_image_{dataset_index}_{index}.jpg"))

    token_file = os.path.join(annotation_dir, "Flickr8k.token.txt")
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(token_file, flags_, 0o750), "w", encoding="utf-8") as filer:
        for index in range(sample_nums):
            filer.write(f"test_image_{dataset_index}_{index}.jpg#\t{dataset_index} {index}\n")

    train_file = os.path.join(annotation_dir, "Flickr_8k.trainImages.txt")
    with os.fdopen(os.open(train_file, flags_, 0o750), "w", encoding="utf-8") as filer:
        for index in range(sample_nums):
            filer.write(f"test_image_{dataset_index}_{index}.jpg\n")

    test_file = os.path.join(annotation_dir, "Flickr_8k.testImages.txt")
    with os.fdopen(os.open(test_file, flags_, 0o750), "w", encoding="utf-8") as filer:
        for index in range(sample_nums):
            filer.write(f"test_image_{dataset_index}_{index}.jpg\n")

    dev_file = os.path.join(annotation_dir, "Flickr_8k.devImages.txt")
    with os.fdopen(os.open(dev_file, flags_, 0o750), "w", encoding="utf-8") as filer:
        for index in range(sample_nums):
            filer.write(f"test_image_{dataset_index}_{index}.jpg\n")
    return dataset_dir


def make_dataset(data_num_list, batch_size=32, samples_count=None, dataset_ratios=None):
    """generate dataset"""
    dataset_root = "./checkpoint_download/Flickr8k_MultiSourceTest"
    os.makedirs(dataset_root, exist_ok=True)

    sub_data_loader = []
    for index, data_num in enumerate(data_num_list):
        filepath = make_flickr_formate_dataset(dataset_root, index, data_num)
        sub_data_loader.append({"type": "Flickr8kDataLoader", "dataset_dir": filepath})

    data_loader = {
        "type": "MultiSourceDataLoader",
        "sub_data_loader_args":
            {
                "stage": "train",
                "column_names": ["image", "text"]
            },
        "sub_data_loader": sub_data_loader,
        "shuffle": "global",
        "shuffle_buffer_size": 256
    }

    if samples_count is not None:
        data_loader["samples_count"] = samples_count

    if dataset_ratios is not None:
        data_loader["dataset_ratios"] = dataset_ratios

    train_dataset = {
        "data_loader": data_loader,
        "num_parallel_workers": 8,
        "python_multiprocessing": False,
        "drop_remainder": True,
        "batch_size": batch_size,
        "repeat": 1,
        "numa_enable": False,
        "prefetch_size": 30,
        "seed": 2022,
        "tokenizer": {
            "type": "CLIPTokenizer",
            "pad_token": "!"
        },
        "text_transforms": {
            "type": "RandomChoiceTokenizerForward",
            "max_length": 77,
            "padding": "max_length",
            "random_seed": 2022
        },
        "transforms":
            [{
                "type": "Resize",
                "size": 224
            }, {
                "type": "CenterCrop",
                "size": 224
            }, {
                "type": "ToTensor"
            }, {
                "type": "Normalize",
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
                "is_hwc": False
            }]
    }
    train_dataset_task = {
        "type": "ContrastiveLanguageImagePretrainDataset",
        "dataset_config": train_dataset
    }
    config = MindFormerConfig(train_dataset=train_dataset, train_dataset_task=train_dataset_task)
    return build_dataset(config.train_dataset_task)


# @pytest.mark.level0
# @pytest.mark.platform_x86_cpu
# @pytest.mark.env_onecard
def test_multi_source_dataloader():
    """
    Feature: Multi-source Dataloader
    Description: Create Multi-source dataloader and iter it
    Expectation: No Exception
    """
    batch_size = 32
    epoch = 3
    dataset_samples_count = [200, 200, 200, 200]

    samples_count = 320
    dataset_ratios = [0.1, 0.2, 0.3, 0.4]
    dataset = make_dataset(dataset_samples_count,
                           samples_count=samples_count,
                           dataset_ratios=dataset_ratios,
                           batch_size=batch_size)
    for cur_epoch in range(epoch):
        iterated = 0
        for image, _ in dataset:
            iterated += image.asnumpy().shape[0]
        logger.info("epoch %s: %s items.", cur_epoch, iterated)
        assert samples_count - iterated < batch_size

    dataset = make_dataset(dataset_samples_count, batch_size=batch_size)
    for cur_epoch in range(epoch):
        iterated = 0
        for image, _ in dataset:
            iterated += image.asnumpy().shape[0]
        logger.info("epoch %s: %s items.", cur_epoch, iterated)
        assert sum(dataset_samples_count) - iterated < batch_size


def op_network_with_epoch(network, step_num):
    iter_num = 0
    network.set_train()
    for _ in range(step_num):
        _ = network()
        iter_num += 1
    return iter_num


def convert_type(shapes, types):
    ms_types = []
    for np_shape, np_type in zip(shapes, types):
        input_np = np.zeros(np_shape, np_type)
        tensor = Tensor(input_np)
        ms_types.append(tensor.dtype)
    return ms_types


def get_dataset_base_value(dataset):
    dataset_size = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()
    return dataset_size, batch_size


def get_dataset_shapes_and_types(dataset):
    dataset_shapes = dataset.output_shapes()
    np_types = dataset.output_types()
    dataset_types = convert_type(dataset_shapes, np_types)
    return dataset_shapes, dataset_types


class SingleOpNetwork(nn.Cell):
    def __init__(self, shapes):
        super(SingleOpNetwork, self).__init__()
        self.shapes = tuple(shapes[0])
        self.reshape = P.Reshape()

    def construct(self, network_input):
        return self.reshape(network_input, self.shapes)


class NetWithTDT(nn.Cell):
    def __init__(self, network, dataset_types, dataset_shapes, shared_name=""):
        super(NetWithTDT, self).__init__()
        self.get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_shapes), shared_name)
        self.network = network

    def construct(self):
        next_input, _ = self.get_next()
        return self.network(next_input)


def op_network_with_step_num(dataset, step_num):
    """run a network with specific step num"""
    dataset_shapes, dataset_types = get_dataset_shapes_and_types(dataset)
    batch_size = dataset.get_batch_size()
    dataset = dataset.device_que()
    queue_name = dataset.queue_name

    logger.info("dataset shapes %s, dataset types %s", dataset_shapes, dataset_types)

    net = SingleOpNetwork(dataset_shapes)
    net_with_dataset = NetWithTDT(net, dataset_types, dataset_shapes, queue_name)
    _cell_graph_executor.init_dataset(dataset.queue_name, 1, batch_size, dataset_types, dataset_shapes, (), "")
    time.sleep(1)
    dataset.send(1)
    return op_network_with_epoch(net_with_dataset, step_num)


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend910b_training
# @pytest.mark.env_onecard
def test_multi_source_dataloader_sink_mode():
    """
    Feature: Multi-source Dataloader sink mode
    Description: Send 10 data into tdt and count number of the out iter
    Expectation: Number of out iter == number of source data
    """
    context.set_context(mode=context.GRAPH_MODE)
    beyond_step_num = 10

    batch_size = 32
    dataset = make_dataset([500, 600, 700, 800], batch_size=batch_size)

    iter_num = op_network_with_step_num(dataset, step_num=beyond_step_num)
    logger.info("out_iter_num：%s", iter_num)
    assert iter_num == beyond_step_num
