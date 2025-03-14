# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test common dataloader"""

import os
import json
from copy import deepcopy
import pytest

from datasets import Dataset

from mindspore.dataset import GeneratorDataset

from mindformers.tools.register.config import MindFormerConfig
from mindformers.dataset.dataloader.common_dataloader import CommonDataLoader
from mindformers.dataset.dataloader.ms_ds_convertor import MSDatasetAdaptor

from tests.utils.path_utils import get_target_file_path

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

llama2_tokenizer = get_target_file_path(
    os.path.abspath(__file__), 5, "utils/llama2_tokenizer/tokenizer.model")
config_dict = dict(
    data_loader=dict(
        type='CommonDataLoader',
        load_func='load_dataset',
        path=None,
        data_files='',
        packing=None,
        handler=None,
        adaptor_config=dict(compress_mask=False),
        shuffle=False),
    input_columns=["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"],
    construct_args_key=["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"],
    num_parallel_workers=8,
    drop_remainder=True,
    repeat=1,
    seed=0,
    prefetch_size=1,
    numa_enable=False,
)
global_config = MindFormerConfig(**config_dict)


def _generate_alpaca_samples(num_samples=10):
    """generate alpaca samples and save to json"""
    sample = [{
        "instruction": "X" * 50,
        "input": "1024",
        "output": "Y" * 200
    }] * num_samples

    with open(f"{WORK_DIR}/samples.json", 'w') as fp:
        json.dump(sample, fp, indent=2)
    return f"{WORK_DIR}/samples.json"


def build_dataloader(config):
    """build dataloader with config"""
    return CommonDataLoader(
        column_names=config.input_columns,
        **config.data_loader
    )


def get_packing_alpaca_config(config, packing):
    """get packing config for alpaca dataset"""
    config.data_loader.path = 'json'
    json_path = _generate_alpaca_samples()

    config.data_loader.data_files = json_path
    config.data_loader.handler = [
        MindFormerConfig(**dict(
            type='AlpacaInstructDataHandler',
            seq_length=256,
            tokenizer=dict(
                unk_token='<unk>',
                bos_token='<s>',
                eos_token='</s>',
                pad_token='<unk>',
                type='LlamaTokenizer',
                vocab_file=llama2_tokenizer
            ),
            output_columns=["input_ids", "labels"]))
    ]
    if packing:
        config.data_loader.handler.append(
            MindFormerConfig(**dict(
                type='PackingHandler',
                seq_length=256,
                output_columns=["input_ids", "labels", "actual_seq_len"]
            )))

    return config


class TestCommonDataLoader:
    """test class for CommonDataLoader"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_json(self):
        """test load json with CommonDataLoader"""
        config = deepcopy(global_config)

        config.data_loader.path = 'json'
        json_path = _generate_alpaca_samples()
        config.data_loader.data_files = json_path

        dataloader = build_dataloader(config)
        assert isinstance(dataloader, GeneratorDataset)
        assert isinstance(dataloader.source, MSDatasetAdaptor)
        assert isinstance(dataloader.source.dataset, Dataset)

        sample = dataloader[0].sort()
        with open(json_path, 'r') as fp:
            src_sample = json.load(fp)[0]
        src_sample = list(src_sample.values()).sort()
        assert sample == src_sample

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_alpaca_handler(self):
        """test alpaca handler with CommonDataLoader"""
        # set config
        config = deepcopy(global_config)
        config = get_packing_alpaca_config(config, False)

        # build dataloader
        dataloader = build_dataloader(config)
        dataset = dataloader.source.dataset
        assert dataset.column_names == ['input_ids', 'labels']
        assert dataset.num_rows == 10

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_alpaca_pack(self):
        """test pack alpaca with CommonDataLoader"""
        # set config
        config = deepcopy(global_config)
        config.data_loader.packing = 'pack'
        config = get_packing_alpaca_config(config, True)

        # build dataloader
        dataloader = build_dataloader(config)
        dataset = dataloader.source.dataset
        assert dataset.column_names == ['input_ids', 'labels', 'actual_seq_len']
        assert dataset.num_rows == 10

        # test pack example
        assert dataset[0]['input_ids'][-1] == 0
        assert dataset[0]['labels'][-1] == -100

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_alpaca_truncate(self):
        """test pack alpaca with CommonDataLoader"""
        # set config
        config = deepcopy(global_config)
        config.data_loader.packing = 'truncate'
        config = get_packing_alpaca_config(config, True)

        # build dataloader
        dataloader = build_dataloader(config)
        dataset = dataloader.source.dataset
        assert dataset.column_names == ['input_ids', 'labels', 'actual_seq_len']
        assert dataset.num_rows == 9

        # test truncate example
        assert dataset[0]['input_ids'][-1] != 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pack_compress_mask(self):
        """test use compress mask in CommonDataLoader"""
        # set config
        config = deepcopy(global_config)
        config.data_loader.packing = 'pack'
        config.data_loader.adaptor_config.compress_mask = True
        config = get_packing_alpaca_config(config, True)

        # build dataloader
        dataloader = build_dataloader(config)
        print(dataloader[0][-1].size)
        assert len(dataloader[0]) == 5
        assert dataloader[0][-1].size == 128

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pack_wo_compress_mask(self):
        """test not use compress mask in CommonDataLoader"""
        # set config
        config = deepcopy(global_config)
        config.data_loader.packing = 'pack'
        config = get_packing_alpaca_config(config, True)

        # build dataloader
        dataloader = build_dataloader(config)
        print(dataloader[0][-1].shape)
        assert len(dataloader[0]) == 5
        assert dataloader[0][-1].shape == (1, 256, 256)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_from_disk(self):
        """test CommonDataLoader load from disk"""
        # generate HF dataset
        config = deepcopy(global_config)
        config.data_loader.packing = 'pack'
        config = get_packing_alpaca_config(config, True)
        dataloader = build_dataloader(config)
        dataset = dataloader.source.dataset
        dataset.save_to_disk(f"{WORK_DIR}/packed_data")

        # set config
        config = deepcopy(global_config)
        config.data_loader.path = f"{WORK_DIR}/packed_data"
        config.data_loader.load_func = 'load_from_disk'

        # build dataloader
        dataloader = build_dataloader(config)
        dataset = dataloader.source.dataset
        assert dataset.column_names == ['input_ids', 'labels', 'actual_seq_len']
        assert dataset.num_rows == 10
