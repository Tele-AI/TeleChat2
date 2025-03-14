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
"""megatron blended dataloader."""

from enum import Enum
from typing import Callable, List, Union

import numpy as np
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore.communication.comm_func import barrier

from mindformers.dataset.blended_datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from mindformers.dataset.blended_datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from mindformers.dataset.blended_datasets.utils import get_blend_from_list
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_dp_from_dataset_strategy, get_real_group_size, get_real_rank


def is_dataset_built_on_rank() -> bool:
    """check which rank need to build dataset."""
    global_rank_id = get_real_rank()
    stage_num = ms.get_auto_parallel_context("pipeline_stages")
    total_device_num = get_real_group_size() // stage_num
    dp = get_dp_from_dataset_strategy()
    tp = int(total_device_num // dp)

    local_stage_num = int(global_rank_id // (dp * tp))

    # when not stage 0 or last stage, no need to build dataset.
    # pylint: disable=R1716
    if local_stage_num > 0 and local_stage_num < (stage_num - 1):
        return False

    # In tp group, only need one card to build dataset, others don't need to build dataset.
    if global_rank_id % tp != 0:
        return False

    return True


class DatasetPhaseType(str, Enum):
    r"""
        The enum of dataset phase.
    """
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class MegatronDatasetBuilder:
    """
    Megatron blended dataset builder.

    Args:
        datasets_type(str): megatron dataset types, e.g. GptDataset
        config (dict): configurations dict.
        tokenzier(Union[dict, Callable]): tokenizer config or tokenizer instance.

    Returns:
        MegatronDataset instance or Fakedataset instance.
    """
    # supported megatron dataset
    supported_megatron_datasets_type_list = ['GPTDataset']

    def __init__(self, datasets_type: str, config: dict, tokenizer=None) -> None:
        if datasets_type not in MegatronDatasetBuilder.supported_megatron_datasets_type_list:
            raise KeyError(f"{datasets_type} is not support now, please choose dataset type  \
                           in {MegatronDatasetBuilder.supported_megatron_datasets_type_list}")
        self.config = config
        self.datasets_type = datasets_type
        self.tokenizer = tokenizer

    def builder(self,
                sizes: List[int],
                column_names,
                shuffle: bool = False,
                num_shards: int = None,
                shard_id: int = None,
                phase: str = "train",
                **kwargs):
        """Megatron dataset builder.Now, only support GptDataset"""
        if self.datasets_type == "GPTDataset":
            return self.build_gptdataset_or_fakedataset(sizes, column_names, shuffle,
                                                        num_shards, shard_id, phase, **kwargs)
        return None

    # pylint: disable=W0613
    def build_gptdataset_or_fakedataset(self,
                                        sizes: List[int],
                                        column_names,
                                        shuffle: bool = False,
                                        num_shards: int = None,
                                        shard_id: int = None,
                                        phase: str = "train",
                                        **kwargs):
        """ build gpt dataset or fake dataset, return dataset obj."""
        # create BlendedMegatronDatasetConfig
        blended_config = self.init_gpt_dataset_config()
        if blended_config.sequence_length is None:
            raise ValueError("sequence_length cannot be None when init megatron dataset.")
        if not isinstance(blended_config.sequence_length, int) or blended_config.sequence_length <= 0:
            raise ValueError(f"sequence_length must be int and sequence_length must be greater than 0. \
                             But get {blended_config.sequence_length}")

        logger.info(f"blended_config: {blended_config}")

        def build_gpt_dataset():
            blended_megatron_dataset_builder = BlendedMegatronDatasetBuilder(GPTDataset, sizes, lambda: True, blended_config).build()

            if phase == DatasetPhaseType.TRAIN:
                blend_dataset = blended_megatron_dataset_builder[0]
            elif phase == DatasetPhaseType.VALID:
                blend_dataset = blended_megatron_dataset_builder[1]
            elif phase == DatasetPhaseType.TEST:
                blend_dataset = blended_megatron_dataset_builder[2]
            else:
                raise ValueError(f"Unsupported blend dataset phase: {phase}")

            gen_dataset = GeneratorDataset(blend_dataset,
                                           column_names=column_names,
                                           shuffle=shuffle,
                                           num_shards=num_shards,
                                           shard_id=shard_id)
            return gen_dataset

        if get_real_group_size() > 1:
            global_rank_id = get_real_rank()
            stage_num = ms.get_auto_parallel_context("pipeline_stages")
            total_device_num = get_real_group_size() // stage_num
            dp = get_dp_from_dataset_strategy()
            tp = total_device_num // dp
            if is_dataset_built_on_rank():
                logger.info(f"This rank is {global_rank_id}, tensor parallel = {tp}, \
                            pipeline stage = {global_rank_id//(dp *tp )}, this rank will build real data.")
                gen_dataset = build_gpt_dataset()
            else:
                logger.info(f"This rank is {global_rank_id}, tensor parallel = {tp}, \
                            pipeline stage = {global_rank_id//(dp *tp )}, this rank will build empty data.")
                source = FakeGptDataset(blended_config)
                gen_dataset = GeneratorDataset(source, column_names=source.cols(), shuffle=False)
                logger.info("barrier 1 start")
                barrier()
                logger.info("barrier 2 start")
                barrier()
                logger.info("barrier 2 completed")
        else:
            gen_dataset = build_gpt_dataset()
        return gen_dataset

    def init_gpt_dataset_config(self) -> GPTDatasetConfig:
        """init GPTDatasetConfig

        Args:
            config (dict): configurations dict.
            tokenizer (optional): tokenizer obj. Defaults to None.

        Returns:
            GPTDatasetConfig: obj.
        """
        return GPTDatasetConfig(
            random_seed=self.config.get("seed", 1234),
            sequence_length=self.config.get("seq_length"),
            blend=get_blend_from_list(self.config.get("data_path", None)),
            blend_per_split=[
                get_blend_from_list(self.config.get("train_data_path", None)),
                get_blend_from_list(self.config.get("valid_data_path", None)),
                get_blend_from_list(self.config.get("test_data_path", None))
            ],
            split=self.config.get("split", None),
            num_dataset_builder_threads=self.config.get("num_dataset_builder_threads", 1),
            path_to_cache=self.config.get("path_to_cache", None),
            mmap_bin_files=self.config.get("mmap_bin_files", None),
            tokenizer=self.tokenizer,
            reset_position_ids=self.config.get("reset_position_ids", False),
            reset_attention_mask=self.config.get("reset_attention_mask", False),
            eod_mask_loss=self.config.get("eod_mask_loss", False),
            create_attention_mask=self.config.get("create_attention_mask", True),
            create_compressed_eod_mask=self.config.get("create_compressed_eod_mask", False),
            s3_cache_path=self.config.get("s3_cache_path", None),
            drop_last_partial_validation_sequence=self.config.get("drop_last_partial_validation_sequence", True),
            add_extra_token_to_sequence=self.config.get("add_extra_token_to_sequence", True),
            eod=self.config.get("eod", -1),
            pad=self.config.get("pad", -1),
        )


class FakeGptDataset:
    """Fake dataset."""
    def __init__(self, config: GPTDatasetConfig):
        self.create_attention_mask = config.create_attention_mask
        self.create_compressed_eod_mask = config.create_compressed_eod_mask
        self.seq_length = config.sequence_length
        self.data_length = 1024  # fake dataset num
        self.input_ids = np.ones((self.seq_length,), dtype=np.int32)
        self.labels = np.ones((self.seq_length,), dtype=np.int32)
        self.loss_mask = np.ones((self.seq_length,), dtype=np.int32)
        self.position_mask = np.ones((self.seq_length,), dtype=np.int32)
        if self.create_compressed_eod_mask:
            self.actual_seq_len = np.ones((config.eod_pad_length,), dtype=np.int32)
        if self.create_attention_mask:
            self.attention_mask = np.ones((1, self.seq_length, self.seq_length,), dtype=np.int32)

    def cols(self):
        # pylint: disable=R1705
        if self.create_compressed_eod_mask:
            return ["input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"]
        elif self.create_attention_mask:
            return ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
        else:
            return ["input_ids", "labels", "loss_mask", "position_ids"]

    def __getitem__(self, i):
        # pylint: disable=R1705
        if self.create_compressed_eod_mask:
            return self.input_ids, self.labels, self.loss_mask, self.position_mask, self.actual_seq_len
        elif self.create_attention_mask:
            return self.input_ids, self.labels, self.loss_mask, self.position_mask, self.attention_mask
        else:
            return self.input_ids, self.labels, self.loss_mask, self.position_mask

    def __len__(self):
        return self.data_length


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class BlendedMegatronDatasetDataLoader():
    """Blended Megatron Dataset DataLoader."""
    _default_column_names = ['input_ids']

    def __new__(cls,
                datasets_type: str,
                sizes: List[int],
                config: dict,
                tokenizer: Union[dict, Callable] = None,
                column_names: list = None,
                shuffle: bool = False,
                num_shards: int = None,
                shard_id: int = None,
                phase: str = "train",
                **kwargs):
        """init Megatron blended dataset dataloader.

        Args:
            datasets_type(str): megatron dataset types, e.g. GptDataset
            sizes(List[int]): datasize of each data.
            tokenzier(Union[dict, Callable]): tokenizer config or tokenizer instance.
            config (dict): configurations dict.
            tokenizer (optional): tokenizer obj. Defaults to None.
            column_names(list): Column names contained in the created dataset.
            shuffle (bool): Whether to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.
            num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
            shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
            phase: The supported keywords are in ["train", "dev", "test"]

        Returns:
            MegatronDataset: obj.
        """
        # init tokenizer
        if isinstance(tokenizer, dict):
            tokenizer = build_tokenizer(tokenizer)

        column_names = cls._default_column_names if column_names is None else column_names

        gen_dataset = MegatronDatasetBuilder(datasets_type=datasets_type, config=config).builder(
            sizes, column_names, shuffle, num_shards, shard_id, phase, **kwargs
        )

        return gen_dataset
