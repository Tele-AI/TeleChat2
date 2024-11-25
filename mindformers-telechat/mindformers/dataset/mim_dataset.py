# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Masked Image Modeling Dataset."""
from typing import Optional, Union, Callable

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map

from .dataloader import build_dataset_loader
from .mask import build_mask
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MIMDataset(BaseDataset):
    """
    Masked Image Modeling Dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        transforms (Union[dict, list]):
            Configurations or objects of one or more transformers.
        mask_policy (Union[dict, list]):
            Indicates the configuration or object of the mask policy.
        sampler (Union[dict, list]):
            Sampler configuration or object.
        input_columns (list):
            Column name before the map function.
        output_columns (list):
            Column name after the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained
            in the last batch is smaller than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        python_multiprocessing (bool):
            Enabling the Python Multi-Process Mode to Accelerate Map Operations. Default: False.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for MIMDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import MIMDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['masked_image_modeling']['mae_vit_base_p16']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/mae.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = MIMDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindspore.dataset import ImageFolderDataset
        >>> from mindspore.dataset.vision import Normalize, HWC2CHW
        >>> from mindformers.dataset import RandomCropDecodeResize, RandomHorizontalFlip, MaeMask, MIMDataset
        >>> data_loader = ImageFolderDataset(dataset_dir="The required task dataset path",
        ...                                  num_parallel_workers=8, shuffle=True)
        >>> transforms = [RandomCropDecodeResize(size=224, scale=[0.2, 1.0], interpolation='cubic'),
        ...               RandomHorizontalFlip(prob=0.5),
        ...               Normalize(mean=[123.675, 118.575, 103.53], std=[58.395, 62.22, 57.375]),
        ...               HWC2CHW()]
        >>> mask_policy = MaeMask(input_size=224, patch_size=16, mask_ratio=0.75)
        >>> dataset_from_param = MIMDataset(data_loader=data_loader, transforms=transforms, mask_policy=mask_policy,
        ...                                 seed=2022, batch_size=64, input_columns=['image'],
        ...                                 output_columns=['image', 'mask', 'ids_restore', 'unmask_index'])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                transforms: Union[dict, list] = None,
                mask_policy: Union[dict, Callable] = None,
                sampler: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Masked Image Modeling Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(
                dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        if (isinstance(dataset_config.transforms, list) and isinstance(dataset_config.transforms[0], dict)) \
                or isinstance(dataset_config.transforms, dict):
            transforms = build_transforms(dataset_config.transforms)
        else:
            transforms = dataset_config.transforms

        if isinstance(dataset_config.mask_policy, dict):
            mask = build_mask(dataset_config.mask_policy)
        else:
            mask = dataset_config.mask_policy

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            for column in dataset_config.input_columns:
                dataset = get_dataset_map(dataset, transforms,
                                          input_columns=column,
                                          num_parallel_workers=dataset_config.num_parallel_workers,
                                          python_multiprocessing=dataset_config.python_multiprocessing)

        if mask is not None:
            dataset = get_dataset_map(dataset, mask,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.project(columns=dataset_config.output_columns)
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
