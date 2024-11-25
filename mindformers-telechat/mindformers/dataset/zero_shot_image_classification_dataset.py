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
"""Zero Shot Image Classification Dataset."""
from typing import Optional, Union, Callable
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .base_dataset import BaseDataset
from ..tools.logger import logger
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models.build_tokenizer import build_tokenizer


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ZeroShotImageClassificationDataset(BaseDataset):
    r"""
    Zero Shot Image Classification Dataset API.
    output image, text, and label columns

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        sampler (Union[dict, list]):
            Sampler configuration or object.
        transforms (Union[dict, list]):
            Configurations or objects of one or more transformers.
        text_transforms (Union[dict, list]):
            Configurations or objects of one or more transformers of text.
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
        A dataset for ZeroShotImageClassificationDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import ZeroShotImageClassificationDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['zero_shot_image_classification']['clip_vit_b_32']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.eval_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/clip.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = ZeroShotImageClassificationDataset(config.eval_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindformers.dataset import ZeroShotImageClassificationDataset, Cifar100DataLoader
        >>> data_loader = Cifar100DataLoader(dataset_dir="The required task dataset path",
        ...                                  column_names=['image', 'text', 'label'],
        ...                                  stage='test', fine_label=True, shuffle=False,
        ...                                  hypothesis_template='a picture of {}')
        >>> dataset_from_param = ZeroShotImageClassificationDataset(data_loader=data_loader, batch_size=40)
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                tokenizer: Union[dict, Callable] = None,
                transforms: Union[dict, list] = None,
                text_transforms: Union[dict, list] = None,
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
        """New method"""
        logger.info("Now Create Zero Shot Image Classification Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'num_shards': device_num, 'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        if isinstance(dataset_config.tokenizer, dict):
            tokenizer = build_tokenizer(dataset_config.tokenizer)
        else:
            tokenizer = dataset_config.tokenizer

        if (isinstance(dataset_config.transforms, list) and isinstance(dataset_config.transforms[0], dict)) \
                or isinstance(dataset_config.transforms, dict):
            transforms = build_transforms(dataset_config.transforms)
        else:
            transforms = dataset_config.transforms

        if (isinstance(dataset_config.text_transforms, list) and isinstance(dataset_config.text_transforms[0], dict)) \
                or isinstance(dataset_config.text_transforms, dict):
            text_transforms = build_transforms(dataset_config.text_transforms, default_args={"tokenizer": tokenizer})
        else:
            text_transforms = dataset_config.text_transforms

        if transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns="image",
                                      operations=transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        if text_transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns="text",
                                      operations=text_transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
