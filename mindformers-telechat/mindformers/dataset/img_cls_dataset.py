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
"""Image Classification Dataset."""
from typing import Optional, Union, Callable

from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map

from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ImageCLSDataset(BaseDataset):
    """
    Image Classification Dataset API.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
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
        transforms (Union[dict, list]):
            Configurations or objects of one or more transformers.
        sampler (Union[dict, list]):
            Sampler configuration or object.
        do_eval (bool):
            Indicates whether to enable evaluation during training.
        mixup_op (dict):
            Configuration of the mixup
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
        A dataset for ImageCLSDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import ImageCLSDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['image_classification']['vit_base_p16']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/vit.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = ImageCLSDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindspore.dataset import ImageFolderDataset
        >>> from mindspore.dataset.vision import Decode, ToPIL, ToTensor, Normalize
        >>> from mindformers.dataset import ImageCLSDataset, RandomResizedCrop, RandomHorizontalFlip
        >>> from mindformers.dataset import rand_augment_transform, RandomErasing
        >>> data_loader = ImageFolderDataset(dataset_dir="The required task dataset path",
        ...                                 num_parallel_workers=8, shuffle=True)
        >>> transforms = [Decode(),
        ...               RandomResizedCrop(size=224, scale=[0.08, 1.0], interpolation='cubic'),
        ...               RandomHorizontalFlip(prob=0.5),
        ...               ToPIL(),
        ...               rand_augment_transform(config_str='rand-m9-mstd0.5-inc1',
        ...                                      hparams={'translate_const': 100,
        ...                                               'img_mean': [124, 116, 104],
        ...                                               'interpolation': 'cubic'}),
        ...               ToTensor(),
        ...               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
        ...               RandomErasing(probability=0.25, mode='pixel', max_count=1)]
        >>> mixup_op = {'mixup_alpha': 0.8, 'cutmix_alpha': 1.0, 'cutmix_minmax': None, 'prob': 1.0,
        ...             'switch_prob': 0.5, 'label_smoothing': 0.1}
        >>> dataset_from_param = ImageCLSDataset(data_loader=data_loader, transforms=transforms, mixup_op=mixup_op,
        ...                                      input_columns=['image', 'label'], output_columns=['image', 'label'],
        ...                                      seed=2022, batch_size=32)
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                transforms: Union[dict, list] = None,
                sampler: Union[dict, Callable] = None,
                do_eval: bool = False,
                mixup_op: dict = None,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Image Classification Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'num_shards': device_num, 'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        if (isinstance(dataset_config.transforms, list) and isinstance(dataset_config.transforms[0], dict)) \
                or isinstance(dataset_config.transforms, dict):
            transforms = build_transforms(dataset_config.transforms)
        else:
            transforms = dataset_config.transforms

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        type_cast_op = TypeCast(mstype.int32)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            dataset = get_dataset_map(dataset, transforms,
                                      input_columns=dataset_config.input_columns[0],
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = get_dataset_map(dataset, type_cast_op,
                                  input_columns=dataset_config.input_columns[1],
                                  num_parallel_workers=dataset_config.num_parallel_workers)

        dataset = dataset.batch(dataset_config.batch_size, drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        if not dataset_config.do_eval and dataset_config.mixup_op is not None:
            mixup_op = build_transforms(class_name="Mixup", **dataset_config.mixup_op)
            dataset = get_dataset_map(dataset, mixup_op,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.output_columns,
                                      num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.project(columns=dataset_config.output_columns)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
