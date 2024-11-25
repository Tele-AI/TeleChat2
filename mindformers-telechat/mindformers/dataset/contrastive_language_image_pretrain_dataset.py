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
"""Contrastive Language Image Pretrain Dataset."""
from typing import Optional, Union, Callable
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset
from ..tools.logger import logger
from ..models.build_tokenizer import build_tokenizer
from ..tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class ContrastiveLanguageImagePretrainDataset(BaseDataset):
    """
    CLIP (Contrastive Language Image Pre-training) dataset.

    The generated dataset has two columns: :py:obj:`[image, text]` .
    The data type of tensor of each column depend on the dataset files format and data transform operations in use.

    Args:
        dataset_config (dict, optional):
            Config for dataset. When `dataset_config` is an empty dict or is None, all arguments below
            will build a non-empty `dataset_config`. Otherwise, they will be ignored. Default: None.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object. When `data_loader` is a `dict`,
            the string "type", "dataset_dir", "stage" and "column_names" are the keys can be parsed.

            - type: Required. Indicates the type of dataset. The value must be string or class type.

            - dataset_dir: Required. The directory of dataset.

            - stage: Optional. The dataset subset to be loaded. The value can be 'train', "test", "dev" and "all".
              Default when missing: 'train'.

            - column_names: Optional. The column names of dataset. Must be a list or tuple of string.
              Default when missing: py:obj:`[image, text]` .

        transforms (Union[dict, list]):
            Configurations or objects of one or more transformers.
            Default: None, no image transform operation to use.
        text_transforms (Union[dict, list]):
            Configurations or objects of one or more transformers of text.
            Default: None, no text transform operation to use.
        tokenizer (Union[dict, Callable]):
            Tokenizer configuration or object. Default: None, no tokenizer to use.
        sampler (Union[dict, Callable]):
            Sampler configuration or object. Default: None, no sampler to use.
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
        Instance of ContrastiveLanguageImagePretrainDataset.

    Examples:
        >>> from mindspore.dataset.vision import CenterCrop, ToTensor, Normalize
        >>> from mindformers import AutoTokenizer
        >>> from mindformers.dataset import Flickr8kDataLoader, ContrastiveLanguageImagePretrainDataset
        >>> from mindformers.dataset import Resize, RandomChoiceTokenizerForward
        >>> tokenizer = AutoTokenizer.from_pretrained("clip_vit_b_32")
        >>> # Note:
        >>> #     `"/dir/to/dataset"` should be replaced with the real directory of dataset.
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/clip.md
        >>> data_loader = Flickr8kDataLoader(dataset_dir="/dir/to/dataset", stage="train",
        ...                                  column_names=["image", "text"])
        >>> text_transforms = RandomChoiceTokenizerForward(max_length=77, padding="max_length", random_seed=2022,
        ...                                                tokenizer=tokenizer)
        >>> transforms = [Resize(size=224), CenterCrop(size=224), ToTensor(),
        ...               Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        ...                         std=[0.26862954, 0.26130258, 0.27577711],
        ...                         is_hwc=False)]
        >>> dataset_from_param = ContrastiveLanguageImagePretrainDataset(data_loader=data_loader,
        ...                                                              text_transforms=text_transforms,
        ...                                                              transforms=transforms,
        ...                                                              tokenizer=tokenizer)
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                transforms: Union[dict, list] = None,
                text_transforms: Union[dict, list] = None,
                tokenizer: Union[dict, Callable] = None,
                sampler: Union[dict, Callable] = None,
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
        """new method"""
        logger.info("Now Create Contrastive Language Image Pretrain Dataset.")
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

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            dataset = get_dataset_map(dataset, transforms,
                                      input_columns="image",
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        if text_transforms is not None:
            dataset = get_dataset_map(dataset, text_transforms,
                                      input_columns="text",
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
