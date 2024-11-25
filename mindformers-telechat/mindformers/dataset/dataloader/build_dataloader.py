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
"""Build DataLoader API."""
import inspect

from mindspore import dataset as ds

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig


def build_dataset_loader(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset_loader', class_name: str = None, **kwargs):
    r"""Build dataset loader For MindFormer.
    Instantiate the dataset loader from MindFormerRegister's registry.

    Args:
        config (dict): The task dataset loader's config. Default: None.
        default_args (dict): The default argument of dataset loader API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'dataset loader'.
        class_name (str): The class name of dataset loader API. Default: None.

    Return:
        The function instance of dataset loader API.

    Examples:
        >>> from mindformers.dataset.dataloader import build_dataset_loader
        >>> dataset_loader_config = {'type': 'ImageFolderDataset', 'dataset_dir': './imagenet/train'}
        >>> # 1) use config dict to build dataset loader
        >>> dataset_loader_from_config = build_dataset_loader(dataset_loader_config)
        >>> # 2) use class name to build dataset loader
        >>> dataset_loader_class_name = build_dataset_loader(
        ...     class_name='ImageFolderDataset', dataset_dir='./imagenet/train')
    """
    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET_LOADER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_dataset_loader():
    """ register MindSpore builtin dataset loader class. """
    for module_name in dir(ds):
        if module_name.startswith('__'):
            continue
        dataset = getattr(ds, module_name)
        if inspect.isclass(dataset):
            MindFormerRegister.register_cls(dataset, MindFormerModuleType.DATASET_LOADER)


register_ms_dataset_loader()
