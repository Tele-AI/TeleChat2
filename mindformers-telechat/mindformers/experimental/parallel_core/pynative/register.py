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
# This file was refer to project
# https://gitee.com/mindspore/vision/blob/master/mindvision/engine/class_factory.py
# ============================================================================
"""Module register."""

import inspect
from enum import Enum
from collections import defaultdict

from mindformers.tools.logger import logger


class ModuleType(Enum):
    """Class module type"""

    GENERAL = "general"
    ACTIVATION_FUNC = "avtivation_func"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"
    GRAD_PROCESS_FUNC = "grad_process_func"
    LOSS_FUNC = "loss_func"


class ModuleRegistry:
    """
    Module register class.
    """

    _registry = defaultdict(lambda: defaultdict(dict))

    @classmethod
    def is_exist(cls, module_type, item_name=None, raise_error=False):
        """check whether a module is in the registry or a item is in the module.

        Args:
            module_type : Module type
            item_name (str, optional): item name, if not set, check whether the module is in the registry.
            raise_error (bool, optional): whether raise error if not exist. Default: False.

        Returns:
            True/False

        Raises:
            ValueError: If the module or item does not exist. Only raise when raise_error is True.
        """
        if module_type not in cls._registry:
            if raise_error:
                raise ValueError(f"Module {module_type} does not exist.")
            return False
        if not item_name:
            return True
        if item_name not in cls._registry.get(module_type):
            if raise_error:
                raise ValueError(f"Module {module_type} with name {item_name} does not exist.")
            return False
        return True

    @classmethod
    # pylint: disable=W0102
    def register(cls, register_item, module_type=ModuleType.GENERAL, item_name=None, meta={}):
        """Register class / function into registry

        Args:
            register_item (Union[class, function]) : class / function need to register
            module_type (ModuleType) : module type name. Default: ModuleType.GENERAL
            item_name (str, optional) : item name, if not set, use class name
            meta (dict, optional) : meta information. Default: {}

        Returns:
            register_item

        Raises:
            ValueError: If the register item is not class or function
        """
        if not inspect.isclass(register_item) and not inspect.isfunction(register_item):
            raise ValueError("Register can only register class or function.")
        item_name = item_name if item_name is not None else register_item.__name__
        if cls.is_exist(module_type, item_name):
            logger.warning(f"Module {module_type} with name {item_name} already exists and will be updated.")
        cls._registry[module_type][item_name]['item'] = register_item
        if meta is not None:
            if not isinstance(meta, dict):
                raise ValueError("Meta should be a dict.")
            cls._registry[module_type][item_name]['meta'] = meta
        return register_item

    @classmethod
    def register_decorator(cls, module_type=ModuleType.GENERAL, item_name=None):
        """Decorator to register class / function into registry

        Args:
            module_type (ModuleType) : module type name, default ModuleType.GENERAL
            item_name (str, optional) : item name, if not set, use class name

        Returns:
            decorator

        Raises:
            ValueError: If the register item is not class or function
        """
        return lambda register_item: cls.register(register_item, module_type, item_name)

    @classmethod
    def get_item(cls, module_type, item_name, return_meta=False):
        """Get class / function from registry

        Args:
            module_type (str) : Module type
            item_name (str) : name of class / function
            return_meta (bool, optional) : whether return meta information. Default: False

        Returns:
            Union[Union[class, function], Tuple[Union[class, function], dict]]:
                class / function or class / function with meta
        """
        cls.is_exist(module_type, item_name, raise_error=True)
        if not return_meta:
            return cls._registry.get(module_type).get(item_name).get('item')
        item = cls._registry.get(module_type).get(item_name)
        return item.get('item'), item.get('meta')

    @classmethod
    def get_item_meta_info(cls, module_type, item_name):
        """Get meta information of class / function from registry

        Args:
            module_type (str) : Module type
            item_name (str) : name of class / function

        Returns:
            dict: meta information
        """
        cls.is_exist(module_type, item_name, raise_error=True)
        return cls._registry.get(module_type).get(item_name).get('meta')

    @classmethod
    def query_registry(cls, module_type=None):
        """Query registry

        Args:
            module_type (str, optional) : module type, if not set, query all module types

        Returns:
            dict[module_type: list[str]]: module type and item names

        Raises:
            ValueError: If the module type does not exist
        """
        if module_type is None:
            results = {module_type: list(cls._registry.get(module_type).keys()) for module_type in cls._registry.keys()}
        else:
            if cls.is_exist(module_type, raise_error=True):
                results = {module_type: list(cls._registry.get(module_type).keys())}
        return results

    @classmethod
    def get_needed_params_for_init(cls, item_cls, params):
        """Get needed params for initiate a class

        Args:
            item_cls (class): class
            params (dict): params

        Returns:
            dict: needed params
        """
        needed_parameters = inspect.signature(item_cls).parameters.keys()
        return {key: params[key] for key in needed_parameters if key in params}
