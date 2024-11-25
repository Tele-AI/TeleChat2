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
# This file was refer to project
# https://gitee.com/mindspore/vision/blob/master/mindvision/engine/class_factory.py
# ============================================================================
""" Class Register Module For MindFormers."""

import inspect
import os

from mindformers.tools.hub.dynamic_module_utils import get_class_from_dynamic_module


class MindFormerModuleType:
    """
    Enumerated class of the MindFormers module type, which includes:

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Enumeration Name
          - Value
        * - CALLBACK
          - 'CALLBACK'
        * - CONFIG
          - 'config'
        * - CONTEXT
          - 'context'
        * - DATASET
          - 'dataset'
        * - DATASET_LOADER
          - 'dataset_loader'
        * - DATASET_SAMPLER
          - 'dataset_sampler'
        * - DATA_HANDLER
          - 'data_handler'
        * - ENCODER
          - 'encoder'
        * - FEATURE_EXTRACTOR
          - 'feature_extractor'
        * - LOSS
          - 'loss'
        * - LR
          - 'lr'
        * - MASK_POLICY
          - 'mask_policy'
        * - METRIC
          - 'metric'
        * - MODELS
          - 'models'
        * - OPTIMIZER
          - 'optimizer'
        * - PIPELINE
          - 'pipeline'
        * - PROCESSOR
          - 'processor'
        * - TOKENIZER
          - 'tokenizer'
        * - TOOLS
          - 'tools'
        * - TRAINER
          - 'trainer'
        * - TRANSFORMS
          - 'transforms'
        * - WRAPPER
          - 'wrapper'

    Examples:
        >>> from mindformers.tools import MindFormerModuleType
        >>>
        >>> print(MindFormerModuleType.MODULES)
        modules
    """

    def __init__(self):
        pass

    TRAINER = 'trainer'
    PIPELINE = 'pipeline'
    PROCESSOR = 'processor'
    TOKENIZER = 'tokenizer'
    DATASET = 'dataset'
    MASK_POLICY = 'mask_policy'
    DATASET_LOADER = 'dataset_loader'
    DATASET_SAMPLER = 'dataset_sampler'
    TRANSFORMS = 'transforms'
    ENCODER = 'encoder'
    MODELS = 'models'
    MODULES = 'modules'
    TRANSFORMER = 'transformer'
    BASE_LAYER = 'base_layer'
    CORE = 'core'
    HEAD = 'head'
    LOSS = 'loss'
    LR = 'lr'
    OPTIMIZER = 'optimizer'
    CONTEXT = 'context'
    CALLBACK = 'callback'
    WRAPPER = 'wrapper'
    METRIC = 'metric'
    CONFIG = 'config'
    TOOLS = 'tools'
    FEATURE_EXTRACTOR = 'feature_extractor'
    DATA_HANDLER = 'data_handler'


class MindFormerRegister:
    """
    The registration interface for MindFormers, provides methods for registering and obtaining the interface.

    Examples:
        >>> from mindformers.tools import MindFormerModuleType, MindFormerRegister
        >>>
        >>>
        >>> # Using decorator to register the class
        >>> @MindFormerRegister.register(MindFormerModuleType.CONFIG)
        >>> class MyConfig:
        ...     def __init__(self, param):
        ...         self.param = param
        >>>
        >>>
        >>> # Using method to register the class
        >>> MindFormerRegister.register_cls(register_class=MyConfig, module_type=MindFormerRegister)
        >>>
        >>> print(MindFormerRegister.is_exist(module_type=MindFormerModuleType.CONFIG, class_name="MyConfig"))
        True
        >>> cls = MindFormerRegister.get_cls(module_type=MindFormerModuleType.CONFIG, class_name="MyConfig")
        >>> print(cls.__name__)
        MyConfig
        >>> instance = MindFormerRegister.get_instance_from_cfg(cfg={'type': 'MyConfig', 'param': 0},
        ...                                                     module_type=MindFormerModuleType.CONFIG)
        >>> print(instance.__class__.__name__)
        MyConfig
        >>> print(instance.param)
        0
        >>> instance = MindFormerRegister.get_instance(module_type=MindFormerModuleType.CONFIG,
        ...                                            class_name="MyConfig",
        ...                                            param=0)
        >>> print(instance.__class__.__name__)
        MyConfig
        >>> print(instance.param)
        0
    """

    def __init__(self):
        pass

    registry = {}

    @classmethod
    def register(cls, module_type=MindFormerModuleType.TOOLS, alias=None):
        """
        A decorator that registers the class in the registry.

        Args:
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            alias (str, optional) : Alias for the class. Default: ``None``.

        Returns:
            Wrapper, decorates the registered class.
        """

        def wrapper(register_class):
            """Register-Class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            class_name = alias if alias is not None else register_class.__name__
            if module_type not in cls.registry:
                cls.registry[module_type] = {class_name: register_class}
            else:
                cls.registry[module_type][class_name] = register_class
            return register_class

        return wrapper

    @classmethod
    def register_cls(cls, register_class, module_type=MindFormerModuleType.TOOLS, alias=None):
        """
        A method that registers a class into the registry.

        Args:
            register_class (type): The class that need to be registered.
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            alias (str, optional) : Alias for the class. Default: ``None``.

        Returns:
            Class, the registered class itself.
        """
        class_name = alias if alias is not None else register_class.__name__
        if module_type not in cls.registry:
            cls.registry[module_type] = {class_name: register_class}
        else:
            cls.registry[module_type][class_name] = register_class
        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """
        Determines whether the given class name is in the current type group. If `class_name` is not given,
        determines if the given class name is in the current registered dictionary.

        Args:
            module_type (MindFormerModuleType): Module type name of MindFormers.
            class_name (str, optional): Class name. Default: ``None``.

        Returns:
            A boolean value, indicating whether it exists or not.
        """
        if not class_name:
            return module_type in cls.registry
        registered = module_type in cls.registry and class_name in cls.registry.get(module_type)
        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """
        Get the class from the registry.

        Args:
            module_type (MindFormerModuleType): Module type name of MindFormers.
            class_name (str, optional): Class name. Default: ``None``.

        Returns:
            A registered class.

        Raises:
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
            ValueError: Can't find type `module_type` in the registry.
        """
        if not cls.is_exist(module_type, class_name):
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(module_type, class_name))

        if not class_name:
            raise ValueError(
                "Can't find class. class type = {}".format(class_name))
        register_class = cls.registry.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=MindFormerModuleType.TOOLS, default_args=None):
        """
        Get instances of the class in the registry via configuration.

        Args:
            cfg (dict): Configuration dictionary. It should contain at least the key "type".
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            default_args (dict, optional): Default initialization arguments. Default: ``None``.

        Returns:
            An instance of the class.

        Raises:
            TypeError: `cfg` must be a configuration.
            KeyError: `cfg` or `default_args` must contain the key "type".
            TypeError: `default_args` must be a dictionary or ``None``.
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
        """
        if not isinstance(cfg, dict):
            raise TypeError(
                "Cfg must be a Config, but got {}".format(type(cfg))
            )

        if 'auto_register' in cfg:
            cls.auto_register(class_reference=cfg.pop('auto_register'), module_type=module_type)

        if 'type' not in cfg:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type",'
                'but got {}\n{}'.format(cfg, default_args)
            )
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError(
                'default_args must be a dict or None'
                'but got {}'.format(type(default_args)))

        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                if k not in args:
                    args.setdefault(k, v)
                else:
                    args[k] = v

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))

    @classmethod
    def get_instance(cls, module_type=MindFormerModuleType.TOOLS, class_name=None, **kwargs):
        """
        Gets an instance of the class in the registry.

        Args:
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            class_name (str, optional): Class name. Default: ``None``.
            kwargs (Any): Additional keyword arguments for constructing instances of the class.

        Returns:
            An instance of the class.

        Raises:
            ValueError: `class_name` cannot be ``None``.
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
        """
        if class_name is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(class_name, str):
            obj_cls = cls.get_cls(module_type, class_name)
        elif inspect.isclass(class_name):
            obj_cls = class_name
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry.".format(type, class_name))

        try:
            return obj_cls(**kwargs)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))

    @classmethod
    def auto_register(cls, class_reference: str, module_type=MindFormerModuleType.TOOLS):
        """
        Auto register function.

        Args:
            class_reference (str): The full name of the class to load.
            module_type (MindFormerModuleType.TOOLS): module type.
        """
        if not isinstance(class_reference, str):
            raise ValueError(f"auto_map must be the type of string, but get {type(class_reference)} ."
                             f"Please fill in the following format: module_file.function_name, such as,"
                             f"llama_model.LlamaForCausalLM")
        register_path = os.getenv("REGISTER_PATH", '')
        if not register_path:
            raise EnvironmentError("When configuring the 'auto_map' automatic registration function, "
                                   "REGISTER_PATH must be specified. "
                                   "It is recommended to complete this action"
                                   "through the official startup script "
                                   "'run_mindformer.py --register_path=module_file_path' "
                                   "or use 'export REGISTER_PATH=module_file_path' to complete this action.")
        if not os.path.realpath(register_path):
            raise EnvironmentError(f"REGISTER_PATH must be real path, but get {register_path}, "
                                   f"please specify the correct directory path.")
        register_path = os.path.realpath(os.getenv("REGISTER_PATH"))
        module_class = get_class_from_dynamic_module(
            class_reference=class_reference, pretrained_model_name_or_path=register_path)
        cls.register_cls(module_class, module_type=module_type)
