mindformers.tools.register.MindFormerRegister
===============================================

.. py:class:: mindformers.tools.register.MindFormerRegister()

    MindFormers的注册接口，提供了接口注册和获取的相关方法。

    .. py:method:: get_cls(module_type, class_name=None)
        :classmethod:

        从注册字典中获取类。

        参数：
            - **module_type** (MindFormerModuleType) - MindFormers的模块类型名。
            - **class_name** (str, 可选) - 类名。默认值： ``None`` 。

        返回：
            一个注册了的类。

        异常：
            - **ValueError** - 在注册字典中未找到类型是 `module_type` 的类 `class_name` 。
            - **ValueError** - 在注册字典中未找到 `module_type` 。

    .. py:method:: get_instance(module_type=MindFormerModuleType.TOOLS, class_name=None, **kwargs)
        :classmethod:

        获取注册字典中类的实例。

        参数：
            - **module_type** (MindFormerModuleType, 可选) - MindFormers的模块类型名。默认值： ``MindFormerModuleType.TOOLS`` 。
            - **class_name** (str, 可选) - 类名。默认值： ``None`` 。
            - **kwargs** (Any) - 额外的关键字参数，用于构造类的实例。

        返回：
            一个类的实例。

        异常：
            - **ValueError** - `class_name` 不能为 ``None`` 。
            - **ValueError** - 在注册字典中未找到类型是 `module_type` 的类 `class_name` 。

    .. py:method:: get_instance_from_cfg(cfg, module_type=MindFormerModuleType.TOOLS, default_args=None)
        :classmethod:

        通过配置获取注册字典中类的实例。

        参数：
            - **cfg** (dict) - 配置字典。应至少包含键 "type" 。
            - **module_type** (MindFormerModuleType, 可选) - MindFormers的模块类型名。默认值： ``MindFormerModuleType.TOOLS`` 。
            - **default_args** (dict, 可选) - 默认的初始化参数。默认值： ``None`` 。

        返回：
            一个类的实例。

        异常：
            - **TypeError** - `cfg` 必须为一个配置。
            - **KeyError** - `cfg` 或 `default_args` 必须包含键 "type" 。
            - **TypeError** - `default_args` 必须为一个字典或为 ``None`` 。
            - **ValueError** - 在注册字典中未找到类型是 `module_type` 的类 `class_name` 。

    .. py:method:: is_exist(module_type, class_name=None)
        :classmethod:

        判断给定的类名是否在当前的类型组中。若 `class_name` 没有给定，则判断给定的类名是否在当前的注册字典中。

        参数：
            - **module_type** (MindFormerModuleType) - MindFormers的模块类型名。
            - **class_name** (str, 可选) - 类名。默认值： ``None`` 。

        返回：
            一个布尔值，表示是否存在。

    .. py:method:: register(module_type=MindFormerModuleType.TOOLS, alias=None)
        :classmethod:

        将类注册至注册字典中的装饰器。

        参数：
            - **module_type** (MindFormerModuleType, 可选) - MindFormers的模块类型名。默认值： ``MindFormerModuleType.TOOLS`` 。
            - **alias** (str, 可选) - 类的别名。默认值： ``None`` 。

        返回：
            包装函数，对注册的类进行装饰。

    .. py:method:: register_cls(register_class, module_type=MindFormerModuleType.TOOLS, alias=None)
        :classmethod:

        将类注册至注册字典中的方法。

        参数：
            - **register_class** (type) - 需要被注册的类。
            - **module_type** (MindFormerModuleType, 可选) - MindFormers的模块类型名。默认值： ``MindFormerModuleType.TOOLS`` 。
            - **alias** (str, 可选) - 类的别名。默认值： ``None`` 。

        返回：
            类，被注册的类本身。

