mindformers.AutoConfig
========================

.. py:class:: mindformers.AutoConfig()

    这是一个通用的配置类，当使用 from_pretrained() 类方法时，它会自动实例化模型Config类，并返回。
    这个类不能直接使用 \_\_init\_\_() 实例化（会抛出异常）。

    .. py:method:: from_pretrained(yaml_name_or_path, **kwargs)
        :classmethod:

        从yaml文件、json文件、文件夹或魔乐社区读取配置信息，实例化为模型Config类，并返回。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **yaml_name_or_path** (str) - yaml文件路径、json文件路径、包含config.json文件的文件夹路径、或魔乐社区上的model_id。后三者为实验特性。
            - **kwargs** (Dict[str, Any], 可选) - 传入的配置信息将会覆盖从yaml_name_or_path读取到的配置信息。

        返回：
            一个继承自PretrainedConfig类的模型Config实例。

    .. py:method:: register(model_type, config, exist_ok=False)
        :classmethod:

        注册一个新的模型Config类到此类中。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **model_type** (str) - 模型简称，类似'bert'或'gpt'。
            - **config** (PretrainedConfig) - 用于注册的类。
            - **exist_ok** (bool, 可选) - 为True时，若model_type已存在也不报错。默认值： ``False`` 。

    .. py:method:: show_support_list()
        :classmethod:

        显示支持的方法列表。