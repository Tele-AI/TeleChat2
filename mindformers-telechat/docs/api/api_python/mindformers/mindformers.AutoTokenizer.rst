mindformers.AutoTokenizer
============================

.. py:class:: mindformers.AutoTokenizer()

    这是一个通用的Tokenizer类，当使用 `from_pretrained()` 类方法时，它会自动实例化Tokenzier类，并返回。
    这个类不能直接使用 \_\_init\_\_() 实例化（会抛出异常）。

    .. py:method:: from_pretrained(yaml_name_or_path, *args, **kwargs)
        :classmethod:

        从文件夹、或魔乐社区读取配置信息，实例化为一个分词器。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **yaml_name_or_path** (str) - 包含yaml文件的文件夹路径、包含json配置文件的文件夹路径、或魔乐社区上的model_id。后两者为实验特性。
            - **args** (Any, 可选) - 会在实例化Tokenizer时，传给Tokenizer的 \_\_init\_\_() 方法。仅在实验特性时生效。
            - **kwargs** (Dict[str, Any], 可选) - 传入的配置信息将会覆盖从yaml_name_or_path读取到的配置信息。

        返回：
            一个Tokenizer实例。

    .. py:method:: register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False)
        :classmethod:

        注册新的Tokenizer类到此类中。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **config_class** (PretrainedConfig) - 模型的Config类。
            - **slow_tokenizer_class** (PreTrainedTokenizer, 可选) - 用于注册的Tokenizer类。
            - **fast_tokenizer_class** (PreTrainedTokenizerFast, 可选) - 用于注册的FastTokenizer类。
            - **exist_ok** (bool, 可选) - 为True时，即使 `config_class` 已存在也不会报错。默认值： ``False`` 。
