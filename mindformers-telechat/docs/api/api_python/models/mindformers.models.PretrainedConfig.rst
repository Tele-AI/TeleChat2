mindformers.models.PretrainedConfig
===================================

.. py:class:: mindformers.models.PretrainedConfig(**kwargs)

    所有配置类的基类。处理所有模型配置的通用参数以及加载、下载、保存配置的方法。

    .. note::
       配置文件可以被加载并保存到磁盘。加载配置文件并使用这个文件初始化模型**不会**加载模型权重。它只影响模型的配置。

    参数：
        - **name_or_path** (str, 可选) - 存储传递给 :func:`mindformers.models.PreTrainedModel.from_pretrained` 的字符串作为 ``pretrained_model_name_or_path`` ，如果配置是用这种方法创建的。默认值： ``""`` 。
        - **checkpoint_name_or_path** (str, 可选) - checkpoint 文件的路径或名称。默认值： ``None`` 。
        - **mindformers_version** (str, 可选) - MindSpore Transformers 的版本。默认值： ``""`` 。

    返回：
        PretrainedConfig类实例。

    .. py:method:: from_dict(config_dict: Dict[str, Any], **kwargs)
        :classmethod:

        从参数字典实例化 PretrainedConfig。

        参数：
            - **config_dict** (Dict[str, Any]) - 用于实例化配置对象的字典。这样的字典可以通过利用 :func:`mindformers.models.PretrainedConfig.get_config_dict` 方法从预训练的检查点检索。

        返回：
            PretrainedConfig, 从这些参数实例化的配置对象。

    .. py:method:: from_json_file(json_file: Union[str, os.PathLike])
        :classmethod:

        从 JSON 文件的路径实例化 PretrainedConfig。

        参数：
            - **json_file** (Union[str, os.PathLike]) - 参数的 JSON 文件路径。

        返回：
            PretrainedConfig, 从该 JSON 文件实例化的配置对象。

    .. py:method:: from_pretrained(yaml_name_or_path, **kwargs)
        :classmethod:

        通过 yaml 名称或路径实例化配置。

        参数：
            - **yaml_name_or_path** (str) - 支持的模型名称或模型配置文件路径（.yaml），支持的模型名称可以从 :func:`mindformers.AutoConfig.show_support_list` 中选择。如果 `yaml_name_or_path` 是模型名称，则支持以 `mindspore` 开头的模型名称或模型名称本身，如 "mindspore/vit_base_p16" 或 "vit_base_p16"。
            - **pretrained_model_name_or_path** (str, 可选) - 等同于 `yaml_name_or_path`，如果设置了 `pretrained_model_name_or_path`，则 `yaml_name_or_path` 无效。默认值： ``None`` 。

        返回：
            PretrainedConfig: 继承自 PretrainedConfig 的模型配置。

    .. py:method:: get_config_dict(pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs)
        :classmethod:

        从 'pretrained_model_name_or_path' 解析到一个参数字典，用于使用 :func:`mindformers.models.PretrainedConfig.from_dict` 实例化 PretrainedConfig。


        参数：
            - **pretrained_model_name_or_path** (Union[str, os.PathLike]) - 预训练检查点的标识符, 我们希望从中获得参数字典。

        返回：
            Tuple[dict, dict]: 用于实例化配置对象的字典。

    .. py:method:: save_pretrained(save_directory=None, save_name="mindspore_model", **kwargs)

        将预训练的配置保存到指定目录。

        参数：
            - **save_directory** (str, 可选) - 配置文件的保存目录。默认值： ``None`` 。
            - **save_name** (str, 可选) - 保存文件的名称，默认值： ``mindspore_model`` 。

    .. py:method:: to_dict()

        将此实例序列化为 Python 字典。

        返回：
            dict, 包含构成此配置实例的所有属性的字典。

    .. py:method:: to_diff_dict()

        移除与默认配置属性对应的所有属性，以提高可读性，并序列化为 Python 字典。

        返回：
            dict: 包含此配置实例的所有属性的字典。

    .. py:method:: to_json_file(json_file_path: Union[str, os.PathLike], use_diff: bool = True)

        将此实例保存到 JSON 文件。

        参数：
            - **json_file_path** (Union[str, os.PathLike]) - 此配置实例参数将被保存的 JSON 文件路径。
            - **use_diff** (bool, 可选) - 如果设置为 True，仅序列化配置实例与默认 :class:`mindformers.models.PretrainedConfig` 的差异到 JSON 文件。默认值： ``True`` 。

    .. py:method:: to_json_string(use_diff: bool = True)

        将此实例序列化为 JSON 字符串。

        参数：
            - **use_diff** (bool, 可选) - 如果设置为 True，仅序列化配置实例与默认 PretrainedConfig() 的差异到 JSON 字符串。默认值： ``True`` 。

        返回：
            str, 包含此配置实例所有属性的 JSON 格式字符串。
