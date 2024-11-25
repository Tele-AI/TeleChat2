mindformers.models.PreTrainedModel
==============================================

.. py:class:: mindformers.models.PreTrainedModel(config: PretrainedConfig, *inputs, **kwargs)

    所有预训练模型的基类。负责存储模型的配置信息，提供加载、下载、保存模型的方法以及调整输入嵌入层大小和在模型的自注意力机制中进行剪枝的通用方法。

    参数：
        - **config** (PretrainedConfig) - 模型架构的配置类。
        - **inputs** (tuple, 可选) - 一个可变数量的位置参数，为待扩展的位置参数预留。
        - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

    .. py:method:: can_generate()
        :classmethod:

        用于判断这个模型是否具备使用 ``.generate()`` 方法生成序列的能力。

        返回：
            Bool类型，True（或False），表示模型可以（或不可以）执行 ``.generate()`` 生成序列。

    .. py:method:: from_pretrained(pretrained_model_name_or_dir: str, *model_args, **kwargs)
        :classmethod:

        通过 ``pretrained_model_name_or_dir`` 实例化模型。如果用户传入模型名称，会下载模型权重，或者在给定路径的目录中加载权重（仅支持单机模式，分布式模式有待开发）。

        参数：
            - **pretrained_model_name_or_dir** (str) - 支持以下两种输入类型：如果 ``pretrained_model_name_or_dir`` 是模型名称，例如"vit_base_p16"和"t5_small"，它将在线下载权重，用户可以通过 ``MindFormerBook.get_model_support_list()`` 从获取到的列表中传递一个模型参数；如果 ``pretrained_model_name_or_dir`` 是本地路径，目录中应该有以 ``.ckpt`` 结尾的模型权重和以 ``yaml`` 结尾的配置文件。
            - **model_args** (str, 可选) - 模型扩展参数。如果包含 ``pretrained_model_name_or_path``，等同于 ``pretrained_model_name_or_dir`` ，如果设置了 ``pretrained_model_name_or_path`` ， ``pretrained_model_name_or_dir`` 就会失效。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。

        返回：
            一个继承 ``PreTrainedModel`` 的模型。

    .. py:method:: post_init()

        在每个Transformer模型初始化结束以及模型需要的模块正确初始化（例如权重初始化）之后执行。

    .. py:method:: register_for_auto_class(auto_class="AutoModel")
        :classmethod:

        使用给定的 ``auto`` 类将当前类进行注册。只适用于自定义模型，标准库中的模型已经与 ``auto`` 类映射好，无需注册。

        .. warning::
            这个API正处于实验阶段，在下一个版本中可能会有一些突破性的变化。

        参数：
            - **auto_class** (Union[str, type], 可选) - 用于注册一个新模型的自动类。默认值： ``AutoModel``。

    .. py:method:: save_pretrained(save_directory: Union[str, os.PathLike], save_name: str = "mindspore_model", **kwargs)

        保存模型权重和配置文件（仅支持单机模式，分布式模式有待开发）。

        参数：
            - **save_directory** (Union[str, os.PathLike]) - 保存模型权重和配置文件的路径。可以通过 ``MindFormerBook.get_default_checkpoint_save_folder()`` 获取路径。
            - **save_name** (str) - 文件存储的名称，包括模型权重和配置文件，默认值： ``mindspore_model`` 。
            - **kwargs** (dict, 可选) - 一个可变数量的关键字参数，为待扩展的关键字参数预留。