mindformers.core.build_context
==============================

.. py:function:: mindformers.core.build_context(config: Union[dict, MindFormerConfig, TrainingArguments])

    基于config创建运行环境的context。

    .. note::
        当config类型是dict时，必须包含context和parallel属性。

    参数：
        - **config** (Union[dict, MindFormerConfig, TrainingArguments]) - 初始化context的配置项，可以是字典类型，MindFormerConfig实例，TrainingArguments实例。

    返回：
        _Context，实例化后的context对象。