mindformers.tools.MindFormerConfig
=======================================

.. py:class:: mindformers.tools.MindFormerConfig(*args, **kwargs)

    一个配置的类，继承于Python的dict类。可以解析来自yaml文件或dict实例的配置参数。

    参数：
        - **args** (Any) - 可扩展参数列表，可以是yaml配置文件路径或配置字典。
        - **kwargs** (Any) - 可扩展参数字典，可以是yaml配置文件路径或配置字典。

    返回：
        一个类的实例。

    .. py:method:: merge_from_dict(options)

        将配置选项合并入配置中。

        参数：
            - **options** (dict) - 需要合并的配置选项。