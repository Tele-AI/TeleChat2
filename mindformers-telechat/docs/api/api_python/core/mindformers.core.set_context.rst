mindformers.core.set_context
============================

.. py:function:: mindformers.core.set_context(run_mode=None, **kwargs)

    设置运行环境的context。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        设置属性时，必须输入属性名称。目前只有run_mode参数属于MindFormers context，Kwargs参数会被传到MindSpore的set_context接口中。

    参数：
        - **run_mode** (str) - 运行模式，必须在列表['train', 'finetune', 'eval', 'predict']中。
        - **kwargs** - MindSpore的context参数。