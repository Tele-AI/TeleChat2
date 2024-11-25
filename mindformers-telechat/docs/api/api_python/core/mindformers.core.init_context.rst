mindformers.core.init_context
===============================

.. py:function:: mindformers.core.init_context(use_parallel=False, context_config=None, parallel_config=None)

    初始化运行环境的context。

    参数：
        - **use_parallel** (bool) - 是否并行。默认值： ``False`` 。
        - **context_config** (Union[dict, ContextConfig]) - contex的配置。默认值： ``None`` 。
        - **parallel_config** (Union[dict, ParallelContextConfig]) - 并行配置。默认值： ``None`` 。

    返回：
        - Int，local_rank序号。
        - Int，总共可用设备数。