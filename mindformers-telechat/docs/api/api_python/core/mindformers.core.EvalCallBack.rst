mindformers.core.EvalCallBack
=============================

.. py:class:: mindformers.core.EvalCallBack(eval_func: Callable, step_interval: int = 100, epoch_interval: int = -1)

    在训练期间评估模型的回调函数。

    参数：
        - **eval_func** (Callable) - 用于评估模型结果的函数，可以根据任务自定义。
        - **step_interval** (int) - 确定每次评估之间的间隔step数。默认值： ``100`` 。注意在数据下层模式下不会生效。
        - **epoch_interval** (int) - 确定每次评估之间的间隔epoch数。默认值： ``-1`` ，表示只在训练epoch结束后进行评估。