mindformers.modules.OpParallelConfig
====================================

.. py:class:: mindformers.modules.OpParallelConfig(data_parallel=1, model_parallel=1, use_seq_parallel=False, context_parallel=1, select_recompute=False, context_parallel_algo: str = "colossalai_cp")

    算子并行的配置，用于设置算子级并行的方式。

    参数：
        - **data_parallel** (int, 可选) - 数据并行的数量。默认值： ``1`` 。
        - **model_parallel** (int, 可选) - 模型并行的数量。默认值： ``1`` 。
        - **use_seq_parallel** (bool, 可选) - 是否使用序列并行。默认值： ``False`` 。
        - **context_parallel** (int, 可选) - 上下文并行的数量。默认值：``1`` 。
        - **select_recompute** (bool, 可选) - 是否选择重计算。默认值： ``False`` 。
        - **context_parallel_algo** (str, 可选) - 上下文并行算法，可选 ``"colossalai_cp"`` 和 ``"ulysses_cp"`` 两种。默认值：``"colossalai_cp"`` 。

    返回：
        `OpParallelConfig` 实例。