mindformers.core.ProfileMonitor
===============================

.. py:class:: mindformers.core.ProfileMonitor(start_step=1, stop_step=10, output_path=None, start_profile=True, profile_rank_ids=None, profile_pipeline = False, profile_communication=False, profile_memory=False, profiler_level=0, with_stack=False, data_simplification=True, config=None, **kwargs)

    监控训练过程的性能分析回调函数。

    参数：
        - **start_step** (int) - 启动profiling的训练步数。默认值： ``1`` 。
        - **stop_step** (int) - 停止profiling的训练步数。默认值： ``10`` 。
        - **output_path** (str) - 保存profiling生成文件的文件夹路径。默认值： ``None`` 。
        - **start_profile** (str) - 是否打开profiling功能。默认值： ``True`` 。
        - **profile_communication** (str) - 在分布式训练期间是否收集通信性能数据。默认值： ``False`` 。
        - **profile_memory** (str) - 是否收集张量的内存数据。默认值： ``False`` 。
        - **profile_rank_ids** (list) - 指定rank ids开启profiling。默认值： ``None``，即该配置不生效，所有rank id均开启profiling。
        - **profile_pipeline** (str) - 是否按流水线并行每个stage的其中一张卡开启profiling。默认值： ``False`` 。
        - **profiler_level** (int) - 采集profiling数据的级别(0, 1, 2)。默认值： ``0`` 。
          - ``0`` - 最精简的采集性能数据级别，只采集计算类算子耗时数据和通信类大算子基础数据。
          - ``1`` - 在level0基础上，额外采集CANN层AscendCL数据、AICORE性能数据以及通信类小算子数据。
          - ``2`` - 在level1基础上，额外采集CANN层中图编译等级为O2和Runtime数据。
        - **with_stack** (str) - 是否收集Python侧的调用栈数据。默认值： ``False`` 。
        - **data_simplification** (str) - 是否开启数据精简，开启后将在导出profiling数据后删除FRAMEWORK目录以及其他多余数据。默认值： ``True`` 。
        - **config** (dict) - 配置项，用于对相关配置信息进行profiling，比如并行配置。默认值： ``None`` 。