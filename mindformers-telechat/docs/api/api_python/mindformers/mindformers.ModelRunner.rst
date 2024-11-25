mindformers.ModelRunner
=====================================

.. py:class:: mindformers.ModelRunner(model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1, npu_device_ids=None, plugin_params=None)

    用于将 MindFormers 的一个模型运行实例作为 MindIEServer 的后端。

    参数：
        - **model_path** (str) - 包含模型配置文件（yaml 文件，tokenizer 文件）的模型路径。
        - **npu_mem_size** (int) - kv-cache 的 NPU 内存大小。
        - **cpu_mem_size** (int) - kv-cache 的 CPU 内存大小。
        - **block_size** (int) - kv-cache 的块大小。
        - **rank_id** (int, 可选) - 用于推理的 rank ID。默认值:  ``0`` 。
        - **world_size** (int, 可选) - 用于推理的 rank 数量。默认值:  ``1`` 。
        - **npu_device_ids** (list[int], 可选) - 从 MindIE 配置中获取的 NPU 设备 ID 列表。默认值:  ``None`` 。
        - **plugin_params** (str, 可选) - 包含额外插件参数的 JSON 字符串。默认值:  ``None`` 。

    返回：
        `MindIEModelRunner` 实例。