mindformers.dataset.CausalLanguageModelDataset
==============================================

.. py:class:: mindformers.dataset.CausalLanguageModelDataset(dataset_config: Optional[dict] = None, data_loader: Union[dict, Callable] = None, input_columns: List[str] = None, output_columns: List[str] = None, batch_size: int = 8, drop_remainder: bool = True, num_parallel_workers: int = 8, python_multiprocessing: bool = False, repeat: int = 1, seed: int = 0, prefetch_size: int = 1, numa_enable: bool = False, eod_reset: bool = False, eod_token_id: Optional[int] = None, auto_tune: bool = False, filepath_prefix: str = './autotune', autotune_per_step: int = 10, profile: bool = False, **kwargs)

    因果语言模型预训练数据集。
    生成的数据集的输出列取决于用户提供的配置信息。 输出列均会被转换为int32类型。

    参数：
        - **dataset_config** (dict, 可选) - 数据集配置信息。当传入的 `dataset_config` 为空字典或 ``None`` 时， `dataset_config` 将由以下所有参数生成；否则以下所有参数被忽略。默认值： ``None`` 。
        - **data_loader** (Union[dict, Callable]) - 必须是包含data loader配置信息的字典，或一个data loader实例。当 `data_loader` 为 `dict` 类型时，字典的键可以是"type"、"dataset_dir"、"dataset_files"和"shuffle"。

          - ``"type"`` - 必选。数据集的类型。值必须是 `str` 或 `type` 类型。当值为"MindDataset"或"TFRecordDataset"时， ``"dataset_dir"`` 与 ``"dataset_files"`` 中必选两者之一，优先使用 ``"dataset_dir"`` ；否则必选 ``"dataset_dir"`` 。
          - ``"dataset_dir"`` - 数据集文件所在路径或目录。当 ``"type"`` 为"MindDataset"或"TFRecordDataset"且 ``"dataset_dir"`` 表示一个目录时，将递归查找目录下所有 `mindrecord` 或 `tfrecord` 格式文件。
          - ``"dataset_files"`` -  `mindrecord` 或 `tfrecord` 格式文件所在路径。当 ``"type"`` 为"MindDataset"或"TFRecordDataset"时生效；否则键被忽略。必须是 `list` 或 `tuple` 类型。
          - ``"shuffle"`` - 可选。指示是否混洗数据集。必须是 `bool` 类型。

        - **input_columns** (list[str]) - 表示映射处理前的数据列名称。
        - **output_columns** (list[str]) - 表示映射处理后的数据列名称。 `eod_reset` 为真时必选；否则该参数被忽略。默认值： ``None`` 。
        - **batch_size** (int) - 每个批次的大小。默认值： ``8`` 。
        - **drop_remainder** (bool) - 是否在最后一个批次的数据项数小于批次大小时，丢弃最后一个批次。默认值： ``True`` 。
        - **num_parallel_workers** (int) - 并行执行数据映射处理的进程/线程数。默认值： ``8`` 。
        - **python_multiprocessing** (bool) - 是否启用Python的Multi-Process模块以加速映射操作。默认值：``False`` 。
        - **repeat** (int) - 数据集重复的次数。默认值： ``1`` 。
        - **seed** (int) - 随机数种子。默认值： ``0`` 。
        - **prefetch_size** (int) - 流水线中每个数据处理操作的缓存队列大小。默认值： ``1`` 。
        - **numa_enable** (bool) - 是否采用NUMA绑定函数。默认值： ``False`` 。
        - **eod_reset** (bool) - 是否重置<EOD>词元。默认值： ``False`` 。
        - **eod_token_id** (int, 可选) - <EOD>词元对应的id。默认值： ``None`` ，表示不手动设置<EOD>词元对应的id。
        - **auto_tune** (bool) - 是否启用数据处理参数自动优化。默认值： ``False`` 。
        - **autotune_per_step** (int) - 设置调整自动数据加速配置步骤的间隔。默认值： ``10`` 。
        - **filepath_prefix** (str) - 保存优化参数配置的路径。默认值： ``'./autotune'`` 。
        - **profile** (bool) - 是否启用数据收集。默认值： ``False`` 。

    返回：
        `CausalLanguageModelDataset` 实例。

    异常：
        - **ValueError** -  当 `dataset_config.eod_reset` 为 ``True`` 且未全量导入数据集时， `dataset_config.batch_size` 不是使用设备数量的整数倍。
        - **ValueError** -  `dataset_config.data_loader` 中没有指定键 `"dataset_dir"` 或 `"dataset_files"` 。
